import argparse
import sys
import os
import onnx
from onnx import numpy_helper
from onnx import onnx_pb as onnx_proto

BIGGEST_KNOWN_VERSION = 13
def _attribute_to_pair(attribute):
    '''
    Convert attribute to kwarg format for use with onnx.helper.make_node.
        :parameter attribute: attribute in AttributeProto format.
        :return: attribute in {key: value} format.
    '''
    if (attribute.type == 0):
        raise ValueError('attribute {} does not have type specified.'.format(attribute.name))

    # Based on attribute type definitions from AttributeProto
    # definition in https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
    if (attribute.type == 1):
        value = attribute.f
    elif (attribute.type == 2):
        value = attribute.i
    elif (attribute.type == 3):
        value = attribute.s
    elif (attribute.type == 4):
        value = attribute.t
    elif (attribute.type == 5):
        value = attribute.g
    elif (attribute.type == 6):
        value = attribute.floats
    elif (attribute.type == 7):
        value = attribute.ints
    elif (attribute.type == 8):
        value = attribute.strings
    elif (attribute.type == 9):
        value = attribute.tensors
    elif (attribute.type == 10):
        value = attribute.graphs
    else:
        raise ValueError('attribute {} has unsupported type {}.'.format(attribute.name, attribute.type))

    return (attribute.name, value)

def PassThroughUpgrad(node, target_opset, all_new_nodes, all_new_initializers):
    return ([node], [])

def SliceV1(node, target_opset, all_new_nodes, all_new_initializers):
    '''
        return (new_nodes, new_initializers, ...)
    '''
    new_nodes = []
    new_initializers = []
    has_axes_attribute = False
    for attribute in node.attribute:
        if (attribute.type == 7):
            if attribute.name == 'axes':
                has_axes_attribute = True
            value = attribute.ints
            tensor_name = node.name + "_" + attribute.name
            initializer = onnx.helper.make_tensor(tensor_name, attribute.type, [len(value)], value)
            new_initializers.append(initializer)
        else:
            raise RuntimeError("Wrong attribute type for Slice node:{}".format(node.name))
    new_node_inputs = [node.input[0], node.name + "_starts", node.name + "_ends"]
    if has_axes_attribute:
        new_node_inputs.append(node.name + "_axes")
    new_nodes.append(onnx.helper.make_node("Slice", new_node_inputs, node.output, name = node.name))
    return (new_nodes, new_initializers)

def ResizeV10(node, target_opset, all_new_nodes, all_new_initializers):
    '''
        return (new_nodes, new_initializers, ...)
    '''
    new_nodes = []
    new_initializers = []

    roi_tensor_name = node.name + "_roi_input"
    initializer = onnx.helper.make_tensor(roi_tensor_name, data_type=onnx.TensorProto.FLOAT, dims=(8,), vals=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    new_initializers.append(initializer)
    #print("***************************{}".format(initializer))

    kwargs = {}
    for attr in node.attribute:
        (key, value) = _attribute_to_pair(attr)
        kwargs.update({key : value})
    nnode = onnx.helper.make_node("Resize", [node.input[0], roi_tensor_name, node.input[1]], node.output, name = node.name, **kwargs)
    #print("===========>{}".format(nnode))
    new_nodes.append(nnode)

    return (new_nodes, new_initializers)


class OpsetUpgrader:
    def __init__(self):
        self._upgrad_table = { }
        self._op_versions = { }
        self._op_versions['Slice'] = [1, 10, 11, 13]
        self.add_rule('Slice', [[1, 10, BIGGEST_KNOWN_VERSION, SliceV1]])
        self._op_versions['Resize'] = [10, 11, 13]
        self.add_rule('Resize', [[10, 11, BIGGEST_KNOWN_VERSION, ResizeV10]])

    def find_real_op_version(self, op_type, opset):
        if op_type in self._op_versions:
            real_versions = self._op_versions[op_type]
            for v in real_versions:
                if v <= opset:
                    return v
        return None

    def get_upgrad_method(self, op_type, src_opset, target_opset):
        real_version = self.find_real_op_version(op_type, src_opset)
        if real_version is None:
            return PassThroughUpgrad
        if op_type in self._upgrad_table:
            rules = self._upgrad_table[op_type]
            for rule in rules:
                [origin_version, target_start, target_last, method] = rule
                if origin_version > real_version:
                    return PassThroughUpgrad
                if origin_version == real_version and target_start <= target_opset and target_last >= target_opset:
                    return method
        return PassThroughUpgrad

    def add_rule(self, op_type, upgrad_rule):
        '''
            Add serials upgrad rules for op_type

            :param op_type: single string for oerator name, like 'Slice'
            :param upgrads: list of [origin_version, target_start, target_last, method],
                            ordered by real_version and target_start
        '''
        self._upgrad_table[op_type] = upgrad_rule

    def upgrade_graph(self, graph, source_opset, target_opset):
        '''
            return upgraded_graph
        '''
        all_new_nodes = []
        all_new_initializers = graph.initializer
        for node in graph.node:
            node = self.upgrad_subgraph_if_any(node, source_opset, target_opset)
            method = self.get_upgrad_method(node.op_type, source_opset, target_opset)
            (new_nodes, new_initializers) = method(node, target_opset, all_new_nodes, all_new_initializers)
            all_new_nodes.extend(new_nodes)
            all_new_initializers.extend(new_initializers)
        graph.ClearField('node')
        graph.node.extend(all_new_nodes)
        graph.ClearField('initializer')
        graph.initializer.extend(all_new_initializers)
        return graph

    def upgrad_subgraph_if_any(self, node, source_opset, target_opset):
        '''
            return new_node with subgraph upgraded (if any, otherwise, original node)
        '''
        graph_attrs = [ attr for attr in node.attribute if attr.type == 5 or attr.type == 10 ]
        if len(graph_attrs) == 0:
            return node
        kwargs = {}
        for attr in node.attribute:
            (key, value) = _attribute_to_pair(attr)
            if attr.type == 5:
                value = self.upgrade_graph(value, source_opset, target_opset)
            elif attr.type == 10:
                upgrades_sub_graphes = []
                for g in value:
                    upgrades_sub_graphes.append(self.upgrade_graph(g, source_opset, target_opset))
                value = applied_sub_graphes
            kwargs.update({key : value})
        return onnx.helper.make_node(node.op_type, node.input, node.output, name = node.name, **kwargs)

    def upgrade_model(self, model, target_opset):
        '''
            return new model
        '''

        opset_infos = [opset for opset in model.opset_import ]
        onnx_ai_opset_info = [ opset for opset in opset_infos if opset.domain == '' or opset.domain == "ai.onnx"]
        if len(onnx_ai_opset_info) != 1:
            raise ValueError("Model have zeor or more than one ai.onnx version")
        source_opset = onnx_ai_opset_info[0].version
        if source_opset >= target_opset:
            print("No need to upgrade from version {} to {}".format(source_opset, target_opset))
            return model
        opset_infos.remove(onnx_ai_opset_info[0])
        opset_infos.extend([onnx.helper.make_opsetid("ai.onnx", target_opset)])
        model.ClearField('opset_import')
        model.opset_import.extend(opset_infos)

        self.upgrade_graph(model.graph, source_opset, target_opset)

        #print('AFTER UPDATE VERSION, COPY MODEL.GRAPH.VALUE_INFO:', model.graph.value_info)

        return model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument("--opset", type=int, default=BIGGEST_KNOWN_VERSION, help="target opset version")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    upgrader = OpsetUpgrader()
    converted_model = upgrader.upgrade_model(onnx.load(args.input_model), args.opset)

    print(f"saving to {args.output_model}")
    onnx.save(converted_model, args.output_model)
