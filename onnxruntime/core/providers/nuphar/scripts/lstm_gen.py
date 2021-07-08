import os

import onnx

from onnxruntime.nuphar.model_editor import convert_to_scan_model
from onnxruntime.nuphar.model_quantizer import convert_matmul_model
from onnxruntime.nuphar.rnn_benchmark import generate_model
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference


def generate_lstm(model_name, **kwargs):
    generate_model("lstm", model_name=model_name, **kwargs)

    scan_model_name = os.path.splitext(model_name)[0] + '_scan.onnx'

    convert_to_scan_model(model_name, scan_model_name)
    # note that symbolic shape inference is needed because model has symbolic batch dim, thus init_state is ConstantOfShape
    onnx.save(SymbolicShapeInference.infer_shapes(onnx.load(scan_model_name)), scan_model_name)

    int8_model_name = os.path.splitext(model_name)[0] + '_int8.onnx'
    convert_matmul_model(scan_model_name, int8_model_name)
    onnx.save(SymbolicShapeInference.infer_shapes(onnx.load(int8_model_name)), int8_model_name)

    model_quant = os.path.splitext(model_name)[0] + '_quant.onnx'
    quantized_model = quantize_dynamic(model_name, model_quant, weight_type=QuantType.QUInt8)


if __name__ == "__main__":
    generate_lstm("BLSTM_i64_h128_l2.onnx", input_dim=64, hidden_dim=128, bidirectional=True, layers=2)
    generate_lstm("LSTM_i256_h1024_l4.onnx", input_dim=256, hidden_dim=1024, bidirectional=False, layers=4)
