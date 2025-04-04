// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

namespace {
std::string_view GetMode(const NodeAttrHelper& helper) {
  // opset 16 used bilinear, nearest, bicubic
  // opset 20+ uses linear, nearest, cubic
  // bilinear is what CoreML uses, so prefer that
  // bicubic/cubic isn't supported
  static const std::string default_mode = "linear";  // static in case we ever return the default as a string_view
  const auto& mode = helper.Get("mode", default_mode);
  if (mode == "linear") {
    return "bilinear";
  }

  return mode;
}
}  // namespace

class GridSampleOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

Status GridSampleOpBuilder::AddToModelBuilderImpl([[maybe_unused]] ModelBuilder& model_builder,
                                                  [[maybe_unused]] const Node& node,
                                                  [[maybe_unused]] const logging::Logger& logger) const {
  using namespace CoreML::Specification::MILSpec;  // NOLINT
  // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.image_resizing.resample

  const auto input_defs = node.InputDefs();
  const auto output_defs = node.OutputDefs();

  // we already checked it and dtype must be existed.
  auto input_dtype = input_defs[0]->TypeAsProto()->tensor_type().elem_type();

  NodeAttrHelper helper(node);
  std::string mode{GetMode(helper)};  //  need a std::string for use in AddScalarConstant
  std::string padding_mode = helper.Get("padding_mode", "zeros");
  const bool align_corners = helper.Get("align_corners", 0);
  const std::string coordinates_mode = "normalized_minus_one_to_one";

  // adjust to coreml equivalents
  if (padding_mode == "zeros") {
    padding_mode = "constant";
  }

  auto op = model_builder.CreateOperation(node, "resample");
  AddOperationInput(*op, "x", input_defs[0]->Name());
  AddOperationInput(*op, "coordinates", input_defs[1]->Name());
  AddOperationInput(*op, "sampling_mode", model_builder.AddScalarConstant(op->type(), "sampling_mode", mode));
  AddOperationInput(*op, "padding_mode", model_builder.AddScalarConstant(op->type(), "padding_mode", padding_mode));
  if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    AddOperationInput(*op, "padding_value", model_builder.AddScalarConstant(op->type(), "padding_value", 0.0f));
  } else {
    AddOperationInput(*op, "padding_value", model_builder.AddScalarConstant(op->type(), "padding_value", MLFloat16(0.0f)));
  }
  AddOperationInput(*op, "coordinates_mode",
                    model_builder.AddScalarConstant(op->type(), "coordinates_mode", coordinates_mode));
  AddOperationInput(*op, "align_corners", model_builder.AddScalarConstant(op->type(), "align_corners", align_corners));

  AddOperationOutput(*op, *output_defs[0]);

  model_builder.AddOperation(std::move(op));
  return Status::OK();
}

bool GridSampleOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                            const logging::Logger& logger) const {
  if (!input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << "GridSample is not supported.";
    return false;
  }

  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    LOGS(logger, VERBOSE) << "GridSample: failed to get input shape";
    return false;
  }

  const auto input_rank = input_shape.size();
  if (input_rank != 4) {
    LOGS(logger, VERBOSE) << "GridSample only supports 4D input. Got:" << input_rank << "D";
    return false;
  }

  NodeAttrHelper helper(node);
  std::string_view mode = GetMode(helper);

  if (mode != "bilinear" && mode != "zeros") {
    LOGS(logger, VERBOSE) << "GridSample does not support mode of " << mode;
    return false;
  }

  // there is one combination of settings where the unit test fails.
  // The ORT unit test values are generated by pytorch so not clear if it's an issue with CoreML.
  // CoreML output is consistent for CPU and non-CPU at least.
  // Disabling until there's a use-case that requires this combination.
  const auto& padding_mode = helper.Get("padding_mode", "zeros");
  const bool align_corners = helper.Get("align_corners", 0);

  if (mode == "bilinear" && padding_mode == "reflection" && align_corners == false) {
    LOGS(logger, VERBOSE) << "GridSample does not support mode:" << mode << " padding_mode:" << padding_mode
                          << " align_corners:" << align_corners
                          << " currently due to output diffs that need to be investigated";
    return false;
  }

  return true;
}

void CreateGridSampleOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GridSampleOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
