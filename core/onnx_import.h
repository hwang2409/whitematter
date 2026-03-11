#ifndef ONNX_IMPORT_H
#define ONNX_IMPORT_H

#include "layer.h"
#include <memory>
#include <string>
#include <vector>

/// Load an ONNX model (subset of ops: Gemm, Conv, Relu, Sigmoid, Tanh, Softmax,
/// Flatten, MaxPool, AveragePool, BatchNormalization, Identity) into a Sequential.
/// Returns nullptr on failure (unsupported op, parse error, or file error).
std::unique_ptr<Sequential> load_onnx(const std::string& filepath);

/// Optional: get the expected input shape from the ONNX graph (e.g. {1, 1, 28, 28}).
/// Empty if not found or not yet parsed.
bool load_onnx_input_shape(const std::string& filepath, std::vector<size_t>& out_shape);

#endif
