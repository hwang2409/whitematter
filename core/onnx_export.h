#ifndef ONNX_EXPORT_H
#define ONNX_EXPORT_H

#include "layer.h"
#include <string>
#include <vector>

// ONNX opset version we target
constexpr int ONNX_OPSET_VERSION = 13;

// Export options
struct ONNXExportOptions {
    std::string model_name = "model";
    std::string producer_name = "whitematter";
    std::string producer_version = "1.0";
    std::vector<size_t> input_shape;  // Required: e.g., {1, 1, 28, 28} for MNIST CNN
    bool verbose = false;
};

// Export a Sequential model to ONNX format
// Returns true on success, false on failure
bool export_onnx(Sequential* model, const std::string& filepath, const ONNXExportOptions& options);

// Convenience overload with just input shape
bool export_onnx(Sequential* model, const std::string& filepath, const std::vector<size_t>& input_shape);

// Get human-readable export info (for debugging)
std::string get_onnx_export_info(Sequential* model, const std::vector<size_t>& input_shape);

#endif
