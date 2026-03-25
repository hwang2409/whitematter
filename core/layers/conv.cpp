#include "../layer.h"
#include <random>
#include <cmath>

static std::mt19937 conv_rng(123);

// Conv2d implementation
Conv2d::Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
               size_t stride, size_t padding)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_size(kernel_size), stride(stride), padding(padding) {
    // Kaiming/He initialization for ReLU
    float std = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    std::normal_distribution<float> dist(0.0f, std);

    weight = Tensor::create({out_channels, in_channels, kernel_size, kernel_size}, true);
    for (size_t i = 0; i < weight->size(); i++) weight->data()[i] = dist(conv_rng);

    bias = Tensor::zeros({out_channels}, true);
}

TensorPtr Conv2d::forward(const TensorPtr& input) {
    return input->conv2d(weight, bias, stride, padding);
}

std::vector<TensorPtr> Conv2d::parameters() {
    return {weight, bias};
}

// ConvTranspose2d implementation
ConvTranspose2d::ConvTranspose2d(size_t in_channels, size_t out_channels, size_t kernel_size,
                                   size_t stride, size_t padding, size_t output_padding)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_size(kernel_size), stride(stride), padding(padding),
      output_padding(output_padding) {
    // Kaiming/He initialization
    float std = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    std::normal_distribution<float> dist(0.0f, std);

    // Weight shape: [in_channels, out_channels, kernel_size, kernel_size]
    // Note: transposed compared to Conv2d
    weight = Tensor::create({in_channels, out_channels, kernel_size, kernel_size}, true);
    for (size_t i = 0; i < weight->size(); i++) weight->data()[i] = dist(conv_rng);

    bias = Tensor::zeros({out_channels}, true);
}

TensorPtr ConvTranspose2d::forward(const TensorPtr& input) {
    return input->conv_transpose2d(weight, bias, stride, padding, output_padding);
}

std::vector<TensorPtr> ConvTranspose2d::parameters() {
    return {weight, bias};
}

// MaxPool2d implementation
MaxPool2d::MaxPool2d(size_t kernel_size, size_t stride)
    : kernel_size(kernel_size), stride(stride == 0 ? kernel_size : stride) {}

TensorPtr MaxPool2d::forward(const TensorPtr& input) {
    return input->maxpool2d(kernel_size, stride);
}

// AvgPool2d implementation
AvgPool2d::AvgPool2d(size_t kernel_size, size_t stride)
    : kernel_size(kernel_size), stride(stride == 0 ? kernel_size : stride) {}

TensorPtr AvgPool2d::forward(const TensorPtr& input) {
    return input->avgpool2d(kernel_size, stride);
}

std::string Conv2d::extra_repr() const {
    return std::to_string(in_channels) + ", " + std::to_string(out_channels) +
           ", kernel_size=" + std::to_string(kernel_size) +
           ", stride=" + std::to_string(stride) +
           ", padding=" + std::to_string(padding);
}

std::string ConvTranspose2d::extra_repr() const {
    return std::to_string(in_channels) + ", " + std::to_string(out_channels) +
           ", kernel_size=" + std::to_string(kernel_size) +
           ", stride=" + std::to_string(stride) +
           ", padding=" + std::to_string(padding) +
           ", output_padding=" + std::to_string(output_padding);
}

std::string MaxPool2d::extra_repr() const {
    return "kernel_size=" + std::to_string(kernel_size) +
           ", stride=" + std::to_string(stride);
}

std::string AvgPool2d::extra_repr() const {
    return "kernel_size=" + std::to_string(kernel_size) +
           ", stride=" + std::to_string(stride);
}

std::vector<size_t> Conv2d::compute_output_shape(const std::vector<size_t>& input_shape) const {
    // [N, C_in, H, W] -> [N, C_out, H', W']
    if (input_shape.size() != 4) return input_shape;
    size_t N = input_shape[0];
    size_t H = input_shape[2];
    size_t W = input_shape[3];
    size_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    size_t W_out = (W + 2 * padding - kernel_size) / stride + 1;
    return {N, out_channels, H_out, W_out};
}

std::vector<size_t> ConvTranspose2d::compute_output_shape(const std::vector<size_t>& input_shape) const {
    // [N, C_in, H, W] -> [N, C_out, H', W']
    // H_out = (H - 1) * stride - 2 * padding + kernel_size + output_padding
    if (input_shape.size() != 4) return input_shape;
    size_t N = input_shape[0];
    size_t H = input_shape[2];
    size_t W = input_shape[3];
    size_t H_out = (H - 1) * stride - 2 * padding + kernel_size + output_padding;
    size_t W_out = (W - 1) * stride - 2 * padding + kernel_size + output_padding;
    return {N, out_channels, H_out, W_out};
}

std::vector<size_t> MaxPool2d::compute_output_shape(const std::vector<size_t>& input_shape) const {
    // [N, C, H, W] -> [N, C, H', W']
    if (input_shape.size() != 4) return input_shape;
    size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H = input_shape[2];
    size_t W = input_shape[3];
    size_t H_out = (H - kernel_size) / stride + 1;
    size_t W_out = (W - kernel_size) / stride + 1;
    return {N, C, H_out, W_out};
}

std::vector<size_t> AvgPool2d::compute_output_shape(const std::vector<size_t>& input_shape) const {
    // [N, C, H, W] -> [N, C, H', W']
    if (input_shape.size() != 4) return input_shape;
    size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H = input_shape[2];
    size_t W = input_shape[3];
    size_t H_out = (H - kernel_size) / stride + 1;
    size_t W_out = (W - kernel_size) / stride + 1;
    return {N, C, H_out, W_out};
}
