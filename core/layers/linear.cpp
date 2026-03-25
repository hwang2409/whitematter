#include "../layer.h"

Linear::Linear(size_t in_features, size_t out_features)
    : in_features(in_features), out_features(out_features) {
    weight = Tensor::xavier(in_features, out_features, true);
    bias = Tensor::zeros({out_features}, true);
}

TensorPtr Linear::forward(const TensorPtr& input) {
    return input->matmul(weight)->add(bias);
}

std::vector<TensorPtr> Linear::parameters() {
    return {weight, bias};
}

std::string Linear::extra_repr() const {
    return "in_features=" + std::to_string(in_features) +
           ", out_features=" + std::to_string(out_features);
}

std::vector<size_t> Linear::compute_output_shape(const std::vector<size_t>& input_shape) const {
    if (input_shape.empty()) return {};
    std::vector<size_t> output_shape = input_shape;
    output_shape.back() = out_features;
    return output_shape;
}
