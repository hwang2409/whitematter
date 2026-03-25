#include "../layer.h"
#include <random>

static std::mt19937 activations_rng(123);

TensorPtr ReLU::forward(const TensorPtr& input) {
    return input->relu();
}

TensorPtr Sigmoid::forward(const TensorPtr& input) {
    return input->sigmoid();
}

TensorPtr Tanh::forward(const TensorPtr& input) {
    return input->tanh_();
}

Softmax::Softmax(int dim) : dim(dim) {}

TensorPtr Softmax::forward(const TensorPtr& input) {
    return input->softmax(dim);
}

LogSoftmax::LogSoftmax(int dim) : dim(dim) {}

TensorPtr LogSoftmax::forward(const TensorPtr& input) {
    return input->log_softmax(dim);
}

Dropout::Dropout(float p) : p(p), training(true) {}

TensorPtr Dropout::forward(const TensorPtr& input) {
    if (!training || p == 0.0f) {
        return input;
    }

    auto result = Tensor::create(input->shape, input->requires_grad);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float scale = 1.0f / (1.0f - p);
    std::vector<float> mask(input->size());

    for (size_t i = 0; i < input->size(); i++) {
        mask[i] = (dist(activations_rng) > p) ? scale : 0.0f;
        result->data()[i] = input->data()[i] * mask[i];
    }

    if (result->requires_grad) {
        result->parents = {input};
        result->grad_fn = [input, result, mask]() {
            for (size_t i = 0; i < input->size(); i++) {
                input->grad()[i] += result->grad()[i] * mask[i];
            }
        };
    }

    return result;
}

std::string Softmax::extra_repr() const {
    return "dim=" + std::to_string(dim);
}

std::string LogSoftmax::extra_repr() const {
    return "dim=" + std::to_string(dim);
}

std::string Dropout::extra_repr() const {
    return "p=" + std::to_string(p);
}

// Flatten implementation
TensorPtr Flatten::forward(const TensorPtr& input) {
    return input->flatten(1);
}

std::vector<size_t> Flatten::compute_output_shape(const std::vector<size_t>& input_shape) const {
    // [N, ...] -> [N, product of rest]
    if (input_shape.size() < 2) return input_shape;
    size_t N = input_shape[0];
    size_t flat_size = 1;
    for (size_t i = 1; i < input_shape.size(); i++) {
        flat_size *= input_shape[i];
    }
    return {N, flat_size};
}
