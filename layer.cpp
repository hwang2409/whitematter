#include "layer.h"
#include <random>
#include <cmath>

static std::mt19937 layer_rng(123);

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
    std::vector<float> mask(input->data.size());

    for (size_t i = 0; i < input->data.size(); i++) {
        mask[i] = (dist(layer_rng) > p) ? scale : 0.0f;
        result->data[i] = input->data[i] * mask[i];
    }

    if (result->requires_grad) {
        result->parents = {input};
        result->grad_fn = [input, result, mask]() {
            for (size_t i = 0; i < input->data.size(); i++) {
                input->grad[i] += result->grad[i] * mask[i];
            }
        };
    }

    return result;
}

Sequential::Sequential(std::initializer_list<Module*> modules) {
    for (auto m : modules) {
        layers.push_back(m);
    }
}

Sequential::~Sequential() {
    for (auto m : layers) {
        delete m;
    }
}

void Sequential::add(Module* module) {
    layers.push_back(module);
}

TensorPtr Sequential::forward(const TensorPtr& input) {
    TensorPtr x = input;
    for (auto& layer : layers) {
        x = layer->forward(x);
    }
    return x;
}

std::vector<TensorPtr> Sequential::parameters() {
    std::vector<TensorPtr> params;
    for (auto& layer : layers) {
        auto layer_params = layer->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

void Sequential::train() {
    for (auto& layer : layers) {
        if (auto dropout = dynamic_cast<Dropout*>(layer)) {
            dropout->train();
        }
    }
}

void Sequential::eval() {
    for (auto& layer : layers) {
        if (auto dropout = dynamic_cast<Dropout*>(layer)) {
            dropout->eval();
        }
    }
}
