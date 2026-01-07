#include "loss.h"
#include <cmath>

TensorPtr MSELoss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    assert(prediction->data.size() == target->data.size());

    auto diff = prediction->sub(target);
    auto sq = diff->mul(diff);
    return sq->mean();
}

TensorPtr CrossEntropyLoss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    assert(prediction->shape.size() == 2);
    assert(target->shape.size() == 1 || (target->shape.size() == 2 && target->shape[1] == 1));

    size_t batch_size = prediction->shape[0];
    size_t num_classes = prediction->shape[1];

    auto log_probs = prediction->log_softmax(-1);

    auto result = Tensor::create({1}, prediction->requires_grad);
    result->data[0] = 0.0f;

    for (size_t i = 0; i < batch_size; i++) {
        size_t label = static_cast<size_t>(target->data[i]);
        result->data[0] -= log_probs->data[i * num_classes + label];
    }
    result->data[0] /= static_cast<float>(batch_size);

    if (result->requires_grad) {
        result->parents = {log_probs};
        result->grad_fn = [log_probs, target, result, batch_size, num_classes]() {
            float scale = result->grad[0] / static_cast<float>(batch_size);
            for (size_t i = 0; i < batch_size; i++) {
                size_t label = static_cast<size_t>(target->data[i]);
                log_probs->grad[i * num_classes + label] -= scale;
            }
        };
    }

    return result;
}

TensorPtr NLLLoss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    assert(prediction->shape.size() == 2);
    assert(target->shape.size() == 1 || (target->shape.size() == 2 && target->shape[1] == 1));

    size_t batch_size = prediction->shape[0];
    size_t num_classes = prediction->shape[1];

    auto result = Tensor::create({1}, prediction->requires_grad);
    result->data[0] = 0.0f;

    for (size_t i = 0; i < batch_size; i++) {
        size_t label = static_cast<size_t>(target->data[i]);
        result->data[0] -= prediction->data[i * num_classes + label];
    }
    result->data[0] /= static_cast<float>(batch_size);

    if (result->requires_grad) {
        auto pred_ptr = prediction;
        result->parents = {pred_ptr};
        result->grad_fn = [pred_ptr, target, result, batch_size, num_classes]() {
            float scale = result->grad[0] / static_cast<float>(batch_size);
            for (size_t i = 0; i < batch_size; i++) {
                size_t label = static_cast<size_t>(target->data[i]);
                pred_ptr->grad[i * num_classes + label] -= scale;
            }
        };
    }

    return result;
}
