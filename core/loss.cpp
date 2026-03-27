#include "loss.h"
#include "device.h"
#include <cmath>

TensorPtr MSELoss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    assert(prediction->size() == target->size());

    auto diff = prediction->sub(target);
    auto sq = diff->mul(diff);
    return sq->mean();
}

TensorPtr L1Loss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    assert(prediction->size() == target->size());

    auto diff = prediction->sub(target);
    auto abs_diff = diff->abs();
    return abs_diff->mean();
}

TensorPtr SmoothL1Loss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    assert(prediction->size() == target->size());

    size_t n = prediction->size();
    auto result = Tensor::create({1}, prediction->requires_grad);
    result->data()[0] = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float diff = prediction->data()[i] - target->data()[i];
        float abs_diff = std::fabs(diff);
        if (abs_diff < beta) {
            result->data()[0] += 0.5f * diff * diff / beta;
        } else {
            result->data()[0] += abs_diff - 0.5f * beta;
        }
    }
    result->data()[0] /= static_cast<float>(n);

    if (result->requires_grad) {
        auto pred_ptr = prediction;
        auto target_ptr = target;
        float beta_val = beta;
        result->parents = {pred_ptr};
        result->grad_fn = [pred_ptr, target_ptr, result, n, beta_val]() {
            float scale = result->grad()[0] / static_cast<float>(n);
            for (size_t i = 0; i < n; i++) {
                float diff = pred_ptr->data()[i] - target_ptr->data()[i];
                float abs_diff = std::fabs(diff);
                if (abs_diff < beta_val) {
                    pred_ptr->grad()[i] += scale * diff / beta_val;
                } else {
                    pred_ptr->grad()[i] += scale * (diff > 0 ? 1.0f : -1.0f);
                }
            }
        };
    }

    return result;
}

TensorPtr CrossEntropyLoss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    assert(prediction->shape.size() == 2);
    assert(target->shape.size() == 1 || (target->shape.size() == 2 && target->shape[1] == 1));

    // CPU fallback for non-CPU tensors
    if (prediction->device != whitematter::DeviceType::CPU) {
        auto cpu_pred = prediction->to(whitematter::DeviceType::CPU);
        auto cpu_tgt = target->to(whitematter::DeviceType::CPU);
        auto cpu_result = forward(cpu_pred, cpu_tgt);
        cpu_result->to_inplace(prediction->device);
        return cpu_result;
    }

    size_t batch_size = prediction->shape[0];
    size_t num_classes = prediction->shape[1];

    auto log_probs = prediction->log_softmax(-1);

    auto result = Tensor::create({1}, prediction->requires_grad);
    result->data()[0] = 0.0f;

    for (size_t i = 0; i < batch_size; i++) {
        size_t label = static_cast<size_t>(target->data()[i]);
        result->data()[0] -= log_probs->data()[i * num_classes + label];
    }
    result->data()[0] /= static_cast<float>(batch_size);

    if (result->requires_grad) {
        result->parents = {log_probs};
        result->grad_fn = [log_probs, target, result, batch_size, num_classes]() {
            float scale = result->grad()[0] / static_cast<float>(batch_size);
            for (size_t i = 0; i < batch_size; i++) {
                size_t label = static_cast<size_t>(target->data()[i]);
                log_probs->grad()[i * num_classes + label] -= scale;
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
    result->data()[0] = 0.0f;

    for (size_t i = 0; i < batch_size; i++) {
        size_t label = static_cast<size_t>(target->data()[i]);
        result->data()[0] -= prediction->data()[i * num_classes + label];
    }
    result->data()[0] /= static_cast<float>(batch_size);

    if (result->requires_grad) {
        auto pred_ptr = prediction;
        result->parents = {pred_ptr};
        result->grad_fn = [pred_ptr, target, result, batch_size, num_classes]() {
            float scale = result->grad()[0] / static_cast<float>(batch_size);
            for (size_t i = 0; i < batch_size; i++) {
                size_t label = static_cast<size_t>(target->data()[i]);
                pred_ptr->grad()[i * num_classes + label] -= scale;
            }
        };
    }

    return result;
}

TensorPtr BCELoss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    assert(prediction->size() == target->size());

    size_t n = prediction->size();
    float eps = 1e-7f;

    auto result = Tensor::create({1}, prediction->requires_grad);
    result->data()[0] = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float p = std::max(std::min(prediction->data()[i], 1.0f - eps), eps);
        float y = target->data()[i];
        result->data()[0] -= y * std::log(p) + (1.0f - y) * std::log(1.0f - p);
    }
    result->data()[0] /= static_cast<float>(n);

    if (result->requires_grad) {
        auto pred_ptr = prediction;
        auto target_ptr = target;
        result->parents = {pred_ptr};
        result->grad_fn = [pred_ptr, target_ptr, result, n, eps]() {
            float scale = result->grad()[0] / static_cast<float>(n);
            for (size_t i = 0; i < n; i++) {
                float p = std::max(std::min(pred_ptr->data()[i], 1.0f - eps), eps);
                float y = target_ptr->data()[i];
                pred_ptr->grad()[i] += scale * (-y / p + (1.0f - y) / (1.0f - p));
            }
        };
    }

    return result;
}

TensorPtr BCEWithLogitsLoss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    assert(prediction->size() == target->size());

    size_t n = prediction->size();

    auto result = Tensor::create({1}, prediction->requires_grad);
    result->data()[0] = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float x = prediction->data()[i];
        float y = target->data()[i];
        float max_val = std::max(x, 0.0f);
        result->data()[0] += max_val - x * y + std::log(1.0f + std::exp(-std::abs(x)));
    }
    result->data()[0] /= static_cast<float>(n);

    if (result->requires_grad) {
        auto pred_ptr = prediction;
        auto target_ptr = target;
        result->parents = {pred_ptr};
        result->grad_fn = [pred_ptr, target_ptr, result, n]() {
            float scale = result->grad()[0] / static_cast<float>(n);
            for (size_t i = 0; i < n; i++) {
                float x = pred_ptr->data()[i];
                float y = target_ptr->data()[i];
                float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
                pred_ptr->grad()[i] += scale * (sigmoid_x - y);
            }
        };
    }

    return result;
}

TensorPtr KLDivLoss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    assert(prediction->size() == target->size());
    assert(prediction->shape.size() >= 1);

    size_t n = prediction->size();
    size_t batch_size = prediction->shape[0];

    auto result = Tensor::create({1}, prediction->requires_grad);
    result->data()[0] = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float t = target->data()[i];
        float log_p = prediction->data()[i];

        if (log_target) {
            float log_t = t;
            float t_prob = std::exp(log_t);
            if (t_prob > 0) {
                result->data()[0] += t_prob * (log_t - log_p);
            }
        } else {
            if (t > 0) {
                result->data()[0] += t * (std::log(t) - log_p);
            }
        }
    }
    result->data()[0] /= static_cast<float>(batch_size);

    if (result->requires_grad) {
        auto pred_ptr = prediction;
        auto target_ptr = target;
        bool log_t = log_target;
        result->parents = {pred_ptr};
        result->grad_fn = [pred_ptr, target_ptr, result, n, batch_size, log_t]() {
            float scale = result->grad()[0] / static_cast<float>(batch_size);
            for (size_t i = 0; i < n; i++) {
                float t = target_ptr->data()[i];
                if (log_t) {
                    pred_ptr->grad()[i] += scale * (-std::exp(t));
                } else {
                    pred_ptr->grad()[i] += scale * (-t);
                }
            }
        };
    }

    return result;
}

TensorPtr FocalLoss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    assert(prediction->shape.size() == 2);
    assert(target->shape.size() == 1 || (target->shape.size() == 2 && target->shape[1] == 1));

    size_t batch_size = prediction->shape[0];
    size_t num_classes = prediction->shape[1];

    auto probs = prediction->softmax(-1);

    auto result = Tensor::create({1}, prediction->requires_grad);
    result->data()[0] = 0.0f;

    float eps = 1e-7f;
    for (size_t i = 0; i < batch_size; i++) {
        size_t label = static_cast<size_t>(target->data()[i]);
        float p_t = std::max(probs->data()[i * num_classes + label], eps);
        float focal_weight = std::pow(1.0f - p_t, gamma);
        float loss_i = -focal_weight * std::log(p_t);
        if (alpha >= 0) {
            loss_i *= alpha;
        }
        result->data()[0] += loss_i;
    }
    result->data()[0] /= static_cast<float>(batch_size);

    if (result->requires_grad) {
        auto pred_ptr = prediction;
        auto probs_ptr = probs;
        auto target_ptr = target;
        float gamma_val = gamma;
        float alpha_val = alpha;
        result->parents = {pred_ptr};
        result->grad_fn = [pred_ptr, probs_ptr, target_ptr, result, batch_size, num_classes, gamma_val, alpha_val, eps]() {
            float scale = result->grad()[0] / static_cast<float>(batch_size);
            for (size_t i = 0; i < batch_size; i++) {
                size_t label = static_cast<size_t>(target_ptr->data()[i]);
                float p_t = std::max(probs_ptr->data()[i * num_classes + label], eps);
                float one_minus_pt = 1.0f - p_t;

                for (size_t c = 0; c < num_classes; c++) {
                    float p_c = probs_ptr->data()[i * num_classes + c];
                    float grad_val;

                    if (c == label) {
                        float term1 = gamma_val * std::log(p_t) * std::pow(one_minus_pt, gamma_val);
                        float term2 = -std::pow(one_minus_pt, gamma_val);
                        grad_val = p_c * (term1 + term2) + std::pow(one_minus_pt, gamma_val);
                    } else {
                        float focal_term = std::pow(one_minus_pt, gamma_val - 1.0f);
                        grad_val = -p_c * (focal_term * (gamma_val * p_t * std::log(p_t) + one_minus_pt));
                    }

                    if (alpha_val >= 0) {
                        grad_val *= alpha_val;
                    }
                    pred_ptr->grad()[i * num_classes + c] += scale * grad_val;
                }
            }
        };
    }

    return result;
}

TensorPtr BinaryFocalLoss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    assert(prediction->size() == target->size());

    size_t n = prediction->size();
    float eps = 1e-7f;

    auto result = Tensor::create({1}, prediction->requires_grad);
    result->data()[0] = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float x = prediction->data()[i];
        float y = target->data()[i];

        float p = 1.0f / (1.0f + std::exp(-x));
        p = std::max(std::min(p, 1.0f - eps), eps);

        float p_t = y * p + (1.0f - y) * (1.0f - p);
        float focal_weight = std::pow(1.0f - p_t, gamma);

        float alpha_t = 1.0f;
        if (alpha >= 0) {
            alpha_t = y * alpha + (1.0f - y) * (1.0f - alpha);
        }

        result->data()[0] -= alpha_t * focal_weight * std::log(p_t);
    }
    result->data()[0] /= static_cast<float>(n);

    if (result->requires_grad) {
        auto pred_ptr = prediction;
        auto target_ptr = target;
        float gamma_val = gamma;
        float alpha_val = alpha;
        result->parents = {pred_ptr};
        result->grad_fn = [pred_ptr, target_ptr, result, n, gamma_val, alpha_val, eps]() {
            float scale = result->grad()[0] / static_cast<float>(n);
            for (size_t i = 0; i < n; i++) {
                float x = pred_ptr->data()[i];
                float y = target_ptr->data()[i];

                float p = 1.0f / (1.0f + std::exp(-x));
                p = std::max(std::min(p, 1.0f - eps), eps);

                float p_t = y * p + (1.0f - y) * (1.0f - p);
                float one_minus_pt = 1.0f - p_t;

                float alpha_t = 1.0f;
                if (alpha_val >= 0) {
                    alpha_t = y * alpha_val + (1.0f - y) * (1.0f - alpha_val);
                }

                float dp_dx = p * (1.0f - p);
                float dp_t_dx = (2.0f * y - 1.0f) * dp_dx;

                float focal_grad;
                if (gamma_val == 0) {
                    focal_grad = -alpha_t / p_t * dp_t_dx;
                } else {
                    float log_pt = std::log(p_t);
                    float bracket = one_minus_pt / p_t - gamma_val * log_pt;
                    focal_grad = -alpha_t * dp_t_dx * std::pow(one_minus_pt, gamma_val - 1.0f) * bracket;
                }

                pred_ptr->grad()[i] += scale * focal_grad;
            }
        };
    }

    return result;
}
