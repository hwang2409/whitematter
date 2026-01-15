#include "loss.h"
#include <cmath>

TensorPtr MSELoss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    assert(prediction->data.size() == target->data.size());

    auto diff = prediction->sub(target);
    auto sq = diff->mul(diff);
    return sq->mean();
}

TensorPtr L1Loss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    // L1 Loss (Mean Absolute Error): mean(|prediction - target|)
    assert(prediction->data.size() == target->data.size());

    auto diff = prediction->sub(target);
    auto abs_diff = diff->abs();
    return abs_diff->mean();
}

TensorPtr SmoothL1Loss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    // Smooth L1 Loss (Huber Loss):
    // If |x| < beta: 0.5 * x^2 / beta
    // Otherwise: |x| - 0.5 * beta
    // Where x = prediction - target
    assert(prediction->data.size() == target->data.size());

    size_t n = prediction->data.size();
    auto result = Tensor::create({1}, prediction->requires_grad);
    result->data[0] = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float diff = prediction->data[i] - target->data[i];
        float abs_diff = std::fabs(diff);
        if (abs_diff < beta) {
            result->data[0] += 0.5f * diff * diff / beta;
        } else {
            result->data[0] += abs_diff - 0.5f * beta;
        }
    }
    result->data[0] /= static_cast<float>(n);

    if (result->requires_grad) {
        auto pred_ptr = prediction;
        auto target_ptr = target;
        float beta_val = beta;
        result->parents = {pred_ptr};
        result->grad_fn = [pred_ptr, target_ptr, result, n, beta_val]() {
            float scale = result->grad[0] / static_cast<float>(n);
            for (size_t i = 0; i < n; i++) {
                float diff = pred_ptr->data[i] - target_ptr->data[i];
                float abs_diff = std::fabs(diff);
                if (abs_diff < beta_val) {
                    // Gradient: x / beta
                    pred_ptr->grad[i] += scale * diff / beta_val;
                } else {
                    // Gradient: sign(x)
                    pred_ptr->grad[i] += scale * (diff > 0 ? 1.0f : -1.0f);
                }
            }
        };
    }

    return result;
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

TensorPtr BCELoss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    // Binary Cross Entropy: -[y * log(p) + (1-y) * log(1-p)]
    // Prediction should already be probabilities (e.g., after sigmoid)
    assert(prediction->data.size() == target->data.size());

    size_t n = prediction->data.size();
    float eps = 1e-7f;  // For numerical stability

    auto result = Tensor::create({1}, prediction->requires_grad);
    result->data[0] = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float p = std::max(std::min(prediction->data[i], 1.0f - eps), eps);
        float y = target->data[i];
        result->data[0] -= y * std::log(p) + (1.0f - y) * std::log(1.0f - p);
    }
    result->data[0] /= static_cast<float>(n);

    if (result->requires_grad) {
        auto pred_ptr = prediction;
        auto target_ptr = target;
        result->parents = {pred_ptr};
        result->grad_fn = [pred_ptr, target_ptr, result, n, eps]() {
            float scale = result->grad[0] / static_cast<float>(n);
            for (size_t i = 0; i < n; i++) {
                float p = std::max(std::min(pred_ptr->data[i], 1.0f - eps), eps);
                float y = target_ptr->data[i];
                // d/dp[-y*log(p) - (1-y)*log(1-p)] = -y/p + (1-y)/(1-p)
                pred_ptr->grad[i] += scale * (-y / p + (1.0f - y) / (1.0f - p));
            }
        };
    }

    return result;
}

TensorPtr BCEWithLogitsLoss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    // Numerically stable BCE with built-in sigmoid
    // loss = max(x, 0) - x*y + log(1 + exp(-|x|))
    assert(prediction->data.size() == target->data.size());

    size_t n = prediction->data.size();

    auto result = Tensor::create({1}, prediction->requires_grad);
    result->data[0] = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float x = prediction->data[i];
        float y = target->data[i];
        // Numerically stable formulation
        float max_val = std::max(x, 0.0f);
        result->data[0] += max_val - x * y + std::log(1.0f + std::exp(-std::abs(x)));
    }
    result->data[0] /= static_cast<float>(n);

    if (result->requires_grad) {
        auto pred_ptr = prediction;
        auto target_ptr = target;
        result->parents = {pred_ptr};
        result->grad_fn = [pred_ptr, target_ptr, result, n]() {
            float scale = result->grad[0] / static_cast<float>(n);
            for (size_t i = 0; i < n; i++) {
                float x = pred_ptr->data[i];
                float y = target_ptr->data[i];
                // Gradient: sigmoid(x) - y
                float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
                pred_ptr->grad[i] += scale * (sigmoid_x - y);
            }
        };
    }

    return result;
}

TensorPtr KLDivLoss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    // KL Divergence: KL(target || prediction)
    // prediction: log probabilities (e.g., output of log_softmax)
    // target: probabilities (or log probabilities if log_target=true)
    // loss = sum(target * (log(target) - prediction)) / batch_size
    assert(prediction->data.size() == target->data.size());
    assert(prediction->shape.size() >= 1);

    size_t n = prediction->data.size();
    size_t batch_size = prediction->shape[0];

    auto result = Tensor::create({1}, prediction->requires_grad);
    result->data[0] = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float t = target->data[i];
        float log_p = prediction->data[i];

        if (log_target) {
            // target is already log probabilities
            float log_t = t;
            float t_prob = std::exp(log_t);
            if (t_prob > 0) {
                result->data[0] += t_prob * (log_t - log_p);
            }
        } else {
            // target is probabilities
            if (t > 0) {
                result->data[0] += t * (std::log(t) - log_p);
            }
        }
    }
    result->data[0] /= static_cast<float>(batch_size);

    if (result->requires_grad) {
        auto pred_ptr = prediction;
        auto target_ptr = target;
        bool log_t = log_target;
        result->parents = {pred_ptr};
        result->grad_fn = [pred_ptr, target_ptr, result, n, batch_size, log_t]() {
            float scale = result->grad[0] / static_cast<float>(batch_size);
            for (size_t i = 0; i < n; i++) {
                float t = target_ptr->data[i];
                // Gradient: -target (or -exp(target) if log_target)
                if (log_t) {
                    pred_ptr->grad[i] += scale * (-std::exp(t));
                } else {
                    pred_ptr->grad[i] += scale * (-t);
                }
            }
        };
    }

    return result;
}

TensorPtr FocalLoss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    // Focal Loss for multi-class classification
    // FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    // Reduces impact of easy examples, focuses on hard ones
    assert(prediction->shape.size() == 2);
    assert(target->shape.size() == 1 || (target->shape.size() == 2 && target->shape[1] == 1));

    size_t batch_size = prediction->shape[0];
    size_t num_classes = prediction->shape[1];

    // Compute softmax probabilities
    auto probs = prediction->softmax(-1);

    auto result = Tensor::create({1}, prediction->requires_grad);
    result->data[0] = 0.0f;

    float eps = 1e-7f;
    for (size_t i = 0; i < batch_size; i++) {
        size_t label = static_cast<size_t>(target->data[i]);
        float p_t = std::max(probs->data[i * num_classes + label], eps);
        float focal_weight = std::pow(1.0f - p_t, gamma);
        float loss_i = -focal_weight * std::log(p_t);
        if (alpha >= 0) {
            loss_i *= alpha;
        }
        result->data[0] += loss_i;
    }
    result->data[0] /= static_cast<float>(batch_size);

    if (result->requires_grad) {
        auto pred_ptr = prediction;
        auto probs_ptr = probs;
        auto target_ptr = target;
        float gamma_val = gamma;
        float alpha_val = alpha;
        result->parents = {pred_ptr};
        result->grad_fn = [pred_ptr, probs_ptr, target_ptr, result, batch_size, num_classes, gamma_val, alpha_val, eps]() {
            float scale = result->grad[0] / static_cast<float>(batch_size);
            for (size_t i = 0; i < batch_size; i++) {
                size_t label = static_cast<size_t>(target_ptr->data[i]);
                float p_t = std::max(probs_ptr->data[i * num_classes + label], eps);
                float one_minus_pt = 1.0f - p_t;

                for (size_t c = 0; c < num_classes; c++) {
                    float p_c = probs_ptr->data[i * num_classes + c];
                    float grad_val;

                    if (c == label) {
                        // Gradient for correct class
                        // d/dx_c FL = p_c * (gamma * (1-p_t)^(gamma-1) * p_t * log(p_t) - (1-p_t)^gamma)
                        //           = (1-p_t)^(gamma-1) * p_t * (gamma * log(p_t) + p_t - 1)
                        float term1 = gamma_val * std::log(p_t) * std::pow(one_minus_pt, gamma_val);
                        float term2 = -std::pow(one_minus_pt, gamma_val);
                        grad_val = p_c * (term1 + term2) + std::pow(one_minus_pt, gamma_val);
                    } else {
                        // Gradient for incorrect class
                        // Uses softmax Jacobian: d(p_t)/d(x_c) = -p_t * p_c for c != t
                        float focal_term = std::pow(one_minus_pt, gamma_val - 1.0f);
                        grad_val = -p_c * (focal_term * (gamma_val * p_t * std::log(p_t) + one_minus_pt));
                    }

                    if (alpha_val >= 0) {
                        grad_val *= alpha_val;
                    }
                    pred_ptr->grad[i * num_classes + c] += scale * grad_val;
                }
            }
        };
    }

    return result;
}

TensorPtr BinaryFocalLoss::forward(const TensorPtr& prediction, const TensorPtr& target) {
    // Binary Focal Loss
    // FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    // prediction: logits (will apply sigmoid internally)
    // target: binary labels (0 or 1)
    assert(prediction->data.size() == target->data.size());

    size_t n = prediction->data.size();
    float eps = 1e-7f;

    auto result = Tensor::create({1}, prediction->requires_grad);
    result->data[0] = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float x = prediction->data[i];
        float y = target->data[i];

        // Numerically stable sigmoid
        float p = 1.0f / (1.0f + std::exp(-x));
        p = std::max(std::min(p, 1.0f - eps), eps);

        // p_t = p if y=1, (1-p) if y=0
        float p_t = y * p + (1.0f - y) * (1.0f - p);
        float focal_weight = std::pow(1.0f - p_t, gamma);

        // alpha_t = alpha if y=1, (1-alpha) if y=0
        float alpha_t = 1.0f;
        if (alpha >= 0) {
            alpha_t = y * alpha + (1.0f - y) * (1.0f - alpha);
        }

        result->data[0] -= alpha_t * focal_weight * std::log(p_t);
    }
    result->data[0] /= static_cast<float>(n);

    if (result->requires_grad) {
        auto pred_ptr = prediction;
        auto target_ptr = target;
        float gamma_val = gamma;
        float alpha_val = alpha;
        result->parents = {pred_ptr};
        result->grad_fn = [pred_ptr, target_ptr, result, n, gamma_val, alpha_val, eps]() {
            float scale = result->grad[0] / static_cast<float>(n);
            for (size_t i = 0; i < n; i++) {
                float x = pred_ptr->data[i];
                float y = target_ptr->data[i];

                float p = 1.0f / (1.0f + std::exp(-x));
                p = std::max(std::min(p, 1.0f - eps), eps);

                float p_t = y * p + (1.0f - y) * (1.0f - p);
                float one_minus_pt = 1.0f - p_t;

                float alpha_t = 1.0f;
                if (alpha_val >= 0) {
                    alpha_t = y * alpha_val + (1.0f - y) * (1.0f - alpha_val);
                }

                // Gradient of focal loss w.r.t. logit x
                // FL = -alpha_t * (1-p_t)^gamma * log(p_t)
                // d(FL)/dx = -alpha_t * d/dx[(1-p_t)^gamma * log(p_t)]
                // Using product rule and chain rule
                float dp_dx = p * (1.0f - p);  // sigmoid derivative
                float dp_t_dx = (2.0f * y - 1.0f) * dp_dx;  // p_t = p if y=1, (1-p) if y=0

                float focal_grad;
                if (gamma_val == 0) {
                    // Standard BCE gradient: d/dx[-alpha_t * log(p_t)] = -alpha_t / p_t * dp_t_dx
                    focal_grad = -alpha_t / p_t * dp_t_dx;
                } else {
                    // Full focal loss gradient
                    float log_pt = std::log(p_t);
                    // d/dx[(1-p_t)^gamma * log(p_t)]
                    // = (1-p_t)^gamma * (1/p_t) * dp_t_dx + log(p_t) * gamma * (1-p_t)^(gamma-1) * (-dp_t_dx)
                    // = dp_t_dx * (1-p_t)^(gamma-1) * [(1-p_t)/p_t - gamma * log(p_t)]
                    float bracket = one_minus_pt / p_t - gamma_val * log_pt;
                    focal_grad = -alpha_t * dp_t_dx * std::pow(one_minus_pt, gamma_val - 1.0f) * bracket;
                }

                pred_ptr->grad[i] += scale * focal_grad;
            }
        };
    }

    return result;
}
