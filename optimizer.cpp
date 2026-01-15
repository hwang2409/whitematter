#include "optimizer.h"
#include <cmath>

Optimizer::Optimizer(const std::vector<TensorPtr>& params, float lr)
    : params(params), lr(lr) {}

void Optimizer::zero_grad() {
    for (auto& p : params) {
        p->zero_grad();
    }
}

SGD::SGD(const std::vector<TensorPtr>& params, float lr, float momentum)
    : Optimizer(params, lr), momentum(momentum) {
    if (momentum > 0.0f) {
        for (const auto& p : params) {
            velocity.push_back(std::vector<float>(p->data.size(), 0.0f));
        }
    }
}

void SGD::step() {
    for (size_t i = 0; i < params.size(); i++) {
        auto& p = params[i];
        if (momentum > 0.0f) {
            for (size_t j = 0; j < p->data.size(); j++) {
                velocity[i][j] = momentum * velocity[i][j] + p->grad[j];
                p->data[j] -= lr * velocity[i][j];
            }
        } else {
            for (size_t j = 0; j < p->data.size(); j++) {
                p->data[j] -= lr * p->grad[j];
            }
        }
    }
}

Adam::Adam(const std::vector<TensorPtr>& params, float lr, float beta1, float beta2, float eps)
    : Optimizer(params, lr), beta1(beta1), beta2(beta2), eps(eps), t(0) {
    for (const auto& p : params) {
        m.push_back(std::vector<float>(p->data.size(), 0.0f));
        v.push_back(std::vector<float>(p->data.size(), 0.0f));
    }
}

void Adam::step() {
    t++;
    float bias_correction1 = 1.0f - std::pow(beta1, t);
    float bias_correction2 = 1.0f - std::pow(beta2, t);

    for (size_t i = 0; i < params.size(); i++) {
        auto& p = params[i];
        for (size_t j = 0; j < p->data.size(); j++) {
            float g = p->grad[j];

            m[i][j] = beta1 * m[i][j] + (1.0f - beta1) * g;
            v[i][j] = beta2 * v[i][j] + (1.0f - beta2) * g * g;

            float m_hat = m[i][j] / bias_correction1;
            float v_hat = v[i][j] / bias_correction2;

            p->data[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
}

AdamW::AdamW(const std::vector<TensorPtr>& params, float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(params, lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay), t(0) {
    for (const auto& p : params) {
        m.push_back(std::vector<float>(p->data.size(), 0.0f));
        v.push_back(std::vector<float>(p->data.size(), 0.0f));
    }
}

void AdamW::step() {
    t++;
    float bias_correction1 = 1.0f - std::pow(beta1, t);
    float bias_correction2 = 1.0f - std::pow(beta2, t);

    for (size_t i = 0; i < params.size(); i++) {
        auto& p = params[i];
        for (size_t j = 0; j < p->data.size(); j++) {
            float g = p->grad[j];

            // Update biased first and second moment estimates
            m[i][j] = beta1 * m[i][j] + (1.0f - beta1) * g;
            v[i][j] = beta2 * v[i][j] + (1.0f - beta2) * g * g;

            // Compute bias-corrected estimates
            float m_hat = m[i][j] / bias_correction1;
            float v_hat = v[i][j] / bias_correction2;

            // AdamW: decoupled weight decay (applied directly to params, not to gradients)
            p->data[j] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * p->data[j]);
        }
    }
}

RMSprop::RMSprop(const std::vector<TensorPtr>& params, float lr, float alpha, float eps, float momentum, float weight_decay)
    : Optimizer(params, lr), alpha(alpha), eps(eps), momentum(momentum), weight_decay(weight_decay) {
    for (const auto& p : params) {
        v.push_back(std::vector<float>(p->data.size(), 0.0f));
        if (momentum > 0.0f) {
            buffer.push_back(std::vector<float>(p->data.size(), 0.0f));
        }
    }
}

void RMSprop::step() {
    for (size_t i = 0; i < params.size(); i++) {
        auto& p = params[i];
        for (size_t j = 0; j < p->data.size(); j++) {
            float g = p->grad[j];

            // Apply weight decay
            if (weight_decay != 0.0f) {
                g += weight_decay * p->data[j];
            }

            // Update running average of squared gradients
            v[i][j] = alpha * v[i][j] + (1.0f - alpha) * g * g;

            float avg = std::sqrt(v[i][j]) + eps;

            if (momentum > 0.0f) {
                // With momentum
                buffer[i][j] = momentum * buffer[i][j] + g / avg;
                p->data[j] -= lr * buffer[i][j];
            } else {
                // Without momentum
                p->data[j] -= lr * g / avg;
            }
        }
    }
}

// ============================================================================
// Gradient Clipping
// ============================================================================

float get_grad_norm(const std::vector<TensorPtr>& params) {
    float total_norm = 0.0f;
    for (const auto& p : params) {
        for (size_t i = 0; i < p->grad.size(); i++) {
            total_norm += p->grad[i] * p->grad[i];
        }
    }
    return std::sqrt(total_norm);
}

void clip_grad_norm_(std::vector<TensorPtr>& params, float max_norm) {
    float total_norm = get_grad_norm(params);
    float clip_coef = max_norm / (total_norm + 1e-6f);
    if (clip_coef < 1.0f) {
        for (auto& p : params) {
            for (size_t i = 0; i < p->grad.size(); i++) {
                p->grad[i] *= clip_coef;
            }
        }
    }
}

void clip_grad_value_(std::vector<TensorPtr>& params, float clip_value) {
    for (auto& p : params) {
        for (size_t i = 0; i < p->grad.size(); i++) {
            if (p->grad[i] > clip_value) {
                p->grad[i] = clip_value;
            } else if (p->grad[i] < -clip_value) {
                p->grad[i] = -clip_value;
            }
        }
    }
}

// ============================================================================
// Learning Rate Schedulers
// ============================================================================

LRScheduler::LRScheduler(Optimizer* optimizer)
    : optimizer(optimizer), base_lr(optimizer->lr), last_epoch(-1) {}

void LRScheduler::step() {
    last_epoch++;
    optimizer->lr = get_lr();
}

// StepLR: decay by gamma every step_size epochs
StepLR::StepLR(Optimizer* optimizer, int step_size, float gamma)
    : LRScheduler(optimizer), step_size(step_size), gamma(gamma) {}

float StepLR::get_lr() {
    if (last_epoch == -1) return base_lr;
    return base_lr * std::pow(gamma, last_epoch / step_size);
}

// ExponentialLR: decay by gamma every epoch
ExponentialLR::ExponentialLR(Optimizer* optimizer, float gamma)
    : LRScheduler(optimizer), gamma(gamma) {}

float ExponentialLR::get_lr() {
    if (last_epoch == -1) return base_lr;
    return base_lr * std::pow(gamma, last_epoch);
}

// CosineAnnealingLR: cosine annealing from base_lr to eta_min
CosineAnnealingLR::CosineAnnealingLR(Optimizer* optimizer, int T_max, float eta_min)
    : LRScheduler(optimizer), T_max(T_max), eta_min(eta_min) {}

float CosineAnnealingLR::get_lr() {
    if (last_epoch == -1) return base_lr;
    return eta_min + (base_lr - eta_min) * (1.0f + std::cos(M_PI * last_epoch / T_max)) / 2.0f;
}

// CosineAnnealingWarmRestarts: cosine annealing with warm restarts
CosineAnnealingWarmRestarts::CosineAnnealingWarmRestarts(Optimizer* optimizer, int T_0, int T_mult, float eta_min)
    : LRScheduler(optimizer), T_0(T_0), T_mult(T_mult), eta_min(eta_min), T_cur(0), T_i(T_0) {}

float CosineAnnealingWarmRestarts::get_lr() {
    return eta_min + (base_lr - eta_min) * (1.0f + std::cos(M_PI * T_cur / T_i)) / 2.0f;
}

void CosineAnnealingWarmRestarts::step() {
    last_epoch++;
    T_cur++;

    if (T_cur >= T_i) {
        T_cur = 0;
        T_i = T_i * T_mult;
    }

    optimizer->lr = get_lr();
}

// ReduceLROnPlateau: reduce LR when metric stops improving
ReduceLROnPlateau::ReduceLROnPlateau(Optimizer* optimizer, float factor, int patience,
                                     float min_lr, bool mode_min)
    : optimizer(optimizer), factor(factor), patience(patience), min_lr(min_lr),
      num_bad_epochs(0), mode_min(mode_min) {
    best = mode_min ? 1e10f : -1e10f;
}

void ReduceLROnPlateau::step(float metric) {
    bool is_better = mode_min ? (metric < best) : (metric > best);

    if (is_better) {
        best = metric;
        num_bad_epochs = 0;
    } else {
        num_bad_epochs++;
    }

    if (num_bad_epochs > patience) {
        float new_lr = optimizer->lr * factor;
        if (new_lr >= min_lr) {
            optimizer->lr = new_lr;
        }
        num_bad_epochs = 0;
    }
}
