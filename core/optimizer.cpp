#include "optimizer.h"
#include <cmath>
#if defined(WHITEMATTER_CUDA)
#include "cuda/cuda_backend.h"
#include "cuda/cuda_memory.h"
#endif

Optimizer::Optimizer(const std::vector<TensorPtr>& params, float lr)
    : params(params), lr(lr) {}

void Optimizer::zero_grad() {
    for (auto& p : params) {
#if defined(WHITEMATTER_CUDA)
        if (p->device == whitematter::DeviceType::CUDA && whitematter::cuda_backend_available()) {
            if (p->grad()) {
                whitematter::CUDABackend::instance().memset_zero(p->grad(), p->size());
            }
            continue;
        }
#endif
        p->zero_grad();
    }
}

SGD::SGD(const std::vector<TensorPtr>& params, float lr, float momentum)
    : Optimizer(params, lr), momentum(momentum) {
    if (momentum > 0.0f) {
        for (const auto& p : params) {
            velocity.push_back(std::vector<float>(p->size(), 0.0f));
        }
    }
}

void SGD::step() {
    for (size_t i = 0; i < params.size(); i++) {
        auto& p = params[i];
        if (!p->requires_grad || !p->grad()) continue;

#if defined(WHITEMATTER_CUDA)
        if (p->device == whitematter::DeviceType::CUDA && whitematter::cuda_backend_available()) {
            if (!cuda_state_initialized_) init_cuda_state();
            whitematter::CUDABackend::instance().sgd_step(
                p->data(), p->grad(), d_state1[i],
                lr, momentum, p->size());
            continue;
        }
#endif
        if (momentum > 0.0f) {
            for (size_t j = 0; j < p->size(); j++) {
                velocity[i][j] = momentum * velocity[i][j] + p->grad()[j];
                p->data()[j] -= lr * velocity[i][j];
            }
        } else {
            for (size_t j = 0; j < p->size(); j++) {
                p->data()[j] -= lr * p->grad()[j];
            }
        }
    }
}

Adam::Adam(const std::vector<TensorPtr>& params, float lr, float beta1, float beta2, float eps)
    : Optimizer(params, lr), beta1(beta1), beta2(beta2), eps(eps), t(0) {
    for (const auto& p : params) {
        m.push_back(std::vector<float>(p->size(), 0.0f));
        v.push_back(std::vector<float>(p->size(), 0.0f));
    }
}

void Adam::step() {
    t++;
    float bias_correction1 = 1.0f - std::pow(beta1, t);
    float bias_correction2 = 1.0f - std::pow(beta2, t);

    for (size_t i = 0; i < params.size(); i++) {
        auto& p = params[i];
        if (!p->requires_grad || !p->grad()) continue;

#if defined(WHITEMATTER_CUDA)
        if (p->device == whitematter::DeviceType::CUDA && whitematter::cuda_backend_available()) {
            if (!cuda_state_initialized_) init_cuda_state();
            whitematter::CUDABackend::instance().adam_step(
                p->data(), p->grad(), d_state1[i], d_state2[i],
                lr, beta1, beta2, eps, bias_correction1, bias_correction2, p->size());
            continue;
        }
#endif
        for (size_t j = 0; j < p->size(); j++) {
            float g = p->grad()[j];

            m[i][j] = beta1 * m[i][j] + (1.0f - beta1) * g;
            v[i][j] = beta2 * v[i][j] + (1.0f - beta2) * g * g;

            float m_hat = m[i][j] / bias_correction1;
            float v_hat = v[i][j] / bias_correction2;

            p->data()[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
}

AdamW::AdamW(const std::vector<TensorPtr>& params, float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(params, lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay), t(0) {
    for (const auto& p : params) {
        m.push_back(std::vector<float>(p->size(), 0.0f));
        v.push_back(std::vector<float>(p->size(), 0.0f));
    }
}

void AdamW::step() {
    t++;
    float bias_correction1 = 1.0f - std::pow(beta1, t);
    float bias_correction2 = 1.0f - std::pow(beta2, t);

    for (size_t i = 0; i < params.size(); i++) {
        auto& p = params[i];
        if (!p->requires_grad || !p->grad()) continue;

#if defined(WHITEMATTER_CUDA)
        if (p->device == whitematter::DeviceType::CUDA && whitematter::cuda_backend_available()) {
            if (!cuda_state_initialized_) init_cuda_state();
            whitematter::CUDABackend::instance().adamw_step(
                p->data(), p->grad(), d_state1[i], d_state2[i],
                lr, beta1, beta2, eps, bias_correction1, bias_correction2,
                weight_decay, p->size());
            continue;
        }
#endif
        for (size_t j = 0; j < p->size(); j++) {
            float g = p->grad()[j];

            m[i][j] = beta1 * m[i][j] + (1.0f - beta1) * g;
            v[i][j] = beta2 * v[i][j] + (1.0f - beta2) * g * g;

            float m_hat = m[i][j] / bias_correction1;
            float v_hat = v[i][j] / bias_correction2;

            p->data()[j] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * p->data()[j]);
        }
    }
}

RMSprop::RMSprop(const std::vector<TensorPtr>& params, float lr, float alpha, float eps, float momentum, float weight_decay)
    : Optimizer(params, lr), alpha(alpha), eps(eps), momentum(momentum), weight_decay(weight_decay) {
    for (const auto& p : params) {
        v.push_back(std::vector<float>(p->size(), 0.0f));
        if (momentum > 0.0f) {
            buffer.push_back(std::vector<float>(p->size(), 0.0f));
        }
    }
}

void RMSprop::step() {
    for (size_t i = 0; i < params.size(); i++) {
        auto& p = params[i];
        if (!p->requires_grad || !p->grad()) continue;

#if defined(WHITEMATTER_CUDA)
        if (p->device == whitematter::DeviceType::CUDA && whitematter::cuda_backend_available()) {
            if (!cuda_state_initialized_) init_cuda_state();
            whitematter::CUDABackend::instance().rmsprop_step(
                p->data(), p->grad(), d_state1[i],
                lr, alpha, eps, momentum, d_state2[i], weight_decay, p->size());
            continue;
        }
#endif
        for (size_t j = 0; j < p->size(); j++) {
            float g = p->grad()[j];

            if (weight_decay != 0.0f) {
                g += weight_decay * p->data()[j];
            }

            v[i][j] = alpha * v[i][j] + (1.0f - alpha) * g * g;

            float avg = std::sqrt(v[i][j]) + eps;

            if (momentum > 0.0f) {
                buffer[i][j] = momentum * buffer[i][j] + g / avg;
                p->data()[j] -= lr * buffer[i][j];
            } else {
                p->data()[j] -= lr * g / avg;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA state initialization
// ---------------------------------------------------------------------------

#if defined(WHITEMATTER_CUDA)
void SGD::init_cuda_state() {
    auto& pool = whitematter::CUDAMemoryPool::instance();
    d_state1.resize(params.size(), nullptr);
    for (size_t i = 0; i < params.size(); i++) {
        if (params[i]->device == whitematter::DeviceType::CUDA && momentum > 0.0f) {
            d_state1[i] = pool.acquire(params[i]->size());
            whitematter::CUDABackend::instance().memset_zero(d_state1[i], params[i]->size());
        }
    }
    cuda_state_initialized_ = true;
}

void Adam::init_cuda_state() {
    auto& pool = whitematter::CUDAMemoryPool::instance();
    d_state1.resize(params.size(), nullptr);
    d_state2.resize(params.size(), nullptr);
    for (size_t i = 0; i < params.size(); i++) {
        if (params[i]->device == whitematter::DeviceType::CUDA) {
            d_state1[i] = pool.acquire(params[i]->size());
            d_state2[i] = pool.acquire(params[i]->size());
            whitematter::CUDABackend::instance().memset_zero(d_state1[i], params[i]->size());
            whitematter::CUDABackend::instance().memset_zero(d_state2[i], params[i]->size());
        }
    }
    cuda_state_initialized_ = true;
}

void AdamW::init_cuda_state() {
    auto& pool = whitematter::CUDAMemoryPool::instance();
    d_state1.resize(params.size(), nullptr);
    d_state2.resize(params.size(), nullptr);
    for (size_t i = 0; i < params.size(); i++) {
        if (params[i]->device == whitematter::DeviceType::CUDA) {
            d_state1[i] = pool.acquire(params[i]->size());
            d_state2[i] = pool.acquire(params[i]->size());
            whitematter::CUDABackend::instance().memset_zero(d_state1[i], params[i]->size());
            whitematter::CUDABackend::instance().memset_zero(d_state2[i], params[i]->size());
        }
    }
    cuda_state_initialized_ = true;
}

void RMSprop::init_cuda_state() {
    auto& pool = whitematter::CUDAMemoryPool::instance();
    d_state1.resize(params.size(), nullptr);
    d_state2.resize(params.size(), nullptr);
    for (size_t i = 0; i < params.size(); i++) {
        if (params[i]->device == whitematter::DeviceType::CUDA) {
            d_state1[i] = pool.acquire(params[i]->size());
            whitematter::CUDABackend::instance().memset_zero(d_state1[i], params[i]->size());
            if (momentum > 0.0f) {
                d_state2[i] = pool.acquire(params[i]->size());
                whitematter::CUDABackend::instance().memset_zero(d_state2[i], params[i]->size());
            }
        }
    }
    cuda_state_initialized_ = true;
}
#endif

float get_grad_norm(const std::vector<TensorPtr>& params) {
    float total_norm = 0.0f;
    for (const auto& p : params) {
        for (size_t i = 0; i < p->grad_size(); i++) {
            total_norm += p->grad()[i] * p->grad()[i];
        }
    }
    return std::sqrt(total_norm);
}

void clip_grad_norm_(std::vector<TensorPtr>& params, float max_norm) {
    float total_norm = get_grad_norm(params);
    float clip_coef = max_norm / (total_norm + 1e-6f);
    if (clip_coef < 1.0f) {
        for (auto& p : params) {
            for (size_t i = 0; i < p->grad_size(); i++) {
                p->grad()[i] *= clip_coef;
            }
        }
    }
}

void clip_grad_value_(std::vector<TensorPtr>& params, float clip_value) {
    for (auto& p : params) {
        for (size_t i = 0; i < p->grad_size(); i++) {
            if (p->grad()[i] > clip_value) {
                p->grad()[i] = clip_value;
            } else if (p->grad()[i] < -clip_value) {
                p->grad()[i] = -clip_value;
            }
        }
    }
}

LRScheduler::LRScheduler(Optimizer* optimizer)
    : optimizer(optimizer), base_lr(optimizer->lr), last_epoch(-1) {}

void LRScheduler::step() {
    last_epoch++;
    optimizer->lr = get_lr();
}

StepLR::StepLR(Optimizer* optimizer, int step_size, float gamma)
    : LRScheduler(optimizer), step_size(step_size), gamma(gamma) {}

float StepLR::get_lr() {
    if (last_epoch == -1) return base_lr;
    return base_lr * std::pow(gamma, last_epoch / step_size);
}

ExponentialLR::ExponentialLR(Optimizer* optimizer, float gamma)
    : LRScheduler(optimizer), gamma(gamma) {}

float ExponentialLR::get_lr() {
    if (last_epoch == -1) return base_lr;
    return base_lr * std::pow(gamma, last_epoch);
}

CosineAnnealingLR::CosineAnnealingLR(Optimizer* optimizer, int T_max, float eta_min)
    : LRScheduler(optimizer), T_max(T_max), eta_min(eta_min) {}

float CosineAnnealingLR::get_lr() {
    if (last_epoch == -1) return base_lr;
    return eta_min + (base_lr - eta_min) * (1.0f + std::cos(M_PI * last_epoch / T_max)) / 2.0f;
}

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

LinearWarmupCosineDecay::LinearWarmupCosineDecay(Optimizer* optimizer, size_t warmup_steps,
                                                 size_t total_steps, float min_lr)
    : optimizer_(optimizer), base_lr_(optimizer->lr), min_lr_(min_lr),
      warmup_steps_(warmup_steps), total_steps_(total_steps), current_step_(0) {}

void LinearWarmupCosineDecay::step() {
    current_step_++;
    if (current_step_ <= warmup_steps_) {
        // Linear warmup
        optimizer_->lr = base_lr_ * static_cast<float>(current_step_) / static_cast<float>(warmup_steps_);
    } else {
        // Cosine decay
        float progress = static_cast<float>(current_step_ - warmup_steps_)
                       / static_cast<float>(total_steps_ - warmup_steps_);
        if (progress > 1.0f) progress = 1.0f;
        optimizer_->lr = min_lr_ + (base_lr_ - min_lr_) * 0.5f * (1.0f + std::cos(M_PI * progress));
    }
}

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
