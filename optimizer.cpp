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
