#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include <vector>
#include <cmath>

class Optimizer {
public:
    std::vector<TensorPtr> params;
    float lr;

    Optimizer(const std::vector<TensorPtr>& params, float lr);
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    void zero_grad();
};

class SGD : public Optimizer {
public:
    float momentum;
    std::vector<std::vector<float>> velocity;

    SGD(const std::vector<TensorPtr>& params, float lr, float momentum = 0.0f);
    void step() override;
};

class Adam : public Optimizer {
public:
    float beta1, beta2, eps;
    int t;
    std::vector<std::vector<float>> m;
    std::vector<std::vector<float>> v;

    Adam(const std::vector<TensorPtr>& params, float lr = 0.001f,
         float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    void step() override;
};

class AdamW : public Optimizer {
public:
    float beta1, beta2, eps, weight_decay;
    int t;
    std::vector<std::vector<float>> m;
    std::vector<std::vector<float>> v;

    AdamW(const std::vector<TensorPtr>& params, float lr = 0.001f,
          float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
          float weight_decay = 0.01f);
    void step() override;
};

class RMSprop : public Optimizer {
public:
    float alpha, eps, momentum, weight_decay;
    std::vector<std::vector<float>> v;
    std::vector<std::vector<float>> buffer;

    RMSprop(const std::vector<TensorPtr>& params, float lr = 0.01f,
            float alpha = 0.99f, float eps = 1e-8f, float momentum = 0.0f,
            float weight_decay = 0.0f);
    void step() override;
};

// Gradient Clipping Utilities
void clip_grad_norm_(std::vector<TensorPtr>& params, float max_norm);
void clip_grad_value_(std::vector<TensorPtr>& params, float clip_value);
float get_grad_norm(const std::vector<TensorPtr>& params);

// Learning Rate Schedulers
class LRScheduler {
public:
    Optimizer* optimizer;
    float base_lr;
    int last_epoch;

    LRScheduler(Optimizer* optimizer);
    virtual ~LRScheduler() = default;

    virtual void step();
    virtual float get_lr() = 0;
    float get_last_lr() const { return optimizer->lr; }
};

// Decays LR by gamma every step_size epochs
class StepLR : public LRScheduler {
public:
    int step_size;
    float gamma;

    StepLR(Optimizer* optimizer, int step_size, float gamma = 0.1f);
    float get_lr() override;
};

// Decays LR by gamma every epoch
class ExponentialLR : public LRScheduler {
public:
    float gamma;

    ExponentialLR(Optimizer* optimizer, float gamma);
    float get_lr() override;
};

// Cosine annealing schedule
class CosineAnnealingLR : public LRScheduler {
public:
    int T_max;
    float eta_min;

    CosineAnnealingLR(Optimizer* optimizer, int T_max, float eta_min = 0.0f);
    float get_lr() override;
};

// Linear warmup followed by cosine decay
class CosineAnnealingWarmRestarts : public LRScheduler {
public:
    int T_0;
    int T_mult;
    float eta_min;
    int T_cur;
    int T_i;

    CosineAnnealingWarmRestarts(Optimizer* optimizer, int T_0, int T_mult = 1, float eta_min = 0.0f);
    float get_lr() override;
    void step() override;
};

// Reduce LR when a metric has stopped improving
class ReduceLROnPlateau {
public:
    Optimizer* optimizer;
    float factor;
    int patience;
    float min_lr;
    int num_bad_epochs;
    float best;
    bool mode_min;

    ReduceLROnPlateau(Optimizer* optimizer, float factor = 0.1f, int patience = 10,
                      float min_lr = 0.0f, bool mode_min = true);

    void step(float metric);
};

#endif
