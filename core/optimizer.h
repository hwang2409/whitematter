#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include <vector>
#include <cmath>
#include <limits>
#include <string>

// Forward declarations for ModelCheckpoint
class Module;
bool save_model(Module* module, const std::string& path);
bool load_model(Module* module, const std::string& path);

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

#if defined(WHITEMATTER_CUDA)
    std::vector<float*> d_state1;  // GPU momentum/velocity
    bool cuda_state_initialized_ = false;
    void init_cuda_state();
#endif
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

#if defined(WHITEMATTER_CUDA)
    std::vector<float*> d_state1;  // GPU m (first moment)
    std::vector<float*> d_state2;  // GPU v (second moment)
    bool cuda_state_initialized_ = false;
    void init_cuda_state();
#endif
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

#if defined(WHITEMATTER_CUDA)
    std::vector<float*> d_state1;  // GPU m (first moment)
    std::vector<float*> d_state2;  // GPU v (second moment)
    bool cuda_state_initialized_ = false;
    void init_cuda_state();
#endif
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

#if defined(WHITEMATTER_CUDA)
    std::vector<float*> d_state1;  // GPU v (running average of squared gradients)
    std::vector<float*> d_state2;  // GPU momentum buffer
    bool cuda_state_initialized_ = false;
    void init_cuda_state();
#endif
};

// Gradient Clipping Utilities
void clip_grad_norm_(std::vector<TensorPtr>& params, float max_norm);
void clip_grad_value_(std::vector<TensorPtr>& params, float clip_value);
float get_grad_norm(const std::vector<TensorPtr>& params);

class GradientAccumulator {
public:
    explicit GradientAccumulator(int accumulation_steps)
        : accumulation_steps_(accumulation_steps),
          current_step_(0),
          scale_factor_(1.0f / accumulation_steps) {
        if (accumulation_steps < 1) {
            accumulation_steps_ = 1;
            scale_factor_ = 1.0f;
        }
    }

    TensorPtr scale(const TensorPtr& loss) const {
        if (accumulation_steps_ == 1) return loss;
        return loss->mul(scale_factor_);
    }

    void backward(const TensorPtr& loss) {
        auto scaled_loss = scale(loss);
        scaled_loss->backward();
        current_step_++;
    }

    bool should_step() const {
        return current_step_ >= accumulation_steps_;
    }

    void increment() {
        current_step_++;
    }

    void reset() {
        current_step_ = 0;
    }

    int current_step() const { return current_step_; }
    int get_accumulation_steps() const { return accumulation_steps_; }
    float get_scale_factor() const { return scale_factor_; }

    bool is_last_step() const {
        return current_step_ == accumulation_steps_ - 1;
    }

private:
    int accumulation_steps_;
    int current_step_;
    float scale_factor_;
};

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

// Linear warmup followed by cosine decay (step-based, not epoch-based)
class LinearWarmupCosineDecay {
public:
    LinearWarmupCosineDecay(Optimizer* optimizer, size_t warmup_steps, size_t total_steps,
                            float min_lr = 0.0f);
    void step();

private:
    Optimizer* optimizer_;
    float base_lr_;
    float min_lr_;
    size_t warmup_steps_;
    size_t total_steps_;
    size_t current_step_;
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

class EarlyStopping {
public:
    EarlyStopping(int patience = 10, float min_delta = 0.0f, bool mode_min = true)
        : patience_(patience),
          min_delta_(min_delta),
          mode_min_(mode_min),
          counter_(0),
          best_(0.0f),
          best_epoch_(-1),
          stopped_epoch_(-1),
          should_stop_(false),
          first_step_(true) {
        if (!mode_min_) {
            min_delta_ = -min_delta_;
        }
    }
    EarlyStopping(int patience, float min_delta, bool mode_min, float baseline)
        : patience_(patience),
          min_delta_(min_delta),
          mode_min_(mode_min),
          counter_(0),
          best_(baseline),
          best_epoch_(-1),
          stopped_epoch_(-1),
          should_stop_(false),
          first_step_(false) {
        if (!mode_min_) {
            min_delta_ = -min_delta_;
        }
    }

    bool step(float metric) {
        if (should_stop_) return true;

        if (first_step_) {
            first_step_ = false;
            best_ = metric;
            best_epoch_ = 0;
            return false;
        }

        bool improved = false;
        if (mode_min_) {
            improved = metric < best_ - min_delta_;
        } else {
            improved = metric > best_ + min_delta_;  // Note: min_delta_ is already negated
        }

        if (improved) {
            best_ = metric;
            best_epoch_ = counter_ + best_epoch_ + 1;  // Track actual epoch
            counter_ = 0;
        } else {
            counter_++;
            if (counter_ >= patience_) {
                should_stop_ = true;
                stopped_epoch_ = best_epoch_ + counter_;
            }
        }

        return should_stop_;
    }

    void reset() {
        counter_ = 0;
        best_ = std::numeric_limits<float>::quiet_NaN();
        best_epoch_ = -1;
        stopped_epoch_ = -1;
        should_stop_ = false;
        first_step_ = true;
    }

    bool should_stop() const { return should_stop_; }
    float best_metric() const { return best_; }
    int best_epoch() const { return best_epoch_; }
    int stopped_epoch() const { return stopped_epoch_; }
    int patience() const { return patience_; }
    int epochs_without_improvement() const { return counter_; }
    void set_patience(int p) { patience_ = p; }

private:
    int patience_;
    float min_delta_;
    bool mode_min_;
    int counter_;
    float best_;
    int best_epoch_;
    int stopped_epoch_;
    bool should_stop_;
    bool first_step_;  // true until first step() when no baseline was provided
};

class ModelCheckpoint {
public:
    ModelCheckpoint(const std::string& filepath, bool mode_min = true, float min_delta = 0.0f)
        : filepath_(filepath),
          mode_min_(mode_min),
          min_delta_(min_delta),
          best_(mode_min ? std::numeric_limits<float>::infinity()
                         : -std::numeric_limits<float>::infinity()),
          saved_(false) {}

    bool step(float metric, Module* model) {
        bool improved = false;
        if (mode_min_) {
            improved = metric < best_ - min_delta_;
        } else {
            improved = metric > best_ + min_delta_;
        }

        if (improved) {
            best_ = metric;
            save_model(model, filepath_);
            saved_ = true;
            return true;
        }
        return false;
    }

    bool restore(Module* model) {
        if (saved_) {
            load_model(model, filepath_);
            return true;
        }
        return false;
    }

    float best_metric() const { return best_; }
    bool has_saved() const { return saved_; }
    const std::string& filepath() const { return filepath_; }

    void reset() {
        best_ = mode_min_ ? std::numeric_limits<float>::infinity()
                          : -std::numeric_limits<float>::infinity();
        saved_ = false;
    }

private:
    std::string filepath_;
    bool mode_min_;
    float min_delta_;
    float best_;
    bool saved_;
};

#endif
