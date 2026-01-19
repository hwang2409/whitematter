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

// =============================================================================
// Gradient Accumulation
// =============================================================================
// Enables training with effectively larger batch sizes when memory is limited.
// Instead of updating weights after every batch, accumulate gradients over
// multiple mini-batches and then perform a single optimizer step.
//
// Example usage:
//   GradientAccumulator accumulator(4);  // Accumulate over 4 mini-batches
//   for (auto [x, y] : dataloader) {
//       auto loss = criterion(model.forward(x), y);
//       accumulator.backward(loss);  // Scales loss and calls backward
//
//       if (accumulator.should_step()) {
//           optimizer.step();
//           optimizer.zero_grad();
//           accumulator.reset();
//       }
//   }

class GradientAccumulator {
public:
    // Create accumulator with specified number of accumulation steps
    // effective_batch_size = mini_batch_size * accumulation_steps
    explicit GradientAccumulator(int accumulation_steps)
        : accumulation_steps_(accumulation_steps),
          current_step_(0),
          scale_factor_(1.0f / accumulation_steps) {
        if (accumulation_steps < 1) {
            accumulation_steps_ = 1;
            scale_factor_ = 1.0f;
        }
    }

    // Scale loss for gradient accumulation (divides by accumulation_steps)
    // Use this before calling backward() manually
    TensorPtr scale(const TensorPtr& loss) const {
        if (accumulation_steps_ == 1) return loss;
        return loss->mul(scale_factor_);
    }

    // Convenience method: scale loss and call backward in one step
    void backward(const TensorPtr& loss) {
        auto scaled_loss = scale(loss);
        scaled_loss->backward();
        current_step_++;
    }

    // Check if we've accumulated enough gradients and should step the optimizer
    bool should_step() const {
        return current_step_ >= accumulation_steps_;
    }

    // Increment step counter (use if calling backward manually)
    void increment() {
        current_step_++;
    }

    // Reset counter after optimizer step (call after optimizer.step() and zero_grad())
    void reset() {
        current_step_ = 0;
    }

    // Get current micro-batch index within accumulation window
    int current_step() const { return current_step_; }

    // Get total number of accumulation steps
    int get_accumulation_steps() const { return accumulation_steps_; }

    // Get the loss scaling factor (1 / accumulation_steps)
    float get_scale_factor() const { return scale_factor_; }

    // Check if this is the last step before optimizer update
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

// =============================================================================
// Early Stopping
// =============================================================================
// Stops training when a monitored metric stops improving.
// Prevents overfitting by halting training at the optimal point.
//
// Example usage:
//   EarlyStopping early_stopping(10);  // patience = 10 epochs
//   for (int epoch = 0; epoch < max_epochs; epoch++) {
//       train_one_epoch();
//       float val_loss = evaluate();
//       if (early_stopping.step(val_loss)) {
//           printf("Early stopping at epoch %d\n", epoch);
//           break;
//       }
//   }

class EarlyStopping {
public:
    // Create early stopping monitor
    // patience: epochs to wait after last improvement before stopping
    // min_delta: minimum change to qualify as improvement
    // mode_min: true = lower is better (loss), false = higher is better (accuracy)
    // baseline: initial best value (optional, uses first metric if not set)
    EarlyStopping(int patience = 10, float min_delta = 0.0f, bool mode_min = true,
                  float baseline = std::numeric_limits<float>::quiet_NaN())
        : patience_(patience),
          min_delta_(min_delta),
          mode_min_(mode_min),
          counter_(0),
          best_(baseline),
          best_epoch_(-1),
          stopped_epoch_(-1),
          should_stop_(false) {
        // Adjust min_delta sign based on mode
        if (!mode_min_) {
            min_delta_ = -min_delta_;
        }
    }

    // Check metric and return true if training should stop
    // Call this at the end of each epoch with validation metric
    bool step(float metric) {
        if (should_stop_) return true;

        // Initialize best on first call if not set
        if (std::isnan(best_)) {
            best_ = metric;
            best_epoch_ = 0;
            return false;
        }

        // Check for improvement
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

    // Reset early stopping state
    void reset() {
        counter_ = 0;
        best_ = std::numeric_limits<float>::quiet_NaN();
        best_epoch_ = -1;
        stopped_epoch_ = -1;
        should_stop_ = false;
    }

    // Getters
    bool should_stop() const { return should_stop_; }
    float best_metric() const { return best_; }
    int best_epoch() const { return best_epoch_; }
    int stopped_epoch() const { return stopped_epoch_; }
    int patience() const { return patience_; }
    int epochs_without_improvement() const { return counter_; }

    // Set patience dynamically
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
};

// =============================================================================
// ModelCheckpoint - Save best model during training
// =============================================================================
// Works with EarlyStopping to save model at best validation metric.
//
// Example usage:
//   EarlyStopping early_stopping(10);
//   ModelCheckpoint checkpoint("best_model.bin");
//   for (int epoch = 0; epoch < max_epochs; epoch++) {
//       train_one_epoch();
//       float val_loss = evaluate();
//       checkpoint.step(val_loss, &model);  // Saves if improved
//       if (early_stopping.step(val_loss)) break;
//   }
//   checkpoint.restore(&model);  // Load best weights

class ModelCheckpoint {
public:
    // Create checkpoint monitor
    // filepath: where to save the best model
    // mode_min: true = lower is better (loss), false = higher is better (accuracy)
    ModelCheckpoint(const std::string& filepath, bool mode_min = true, float min_delta = 0.0f)
        : filepath_(filepath),
          mode_min_(mode_min),
          min_delta_(min_delta),
          best_(mode_min ? std::numeric_limits<float>::infinity()
                         : -std::numeric_limits<float>::infinity()),
          saved_(false) {}

    // Check metric and save model if improved
    // Returns true if model was saved
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

    // Restore best model weights
    bool restore(Module* model) {
        if (saved_) {
            load_model(model, filepath_);
            return true;
        }
        return false;
    }

    // Getters
    float best_metric() const { return best_; }
    bool has_saved() const { return saved_; }
    const std::string& filepath() const { return filepath_; }

    // Reset state
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
