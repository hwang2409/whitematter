#ifndef AMP_H
#define AMP_H

#include "tensor.h"
#include "optimizer.h"
#include <cstdint>
#include <cmath>
#include <limits>

// =============================================================================
// Half-precision (fp16) utilities
// =============================================================================

// IEEE 754 half-precision format conversion
inline uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float));

    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exponent = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = x & 0x7FFFFF;

    if (exponent <= 0) {
        // Denormalized or zero
        if (exponent < -10) return sign;  // Too small, return signed zero
        mantissa = (mantissa | 0x800000) >> (1 - exponent);
        return sign | (mantissa >> 13);
    } else if (exponent == 0xFF - 127 + 15) {
        // Inf or NaN
        if (mantissa == 0) return sign | 0x7C00;  // Inf
        return sign | 0x7C00 | (mantissa >> 13);  // NaN
    } else if (exponent > 30) {
        // Overflow to Inf
        return sign | 0x7C00;
    }

    return sign | (exponent << 10) | (mantissa >> 13);
}

inline float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            uint32_t result = sign;
            float f;
            std::memcpy(&f, &result, sizeof(float));
            return f;
        }
        // Denormalized
        while ((mantissa & 0x400) == 0) {
            mantissa <<= 1;
            exponent--;
        }
        exponent++;
        mantissa &= ~0x400;
    } else if (exponent == 31) {
        // Inf or NaN
        uint32_t result = sign | 0x7F800000 | (mantissa << 13);
        float f;
        std::memcpy(&f, &result, sizeof(float));
        return f;
    }

    uint32_t result = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    float f;
    std::memcpy(&f, &result, sizeof(float));
    return f;
}

// Convert tensor data to fp16 storage
inline std::vector<uint16_t> to_half(const std::vector<float>& data) {
    std::vector<uint16_t> half_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        half_data[i] = float_to_half(data[i]);
    }
    return half_data;
}

// Convert fp16 storage back to fp32
inline std::vector<float> from_half(const std::vector<uint16_t>& half_data) {
    std::vector<float> data(half_data.size());
    for (size_t i = 0; i < half_data.size(); i++) {
        data[i] = half_to_float(half_data[i]);
    }
    return data;
}

// =============================================================================
// GradScaler - Dynamic loss scaling for mixed precision training
// =============================================================================

class GradScaler {
public:
    // Constructor with initial scale and growth/backoff factors
    GradScaler(float init_scale = 65536.0f,
               float growth_factor = 2.0f,
               float backoff_factor = 0.5f,
               int growth_interval = 2000,
               bool enabled = true)
        : scale_(init_scale),
          growth_factor_(growth_factor),
          backoff_factor_(backoff_factor),
          growth_interval_(growth_interval),
          growth_tracker_(0),
          enabled_(enabled) {}

    // Get current scale factor
    float get_scale() const { return enabled_ ? scale_ : 1.0f; }

    // Check if scaler is enabled
    bool is_enabled() const { return enabled_; }

    // Scale a loss tensor for backward pass
    TensorPtr scale(const TensorPtr& loss) const {
        if (!enabled_) return loss;
        return loss->mul(scale_);
    }

    // Unscale gradients of optimizer parameters
    // Returns true if gradients are finite, false if inf/nan detected
    bool unscale(Optimizer* optimizer) {
        if (!enabled_) return true;

        float inv_scale = 1.0f / scale_;
        bool found_inf = false;

        for (auto& param : optimizer->params) {
            if (param->grad.empty()) continue;

            for (size_t i = 0; i < param->grad.size(); i++) {
                float g = param->grad[i] * inv_scale;
                if (!std::isfinite(g)) {
                    found_inf = true;
                    break;
                }
                param->grad[i] = g;
            }
            if (found_inf) break;
        }

        found_inf_or_nan_ = found_inf;
        return !found_inf;
    }

    // Step the optimizer only if gradients are finite
    // Call this after unscale() or let it unscale automatically
    void step(Optimizer* optimizer, bool already_unscaled = false) {
        if (!enabled_) {
            optimizer->step();
            return;
        }

        if (!already_unscaled) {
            unscale(optimizer);
        }

        if (!found_inf_or_nan_) {
            optimizer->step();
        }
    }

    // Update scale factor based on gradient overflow history
    void update() {
        if (!enabled_) return;

        if (found_inf_or_nan_) {
            // Reduce scale on overflow
            scale_ *= backoff_factor_;
            growth_tracker_ = 0;
        } else {
            // Increase scale after growth_interval successful steps
            growth_tracker_++;
            if (growth_tracker_ >= growth_interval_) {
                scale_ *= growth_factor_;
                growth_tracker_ = 0;
            }
        }

        // Clamp scale to reasonable range
        scale_ = std::max(1.0f, std::min(scale_, 65536.0f * 65536.0f));
        found_inf_or_nan_ = false;
    }

    // Get state for checkpointing
    float scale() const { return scale_; }
    int growth_tracker() const { return growth_tracker_; }

    // Set state for loading checkpoint
    void set_scale(float s) { scale_ = s; }
    void set_growth_tracker(int t) { growth_tracker_ = t; }

private:
    float scale_;
    float growth_factor_;
    float backoff_factor_;
    int growth_interval_;
    int growth_tracker_;
    bool enabled_;
    bool found_inf_or_nan_ = false;
};

// =============================================================================
// AmpContext - Automatic Mixed Precision context
// =============================================================================

class AmpContext {
public:
    static AmpContext& instance() {
        static AmpContext ctx;
        return ctx;
    }

    // Enable/disable autocast
    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }

    // Set default dtype for autocast regions
    enum class Dtype { Float32, Float16, BFloat16 };
    void set_dtype(Dtype dtype) { dtype_ = dtype; }
    Dtype get_dtype() const { return dtype_; }

private:
    AmpContext() : enabled_(false), dtype_(Dtype::Float16) {}
    bool enabled_;
    Dtype dtype_;
};

// RAII guard for autocast regions
class AutocastGuard {
public:
    AutocastGuard(bool enabled = true, AmpContext::Dtype dtype = AmpContext::Dtype::Float16) {
        prev_enabled_ = AmpContext::instance().is_enabled();
        prev_dtype_ = AmpContext::instance().get_dtype();
        AmpContext::instance().set_enabled(enabled);
        AmpContext::instance().set_dtype(dtype);
    }

    ~AutocastGuard() {
        AmpContext::instance().set_enabled(prev_enabled_);
        AmpContext::instance().set_dtype(prev_dtype_);
    }

private:
    bool prev_enabled_;
    AmpContext::Dtype prev_dtype_;
};

// =============================================================================
// Half-precision tensor wrapper (for storage optimization)
// =============================================================================

class HalfTensor {
public:
    std::vector<uint16_t> data;
    std::vector<size_t> shape;

    HalfTensor() = default;

    // Create from float tensor
    explicit HalfTensor(const TensorPtr& tensor) {
        shape = tensor->shape;
        data = to_half(tensor->data);
    }

    // Convert back to float tensor
    TensorPtr to_float(bool requires_grad = false) const {
        auto result = Tensor::create(shape, requires_grad);
        result->data = from_half(data);
        return result;
    }

    size_t size() const {
        size_t s = 1;
        for (auto d : shape) s *= d;
        return s;
    }

    // Memory savings: returns bytes saved compared to fp32
    size_t memory_saved() const {
        return size() * sizeof(float) - size() * sizeof(uint16_t);
    }
};

// =============================================================================
// Mixed precision training utilities
// =============================================================================

// Check if gradients contain inf/nan
inline bool check_gradients_finite(const std::vector<TensorPtr>& params) {
    for (const auto& p : params) {
        for (float g : p->grad) {
            if (!std::isfinite(g)) return false;
        }
    }
    return true;
}

// Clip gradients by global norm (useful with mixed precision)
inline float clip_grad_norm_amp(const std::vector<TensorPtr>& params, float max_norm, float scale = 1.0f) {
    // Compute total norm with scale factor
    float total_norm = 0.0f;
    float inv_scale = 1.0f / scale;

    for (const auto& p : params) {
        for (float g : p->grad) {
            float unscaled = g * inv_scale;
            total_norm += unscaled * unscaled;
        }
    }
    total_norm = std::sqrt(total_norm);

    // Clip if needed
    float clip_coef = max_norm / (total_norm + 1e-6f);
    if (clip_coef < 1.0f) {
        for (auto& p : params) {
            for (float& g : p->grad) {
                g *= clip_coef;
            }
        }
    }

    return total_norm;
}

#endif // AMP_H
