#include "../layer.h"
#include <cmath>
#include <cassert>
#include <cstring>
#if defined(WHITEMATTER_CUDA)
#include "../cuda/cuda_backend.h"
#include "../cuda/cuda_memory.h"
#endif

BatchNorm2d::BatchNorm2d(size_t num_features, float eps, float momentum)
    : num_features(num_features), eps(eps), momentum(momentum), training(true) {
    gamma = Tensor::ones({num_features}, true);
    beta = Tensor::zeros({num_features}, true);
    running_mean = Tensor::zeros({num_features}, false);
    running_var = Tensor::ones({num_features}, false);
}

TensorPtr BatchNorm2d::forward(const TensorPtr& input) {
    assert(input->shape.size() == 4);
    assert(input->shape[1] == num_features);

#if defined(WHITEMATTER_CUDA)
    // cuDNN BatchNorm dispatch
    if (whitematter::cuda_backend_available()) {
        size_t batch = input->shape[0];
        size_t channels = input->shape[1];
        size_t spatial = input->shape[2] * input->shape[3];

        bool track = input->requires_grad && GradMode::is_enabled();
        auto result = Tensor::create(input->shape, track);

        // Save mean/inv_var for backward
        auto save_mean = std::make_shared<std::vector<float>>(channels);
        auto save_inv_var = std::make_shared<std::vector<float>>(channels);

        whitematter::CUDABackend::instance().batchnorm_forward(
            input->data(), result->data(),
            gamma->data(), beta->data(),
            running_mean->data(), running_var->data(),
            save_mean->data(), save_inv_var->data(),
            batch, channels, spatial, eps, momentum, training);

        if (track) {
            auto input_ptr = std::const_pointer_cast<Tensor>(input->shared_from_this());
            auto gamma_ptr = gamma;
            auto beta_ptr = beta;
            result->parents = {input_ptr, gamma_ptr, beta_ptr};
            result->grad_fn = [input_ptr, gamma_ptr, beta_ptr, result,
                               save_mean, save_inv_var,
                               batch, channels, spatial, eps = this->eps]() {
                whitematter::CUDABackend::instance().batchnorm_backward(
                    input_ptr->data(), result->grad(),
                    input_ptr->requires_grad ? input_ptr->grad() : nullptr,
                    gamma_ptr->grad(), beta_ptr->grad(),
                    gamma_ptr->data(), save_mean->data(), save_inv_var->data(),
                    batch, channels, spatial, eps);
            };
        }
        return result;
    }
#endif

    size_t batch = input->shape[0];
    size_t channels = input->shape[1];
    size_t height = input->shape[2];
    size_t width = input->shape[3];
    size_t spatial_size = height * width;
    size_t n = batch * spatial_size;

    bool track = input->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create(input->shape, track);

    std::vector<float> mean(channels, 0.0f);
    std::vector<float> var(channels, 0.0f);

    if (training) {
        // Single-pass mean + variance using E[x^2] - E[x]^2
        // Iterate (b, c, h, w) for contiguous memory access
        std::vector<float> sum_sq(channels, 0.0f);
        for (size_t b = 0; b < batch; b++) {
            for (size_t c = 0; c < channels; c++) {
                const float* ptr = input->data() + b * (channels * spatial_size) + c * spatial_size;
                for (size_t i = 0; i < spatial_size; i++) {
                    float val = ptr[i];
                    mean[c] += val;
                    sum_sq[c] += val * val;
                }
            }
        }
        float inv_n = 1.0f / static_cast<float>(n);
        for (size_t c = 0; c < channels; c++) {
            mean[c] *= inv_n;
            var[c] = sum_sq[c] * inv_n - mean[c] * mean[c];
        }

        for (size_t c = 0; c < channels; c++) {
            running_mean->data()[c] = (1.0f - momentum) * running_mean->data()[c] + momentum * mean[c];
            running_var->data()[c] = (1.0f - momentum) * running_var->data()[c] + momentum * var[c];
        }
    } else {
        for (size_t c = 0; c < channels; c++) {
            mean[c] = running_mean->data()[c];
            var[c] = running_var->data()[c];
        }
    }

    std::vector<float> inv_std(channels);
    for (size_t c = 0; c < channels; c++) {
        inv_std[c] = 1.0f / std::sqrt(var[c] + eps);
    }

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    size_t idx = b * (channels * spatial_size) + c * spatial_size + h * width + w;
                    float x_norm = (input->data()[idx] - mean[c]) * inv_std[c];
                    result->data()[idx] = gamma->data()[c] * x_norm + beta->data()[c];
                }
            }
        }
    }

    if (track) {
        auto input_ptr = input;
        auto gamma_ptr = gamma;
        auto beta_ptr = beta;
        result->parents = {input_ptr, gamma_ptr, beta_ptr};

        result->grad_fn = [input_ptr, gamma_ptr, beta_ptr, result,
                           mean, inv_std, batch, channels, height, width, spatial_size, n]() {
            std::vector<float> dgamma(channels, 0.0f);
            std::vector<float> dbeta(channels, 0.0f);
            std::vector<float> dmean(channels, 0.0f);
            std::vector<float> dvar(channels, 0.0f);

            // Accumulate dgamma, dbeta, dvar — (b, c, h, w) order for cache locality
            std::vector<float> sum_dx_norm(channels, 0.0f);
            std::vector<float> sum_x_diff(channels, 0.0f);
            for (size_t b = 0; b < batch; b++) {
                for (size_t c = 0; c < channels; c++) {
                    size_t base = b * (channels * spatial_size) + c * spatial_size;
                    float g = gamma_ptr->data()[c];
                    float m = mean[c];
                    float is = inv_std[c];
                    float is3 = -0.5f * is * is * is;
                    for (size_t i = 0; i < spatial_size; i++) {
                        size_t idx = base + i;
                        float dy = result->grad()[idx];
                        float x_diff = input_ptr->data()[idx] - m;
                        float x_norm = x_diff * is;
                        dgamma[c] += dy * x_norm;
                        dbeta[c] += dy;
                        float dx_norm = dy * g;
                        dvar[c] += dx_norm * x_diff * is3;
                        sum_dx_norm[c] += dx_norm * (-is);
                        sum_x_diff[c] += -2.0f * x_diff;
                    }
                }
            }
            // dmean depends on final dvar
            for (size_t c = 0; c < channels; c++) {
                dmean[c] = sum_dx_norm[c] + dvar[c] * sum_x_diff[c] / static_cast<float>(n);
            }

            if (input_ptr->requires_grad) {
                for (size_t b = 0; b < batch; b++) {
                    for (size_t c = 0; c < channels; c++) {
                        for (size_t h = 0; h < height; h++) {
                            for (size_t w = 0; w < width; w++) {
                                size_t idx = b * (channels * spatial_size) + c * spatial_size + h * width + w;
                                float dx_norm = result->grad()[idx] * gamma_ptr->data()[c];
                                input_ptr->grad()[idx] += dx_norm * inv_std[c]
                                    + dvar[c] * 2.0f * (input_ptr->data()[idx] - mean[c]) / static_cast<float>(n)
                                    + dmean[c] / static_cast<float>(n);
                            }
                        }
                    }
                }
            }

            if (gamma_ptr->requires_grad) {
                for (size_t c = 0; c < channels; c++) {
                    gamma_ptr->grad()[c] += dgamma[c];
                }
            }
            if (beta_ptr->requires_grad) {
                for (size_t c = 0; c < channels; c++) {
                    beta_ptr->grad()[c] += dbeta[c];
                }
            }
        };
    }

    return result;
}

std::vector<TensorPtr> BatchNorm2d::parameters() {
    return {gamma, beta};
}

LayerNorm::LayerNorm(std::vector<size_t> normalized_shape, float eps)
    : normalized_shape(normalized_shape), eps(eps) {
    size_t size = 1;
    for (auto s : normalized_shape) size *= s;

    gamma = Tensor::ones({size}, true);
    beta = Tensor::zeros({size}, true);
}

LayerNorm::LayerNorm(size_t dim, float eps)
    : LayerNorm(std::vector<size_t>{dim}, eps) {}

TensorPtr LayerNorm::forward(const TensorPtr& input) {
    if (input->device != whitematter::DeviceType::CPU) {
#if defined(WHITEMATTER_CUDA)
        auto orig_device = input->device;

        // Transfer input to CPU for computation
        auto cpu_input = input->to(whitematter::DeviceType::CPU);
        cpu_input->requires_grad = input->requires_grad;

        // Temporarily move params to CPU
        gamma->to_inplace(whitematter::DeviceType::CPU);
        beta->to_inplace(whitematter::DeviceType::CPU);

        auto cpu_result = forward(cpu_input);  // recursive call on CPU path

        // Move params back to original device
        gamma->to_inplace(orig_device);
        beta->to_inplace(orig_device);

        // Create CUDA result and copy forward data
        auto result = Tensor::create_on_device(cpu_result->shape, cpu_result->requires_grad, orig_device);
        whitematter::CUDABackend::instance().memcpy_h2d(result->data(), cpu_result->data(), cpu_result->size());

        if (result->requires_grad && GradMode::is_enabled()) {
            auto input_sptr = std::const_pointer_cast<Tensor>(input);
            auto gamma_ptr = gamma;
            auto beta_ptr = beta;
            result->parents = {input_sptr, gamma_ptr, beta_ptr};

            result->grad_fn = [input_sptr, gamma_ptr, beta_ptr,
                               cpu_input, cpu_result, result, orig_device]() {
                // Move gamma/beta to CPU so the CPU grad_fn can read/write them
                gamma_ptr->to_inplace(whitematter::DeviceType::CPU);
                beta_ptr->to_inplace(whitematter::DeviceType::CPU);

                // Copy grad from CUDA result to CPU result
                whitematter::CUDABackend::instance().memcpy_d2h(
                    cpu_result->grad(), result->grad(), result->size());

                // Run the CPU backward pass for this node
                if (cpu_result->grad_fn) cpu_result->grad_fn();

                // Move gamma/beta back to CUDA (grads now updated on CPU, transferred with to_inplace)
                gamma_ptr->to_inplace(orig_device);
                beta_ptr->to_inplace(orig_device);

                // Copy input grad from CPU to CUDA parent (accumulate)
                if (input_sptr->requires_grad && cpu_input->grad()) {
                    size_t n = input_sptr->size();
                    std::vector<float> cpu_grad(n);
                    std::memcpy(cpu_grad.data(), cpu_input->grad(), n * sizeof(float));
                    std::vector<float> existing(n);
                    whitematter::CUDABackend::instance().memcpy_d2h(
                        existing.data(), input_sptr->grad(), n);
                    for (size_t i = 0; i < n; i++) existing[i] += cpu_grad[i];
                    whitematter::CUDABackend::instance().memcpy_h2d(
                        input_sptr->grad(), existing.data(), n);
                }
            };
        }

        return result;
#endif
    }

    size_t norm_size = 1;
    for (auto s : normalized_shape) norm_size *= s;

    size_t ndim = normalized_shape.size();
    assert(input->shape.size() >= ndim);
    for (size_t i = 0; i < ndim; i++) {
        assert(input->shape[input->shape.size() - ndim + i] == normalized_shape[i]);
    }

    size_t num_instances = 1;
    for (size_t i = 0; i < input->shape.size() - ndim; i++) {
        num_instances *= input->shape[i];
    }

    bool track = input->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create(input->shape, track);

    std::vector<float> mean(num_instances, 0.0f);
    std::vector<float> inv_std(num_instances, 0.0f);

    for (size_t n = 0; n < num_instances; n++) {
        float sum = 0.0f;
        for (size_t i = 0; i < norm_size; i++) {
            sum += input->data()[n * norm_size + i];
        }
        mean[n] = sum / static_cast<float>(norm_size);

        float var_sum = 0.0f;
        for (size_t i = 0; i < norm_size; i++) {
            float diff = input->data()[n * norm_size + i] - mean[n];
            var_sum += diff * diff;
        }
        float var = var_sum / static_cast<float>(norm_size);
        inv_std[n] = 1.0f / std::sqrt(var + eps);
    }

    for (size_t n = 0; n < num_instances; n++) {
        for (size_t i = 0; i < norm_size; i++) {
            float x_norm = (input->data()[n * norm_size + i] - mean[n]) * inv_std[n];
            result->data()[n * norm_size + i] = gamma->data()[i] * x_norm + beta->data()[i];
        }
    }

    if (track) {
        auto input_ptr = input;
        auto gamma_ptr = gamma;
        auto beta_ptr = beta;
        result->parents = {input_ptr, gamma_ptr, beta_ptr};

        result->grad_fn = [input_ptr, gamma_ptr, beta_ptr, result,
                           mean, inv_std, num_instances, norm_size]() {
            std::vector<float> dgamma(norm_size, 0.0f);
            std::vector<float> dbeta(norm_size, 0.0f);

            for (size_t n = 0; n < num_instances; n++) {
                for (size_t i = 0; i < norm_size; i++) {
                    float x_norm = (input_ptr->data()[n * norm_size + i] - mean[n]) * inv_std[n];
                    dgamma[i] += result->grad()[n * norm_size + i] * x_norm;
                    dbeta[i] += result->grad()[n * norm_size + i];
                }

                float sum_dy_gamma = 0.0f;
                float sum_dy_gamma_xnorm = 0.0f;

                for (size_t i = 0; i < norm_size; i++) {
                    float dy = result->grad()[n * norm_size + i];
                    float x_norm = (input_ptr->data()[n * norm_size + i] - mean[n]) * inv_std[n];
                    sum_dy_gamma += dy * gamma_ptr->data()[i];
                    sum_dy_gamma_xnorm += dy * gamma_ptr->data()[i] * x_norm;
                }

                float mean_dy_gamma = sum_dy_gamma / static_cast<float>(norm_size);
                float mean_dy_gamma_xnorm = sum_dy_gamma_xnorm / static_cast<float>(norm_size);

                if (input_ptr->requires_grad) {
                    for (size_t i = 0; i < norm_size; i++) {
                        float dy = result->grad()[n * norm_size + i];
                        float x_norm = (input_ptr->data()[n * norm_size + i] - mean[n]) * inv_std[n];
                        input_ptr->grad()[n * norm_size + i] += inv_std[n] *
                            (dy * gamma_ptr->data()[i] - mean_dy_gamma - x_norm * mean_dy_gamma_xnorm);
                    }
                }
            }

            if (gamma_ptr->requires_grad) {
                for (size_t i = 0; i < norm_size; i++) {
                    gamma_ptr->grad()[i] += dgamma[i];
                }
            }
            if (beta_ptr->requires_grad) {
                for (size_t i = 0; i < norm_size; i++) {
                    beta_ptr->grad()[i] += dbeta[i];
                }
            }
        };
    }

    return result;
}

std::vector<TensorPtr> LayerNorm::parameters() {
    return {gamma, beta};
}

std::string BatchNorm2d::extra_repr() const {
    return std::to_string(num_features) +
           ", eps=" + std::to_string(eps) +
           ", momentum=" + std::to_string(momentum);
}

std::string LayerNorm::extra_repr() const {
    std::string shape_str = "[";
    for (size_t i = 0; i < normalized_shape.size(); i++) {
        if (i > 0) shape_str += ", ";
        shape_str += std::to_string(normalized_shape[i]);
    }
    shape_str += "]";
    return shape_str + ", eps=" + std::to_string(eps);
}

// =============================================================================
// GroupNorm
// =============================================================================

GroupNorm::GroupNorm(size_t num_groups, size_t num_channels, float eps)
    : num_groups(num_groups), num_channels(num_channels), eps(eps) {
    assert(num_channels % num_groups == 0);
    gamma = Tensor::ones({num_channels}, true);
    beta = Tensor::zeros({num_channels}, true);
}

TensorPtr GroupNorm::forward(const TensorPtr& input) {
    // Supports 3D [batch, channels, length] or 4D [batch, channels, H, W]
    assert(input->shape.size() >= 3);
    assert(input->shape[1] == num_channels);

    size_t batch = input->shape[0];
    size_t channels = input->shape[1];
    size_t channels_per_group = channels / num_groups;

    // Compute spatial size (everything after the channel dim)
    size_t spatial_size = 1;
    for (size_t d = 2; d < input->shape.size(); d++) {
        spatial_size *= input->shape[d];
    }

    size_t group_size = channels_per_group * spatial_size;  // elements per group

    bool track = input->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create(input->shape, track);

    // Per (batch, group) mean and inv_std
    size_t num_stats = batch * num_groups;
    std::vector<float> mean(num_stats, 0.0f);
    std::vector<float> inv_std(num_stats, 0.0f);

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < num_groups; g++) {
            size_t stat_idx = b * num_groups + g;
            float sum = 0.0f;
            for (size_t c = g * channels_per_group; c < (g + 1) * channels_per_group; c++) {
                size_t base = b * channels * spatial_size + c * spatial_size;
                for (size_t i = 0; i < spatial_size; i++) {
                    sum += input->data()[base + i];
                }
            }
            mean[stat_idx] = sum / static_cast<float>(group_size);

            float var_sum = 0.0f;
            for (size_t c = g * channels_per_group; c < (g + 1) * channels_per_group; c++) {
                size_t base = b * channels * spatial_size + c * spatial_size;
                for (size_t i = 0; i < spatial_size; i++) {
                    float diff = input->data()[base + i] - mean[stat_idx];
                    var_sum += diff * diff;
                }
            }
            float var = var_sum / static_cast<float>(group_size);
            inv_std[stat_idx] = 1.0f / std::sqrt(var + eps);
        }
    }

    // Normalize and apply affine
    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            size_t g = c / channels_per_group;
            size_t stat_idx = b * num_groups + g;
            size_t base = b * channels * spatial_size + c * spatial_size;
            for (size_t i = 0; i < spatial_size; i++) {
                float x_norm = (input->data()[base + i] - mean[stat_idx]) * inv_std[stat_idx];
                result->data()[base + i] = gamma->data()[c] * x_norm + beta->data()[c];
            }
        }
    }

    if (track) {
        auto input_ptr = input;
        auto gamma_ptr = gamma;
        auto beta_ptr = beta;
        result->parents = {input_ptr, gamma_ptr, beta_ptr};

        auto num_groups_ = num_groups;
        auto channels_per_group_ = channels_per_group;

        result->grad_fn = [input_ptr, gamma_ptr, beta_ptr, result,
                           mean, inv_std, batch, channels, spatial_size,
                           num_groups_, channels_per_group_]() {
            size_t group_size = channels_per_group_ * spatial_size;

            std::vector<float> dgamma(channels, 0.0f);
            std::vector<float> dbeta(channels, 0.0f);

            for (size_t b = 0; b < batch; b++) {
                for (size_t g = 0; g < num_groups_; g++) {
                    size_t stat_idx = b * num_groups_ + g;
                    float is = inv_std[stat_idx];
                    float m = mean[stat_idx];

                    // Accumulate sums for efficient backward
                    float sum_dy_gamma = 0.0f;
                    float sum_dy_gamma_xnorm = 0.0f;

                    for (size_t c = g * channels_per_group_; c < (g + 1) * channels_per_group_; c++) {
                        size_t base = b * channels * spatial_size + c * spatial_size;
                        for (size_t i = 0; i < spatial_size; i++) {
                            float dy = result->grad()[base + i];
                            float x_norm = (input_ptr->data()[base + i] - m) * is;
                            dgamma[c] += dy * x_norm;
                            dbeta[c] += dy;
                            sum_dy_gamma += dy * gamma_ptr->data()[c];
                            sum_dy_gamma_xnorm += dy * gamma_ptr->data()[c] * x_norm;
                        }
                    }

                    float mean_dy_gamma = sum_dy_gamma / static_cast<float>(group_size);
                    float mean_dy_gamma_xnorm = sum_dy_gamma_xnorm / static_cast<float>(group_size);

                    if (input_ptr->requires_grad) {
                        for (size_t c = g * channels_per_group_; c < (g + 1) * channels_per_group_; c++) {
                            size_t base = b * channels * spatial_size + c * spatial_size;
                            for (size_t i = 0; i < spatial_size; i++) {
                                float dy = result->grad()[base + i];
                                float x_norm = (input_ptr->data()[base + i] - m) * is;
                                input_ptr->grad()[base + i] += is *
                                    (dy * gamma_ptr->data()[c] - mean_dy_gamma - x_norm * mean_dy_gamma_xnorm);
                            }
                        }
                    }
                }
            }

            if (gamma_ptr->requires_grad) {
                for (size_t c = 0; c < channels; c++) {
                    gamma_ptr->grad()[c] += dgamma[c];
                }
            }
            if (beta_ptr->requires_grad) {
                for (size_t c = 0; c < channels; c++) {
                    beta_ptr->grad()[c] += dbeta[c];
                }
            }
        };
    }

    return result;
}

std::vector<TensorPtr> GroupNorm::parameters() {
    return {gamma, beta};
}

std::string GroupNorm::extra_repr() const {
    return std::to_string(num_groups) + ", " + std::to_string(num_channels) +
           ", eps=" + std::to_string(eps);
}

// =============================================================================
// RMSNorm
// =============================================================================

RMSNorm::RMSNorm(size_t dim, float eps)
    : dim(dim), eps(eps) {
    gamma = Tensor::ones({dim}, true);
}

TensorPtr RMSNorm::forward(const TensorPtr& input) {
    // Normalizes over the last dimension
    assert(input->shape.back() == dim);

    size_t num_instances = 1;
    for (size_t i = 0; i < input->shape.size() - 1; i++) {
        num_instances *= input->shape[i];
    }

    bool track = input->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create(input->shape, track);

    std::vector<float> rms(num_instances, 0.0f);

    for (size_t n = 0; n < num_instances; n++) {
        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            float val = input->data()[n * dim + i];
            sum_sq += val * val;
        }
        rms[n] = std::sqrt(sum_sq / static_cast<float>(dim) + eps);
    }

    for (size_t n = 0; n < num_instances; n++) {
        float inv_rms = 1.0f / rms[n];
        for (size_t i = 0; i < dim; i++) {
            result->data()[n * dim + i] = gamma->data()[i] * input->data()[n * dim + i] * inv_rms;
        }
    }

    if (track) {
        auto input_ptr = input;
        auto gamma_ptr = gamma;
        result->parents = {input_ptr, gamma_ptr};

        auto dim_ = dim;

        result->grad_fn = [input_ptr, gamma_ptr, result, rms, num_instances, dim_]() {
            std::vector<float> dgamma(dim_, 0.0f);

            for (size_t n = 0; n < num_instances; n++) {
                float inv_rms = 1.0f / rms[n];
                float inv_rms3 = inv_rms * inv_rms * inv_rms;

                // Compute sum(dy * gamma * x) for the chain rule through rms
                float sum_dy_gamma_x = 0.0f;
                for (size_t i = 0; i < dim_; i++) {
                    float dy = result->grad()[n * dim_ + i];
                    sum_dy_gamma_x += dy * gamma_ptr->data()[i] * input_ptr->data()[n * dim_ + i];
                }

                // dgamma accumulation
                for (size_t i = 0; i < dim_; i++) {
                    dgamma[i] += result->grad()[n * dim_ + i] * input_ptr->data()[n * dim_ + i] * inv_rms;
                }

                // dx = gamma * (dy / rms - x * sum(dy * gamma * x) / (dim * rms^3))
                if (input_ptr->requires_grad) {
                    for (size_t i = 0; i < dim_; i++) {
                        float dy = result->grad()[n * dim_ + i];
                        float x = input_ptr->data()[n * dim_ + i];
                        input_ptr->grad()[n * dim_ + i] += gamma_ptr->data()[i] *
                            (dy * inv_rms - x * sum_dy_gamma_x * inv_rms3 / static_cast<float>(dim_));
                    }
                }
            }

            if (gamma_ptr->requires_grad) {
                for (size_t i = 0; i < dim_; i++) {
                    gamma_ptr->grad()[i] += dgamma[i];
                }
            }
        };
    }

    return result;
}

std::vector<TensorPtr> RMSNorm::parameters() {
    return {gamma};
}

std::string RMSNorm::extra_repr() const {
    return std::to_string(dim) + ", eps=" + std::to_string(eps);
}
