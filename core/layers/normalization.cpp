#include "../layer.h"
#include <cmath>
#include <cassert>

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
