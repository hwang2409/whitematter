#include "layer.h"
#include <random>
#include <cmath>

static std::mt19937 layer_rng(123);

Linear::Linear(size_t in_features, size_t out_features)
    : in_features(in_features), out_features(out_features) {
    weight = Tensor::xavier(in_features, out_features, true);
    bias = Tensor::zeros({out_features}, true);
}

TensorPtr Linear::forward(const TensorPtr& input) {
    return input->matmul(weight)->add(bias);
}

std::vector<TensorPtr> Linear::parameters() {
    return {weight, bias};
}

TensorPtr ReLU::forward(const TensorPtr& input) {
    return input->relu();
}

TensorPtr Sigmoid::forward(const TensorPtr& input) {
    return input->sigmoid();
}

TensorPtr Tanh::forward(const TensorPtr& input) {
    return input->tanh_();
}

Softmax::Softmax(int dim) : dim(dim) {}

TensorPtr Softmax::forward(const TensorPtr& input) {
    return input->softmax(dim);
}

LogSoftmax::LogSoftmax(int dim) : dim(dim) {}

TensorPtr LogSoftmax::forward(const TensorPtr& input) {
    return input->log_softmax(dim);
}

Dropout::Dropout(float p) : p(p), training(true) {}

TensorPtr Dropout::forward(const TensorPtr& input) {
    if (!training || p == 0.0f) {
        return input;
    }

    auto result = Tensor::create(input->shape, input->requires_grad);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float scale = 1.0f / (1.0f - p);
    std::vector<float> mask(input->data.size());

    for (size_t i = 0; i < input->data.size(); i++) {
        mask[i] = (dist(layer_rng) > p) ? scale : 0.0f;
        result->data[i] = input->data[i] * mask[i];
    }

    if (result->requires_grad) {
        result->parents = {input};
        result->grad_fn = [input, result, mask]() {
            for (size_t i = 0; i < input->data.size(); i++) {
                input->grad[i] += result->grad[i] * mask[i];
            }
        };
    }

    return result;
}

Sequential::Sequential(std::initializer_list<Module*> modules) {
    for (auto m : modules) {
        layers.push_back(m);
    }
}

Sequential::~Sequential() {
    for (auto m : layers) {
        delete m;
    }
}

void Sequential::add(Module* module) {
    layers.push_back(module);
}

TensorPtr Sequential::forward(const TensorPtr& input) {
    TensorPtr x = input;
    for (auto& layer : layers) {
        x = layer->forward(x);
    }
    return x;
}

std::vector<TensorPtr> Sequential::parameters() {
    std::vector<TensorPtr> params;
    for (auto& layer : layers) {
        auto layer_params = layer->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

void Sequential::train() {
    for (auto& layer : layers) {
        if (auto dropout = dynamic_cast<Dropout*>(layer)) {
            dropout->train();
        } else if (auto bn = dynamic_cast<BatchNorm2d*>(layer)) {
            bn->train();
        }
    }
}

void Sequential::eval() {
    for (auto& layer : layers) {
        if (auto dropout = dynamic_cast<Dropout*>(layer)) {
            dropout->eval();
        } else if (auto bn = dynamic_cast<BatchNorm2d*>(layer)) {
            bn->eval();
        }
    }
}

// Conv2d implementation
Conv2d::Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
               size_t stride, size_t padding)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_size(kernel_size), stride(stride), padding(padding) {
    // Kaiming/He initialization for ReLU
    float std = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    std::normal_distribution<float> dist(0.0f, std);

    weight = Tensor::create({out_channels, in_channels, kernel_size, kernel_size}, true);
    for (auto& v : weight->data) v = dist(layer_rng);

    bias = Tensor::zeros({out_channels}, true);
}

TensorPtr Conv2d::forward(const TensorPtr& input) {
    return input->conv2d(weight, bias, stride, padding);
}

std::vector<TensorPtr> Conv2d::parameters() {
    return {weight, bias};
}

// MaxPool2d implementation
MaxPool2d::MaxPool2d(size_t kernel_size, size_t stride)
    : kernel_size(kernel_size), stride(stride == 0 ? kernel_size : stride) {}

TensorPtr MaxPool2d::forward(const TensorPtr& input) {
    return input->maxpool2d(kernel_size, stride);
}

// AvgPool2d implementation
AvgPool2d::AvgPool2d(size_t kernel_size, size_t stride)
    : kernel_size(kernel_size), stride(stride == 0 ? kernel_size : stride) {}

TensorPtr AvgPool2d::forward(const TensorPtr& input) {
    return input->avgpool2d(kernel_size, stride);
}

// BatchNorm2d implementation
BatchNorm2d::BatchNorm2d(size_t num_features, float eps, float momentum)
    : num_features(num_features), eps(eps), momentum(momentum), training(true) {
    gamma = Tensor::ones({num_features}, true);
    beta = Tensor::zeros({num_features}, true);
    running_mean = Tensor::zeros({num_features}, false);
    running_var = Tensor::ones({num_features}, false);
}

TensorPtr BatchNorm2d::forward(const TensorPtr& input) {
    // Input shape: (batch, channels, height, width)
    assert(input->shape.size() == 4);
    assert(input->shape[1] == num_features);

    size_t batch = input->shape[0];
    size_t channels = input->shape[1];
    size_t height = input->shape[2];
    size_t width = input->shape[3];
    size_t spatial_size = height * width;
    size_t n = batch * spatial_size;  // number of elements per channel

    bool track = input->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create(input->shape, track);

    // Store mean and var for backward pass
    std::vector<float> mean(channels, 0.0f);
    std::vector<float> var(channels, 0.0f);

    if (training) {
        // Compute batch mean and variance per channel
        for (size_t c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (size_t b = 0; b < batch; b++) {
                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        size_t idx = b * (channels * spatial_size) + c * spatial_size + h * width + w;
                        sum += input->data[idx];
                    }
                }
            }
            mean[c] = sum / static_cast<float>(n);
        }

        for (size_t c = 0; c < channels; c++) {
            float sum_sq = 0.0f;
            for (size_t b = 0; b < batch; b++) {
                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        size_t idx = b * (channels * spatial_size) + c * spatial_size + h * width + w;
                        float diff = input->data[idx] - mean[c];
                        sum_sq += diff * diff;
                    }
                }
            }
            var[c] = sum_sq / static_cast<float>(n);
        }

        // Update running statistics
        for (size_t c = 0; c < channels; c++) {
            running_mean->data[c] = (1.0f - momentum) * running_mean->data[c] + momentum * mean[c];
            running_var->data[c] = (1.0f - momentum) * running_var->data[c] + momentum * var[c];
        }
    } else {
        // Use running statistics for inference
        for (size_t c = 0; c < channels; c++) {
            mean[c] = running_mean->data[c];
            var[c] = running_var->data[c];
        }
    }

    // Normalize and apply scale/shift
    std::vector<float> inv_std(channels);
    for (size_t c = 0; c < channels; c++) {
        inv_std[c] = 1.0f / std::sqrt(var[c] + eps);
    }

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    size_t idx = b * (channels * spatial_size) + c * spatial_size + h * width + w;
                    float x_norm = (input->data[idx] - mean[c]) * inv_std[c];
                    result->data[idx] = gamma->data[c] * x_norm + beta->data[c];
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
            // Gradients for BatchNorm
            // dx_norm = dout * gamma
            // dvar = sum(dx_norm * (x - mean) * -0.5 * (var + eps)^(-1.5))
            // dmean = sum(dx_norm * -inv_std) + dvar * sum(-2 * (x - mean)) / n
            // dx = dx_norm * inv_std + dvar * 2 * (x - mean) / n + dmean / n

            std::vector<float> dgamma(channels, 0.0f);
            std::vector<float> dbeta(channels, 0.0f);
            std::vector<float> dmean(channels, 0.0f);
            std::vector<float> dvar(channels, 0.0f);

            // Compute dgamma, dbeta, and intermediate gradients
            for (size_t c = 0; c < channels; c++) {
                for (size_t b = 0; b < batch; b++) {
                    for (size_t h = 0; h < height; h++) {
                        for (size_t w = 0; w < width; w++) {
                            size_t idx = b * (channels * spatial_size) + c * spatial_size + h * width + w;
                            float x_norm = (input_ptr->data[idx] - mean[c]) * inv_std[c];
                            dgamma[c] += result->grad[idx] * x_norm;
                            dbeta[c] += result->grad[idx];
                        }
                    }
                }
            }

            // Compute dvar
            for (size_t c = 0; c < channels; c++) {
                for (size_t b = 0; b < batch; b++) {
                    for (size_t h = 0; h < height; h++) {
                        for (size_t w = 0; w < width; w++) {
                            size_t idx = b * (channels * spatial_size) + c * spatial_size + h * width + w;
                            float dx_norm = result->grad[idx] * gamma_ptr->data[c];
                            dvar[c] += dx_norm * (input_ptr->data[idx] - mean[c]) * -0.5f * inv_std[c] * inv_std[c] * inv_std[c];
                        }
                    }
                }
            }

            // Compute dmean
            for (size_t c = 0; c < channels; c++) {
                float sum_dx_norm = 0.0f;
                float sum_x_diff = 0.0f;
                for (size_t b = 0; b < batch; b++) {
                    for (size_t h = 0; h < height; h++) {
                        for (size_t w = 0; w < width; w++) {
                            size_t idx = b * (channels * spatial_size) + c * spatial_size + h * width + w;
                            sum_dx_norm += result->grad[idx] * gamma_ptr->data[c] * (-inv_std[c]);
                            sum_x_diff += -2.0f * (input_ptr->data[idx] - mean[c]);
                        }
                    }
                }
                dmean[c] = sum_dx_norm + dvar[c] * sum_x_diff / static_cast<float>(n);
            }

            // Compute dx
            if (input_ptr->requires_grad) {
                for (size_t b = 0; b < batch; b++) {
                    for (size_t c = 0; c < channels; c++) {
                        for (size_t h = 0; h < height; h++) {
                            for (size_t w = 0; w < width; w++) {
                                size_t idx = b * (channels * spatial_size) + c * spatial_size + h * width + w;
                                float dx_norm = result->grad[idx] * gamma_ptr->data[c];
                                input_ptr->grad[idx] += dx_norm * inv_std[c]
                                    + dvar[c] * 2.0f * (input_ptr->data[idx] - mean[c]) / static_cast<float>(n)
                                    + dmean[c] / static_cast<float>(n);
                            }
                        }
                    }
                }
            }

            // Apply gradients to gamma and beta
            if (gamma_ptr->requires_grad) {
                for (size_t c = 0; c < channels; c++) {
                    gamma_ptr->grad[c] += dgamma[c];
                }
            }
            if (beta_ptr->requires_grad) {
                for (size_t c = 0; c < channels; c++) {
                    beta_ptr->grad[c] += dbeta[c];
                }
            }
        };
    }

    return result;
}

std::vector<TensorPtr> BatchNorm2d::parameters() {
    return {gamma, beta};
}

// LayerNorm implementation
LayerNorm::LayerNorm(std::vector<size_t> normalized_shape, float eps)
    : normalized_shape(normalized_shape), eps(eps) {
    // Compute total size of normalized dimensions
    size_t size = 1;
    for (auto s : normalized_shape) size *= s;

    gamma = Tensor::ones({size}, true);
    beta = Tensor::zeros({size}, true);
}

LayerNorm::LayerNorm(size_t dim, float eps)
    : LayerNorm(std::vector<size_t>{dim}, eps) {}

TensorPtr LayerNorm::forward(const TensorPtr& input) {
    // LayerNorm normalizes over the last len(normalized_shape) dimensions
    // Input can be any shape, we normalize over the trailing dimensions

    size_t norm_size = 1;
    for (auto s : normalized_shape) norm_size *= s;

    // Verify input shape matches normalized_shape at the end
    size_t ndim = normalized_shape.size();
    assert(input->shape.size() >= ndim);
    for (size_t i = 0; i < ndim; i++) {
        assert(input->shape[input->shape.size() - ndim + i] == normalized_shape[i]);
    }

    // Compute number of instances to normalize (product of leading dimensions)
    size_t num_instances = 1;
    for (size_t i = 0; i < input->shape.size() - ndim; i++) {
        num_instances *= input->shape[i];
    }

    bool track = input->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create(input->shape, track);

    // Store mean and inv_std for backward pass
    std::vector<float> mean(num_instances, 0.0f);
    std::vector<float> inv_std(num_instances, 0.0f);

    // Compute mean and variance for each instance
    for (size_t n = 0; n < num_instances; n++) {
        float sum = 0.0f;
        for (size_t i = 0; i < norm_size; i++) {
            sum += input->data[n * norm_size + i];
        }
        mean[n] = sum / static_cast<float>(norm_size);

        float var_sum = 0.0f;
        for (size_t i = 0; i < norm_size; i++) {
            float diff = input->data[n * norm_size + i] - mean[n];
            var_sum += diff * diff;
        }
        float var = var_sum / static_cast<float>(norm_size);
        inv_std[n] = 1.0f / std::sqrt(var + eps);
    }

    // Normalize and apply scale/shift
    for (size_t n = 0; n < num_instances; n++) {
        for (size_t i = 0; i < norm_size; i++) {
            float x_norm = (input->data[n * norm_size + i] - mean[n]) * inv_std[n];
            result->data[n * norm_size + i] = gamma->data[i] * x_norm + beta->data[i];
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
                // Compute dgamma and dbeta
                for (size_t i = 0; i < norm_size; i++) {
                    float x_norm = (input_ptr->data[n * norm_size + i] - mean[n]) * inv_std[n];
                    dgamma[i] += result->grad[n * norm_size + i] * x_norm;
                    dbeta[i] += result->grad[n * norm_size + i];
                }

                // Compute dx using the LayerNorm backward formula
                // dx = (1/std) * (dout * gamma - mean(dout * gamma) - x_norm * mean(dout * gamma * x_norm))
                float sum_dy_gamma = 0.0f;
                float sum_dy_gamma_xnorm = 0.0f;

                for (size_t i = 0; i < norm_size; i++) {
                    float dy = result->grad[n * norm_size + i];
                    float x_norm = (input_ptr->data[n * norm_size + i] - mean[n]) * inv_std[n];
                    sum_dy_gamma += dy * gamma_ptr->data[i];
                    sum_dy_gamma_xnorm += dy * gamma_ptr->data[i] * x_norm;
                }

                float mean_dy_gamma = sum_dy_gamma / static_cast<float>(norm_size);
                float mean_dy_gamma_xnorm = sum_dy_gamma_xnorm / static_cast<float>(norm_size);

                if (input_ptr->requires_grad) {
                    for (size_t i = 0; i < norm_size; i++) {
                        float dy = result->grad[n * norm_size + i];
                        float x_norm = (input_ptr->data[n * norm_size + i] - mean[n]) * inv_std[n];
                        input_ptr->grad[n * norm_size + i] += inv_std[n] *
                            (dy * gamma_ptr->data[i] - mean_dy_gamma - x_norm * mean_dy_gamma_xnorm);
                    }
                }
            }

            // Apply gradients to gamma and beta
            if (gamma_ptr->requires_grad) {
                for (size_t i = 0; i < norm_size; i++) {
                    gamma_ptr->grad[i] += dgamma[i];
                }
            }
            if (beta_ptr->requires_grad) {
                for (size_t i = 0; i < norm_size; i++) {
                    beta_ptr->grad[i] += dbeta[i];
                }
            }
        };
    }

    return result;
}

std::vector<TensorPtr> LayerNorm::parameters() {
    return {gamma, beta};
}

// Flatten implementation
TensorPtr Flatten::forward(const TensorPtr& input) {
    return input->flatten(1);
}

// Embedding implementation
Embedding::Embedding(size_t num_embeddings, size_t embedding_dim)
    : num_embeddings(num_embeddings), embedding_dim(embedding_dim) {
    // Initialize with normal distribution (like PyTorch)
    weight = Tensor::randn({num_embeddings, embedding_dim}, true);
}

TensorPtr Embedding::forward(const TensorPtr& indices) {
    // Input: indices of shape [batch_size, seq_len] or [seq_len]
    // Output: embeddings of shape [batch_size, seq_len, embedding_dim] or [seq_len, embedding_dim]

    std::vector<size_t> out_shape;
    for (size_t dim : indices->shape) {
        out_shape.push_back(dim);
    }
    out_shape.push_back(embedding_dim);

    size_t num_indices = indices->data.size();
    bool track = weight->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create(out_shape, track);

    // Forward: lookup embeddings
    for (size_t i = 0; i < num_indices; i++) {
        size_t idx = static_cast<size_t>(indices->data[i]);
        assert(idx < num_embeddings);
        for (size_t j = 0; j < embedding_dim; j++) {
            result->data[i * embedding_dim + j] = weight->data[idx * embedding_dim + j];
        }
    }

    if (track) {
        auto weight_ptr = weight;
        auto indices_ptr = indices;
        result->parents = {weight_ptr};
        result->grad_fn = [weight_ptr, indices_ptr, result, num_indices, this]() {
            // Backward: accumulate gradients for each embedding
            for (size_t i = 0; i < num_indices; i++) {
                size_t idx = static_cast<size_t>(indices_ptr->data[i]);
                for (size_t j = 0; j < embedding_dim; j++) {
                    weight_ptr->grad[idx * embedding_dim + j] += result->grad[i * embedding_dim + j];
                }
            }
        };
    }

    return result;
}

std::vector<TensorPtr> Embedding::parameters() {
    return {weight};
}

// LSTM implementation
LSTM::LSTM(size_t input_size, size_t hidden_size, bool batch_first)
    : input_size(input_size), hidden_size(hidden_size), batch_first(batch_first) {
    // Xavier/Glorot initialization
    float std_ih = std::sqrt(2.0f / (input_size + hidden_size));
    float std_hh = std::sqrt(2.0f / (hidden_size + hidden_size));
    std::normal_distribution<float> dist_ih(0.0f, std_ih);
    std::normal_distribution<float> dist_hh(0.0f, std_hh);

    // Weight matrices: [4*hidden_size, input_size] and [4*hidden_size, hidden_size]
    weight_ih = Tensor::create({4 * hidden_size, input_size}, true);
    weight_hh = Tensor::create({4 * hidden_size, hidden_size}, true);
    bias_ih = Tensor::zeros({4 * hidden_size}, true);
    bias_hh = Tensor::zeros({4 * hidden_size}, true);

    for (auto& v : weight_ih->data) v = dist_ih(layer_rng);
    for (auto& v : weight_hh->data) v = dist_hh(layer_rng);

    // Initialize forget gate bias to 1.0 for better gradient flow
    for (size_t i = hidden_size; i < 2 * hidden_size; i++) {
        bias_ih->data[i] = 1.0f;
    }
}

TensorPtr LSTM::forward(const TensorPtr& input) {
    // Initialize hidden states to zeros
    size_t batch_size = batch_first ? input->shape[0] : input->shape[1];
    auto h0 = Tensor::zeros({batch_size, hidden_size}, false);
    auto c0 = Tensor::zeros({batch_size, hidden_size}, false);
    return forward(input, h0, c0);
}

TensorPtr LSTM::forward(const TensorPtr& input, const TensorPtr& h0, const TensorPtr& c0) {
    // Input: [batch, seq, input_size] if batch_first else [seq, batch, input_size]
    // Output: [batch, seq, hidden_size] if batch_first else [seq, batch, hidden_size]
    assert(input->shape.size() == 3);

    size_t batch_size, seq_len;
    if (batch_first) {
        batch_size = input->shape[0];
        seq_len = input->shape[1];
        assert(input->shape[2] == input_size);
    } else {
        seq_len = input->shape[0];
        batch_size = input->shape[1];
        assert(input->shape[2] == input_size);
    }

    bool track = input->requires_grad && GradMode::is_enabled();

    // Output tensor for all hidden states
    std::vector<size_t> out_shape;
    if (batch_first) {
        out_shape = {batch_size, seq_len, hidden_size};
    } else {
        out_shape = {seq_len, batch_size, hidden_size};
    }
    auto output = Tensor::create(out_shape, track);

    // Store intermediate values for backward pass
    std::vector<std::vector<float>> all_i(seq_len), all_f(seq_len), all_g(seq_len), all_o(seq_len);
    std::vector<std::vector<float>> all_c(seq_len + 1), all_h(seq_len + 1);
    std::vector<std::vector<float>> all_tanh_c(seq_len);

    // Initialize h and c from h0 and c0
    all_h[0].resize(batch_size * hidden_size);
    all_c[0].resize(batch_size * hidden_size);
    for (size_t i = 0; i < batch_size * hidden_size; i++) {
        all_h[0][i] = h0->data[i];
        all_c[0][i] = c0->data[i];
    }

    // Forward through time
    for (size_t t = 0; t < seq_len; t++) {
        all_i[t].resize(batch_size * hidden_size);
        all_f[t].resize(batch_size * hidden_size);
        all_g[t].resize(batch_size * hidden_size);
        all_o[t].resize(batch_size * hidden_size);
        all_c[t + 1].resize(batch_size * hidden_size);
        all_h[t + 1].resize(batch_size * hidden_size);
        all_tanh_c[t].resize(batch_size * hidden_size);

        for (size_t b = 0; b < batch_size; b++) {
            // Compute gates for each batch element
            // gates = x_t @ W_ih^T + h_{t-1} @ W_hh^T + b_ih + b_hh
            std::vector<float> gates(4 * hidden_size, 0.0f);

            // x_t @ W_ih^T
            size_t x_offset = batch_first ?
                (b * seq_len * input_size + t * input_size) :
                (t * batch_size * input_size + b * input_size);

            for (size_t g = 0; g < 4 * hidden_size; g++) {
                for (size_t i = 0; i < input_size; i++) {
                    gates[g] += input->data[x_offset + i] * weight_ih->data[g * input_size + i];
                }
                gates[g] += bias_ih->data[g];
            }

            // h_{t-1} @ W_hh^T
            for (size_t g = 0; g < 4 * hidden_size; g++) {
                for (size_t h = 0; h < hidden_size; h++) {
                    gates[g] += all_h[t][b * hidden_size + h] * weight_hh->data[g * hidden_size + h];
                }
                gates[g] += bias_hh->data[g];
            }

            // Split into i, f, g, o and apply activations
            for (size_t h = 0; h < hidden_size; h++) {
                size_t idx = b * hidden_size + h;
                float i_gate = 1.0f / (1.0f + std::exp(-gates[h]));                      // sigmoid
                float f_gate = 1.0f / (1.0f + std::exp(-gates[hidden_size + h]));        // sigmoid
                float g_gate = std::tanh(gates[2 * hidden_size + h]);                     // tanh
                float o_gate = 1.0f / (1.0f + std::exp(-gates[3 * hidden_size + h]));    // sigmoid

                all_i[t][idx] = i_gate;
                all_f[t][idx] = f_gate;
                all_g[t][idx] = g_gate;
                all_o[t][idx] = o_gate;

                // c_t = f_t * c_{t-1} + i_t * g_t
                float c_new = f_gate * all_c[t][idx] + i_gate * g_gate;
                all_c[t + 1][idx] = c_new;

                // h_t = o_t * tanh(c_t)
                float tanh_c = std::tanh(c_new);
                all_tanh_c[t][idx] = tanh_c;
                float h_new = o_gate * tanh_c;
                all_h[t + 1][idx] = h_new;

                // Store in output
                size_t out_offset = batch_first ?
                    (b * seq_len * hidden_size + t * hidden_size + h) :
                    (t * batch_size * hidden_size + b * hidden_size + h);
                output->data[out_offset] = h_new;
            }
        }
    }

    // Store final hidden and cell states
    h_n = Tensor::create({batch_size, hidden_size}, false);
    c_n = Tensor::create({batch_size, hidden_size}, false);
    for (size_t i = 0; i < batch_size * hidden_size; i++) {
        h_n->data[i] = all_h[seq_len][i];
        c_n->data[i] = all_c[seq_len][i];
    }

    if (track) {
        auto input_ptr = input;
        auto weight_ih_ptr = weight_ih;
        auto weight_hh_ptr = weight_hh;
        auto bias_ih_ptr = bias_ih;
        auto bias_hh_ptr = bias_hh;
        size_t hs = hidden_size;
        size_t is = input_size;
        bool bf = batch_first;

        output->parents = {input_ptr, weight_ih_ptr, weight_hh_ptr, bias_ih_ptr, bias_hh_ptr};
        output->grad_fn = [=]() mutable {
            // Backpropagation through time (BPTT)
            std::vector<float> dh_next(batch_size * hs, 0.0f);
            std::vector<float> dc_next(batch_size * hs, 0.0f);

            for (int t = static_cast<int>(seq_len) - 1; t >= 0; t--) {
                for (size_t b = 0; b < batch_size; b++) {
                    for (size_t h = 0; h < hs; h++) {
                        size_t idx = b * hs + h;
                        size_t out_offset = bf ?
                            (b * seq_len * hs + t * hs + h) :
                            (t * batch_size * hs + b * hs + h);

                        // Gradient from output and from next time step
                        float dh = output->grad[out_offset] + dh_next[idx];

                        // h_t = o_t * tanh(c_t)
                        float do_gate = dh * all_tanh_c[t][idx];
                        float dtanh_c = dh * all_o[t][idx];

                        // tanh'(c_t) = 1 - tanh^2(c_t)
                        float dc = dtanh_c * (1.0f - all_tanh_c[t][idx] * all_tanh_c[t][idx]) + dc_next[idx];

                        // c_t = f_t * c_{t-1} + i_t * g_t
                        float di_gate = dc * all_g[t][idx];
                        float df_gate = dc * all_c[t][idx];
                        float dg_gate = dc * all_i[t][idx];
                        dc_next[idx] = dc * all_f[t][idx];

                        // Gate gradients (pre-activation)
                        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
                        // tanh'(x) = 1 - tanh^2(x)
                        float di_pre = di_gate * all_i[t][idx] * (1.0f - all_i[t][idx]);
                        float df_pre = df_gate * all_f[t][idx] * (1.0f - all_f[t][idx]);
                        float dg_pre = dg_gate * (1.0f - all_g[t][idx] * all_g[t][idx]);
                        float do_pre = do_gate * all_o[t][idx] * (1.0f - all_o[t][idx]);

                        // Gradients for weights and biases
                        size_t x_offset = bf ?
                            (b * seq_len * is + t * is) :
                            (t * batch_size * is + b * is);

                        // d_bias_ih and d_bias_hh
                        bias_ih_ptr->grad[h] += di_pre;
                        bias_ih_ptr->grad[hs + h] += df_pre;
                        bias_ih_ptr->grad[2 * hs + h] += dg_pre;
                        bias_ih_ptr->grad[3 * hs + h] += do_pre;

                        bias_hh_ptr->grad[h] += di_pre;
                        bias_hh_ptr->grad[hs + h] += df_pre;
                        bias_hh_ptr->grad[2 * hs + h] += dg_pre;
                        bias_hh_ptr->grad[3 * hs + h] += do_pre;

                        // d_weight_ih: [4*hs, is]
                        for (size_t i = 0; i < is; i++) {
                            weight_ih_ptr->grad[h * is + i] += di_pre * input_ptr->data[x_offset + i];
                            weight_ih_ptr->grad[(hs + h) * is + i] += df_pre * input_ptr->data[x_offset + i];
                            weight_ih_ptr->grad[(2 * hs + h) * is + i] += dg_pre * input_ptr->data[x_offset + i];
                            weight_ih_ptr->grad[(3 * hs + h) * is + i] += do_pre * input_ptr->data[x_offset + i];
                        }

                        // d_weight_hh: [4*hs, hs]
                        for (size_t hh = 0; hh < hs; hh++) {
                            weight_hh_ptr->grad[h * hs + hh] += di_pre * all_h[t][b * hs + hh];
                            weight_hh_ptr->grad[(hs + h) * hs + hh] += df_pre * all_h[t][b * hs + hh];
                            weight_hh_ptr->grad[(2 * hs + h) * hs + hh] += dg_pre * all_h[t][b * hs + hh];
                            weight_hh_ptr->grad[(3 * hs + h) * hs + hh] += do_pre * all_h[t][b * hs + hh];
                        }

                        // d_input
                        if (input_ptr->requires_grad) {
                            for (size_t i = 0; i < is; i++) {
                                input_ptr->grad[x_offset + i] +=
                                    di_pre * weight_ih_ptr->data[h * is + i] +
                                    df_pre * weight_ih_ptr->data[(hs + h) * is + i] +
                                    dg_pre * weight_ih_ptr->data[(2 * hs + h) * is + i] +
                                    do_pre * weight_ih_ptr->data[(3 * hs + h) * is + i];
                            }
                        }

                        // d_h_prev
                        dh_next[idx] = 0.0f;
                        for (size_t hh = 0; hh < hs; hh++) {
                            dh_next[b * hs + hh] +=
                                di_pre * weight_hh_ptr->data[h * hs + hh] +
                                df_pre * weight_hh_ptr->data[(hs + h) * hs + hh] +
                                dg_pre * weight_hh_ptr->data[(2 * hs + h) * hs + hh] +
                                do_pre * weight_hh_ptr->data[(3 * hs + h) * hs + hh];
                        }
                    }
                }
            }
        };
    }

    return output;
}

std::vector<TensorPtr> LSTM::parameters() {
    return {weight_ih, weight_hh, bias_ih, bias_hh};
}

// GRU implementation
GRU::GRU(size_t input_size, size_t hidden_size, bool batch_first)
    : input_size(input_size), hidden_size(hidden_size), batch_first(batch_first) {
    // Xavier/Glorot initialization
    float std_ih = std::sqrt(2.0f / (input_size + hidden_size));
    float std_hh = std::sqrt(2.0f / (hidden_size + hidden_size));
    std::normal_distribution<float> dist_ih(0.0f, std_ih);
    std::normal_distribution<float> dist_hh(0.0f, std_hh);

    // Weight matrices: [3*hidden_size, input_size] and [3*hidden_size, hidden_size]
    weight_ih = Tensor::create({3 * hidden_size, input_size}, true);
    weight_hh = Tensor::create({3 * hidden_size, hidden_size}, true);
    bias_ih = Tensor::zeros({3 * hidden_size}, true);
    bias_hh = Tensor::zeros({3 * hidden_size}, true);

    for (auto& v : weight_ih->data) v = dist_ih(layer_rng);
    for (auto& v : weight_hh->data) v = dist_hh(layer_rng);
}

TensorPtr GRU::forward(const TensorPtr& input) {
    // Initialize hidden state to zeros
    size_t batch_size = batch_first ? input->shape[0] : input->shape[1];
    auto h0 = Tensor::zeros({batch_size, hidden_size}, false);
    return forward(input, h0);
}

TensorPtr GRU::forward(const TensorPtr& input, const TensorPtr& h0) {
    // Input: [batch, seq, input_size] if batch_first else [seq, batch, input_size]
    // Output: [batch, seq, hidden_size] if batch_first else [seq, batch, hidden_size]
    assert(input->shape.size() == 3);

    size_t batch_size, seq_len;
    if (batch_first) {
        batch_size = input->shape[0];
        seq_len = input->shape[1];
        assert(input->shape[2] == input_size);
    } else {
        seq_len = input->shape[0];
        batch_size = input->shape[1];
        assert(input->shape[2] == input_size);
    }

    bool track = input->requires_grad && GradMode::is_enabled();

    // Output tensor for all hidden states
    std::vector<size_t> out_shape;
    if (batch_first) {
        out_shape = {batch_size, seq_len, hidden_size};
    } else {
        out_shape = {seq_len, batch_size, hidden_size};
    }
    auto output = Tensor::create(out_shape, track);

    // Store intermediate values for backward pass
    std::vector<std::vector<float>> all_r(seq_len), all_z(seq_len), all_n(seq_len);
    std::vector<std::vector<float>> all_h(seq_len + 1);
    std::vector<std::vector<float>> all_hh_n(seq_len);  // W_hn @ h_{t-1} + b_hn before reset gate

    // Initialize h from h0
    all_h[0].resize(batch_size * hidden_size);
    for (size_t i = 0; i < batch_size * hidden_size; i++) {
        all_h[0][i] = h0->data[i];
    }

    // Forward through time
    for (size_t t = 0; t < seq_len; t++) {
        all_r[t].resize(batch_size * hidden_size);
        all_z[t].resize(batch_size * hidden_size);
        all_n[t].resize(batch_size * hidden_size);
        all_h[t + 1].resize(batch_size * hidden_size);
        all_hh_n[t].resize(batch_size * hidden_size);

        for (size_t b = 0; b < batch_size; b++) {
            // Compute gates for each batch element
            // r = sigmoid(x @ W_ir^T + b_ir + h @ W_hr^T + b_hr)
            // z = sigmoid(x @ W_iz^T + b_iz + h @ W_hz^T + b_hz)
            // n = tanh(x @ W_in^T + b_in + r * (h @ W_hn^T + b_hn))
            // h_new = (1 - z) * n + z * h

            std::vector<float> gates_ih(3 * hidden_size, 0.0f);
            std::vector<float> gates_hh(3 * hidden_size, 0.0f);

            size_t x_offset = batch_first ?
                (b * seq_len * input_size + t * input_size) :
                (t * batch_size * input_size + b * input_size);

            // x @ W_ih^T + b_ih
            for (size_t g = 0; g < 3 * hidden_size; g++) {
                for (size_t i = 0; i < input_size; i++) {
                    gates_ih[g] += input->data[x_offset + i] * weight_ih->data[g * input_size + i];
                }
                gates_ih[g] += bias_ih->data[g];
            }

            // h @ W_hh^T + b_hh
            for (size_t g = 0; g < 3 * hidden_size; g++) {
                for (size_t h = 0; h < hidden_size; h++) {
                    gates_hh[g] += all_h[t][b * hidden_size + h] * weight_hh->data[g * hidden_size + h];
                }
                gates_hh[g] += bias_hh->data[g];
            }

            // Apply gates
            for (size_t h = 0; h < hidden_size; h++) {
                size_t idx = b * hidden_size + h;

                // r = sigmoid(gates_ih[r] + gates_hh[r])
                float r_gate = 1.0f / (1.0f + std::exp(-(gates_ih[h] + gates_hh[h])));

                // z = sigmoid(gates_ih[z] + gates_hh[z])
                float z_gate = 1.0f / (1.0f + std::exp(-(gates_ih[hidden_size + h] + gates_hh[hidden_size + h])));

                // Store hh_n for backward pass (before reset gate multiplication)
                all_hh_n[t][idx] = gates_hh[2 * hidden_size + h];

                // n = tanh(gates_ih[n] + r * gates_hh[n])
                float n_gate = std::tanh(gates_ih[2 * hidden_size + h] + r_gate * gates_hh[2 * hidden_size + h]);

                // h_new = (1 - z) * n + z * h_prev
                float h_new = (1.0f - z_gate) * n_gate + z_gate * all_h[t][idx];

                all_r[t][idx] = r_gate;
                all_z[t][idx] = z_gate;
                all_n[t][idx] = n_gate;
                all_h[t + 1][idx] = h_new;

                // Store in output
                size_t out_offset = batch_first ?
                    (b * seq_len * hidden_size + t * hidden_size + h) :
                    (t * batch_size * hidden_size + b * hidden_size + h);
                output->data[out_offset] = h_new;
            }
        }
    }

    // Store final hidden state
    h_n = Tensor::create({batch_size, hidden_size}, false);
    for (size_t i = 0; i < batch_size * hidden_size; i++) {
        h_n->data[i] = all_h[seq_len][i];
    }

    if (track) {
        auto input_ptr = input;
        auto weight_ih_ptr = weight_ih;
        auto weight_hh_ptr = weight_hh;
        auto bias_ih_ptr = bias_ih;
        auto bias_hh_ptr = bias_hh;
        size_t hs = hidden_size;
        size_t is = input_size;
        bool bf = batch_first;

        output->parents = {input_ptr, weight_ih_ptr, weight_hh_ptr, bias_ih_ptr, bias_hh_ptr};
        output->grad_fn = [=]() mutable {
            // Backpropagation through time (BPTT)
            std::vector<float> dh_next(batch_size * hs, 0.0f);

            for (int t = static_cast<int>(seq_len) - 1; t >= 0; t--) {
                for (size_t b = 0; b < batch_size; b++) {
                    for (size_t h = 0; h < hs; h++) {
                        size_t idx = b * hs + h;
                        size_t out_offset = bf ?
                            (b * seq_len * hs + t * hs + h) :
                            (t * batch_size * hs + b * hs + h);

                        // Gradient from output and from next time step
                        float dh = output->grad[out_offset] + dh_next[idx];

                        // h_t = (1 - z_t) * n_t + z_t * h_{t-1}
                        float dz = dh * (all_h[t][idx] - all_n[t][idx]);
                        float dn = dh * (1.0f - all_z[t][idx]);
                        dh_next[idx] = dh * all_z[t][idx];

                        // n_t = tanh(...), so dn_pre = dn * (1 - n^2)
                        float dn_pre = dn * (1.0f - all_n[t][idx] * all_n[t][idx]);

                        // n = tanh(x @ W_in^T + b_in + r * (h @ W_hn^T + b_hn))
                        float dr_from_n = dn_pre * all_hh_n[t][idx];

                        // r_t = sigmoid(...), so dr_pre = dr * r * (1 - r)
                        float dr = dr_from_n;
                        float dr_pre = dr * all_r[t][idx] * (1.0f - all_r[t][idx]);

                        // z_t = sigmoid(...), so dz_pre = dz * z * (1 - z)
                        float dz_pre = dz * all_z[t][idx] * (1.0f - all_z[t][idx]);

                        size_t x_offset = bf ?
                            (b * seq_len * is + t * is) :
                            (t * batch_size * is + b * is);

                        // Gradients for biases
                        bias_ih_ptr->grad[h] += dr_pre;
                        bias_ih_ptr->grad[hs + h] += dz_pre;
                        bias_ih_ptr->grad[2 * hs + h] += dn_pre;

                        bias_hh_ptr->grad[h] += dr_pre;
                        bias_hh_ptr->grad[hs + h] += dz_pre;
                        bias_hh_ptr->grad[2 * hs + h] += dn_pre * all_r[t][idx];

                        // Gradients for weight_ih
                        for (size_t i = 0; i < is; i++) {
                            weight_ih_ptr->grad[h * is + i] += dr_pre * input_ptr->data[x_offset + i];
                            weight_ih_ptr->grad[(hs + h) * is + i] += dz_pre * input_ptr->data[x_offset + i];
                            weight_ih_ptr->grad[(2 * hs + h) * is + i] += dn_pre * input_ptr->data[x_offset + i];
                        }

                        // Gradients for weight_hh
                        for (size_t hh = 0; hh < hs; hh++) {
                            weight_hh_ptr->grad[h * hs + hh] += dr_pre * all_h[t][b * hs + hh];
                            weight_hh_ptr->grad[(hs + h) * hs + hh] += dz_pre * all_h[t][b * hs + hh];
                            weight_hh_ptr->grad[(2 * hs + h) * hs + hh] += dn_pre * all_r[t][idx] * all_h[t][b * hs + hh];
                        }

                        // Gradient for input
                        if (input_ptr->requires_grad) {
                            for (size_t i = 0; i < is; i++) {
                                input_ptr->grad[x_offset + i] +=
                                    dr_pre * weight_ih_ptr->data[h * is + i] +
                                    dz_pre * weight_ih_ptr->data[(hs + h) * is + i] +
                                    dn_pre * weight_ih_ptr->data[(2 * hs + h) * is + i];
                            }
                        }

                        // Gradient for h_{t-1} (accumulate for next iteration)
                        for (size_t hh = 0; hh < hs; hh++) {
                            dh_next[b * hs + hh] +=
                                dr_pre * weight_hh_ptr->data[h * hs + hh] +
                                dz_pre * weight_hh_ptr->data[(hs + h) * hs + hh] +
                                dn_pre * all_r[t][idx] * weight_hh_ptr->data[(2 * hs + h) * hs + hh];
                        }
                    }
                }
            }
        };
    }

    return output;
}

std::vector<TensorPtr> GRU::parameters() {
    return {weight_ih, weight_hh, bias_ih, bias_hh};
}

// MultiHeadAttention implementation
MultiHeadAttention::MultiHeadAttention(size_t embed_dim, size_t num_heads)
    : embed_dim(embed_dim), num_heads(num_heads) {
    assert(embed_dim % num_heads == 0 && "embed_dim must be divisible by num_heads");
    head_dim = embed_dim / num_heads;

    // Xavier initialization
    float std = std::sqrt(2.0f / (embed_dim + embed_dim));
    std::normal_distribution<float> dist(0.0f, std);

    // Q, K, V projection weights: [embed_dim, embed_dim]
    W_q = Tensor::create({embed_dim, embed_dim}, true);
    W_k = Tensor::create({embed_dim, embed_dim}, true);
    W_v = Tensor::create({embed_dim, embed_dim}, true);
    W_o = Tensor::create({embed_dim, embed_dim}, true);

    for (auto& v : W_q->data) v = dist(layer_rng);
    for (auto& v : W_k->data) v = dist(layer_rng);
    for (auto& v : W_v->data) v = dist(layer_rng);
    for (auto& v : W_o->data) v = dist(layer_rng);

    // Biases
    b_q = Tensor::zeros({embed_dim}, true);
    b_k = Tensor::zeros({embed_dim}, true);
    b_v = Tensor::zeros({embed_dim}, true);
    b_o = Tensor::zeros({embed_dim}, true);
}

TensorPtr MultiHeadAttention::forward(const TensorPtr& input) {
    return forward(input, input, input, nullptr);
}

TensorPtr MultiHeadAttention::forward(const TensorPtr& query, const TensorPtr& key,
                                       const TensorPtr& value, const TensorPtr& mask) {
    // query: [batch, seq_q, embed_dim]
    // key:   [batch, seq_k, embed_dim]
    // value: [batch, seq_k, embed_dim]
    // mask:  [batch, 1, seq_q, seq_k] or nullptr

    assert(query->shape.size() == 3);
    assert(key->shape.size() == 3);
    assert(value->shape.size() == 3);
    assert(query->shape[2] == embed_dim);
    assert(key->shape[2] == embed_dim);
    assert(value->shape[2] == embed_dim);

    size_t batch = query->shape[0];
    size_t seq_q = query->shape[1];
    size_t seq_k = key->shape[1];

    bool track = (query->requires_grad || key->requires_grad || value->requires_grad)
                 && GradMode::is_enabled();

    // Linear projections: Q, K, V
    // Q = query @ W_q + b_q  -> [batch, seq_q, embed_dim]
    // K = key @ W_k + b_k    -> [batch, seq_k, embed_dim]
    // V = value @ W_v + b_v  -> [batch, seq_k, embed_dim]

    auto Q = Tensor::create({batch, seq_q, embed_dim}, track);
    auto K = Tensor::create({batch, seq_k, embed_dim}, track);
    auto V = Tensor::create({batch, seq_k, embed_dim}, track);

    // Compute Q = query @ W_q^T + b_q
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_q; s++) {
            for (size_t d = 0; d < embed_dim; d++) {
                float sum = b_q->data[d];
                for (size_t k = 0; k < embed_dim; k++) {
                    sum += query->data[b * seq_q * embed_dim + s * embed_dim + k] *
                           W_q->data[d * embed_dim + k];
                }
                Q->data[b * seq_q * embed_dim + s * embed_dim + d] = sum;
            }
        }
    }

    // Compute K = key @ W_k^T + b_k
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_k; s++) {
            for (size_t d = 0; d < embed_dim; d++) {
                float sum = b_k->data[d];
                for (size_t k = 0; k < embed_dim; k++) {
                    sum += key->data[b * seq_k * embed_dim + s * embed_dim + k] *
                           W_k->data[d * embed_dim + k];
                }
                K->data[b * seq_k * embed_dim + s * embed_dim + d] = sum;
            }
        }
    }

    // Compute V = value @ W_v^T + b_v
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_k; s++) {
            for (size_t d = 0; d < embed_dim; d++) {
                float sum = b_v->data[d];
                for (size_t k = 0; k < embed_dim; k++) {
                    sum += value->data[b * seq_k * embed_dim + s * embed_dim + k] *
                           W_v->data[d * embed_dim + k];
                }
                V->data[b * seq_k * embed_dim + s * embed_dim + d] = sum;
            }
        }
    }

    // Reshape to [batch, num_heads, seq, head_dim]
    // Then compute attention: softmax(Q @ K^T / sqrt(head_dim)) @ V

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Attention scores: [batch, num_heads, seq_q, seq_k]
    auto scores = Tensor::create({batch, num_heads, seq_q, seq_k}, track);

    // Compute Q @ K^T / sqrt(head_dim) for each head
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t i = 0; i < seq_q; i++) {
                for (size_t j = 0; j < seq_k; j++) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; d++) {
                        // Q[b, i, h*head_dim + d] * K[b, j, h*head_dim + d]
                        size_t q_idx = b * seq_q * embed_dim + i * embed_dim + h * head_dim + d;
                        size_t k_idx = b * seq_k * embed_dim + j * embed_dim + h * head_dim + d;
                        dot += Q->data[q_idx] * K->data[k_idx];
                    }
                    size_t score_idx = b * num_heads * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                    scores->data[score_idx] = dot * scale;

                    // Apply mask if provided
                    if (mask != nullptr) {
                        // mask shape: [batch, 1, seq_q, seq_k] or [1, 1, seq_q, seq_k]
                        size_t mb = (mask->shape[0] == 1) ? 0 : b;
                        size_t mask_idx = mb * seq_q * seq_k + i * seq_k + j;
                        scores->data[score_idx] += mask->data[mask_idx];
                    }
                }
            }
        }
    }

    // Softmax over seq_k dimension
    auto attn = Tensor::create({batch, num_heads, seq_q, seq_k}, track);
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t i = 0; i < seq_q; i++) {
                // Find max for numerical stability
                float max_val = -std::numeric_limits<float>::max();
                for (size_t j = 0; j < seq_k; j++) {
                    size_t idx = b * num_heads * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                    max_val = std::max(max_val, scores->data[idx]);
                }

                // Compute exp and sum
                float sum_exp = 0.0f;
                for (size_t j = 0; j < seq_k; j++) {
                    size_t idx = b * num_heads * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                    attn->data[idx] = std::exp(scores->data[idx] - max_val);
                    sum_exp += attn->data[idx];
                }

                // Normalize
                for (size_t j = 0; j < seq_k; j++) {
                    size_t idx = b * num_heads * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                    attn->data[idx] /= sum_exp;
                }
            }
        }
    }

    // Store attention weights for visualization
    attn_weights = attn;

    // Compute attn @ V -> [batch, num_heads, seq_q, head_dim]
    // Then reshape to [batch, seq_q, embed_dim]
    auto context = Tensor::create({batch, seq_q, embed_dim}, track);

    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t i = 0; i < seq_q; i++) {
                for (size_t d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < seq_k; j++) {
                        size_t attn_idx = b * num_heads * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                        size_t v_idx = b * seq_k * embed_dim + j * embed_dim + h * head_dim + d;
                        sum += attn->data[attn_idx] * V->data[v_idx];
                    }
                    // Store in [batch, seq_q, embed_dim] format
                    size_t out_idx = b * seq_q * embed_dim + i * embed_dim + h * head_dim + d;
                    context->data[out_idx] = sum;
                }
            }
        }
    }

    // Output projection: context @ W_o^T + b_o
    auto output = Tensor::create({batch, seq_q, embed_dim}, track);

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_q; s++) {
            for (size_t d = 0; d < embed_dim; d++) {
                float sum = b_o->data[d];
                for (size_t k = 0; k < embed_dim; k++) {
                    sum += context->data[b * seq_q * embed_dim + s * embed_dim + k] *
                           W_o->data[d * embed_dim + k];
                }
                output->data[b * seq_q * embed_dim + s * embed_dim + d] = sum;
            }
        }
    }

    if (track) {
        auto query_ptr = query;
        auto key_ptr = key;
        auto value_ptr = value;
        auto W_q_ptr = W_q;
        auto W_k_ptr = W_k;
        auto W_v_ptr = W_v;
        auto W_o_ptr = W_o;
        auto b_q_ptr = b_q;
        auto b_k_ptr = b_k;
        auto b_v_ptr = b_v;
        auto b_o_ptr = b_o;
        size_t ed = embed_dim;
        size_t nh = num_heads;
        size_t hd = head_dim;

        output->parents = {query_ptr, key_ptr, value_ptr, W_q_ptr, W_k_ptr, W_v_ptr, W_o_ptr,
                          b_q_ptr, b_k_ptr, b_v_ptr, b_o_ptr};

        output->grad_fn = [=]() mutable {
            // Backward pass through MultiHeadAttention
            // This is complex, so we'll compute it step by step

            // Gradient of output projection
            auto d_context = Tensor::create({batch, seq_q, ed}, false);
            for (size_t i = 0; i < d_context->data.size(); i++) d_context->data[i] = 0;

            for (size_t b = 0; b < batch; b++) {
                for (size_t s = 0; s < seq_q; s++) {
                    for (size_t d = 0; d < ed; d++) {
                        float dout = output->grad[b * seq_q * ed + s * ed + d];
                        b_o_ptr->grad[d] += dout;

                        for (size_t k = 0; k < ed; k++) {
                            W_o_ptr->grad[d * ed + k] += dout * context->data[b * seq_q * ed + s * ed + k];
                            d_context->data[b * seq_q * ed + s * ed + k] += dout * W_o_ptr->data[d * ed + k];
                        }
                    }
                }
            }

            // Gradient of attn @ V
            auto d_attn = Tensor::create({batch, nh, seq_q, seq_k}, false);
            auto d_V = Tensor::create({batch, seq_k, ed}, false);
            for (size_t i = 0; i < d_attn->data.size(); i++) d_attn->data[i] = 0;
            for (size_t i = 0; i < d_V->data.size(); i++) d_V->data[i] = 0;

            for (size_t b = 0; b < batch; b++) {
                for (size_t h = 0; h < nh; h++) {
                    for (size_t i = 0; i < seq_q; i++) {
                        for (size_t d = 0; d < hd; d++) {
                            float d_ctx = d_context->data[b * seq_q * ed + i * ed + h * hd + d];
                            for (size_t j = 0; j < seq_k; j++) {
                                size_t attn_idx = b * nh * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                                size_t v_idx = b * seq_k * ed + j * ed + h * hd + d;
                                d_attn->data[attn_idx] += d_ctx * V->data[v_idx];
                                d_V->data[v_idx] += d_ctx * attn->data[attn_idx];
                            }
                        }
                    }
                }
            }

            // Gradient through softmax
            auto d_scores = Tensor::create({batch, nh, seq_q, seq_k}, false);
            for (size_t b = 0; b < batch; b++) {
                for (size_t h = 0; h < nh; h++) {
                    for (size_t i = 0; i < seq_q; i++) {
                        // softmax gradient: d_score[j] = attn[j] * (d_attn[j] - sum(attn * d_attn))
                        float dot_sum = 0.0f;
                        for (size_t j = 0; j < seq_k; j++) {
                            size_t idx = b * nh * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                            dot_sum += attn->data[idx] * d_attn->data[idx];
                        }
                        for (size_t j = 0; j < seq_k; j++) {
                            size_t idx = b * nh * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                            d_scores->data[idx] = attn->data[idx] * (d_attn->data[idx] - dot_sum);
                        }
                    }
                }
            }

            // Gradient of Q @ K^T / sqrt(head_dim)
            auto d_Q = Tensor::create({batch, seq_q, ed}, false);
            auto d_K = Tensor::create({batch, seq_k, ed}, false);
            for (size_t i = 0; i < d_Q->data.size(); i++) d_Q->data[i] = 0;
            for (size_t i = 0; i < d_K->data.size(); i++) d_K->data[i] = 0;

            for (size_t b = 0; b < batch; b++) {
                for (size_t h = 0; h < nh; h++) {
                    for (size_t i = 0; i < seq_q; i++) {
                        for (size_t j = 0; j < seq_k; j++) {
                            size_t score_idx = b * nh * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                            float d_s = d_scores->data[score_idx] * scale;
                            for (size_t d = 0; d < hd; d++) {
                                size_t q_idx = b * seq_q * ed + i * ed + h * hd + d;
                                size_t k_idx = b * seq_k * ed + j * ed + h * hd + d;
                                d_Q->data[q_idx] += d_s * K->data[k_idx];
                                d_K->data[k_idx] += d_s * Q->data[q_idx];
                            }
                        }
                    }
                }
            }

            // Gradient through V projection
            for (size_t b = 0; b < batch; b++) {
                for (size_t s = 0; s < seq_k; s++) {
                    for (size_t d = 0; d < ed; d++) {
                        float dv = d_V->data[b * seq_k * ed + s * ed + d];
                        b_v_ptr->grad[d] += dv;

                        for (size_t k = 0; k < ed; k++) {
                            W_v_ptr->grad[d * ed + k] += dv * value_ptr->data[b * seq_k * ed + s * ed + k];
                            if (value_ptr->requires_grad) {
                                value_ptr->grad[b * seq_k * ed + s * ed + k] += dv * W_v_ptr->data[d * ed + k];
                            }
                        }
                    }
                }
            }

            // Gradient through K projection
            for (size_t b = 0; b < batch; b++) {
                for (size_t s = 0; s < seq_k; s++) {
                    for (size_t d = 0; d < ed; d++) {
                        float dk = d_K->data[b * seq_k * ed + s * ed + d];
                        b_k_ptr->grad[d] += dk;

                        for (size_t k = 0; k < ed; k++) {
                            W_k_ptr->grad[d * ed + k] += dk * key_ptr->data[b * seq_k * ed + s * ed + k];
                            if (key_ptr->requires_grad) {
                                key_ptr->grad[b * seq_k * ed + s * ed + k] += dk * W_k_ptr->data[d * ed + k];
                            }
                        }
                    }
                }
            }

            // Gradient through Q projection
            for (size_t b = 0; b < batch; b++) {
                for (size_t s = 0; s < seq_q; s++) {
                    for (size_t d = 0; d < ed; d++) {
                        float dq = d_Q->data[b * seq_q * ed + s * ed + d];
                        b_q_ptr->grad[d] += dq;

                        for (size_t k = 0; k < ed; k++) {
                            W_q_ptr->grad[d * ed + k] += dq * query_ptr->data[b * seq_q * ed + s * ed + k];
                            if (query_ptr->requires_grad) {
                                query_ptr->grad[b * seq_q * ed + s * ed + k] += dq * W_q_ptr->data[d * ed + k];
                            }
                        }
                    }
                }
            }
        };
    }

    return output;
}

std::vector<TensorPtr> MultiHeadAttention::parameters() {
    return {W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o};
}

TensorPtr MultiHeadAttention::causal_mask(size_t seq_len) {
    // Create a causal mask: positions can only attend to previous positions
    // Shape: [1, 1, seq_len, seq_len]
    // Upper triangular (above diagonal) = -inf, lower triangular (including diagonal) = 0
    auto mask = Tensor::create({1, 1, seq_len, seq_len}, false);

    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            size_t idx = i * seq_len + j;
            if (j > i) {
                mask->data[idx] = -1e9f;  // Large negative value (effectively -inf for softmax)
            } else {
                mask->data[idx] = 0.0f;
            }
        }
    }

    return mask;
}
