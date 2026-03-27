#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include <memory>
#include <vector>
#include <initializer_list>

class Module {
public:
    virtual ~Module() = default;
    virtual TensorPtr forward(const TensorPtr& input) = 0;
    virtual std::vector<TensorPtr> parameters() { return {}; }

    virtual size_t num_parameters() const;
    virtual size_t num_trainable_parameters() const;

    virtual std::string name() const { return "Module"; }
    virtual std::string extra_repr() const { return ""; }

    virtual std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const {
        return input_shape;
    }

    virtual void to(whitematter::DeviceType device) {
        for (auto& p : parameters()) {
            p->to_inplace(device);
        }
    }

    TensorPtr operator()(const TensorPtr& input) {
        return forward(input);
    }
};

using ModulePtr = std::shared_ptr<Module>;

class Linear : public Module {
public:
    TensorPtr weight;
    TensorPtr bias;
    size_t in_features, out_features;

    Linear(size_t in_features, size_t out_features);

    TensorPtr forward(const TensorPtr& input) override;
    std::vector<TensorPtr> parameters() override;
    std::string name() const override { return "Linear"; }
    std::string extra_repr() const override;
    std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const override;
};

class ReLU : public Module {
public:
    TensorPtr forward(const TensorPtr& input) override;
    std::string name() const override { return "ReLU"; }
};

class Sigmoid : public Module {
public:
    TensorPtr forward(const TensorPtr& input) override;
    std::string name() const override { return "Sigmoid"; }
};

class Tanh : public Module {
public:
    TensorPtr forward(const TensorPtr& input) override;
    std::string name() const override { return "Tanh"; }
};

class SiLU : public Module {
public:
    TensorPtr forward(const TensorPtr& input) override;
    std::string name() const override { return "SiLU"; }
};

class GELU : public Module {
public:
    TensorPtr forward(const TensorPtr& input) override;
    std::string name() const override { return "GELU"; }
};

class Mish : public Module {
public:
    TensorPtr forward(const TensorPtr& input) override;
    std::string name() const override { return "Mish"; }
};

class Softmax : public Module {
public:
    int dim;
    Softmax(int dim = -1);
    TensorPtr forward(const TensorPtr& input) override;
    std::string name() const override { return "Softmax"; }
    std::string extra_repr() const override;
};

class LogSoftmax : public Module {
public:
    int dim;
    LogSoftmax(int dim = -1);
    TensorPtr forward(const TensorPtr& input) override;
    std::string name() const override { return "LogSoftmax"; }
    std::string extra_repr() const override;
};

class Dropout : public Module {
public:
    float p;
    bool training;

    Dropout(float p = 0.5f);
    TensorPtr forward(const TensorPtr& input) override;
    void train() { training = true; }
    void eval() { training = false; }
    std::string name() const override { return "Dropout"; }
    std::string extra_repr() const override;
};

class Conv2d : public Module {
public:
    TensorPtr weight;
    TensorPtr bias;
    size_t in_channels, out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    size_t groups;
    size_t dilation;

    Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
           size_t stride = 1, size_t padding = 0, size_t groups = 1, size_t dilation = 1);

    TensorPtr forward(const TensorPtr& input) override;
    std::vector<TensorPtr> parameters() override;
    std::string name() const override { return "Conv2d"; }
    std::string extra_repr() const override;
    std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const override;
};

class ConvTranspose2d : public Module {
public:
    TensorPtr weight;
    TensorPtr bias;
    size_t in_channels, out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    size_t output_padding;

    ConvTranspose2d(size_t in_channels, size_t out_channels, size_t kernel_size,
                    size_t stride = 1, size_t padding = 0, size_t output_padding = 0);

    TensorPtr forward(const TensorPtr& input) override;
    std::vector<TensorPtr> parameters() override;
    std::string name() const override { return "ConvTranspose2d"; }
    std::string extra_repr() const override;
    std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const override;
};

class MaxPool2d : public Module {
public:
    size_t kernel_size;
    size_t stride;

    MaxPool2d(size_t kernel_size, size_t stride = 0);
    TensorPtr forward(const TensorPtr& input) override;
    std::string name() const override { return "MaxPool2d"; }
    std::string extra_repr() const override;
    std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const override;
};

class AvgPool2d : public Module {
public:
    size_t kernel_size;
    size_t stride;

    AvgPool2d(size_t kernel_size, size_t stride = 0);
    TensorPtr forward(const TensorPtr& input) override;
    std::string name() const override { return "AvgPool2d"; }
    std::string extra_repr() const override;
    std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const override;
};

class BatchNorm2d : public Module {
public:
    size_t num_features;
    float eps;
    float momentum;
    bool training;

    TensorPtr gamma;       // learnable scale
    TensorPtr beta;        // learnable shift
    TensorPtr running_mean;
    TensorPtr running_var;

    BatchNorm2d(size_t num_features, float eps = 1e-5f, float momentum = 0.1f);

    TensorPtr forward(const TensorPtr& input) override;
    std::vector<TensorPtr> parameters() override;
    void train() { training = true; }
    void eval() { training = false; }
    std::string name() const override { return "BatchNorm2d"; }
    std::string extra_repr() const override;
};

class LayerNorm : public Module {
public:
    std::vector<size_t> normalized_shape;
    float eps;

    TensorPtr gamma;  // learnable scale (weight)
    TensorPtr beta;   // learnable shift (bias)

    LayerNorm(std::vector<size_t> normalized_shape, float eps = 1e-5f);
    LayerNorm(size_t normalized_shape, float eps = 1e-5f);  // Convenience for 1D

    TensorPtr forward(const TensorPtr& input) override;
    std::vector<TensorPtr> parameters() override;
    std::string name() const override { return "LayerNorm"; }
    std::string extra_repr() const override;
};

class GroupNorm : public Module {
public:
    size_t num_groups, num_channels;
    float eps;
    TensorPtr gamma, beta;

    GroupNorm(size_t num_groups, size_t num_channels, float eps = 1e-5f);

    TensorPtr forward(const TensorPtr& input) override;
    std::vector<TensorPtr> parameters() override;
    std::string name() const override { return "GroupNorm"; }
    std::string extra_repr() const override;
};

class RMSNorm : public Module {
public:
    size_t dim;
    float eps;
    TensorPtr gamma;

    RMSNorm(size_t dim, float eps = 1e-8f);

    TensorPtr forward(const TensorPtr& input) override;
    std::vector<TensorPtr> parameters() override;
    std::string name() const override { return "RMSNorm"; }
    std::string extra_repr() const override;
};

class SinusoidalPositionalEncoding : public Module {
public:
    SinusoidalPositionalEncoding(size_t max_seq_len, size_t embed_dim);

    TensorPtr forward(const TensorPtr& input) override;
    std::string name() const override { return "SinusoidalPE"; }

private:
    TensorPtr pe_table;  // [max_seq_len, embed_dim], not learnable
    size_t max_seq_len_, embed_dim_;
};

// Apply Rotary Positional Embedding (RoPE) to Q or K tensor
// qk shape: [batch, seq_len, embed] where embed = num_heads * head_dim
void apply_rope(TensorPtr& qk, size_t seq_len, size_t num_heads, size_t head_dim);

class Upsample : public Module {
public:
    size_t scale_factor;
    std::string mode;  // "nearest" or "bilinear"

    Upsample(size_t scale_factor, std::string mode = "nearest");

    TensorPtr forward(const TensorPtr& input) override;
    std::string name() const override { return "Upsample"; }
    std::string extra_repr() const override;
    std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const override;
};

class Conv1d : public Module {
public:
    TensorPtr weight;
    TensorPtr bias;
    size_t in_channels, out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;

    Conv1d(size_t in_channels, size_t out_channels, size_t kernel_size,
           size_t stride = 1, size_t padding = 0);

    TensorPtr forward(const TensorPtr& input) override;
    std::vector<TensorPtr> parameters() override;
    std::string name() const override { return "Conv1d"; }
    std::string extra_repr() const override;
    std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const override;
};

class AdaptiveAvgPool2d : public Module {
public:
    size_t output_h, output_w;

    AdaptiveAvgPool2d(size_t output_h, size_t output_w);

    TensorPtr forward(const TensorPtr& input) override;
    std::string name() const override { return "AdaptiveAvgPool2d"; }
    std::string extra_repr() const override;
    std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const override;
};

class Flatten : public Module {
public:
    TensorPtr forward(const TensorPtr& input) override;
    std::string name() const override { return "Flatten"; }
    std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const override;
};

class Embedding : public Module {
public:
    TensorPtr weight;
    size_t num_embeddings;
    size_t embedding_dim;

    Embedding(size_t num_embeddings, size_t embedding_dim);

    TensorPtr forward(const TensorPtr& indices) override;
    std::vector<TensorPtr> parameters() override;
    std::string name() const override { return "Embedding"; }
    std::string extra_repr() const override;
    std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const override;
};

class LSTM : public Module {
public:
    size_t input_size;
    size_t hidden_size;
    bool batch_first;

    // Weights: [4*hidden_size, input_size] and [4*hidden_size, hidden_size]
    // Gates order: input, forget, cell, output (i, f, g, o)
    TensorPtr weight_ih;  // input-to-hidden weights
    TensorPtr weight_hh;  // hidden-to-hidden weights
    TensorPtr bias_ih;    // input-to-hidden bias
    TensorPtr bias_hh;    // hidden-to-hidden bias

    // Last hidden and cell states (set after forward pass)
    TensorPtr h_n;
    TensorPtr c_n;

    LSTM(size_t input_size, size_t hidden_size, bool batch_first = true);

    // Standard forward: input shape [batch, seq, input] or [seq, batch, input]
    // Returns all hidden states: [batch, seq, hidden] or [seq, batch, hidden]
    TensorPtr forward(const TensorPtr& input) override;

    // Forward with initial hidden state
    TensorPtr forward(const TensorPtr& input, const TensorPtr& h0, const TensorPtr& c0);

    std::vector<TensorPtr> parameters() override;
    std::string name() const override { return "LSTM"; }
    std::string extra_repr() const override;
    std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const override;
};

class GRU : public Module {
public:
    size_t input_size;
    size_t hidden_size;
    bool batch_first;

    // Weights: [3*hidden_size, input_size] and [3*hidden_size, hidden_size]
    // Gates order: reset, update, new (r, z, n)
    TensorPtr weight_ih;  // input-to-hidden weights
    TensorPtr weight_hh;  // hidden-to-hidden weights
    TensorPtr bias_ih;    // input-to-hidden bias
    TensorPtr bias_hh;    // hidden-to-hidden bias

    // Last hidden state (set after forward pass)
    TensorPtr h_n;

    GRU(size_t input_size, size_t hidden_size, bool batch_first = true);

    // Standard forward: input shape [batch, seq, input] or [seq, batch, input]
    // Returns all hidden states: [batch, seq, hidden] or [seq, batch, hidden]
    TensorPtr forward(const TensorPtr& input) override;

    // Forward with initial hidden state
    TensorPtr forward(const TensorPtr& input, const TensorPtr& h0);

    std::vector<TensorPtr> parameters() override;
    std::string name() const override { return "GRU"; }
    std::string extra_repr() const override;
    std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const override;
};

class MultiHeadAttention : public Module {
public:
    size_t embed_dim;
    size_t num_heads;
    size_t head_dim;

    // Projection weights: Q, K, V and output
    TensorPtr W_q, W_k, W_v, W_o;
    TensorPtr b_q, b_k, b_v, b_o;

    // Stored attention weights (for visualization/debugging)
    TensorPtr attn_weights;

    MultiHeadAttention(size_t embed_dim, size_t num_heads);

    // Self-attention: Q=K=V=input
    // Input: [batch, seq_len, embed_dim]
    // Output: [batch, seq_len, embed_dim]
    TensorPtr forward(const TensorPtr& input) override;

    // Cross-attention or self-attention with explicit Q, K, V
    // Optional mask: [batch, 1, seq_len, seq_len] or [1, 1, seq_len, seq_len]
    // Mask values: 0 = attend, -inf (large negative) = don't attend
    TensorPtr forward(const TensorPtr& query, const TensorPtr& key, const TensorPtr& value,
                      const TensorPtr& mask = nullptr);

    std::vector<TensorPtr> parameters() override;
    std::string name() const override { return "MultiHeadAttention"; }
    std::string extra_repr() const override;

    // Helper to create causal mask for autoregressive models
    static TensorPtr causal_mask(size_t seq_len);
};

class KVCache {
public:
    KVCache(size_t max_seq_len, size_t num_heads, size_t head_dim);

    // Append new key/value for the current step
    // new_keys/new_values shape: [batch, new_tokens, embed_dim] where embed_dim = num_heads * head_dim
    void append(const TensorPtr& new_keys, const TensorPtr& new_values);

    // Get all cached keys/values up to current position
    TensorPtr keys() const;    // [batch, cached_len, embed_dim]
    TensorPtr values() const;  // [batch, cached_len, embed_dim]

    size_t length() const;  // Current cached sequence length
    void clear();           // Reset cache

private:
    TensorPtr key_cache_;    // Pre-allocated [batch, max_seq_len, embed_dim]
    TensorPtr value_cache_;
    size_t current_len_;
    size_t max_seq_len_;
    size_t embed_dim_;
    size_t batch_size_;
};

class GroupedQueryAttention : public Module {
public:
    size_t embed_dim;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_dim;

    // Projection weights: Q, K, V and output
    // W_q: [num_heads * head_dim, embed_dim]
    // W_k, W_v: [num_kv_heads * head_dim, embed_dim] (smaller when num_kv_heads < num_heads)
    // W_o: [embed_dim, embed_dim]
    TensorPtr W_q, W_k, W_v, W_o;
    TensorPtr b_q, b_k, b_v, b_o;

    // Stored attention weights (for visualization/debugging)
    TensorPtr attn_weights;

    GroupedQueryAttention(size_t embed_dim, size_t num_heads, size_t num_kv_heads);

    // Self-attention: Q=K=V=input
    TensorPtr forward(const TensorPtr& input) override;

    // Cross-attention or self-attention with explicit Q, K, V
    TensorPtr forward(const TensorPtr& query, const TensorPtr& key, const TensorPtr& value,
                      const TensorPtr& mask = nullptr);

    std::vector<TensorPtr> parameters() override;
    std::string name() const override { return "GroupedQueryAttention"; }
    std::string extra_repr() const override;
};

class Sequential : public Module {
public:
    std::vector<std::shared_ptr<Module>> layers;

    Sequential() = default;
    Sequential(std::initializer_list<Module*> modules);
    ~Sequential() = default;

    Sequential(Sequential&&) = default;
    Sequential& operator=(Sequential&&) = default;
    Sequential(const Sequential&) = delete;
    Sequential& operator=(const Sequential&) = delete;

    void add(Module* module);
    void add(std::shared_ptr<Module> module);
    TensorPtr forward(const TensorPtr& input) override;
    std::vector<TensorPtr> parameters() override;
    std::string name() const override { return "Sequential"; }

    void train();
    void eval();

    // Model summary (PyTorch-style)
    // Pass input_shape to enable output shape tracking (e.g., {1, 1, 28, 28} for MNIST)
    void summary(const std::vector<size_t>& input_shape = {}) const;
};

struct ModelSummary {
    size_t total_params;
    size_t trainable_params;
    size_t non_trainable_params;
    size_t param_memory_bytes;      // Memory for parameters (fp32)
    size_t param_memory_fp16_bytes; // Memory if using fp16
    size_t grad_memory_bytes;       // Memory for gradients (fp32)
    size_t total_memory_bytes;      // params + gradients
    size_t num_layers;
};

ModelSummary get_model_summary(Module* model);

inline size_t count_parameters(Module* model) {
    return model->num_parameters();
}

inline size_t count_trainable_parameters(Module* model) {
    return model->num_trainable_parameters();
}

std::string format_memory(size_t bytes);
std::string format_number(size_t n);
void print_model_info(Module* model, const std::string& name = "Model");

#endif
