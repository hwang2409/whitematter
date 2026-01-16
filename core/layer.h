#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include <vector>
#include <initializer_list>

class Module {
public:
    virtual ~Module() = default;
    virtual TensorPtr forward(const TensorPtr& input) = 0;
    virtual std::vector<TensorPtr> parameters() { return {}; }

    // Parameter counting
    virtual size_t num_parameters() const;
    virtual size_t num_trainable_parameters() const;

    // Layer info for summary
    virtual std::string name() const { return "Module"; }
    virtual std::string extra_repr() const { return ""; }

    // Output shape computation for summary (default: pass-through)
    virtual std::vector<size_t> compute_output_shape(const std::vector<size_t>& input_shape) const {
        return input_shape;
    }

    TensorPtr operator()(const TensorPtr& input) {
        return forward(input);
    }
};

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

    Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
           size_t stride = 1, size_t padding = 0);

    TensorPtr forward(const TensorPtr& input) override;
    std::vector<TensorPtr> parameters() override;
    std::string name() const override { return "Conv2d"; }
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

class Sequential : public Module {
public:
    std::vector<Module*> layers;

    Sequential() = default;
    Sequential(std::initializer_list<Module*> modules);
    ~Sequential();

    void add(Module* module);
    TensorPtr forward(const TensorPtr& input) override;
    std::vector<TensorPtr> parameters() override;
    std::string name() const override { return "Sequential"; }

    void train();
    void eval();

    // Model summary (PyTorch-style)
    // Pass input_shape to enable output shape tracking (e.g., {1, 1, 28, 28} for MNIST)
    void summary(const std::vector<size_t>& input_shape = {}) const;
};

#endif
