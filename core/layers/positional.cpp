#include "../layer.h"
#include <cmath>
#include <cassert>

// =============================================================================
// SinusoidalPositionalEncoding
// =============================================================================

SinusoidalPositionalEncoding::SinusoidalPositionalEncoding(size_t max_seq_len, size_t embed_dim)
    : max_seq_len_(max_seq_len), embed_dim_(embed_dim) {
    // Pre-compute the PE table: [max_seq_len, embed_dim]
    pe_table = Tensor::zeros({max_seq_len, embed_dim}, false);

    for (size_t pos = 0; pos < max_seq_len; pos++) {
        for (size_t i = 0; i < embed_dim / 2; i++) {
            double theta = static_cast<double>(pos) /
                std::pow(10000.0, static_cast<double>(2 * i) / static_cast<double>(embed_dim));
            pe_table->data()[pos * embed_dim + 2 * i]     = static_cast<float>(std::sin(theta));
            pe_table->data()[pos * embed_dim + 2 * i + 1] = static_cast<float>(std::cos(theta));
        }
        // If embed_dim is odd, the last position stays zero (already zeroed)
    }
}

TensorPtr SinusoidalPositionalEncoding::forward(const TensorPtr& input) {
    // input shape: [batch, seq_len, embed_dim]
    assert(input->shape.size() == 3);
    assert(input->shape[2] == embed_dim_);
    assert(input->shape[1] <= max_seq_len_);

    size_t batch = input->shape[0];
    size_t seq_len = input->shape[1];

    bool track = input->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create(input->shape, track);

    // output = input + pe_table[:seq_len, :]
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            size_t in_base = b * seq_len * embed_dim_ + s * embed_dim_;
            size_t pe_base = s * embed_dim_;
            for (size_t d = 0; d < embed_dim_; d++) {
                result->data()[in_base + d] = input->data()[in_base + d] + pe_table->data()[pe_base + d];
            }
        }
    }

    if (track) {
        auto input_ptr = input;
        result->parents = {input_ptr};

        result->grad_fn = [input_ptr, result]() {
            if (input_ptr->requires_grad) {
                for (size_t i = 0; i < input_ptr->size(); i++) {
                    input_ptr->grad()[i] += result->grad()[i];
                }
            }
            // pe_table is not learnable — no gradient
        };
    }

    return result;
}

// =============================================================================
// RoPE (Rotary Positional Embedding)
// =============================================================================

void apply_rope(TensorPtr& qk, size_t seq_len, size_t num_heads, size_t head_dim) {
    // qk shape: [batch, seq_len, embed] where embed = num_heads * head_dim
    assert(qk->shape.size() == 3);
    assert(qk->shape[1] == seq_len);
    assert(qk->shape[2] == num_heads * head_dim);

    size_t batch = qk->shape[0];
    size_t embed = num_heads * head_dim;

    for (size_t b = 0; b < batch; b++) {
        for (size_t pos = 0; pos < seq_len; pos++) {
            for (size_t h = 0; h < num_heads; h++) {
                for (size_t i = 0; i < head_dim / 2; i++) {
                    double theta = static_cast<double>(pos) /
                        std::pow(10000.0, static_cast<double>(2 * i) / static_cast<double>(head_dim));
                    float cos_theta = static_cast<float>(std::cos(theta));
                    float sin_theta = static_cast<float>(std::sin(theta));

                    size_t base = b * seq_len * embed + pos * embed + h * head_dim;
                    size_t idx0 = base + 2 * i;
                    size_t idx1 = base + 2 * i + 1;

                    float x0 = qk->data()[idx0];
                    float x1 = qk->data()[idx1];

                    qk->data()[idx0] = x0 * cos_theta - x1 * sin_theta;
                    qk->data()[idx1] = x0 * sin_theta + x1 * cos_theta;
                }
            }
        }
    }
}
