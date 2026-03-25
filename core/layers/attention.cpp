#include "../layer.h"
#include <random>
#include <cmath>
#include <cassert>
#include <limits>

static std::mt19937 attention_rng(123);

MultiHeadAttention::MultiHeadAttention(size_t embed_dim, size_t num_heads)
    : embed_dim(embed_dim), num_heads(num_heads) {
    assert(embed_dim % num_heads == 0 && "embed_dim must be divisible by num_heads");
    head_dim = embed_dim / num_heads;

    float std = std::sqrt(2.0f / (embed_dim + embed_dim));
    std::normal_distribution<float> dist(0.0f, std);

    W_q = Tensor::create({embed_dim, embed_dim}, true);
    W_k = Tensor::create({embed_dim, embed_dim}, true);
    W_v = Tensor::create({embed_dim, embed_dim}, true);
    W_o = Tensor::create({embed_dim, embed_dim}, true);

    for (size_t i = 0; i < W_q->size(); i++) W_q->data()[i] = dist(attention_rng);
    for (size_t i = 0; i < W_k->size(); i++) W_k->data()[i] = dist(attention_rng);
    for (size_t i = 0; i < W_v->size(); i++) W_v->data()[i] = dist(attention_rng);
    for (size_t i = 0; i < W_o->size(); i++) W_o->data()[i] = dist(attention_rng);

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

    auto Q = Tensor::create({batch, seq_q, embed_dim}, track);
    auto K = Tensor::create({batch, seq_k, embed_dim}, track);
    auto V = Tensor::create({batch, seq_k, embed_dim}, track);

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_q; s++) {
            for (size_t d = 0; d < embed_dim; d++) {
                float sum = b_q->data()[d];
                for (size_t k = 0; k < embed_dim; k++) {
                    sum += query->data()[b * seq_q * embed_dim + s * embed_dim + k] *
                           W_q->data()[d * embed_dim + k];
                }
                Q->data()[b * seq_q * embed_dim + s * embed_dim + d] = sum;
            }
        }
    }

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_k; s++) {
            for (size_t d = 0; d < embed_dim; d++) {
                float sum = b_k->data()[d];
                for (size_t k = 0; k < embed_dim; k++) {
                    sum += key->data()[b * seq_k * embed_dim + s * embed_dim + k] *
                           W_k->data()[d * embed_dim + k];
                }
                K->data()[b * seq_k * embed_dim + s * embed_dim + d] = sum;
            }
        }
    }

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_k; s++) {
            for (size_t d = 0; d < embed_dim; d++) {
                float sum = b_v->data()[d];
                for (size_t k = 0; k < embed_dim; k++) {
                    sum += value->data()[b * seq_k * embed_dim + s * embed_dim + k] *
                           W_v->data()[d * embed_dim + k];
                }
                V->data()[b * seq_k * embed_dim + s * embed_dim + d] = sum;
            }
        }
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto scores = Tensor::create({batch, num_heads, seq_q, seq_k}, track);

    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t i = 0; i < seq_q; i++) {
                for (size_t j = 0; j < seq_k; j++) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; d++) {
                        size_t q_idx = b * seq_q * embed_dim + i * embed_dim + h * head_dim + d;
                        size_t k_idx = b * seq_k * embed_dim + j * embed_dim + h * head_dim + d;
                        dot += Q->data()[q_idx] * K->data()[k_idx];
                    }
                    size_t score_idx = b * num_heads * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                    scores->data()[score_idx] = dot * scale;

                    if (mask != nullptr) {
                        size_t mb = (mask->shape[0] == 1) ? 0 : b;
                        size_t mask_idx = mb * seq_q * seq_k + i * seq_k + j;
                        scores->data()[score_idx] += mask->data()[mask_idx];
                    }
                }
            }
        }
    }

    auto attn = Tensor::create({batch, num_heads, seq_q, seq_k}, track);
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t i = 0; i < seq_q; i++) {
                float max_val = -std::numeric_limits<float>::max();
                for (size_t j = 0; j < seq_k; j++) {
                    size_t idx = b * num_heads * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                    max_val = std::max(max_val, scores->data()[idx]);
                }

                float sum_exp = 0.0f;
                for (size_t j = 0; j < seq_k; j++) {
                    size_t idx = b * num_heads * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                    attn->data()[idx] = std::exp(scores->data()[idx] - max_val);
                    sum_exp += attn->data()[idx];
                }

                for (size_t j = 0; j < seq_k; j++) {
                    size_t idx = b * num_heads * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                    attn->data()[idx] /= sum_exp;
                }
            }
        }
    }

    attn_weights = attn;

    auto context = Tensor::create({batch, seq_q, embed_dim}, track);

    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t i = 0; i < seq_q; i++) {
                for (size_t d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < seq_k; j++) {
                        size_t attn_idx = b * num_heads * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                        size_t v_idx = b * seq_k * embed_dim + j * embed_dim + h * head_dim + d;
                        sum += attn->data()[attn_idx] * V->data()[v_idx];
                    }
                    size_t out_idx = b * seq_q * embed_dim + i * embed_dim + h * head_dim + d;
                    context->data()[out_idx] = sum;
                }
            }
        }
    }

    auto output = Tensor::create({batch, seq_q, embed_dim}, track);

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_q; s++) {
            for (size_t d = 0; d < embed_dim; d++) {
                float sum = b_o->data()[d];
                for (size_t k = 0; k < embed_dim; k++) {
                    sum += context->data()[b * seq_q * embed_dim + s * embed_dim + k] *
                           W_o->data()[d * embed_dim + k];
                }
                output->data()[b * seq_q * embed_dim + s * embed_dim + d] = sum;
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
            auto d_context = Tensor::create({batch, seq_q, ed}, false);
            for (size_t i = 0; i < d_context->size(); i++) d_context->data()[i] = 0;

            for (size_t b = 0; b < batch; b++) {
                for (size_t s = 0; s < seq_q; s++) {
                    for (size_t d = 0; d < ed; d++) {
                        float dout = output->grad()[b * seq_q * ed + s * ed + d];
                        b_o_ptr->grad()[d] += dout;

                        for (size_t k = 0; k < ed; k++) {
                            W_o_ptr->grad()[d * ed + k] += dout * context->data()[b * seq_q * ed + s * ed + k];
                            d_context->data()[b * seq_q * ed + s * ed + k] += dout * W_o_ptr->data()[d * ed + k];
                        }
                    }
                }
            }

            auto d_attn = Tensor::create({batch, nh, seq_q, seq_k}, false);
            auto d_V = Tensor::create({batch, seq_k, ed}, false);
            for (size_t i = 0; i < d_attn->size(); i++) d_attn->data()[i] = 0;
            for (size_t i = 0; i < d_V->size(); i++) d_V->data()[i] = 0;

            for (size_t b = 0; b < batch; b++) {
                for (size_t h = 0; h < nh; h++) {
                    for (size_t i = 0; i < seq_q; i++) {
                        for (size_t d = 0; d < hd; d++) {
                            float d_ctx = d_context->data()[b * seq_q * ed + i * ed + h * hd + d];
                            for (size_t j = 0; j < seq_k; j++) {
                                size_t attn_idx = b * nh * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                                size_t v_idx = b * seq_k * ed + j * ed + h * hd + d;
                                d_attn->data()[attn_idx] += d_ctx * V->data()[v_idx];
                                d_V->data()[v_idx] += d_ctx * attn->data()[attn_idx];
                            }
                        }
                    }
                }
            }

            auto d_scores = Tensor::create({batch, nh, seq_q, seq_k}, false);
            for (size_t b = 0; b < batch; b++) {
                for (size_t h = 0; h < nh; h++) {
                    for (size_t i = 0; i < seq_q; i++) {
                        // softmax gradient: d_score[j] = attn[j] * (d_attn[j] - sum(attn * d_attn))
                        float dot_sum = 0.0f;
                        for (size_t j = 0; j < seq_k; j++) {
                            size_t idx = b * nh * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                            dot_sum += attn->data()[idx] * d_attn->data()[idx];
                        }
                        for (size_t j = 0; j < seq_k; j++) {
                            size_t idx = b * nh * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                            d_scores->data()[idx] = attn->data()[idx] * (d_attn->data()[idx] - dot_sum);
                        }
                    }
                }
            }

            auto d_Q = Tensor::create({batch, seq_q, ed}, false);
            auto d_K = Tensor::create({batch, seq_k, ed}, false);
            for (size_t i = 0; i < d_Q->size(); i++) d_Q->data()[i] = 0;
            for (size_t i = 0; i < d_K->size(); i++) d_K->data()[i] = 0;

            for (size_t b = 0; b < batch; b++) {
                for (size_t h = 0; h < nh; h++) {
                    for (size_t i = 0; i < seq_q; i++) {
                        for (size_t j = 0; j < seq_k; j++) {
                            size_t score_idx = b * nh * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                            float d_s = d_scores->data()[score_idx] * scale;
                            for (size_t d = 0; d < hd; d++) {
                                size_t q_idx = b * seq_q * ed + i * ed + h * hd + d;
                                size_t k_idx = b * seq_k * ed + j * ed + h * hd + d;
                                d_Q->data()[q_idx] += d_s * K->data()[k_idx];
                                d_K->data()[k_idx] += d_s * Q->data()[q_idx];
                            }
                        }
                    }
                }
            }

            for (size_t b = 0; b < batch; b++) {
                for (size_t s = 0; s < seq_k; s++) {
                    for (size_t d = 0; d < ed; d++) {
                        float dv = d_V->data()[b * seq_k * ed + s * ed + d];
                        b_v_ptr->grad()[d] += dv;

                        for (size_t k = 0; k < ed; k++) {
                            W_v_ptr->grad()[d * ed + k] += dv * value_ptr->data()[b * seq_k * ed + s * ed + k];
                            if (value_ptr->requires_grad) {
                                value_ptr->grad()[b * seq_k * ed + s * ed + k] += dv * W_v_ptr->data()[d * ed + k];
                            }
                        }
                    }
                }
            }

            for (size_t b = 0; b < batch; b++) {
                for (size_t s = 0; s < seq_k; s++) {
                    for (size_t d = 0; d < ed; d++) {
                        float dk = d_K->data()[b * seq_k * ed + s * ed + d];
                        b_k_ptr->grad()[d] += dk;

                        for (size_t k = 0; k < ed; k++) {
                            W_k_ptr->grad()[d * ed + k] += dk * key_ptr->data()[b * seq_k * ed + s * ed + k];
                            if (key_ptr->requires_grad) {
                                key_ptr->grad()[b * seq_k * ed + s * ed + k] += dk * W_k_ptr->data()[d * ed + k];
                            }
                        }
                    }
                }
            }

            for (size_t b = 0; b < batch; b++) {
                for (size_t s = 0; s < seq_q; s++) {
                    for (size_t d = 0; d < ed; d++) {
                        float dq = d_Q->data()[b * seq_q * ed + s * ed + d];
                        b_q_ptr->grad()[d] += dq;

                        for (size_t k = 0; k < ed; k++) {
                            W_q_ptr->grad()[d * ed + k] += dq * query_ptr->data()[b * seq_q * ed + s * ed + k];
                            if (query_ptr->requires_grad) {
                                query_ptr->grad()[b * seq_q * ed + s * ed + k] += dq * W_q_ptr->data()[d * ed + k];
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
                mask->data()[idx] = -1e9f;  // Large negative value (effectively -inf for softmax)
            } else {
                mask->data()[idx] = 0.0f;
            }
        }
    }

    return mask;
}

std::string MultiHeadAttention::extra_repr() const {
    return "embed_dim=" + std::to_string(embed_dim) +
           ", num_heads=" + std::to_string(num_heads);
}
