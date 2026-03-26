#include "../layer.h"
#include "../ops/matmul_cpu.h"
#include <random>
#include <cmath>
#include <cassert>
#include <limits>
#include <cstring>
#include <vector>

static thread_local std::mt19937 attention_rng(123);

// C[M×N] = A[M×K] @ B^T[K×N]  (B stored as [N×K])
static void matmul_ABt(float* C, const float* A, const float* B,
                       size_t M, size_t K, size_t N) {
    // Transpose B into scratch, then call matmul_blocked
    thread_local std::vector<float> Bt_buf;
    if (Bt_buf.size() < K * N) Bt_buf.resize(K * N);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < K; j++)
            Bt_buf[j * N + i] = B[i * K + j];
    matmul_blocked(C, A, Bt_buf.data(), M, K, N);
}

// Reshape [batch, seq, embed] (heads interleaved) → [batch*heads, seq, head_dim]
static void deinterleave_heads(float* dst, const float* src,
                               size_t batch, size_t seq, size_t num_heads, size_t head_dim) {
    size_t embed = num_heads * head_dim;
    for (size_t b = 0; b < batch; b++)
        for (size_t h = 0; h < num_heads; h++)
            for (size_t s = 0; s < seq; s++)
                for (size_t d = 0; d < head_dim; d++)
                    dst[(b * num_heads + h) * seq * head_dim + s * head_dim + d] =
                        src[b * seq * embed + s * embed + h * head_dim + d];
}

// Reshape [batch*heads, seq, head_dim] → [batch, seq, embed] (heads interleaved)
static void interleave_heads(float* dst, const float* src,
                             size_t batch, size_t seq, size_t num_heads, size_t head_dim) {
    size_t embed = num_heads * head_dim;
    for (size_t b = 0; b < batch; b++)
        for (size_t h = 0; h < num_heads; h++)
            for (size_t s = 0; s < seq; s++)
                for (size_t d = 0; d < head_dim; d++)
                    dst[b * seq * embed + s * embed + h * head_dim + d] =
                        src[(b * num_heads + h) * seq * head_dim + s * head_dim + d];
}

// Add bias to each row: out[s,d] += bias[d] for all rows in [seq, embed]
static void add_bias(float* out, const float* bias, size_t rows, size_t cols) {
    for (size_t s = 0; s < rows; s++)
        for (size_t d = 0; d < cols; d++)
            out[s * cols + d] += bias[d];
}

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
    assert(query->shape.size() == 3);
    assert(key->shape.size() == 3);
    assert(value->shape.size() == 3);
    assert(query->shape[2] == embed_dim);
    assert(key->shape[2] == embed_dim);
    assert(value->shape[2] == embed_dim);

    size_t batch = query->shape[0];
    size_t seq_q = query->shape[1];
    size_t seq_k = key->shape[1];
    size_t E = embed_dim;
    size_t nh = num_heads;
    size_t hd = head_dim;

    bool track = (query->requires_grad || key->requires_grad || value->requires_grad)
                 && GradMode::is_enabled();

    // --- 1. Linear projections: Q/K/V = input @ W^T + b ---
    auto Q = Tensor::create({batch, seq_q, E}, track);
    auto K = Tensor::create({batch, seq_k, E}, track);
    auto V = Tensor::create({batch, seq_k, E}, track);

    for (size_t b = 0; b < batch; b++) {
        matmul_ABt(Q->data() + b * seq_q * E,
                   query->data() + b * seq_q * E,
                   W_q->data(), seq_q, E, E);
        add_bias(Q->data() + b * seq_q * E, b_q->data(), seq_q, E);

        matmul_ABt(K->data() + b * seq_k * E,
                   key->data() + b * seq_k * E,
                   W_k->data(), seq_k, E, E);
        add_bias(K->data() + b * seq_k * E, b_k->data(), seq_k, E);

        matmul_ABt(V->data() + b * seq_k * E,
                   value->data() + b * seq_k * E,
                   W_v->data(), seq_k, E, E);
        add_bias(V->data() + b * seq_k * E, b_v->data(), seq_k, E);
    }

    // --- 2. Reshape Q/K/V to [batch*heads, seq, head_dim] ---
    size_t BH = batch * nh;
    std::vector<float> Qh(BH * seq_q * hd), Kh(BH * seq_k * hd), Vh(BH * seq_k * hd);
    deinterleave_heads(Qh.data(), Q->data(), batch, seq_q, nh, hd);
    deinterleave_heads(Kh.data(), K->data(), batch, seq_k, nh, hd);
    deinterleave_heads(Vh.data(), V->data(), batch, seq_k, nh, hd);

    // --- 3. Attention scores: scores[bh] = Qh[bh] @ Kh[bh]^T * scale ---
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    auto scores = Tensor::create({batch, nh, seq_q, seq_k}, track);

    // Transpose K for each (batch*head) and matmul
    thread_local std::vector<float> Kt_buf;
    if (Kt_buf.size() < hd * seq_k) Kt_buf.resize(hd * seq_k);

    for (size_t bh = 0; bh < BH; bh++) {
        const float* Kh_ptr = Kh.data() + bh * seq_k * hd;
        // Transpose K[seq_k, hd] → Kt[hd, seq_k]
        for (size_t i = 0; i < seq_k; i++)
            for (size_t j = 0; j < hd; j++)
                Kt_buf[j * seq_k + i] = Kh_ptr[i * hd + j];

        matmul_blocked(scores->data() + bh * seq_q * seq_k,
                       Qh.data() + bh * seq_q * hd,
                       Kt_buf.data(), seq_q, hd, seq_k);
    }

    // Apply scale and mask
    for (size_t i = 0; i < scores->size(); i++)
        scores->data()[i] *= scale;

    if (mask != nullptr) {
        for (size_t b = 0; b < batch; b++) {
            size_t mb = (mask->shape[0] == 1) ? 0 : b;
            for (size_t h = 0; h < nh; h++) {
                for (size_t i = 0; i < seq_q; i++) {
                    for (size_t j = 0; j < seq_k; j++) {
                        size_t score_idx = b * nh * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                        size_t mask_idx = mb * seq_q * seq_k + i * seq_k + j;
                        scores->data()[score_idx] += mask->data()[mask_idx];
                    }
                }
            }
        }
    }

    // --- 4. Softmax over seq_k dimension ---
    auto attn = Tensor::create({batch, nh, seq_q, seq_k}, track);
    for (size_t bh = 0; bh < BH; bh++) {
        for (size_t i = 0; i < seq_q; i++) {
            const float* row = scores->data() + bh * seq_q * seq_k + i * seq_k;
            float* out_row = attn->data() + bh * seq_q * seq_k + i * seq_k;

            float max_val = row[0];
            for (size_t j = 1; j < seq_k; j++)
                max_val = std::max(max_val, row[j]);

            float sum_exp = 0.0f;
            for (size_t j = 0; j < seq_k; j++) {
                out_row[j] = std::exp(row[j] - max_val);
                sum_exp += out_row[j];
            }
            for (size_t j = 0; j < seq_k; j++)
                out_row[j] /= sum_exp;
        }
    }

    attn_weights = attn;

    // --- 5. Context: ctx_heads[bh] = attn[bh] @ Vh[bh] ---
    std::vector<float> ctx_heads(BH * seq_q * hd);
    for (size_t bh = 0; bh < BH; bh++) {
        matmul_blocked(ctx_heads.data() + bh * seq_q * hd,
                       attn->data() + bh * seq_q * seq_k,
                       Vh.data() + bh * seq_k * hd,
                       seq_q, seq_k, hd);
    }

    // --- 6. Reshape context back to [batch, seq_q, embed_dim] ---
    auto context = Tensor::create({batch, seq_q, E}, track);
    interleave_heads(context->data(), ctx_heads.data(), batch, seq_q, nh, hd);

    // --- 7. Output projection: output[b] = context[b] @ W_o^T + b_o ---
    auto output = Tensor::create({batch, seq_q, E}, track);
    for (size_t b = 0; b < batch; b++) {
        matmul_ABt(output->data() + b * seq_q * E,
                   context->data() + b * seq_q * E,
                   W_o->data(), seq_q, E, E);
        add_bias(output->data() + b * seq_q * E, b_o->data(), seq_q, E);
    }

    // --- Backward pass ---
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

        output->parents = {query_ptr, key_ptr, value_ptr, W_q_ptr, W_k_ptr, W_v_ptr, W_o_ptr,
                          b_q_ptr, b_k_ptr, b_v_ptr, b_o_ptr};

        output->grad_fn = [=]() mutable {
            // --- Backward through output projection ---
            // d_context[b] = d_output[b] @ W_o (not transposed: d_out[seq,E] @ W_o[E,E])
            auto d_context = Tensor::create({batch, seq_q, E}, false);
            std::memset(d_context->data(), 0, d_context->size() * sizeof(float));

            thread_local std::vector<float> grad_tmp;
            thread_local std::vector<float> tmp_T;
            if (grad_tmp.size() < E * E) grad_tmp.resize(E * E);

            for (size_t b = 0; b < batch; b++) {
                const float* d_out = output->grad() + b * seq_q * E;
                float* d_ctx = d_context->data() + b * seq_q * E;

                // d_ctx[b] = d_out[b] @ W_o  (M=seq_q, K=E, N=E)
                matmul_blocked(d_ctx, d_out, W_o_ptr->data(), seq_q, E, E);

                // d_W_o += d_out^T @ context  (E×seq_q @ seq_q×E = E×E)
                if (tmp_T.size() < seq_q * E) tmp_T.resize(seq_q * E);
                for (size_t i = 0; i < seq_q; i++)
                    for (size_t j = 0; j < E; j++)
                        tmp_T[j * seq_q + i] = d_out[i * E + j];
                matmul_blocked(grad_tmp.data(), tmp_T.data(),
                               context->data() + b * seq_q * E, E, seq_q, E);
                for (size_t i = 0; i < E * E; i++)
                    W_o_ptr->grad()[i] += grad_tmp[i];

                // b_o grad: sum over seq dimension
                for (size_t s = 0; s < seq_q; s++)
                    for (size_t d = 0; d < E; d++)
                        b_o_ptr->grad()[d] += d_out[s * E + d];
            }

            // --- Backward through head reshape: d_ctx_heads ---
            std::vector<float> d_ctx_heads(BH * seq_q * hd);
            deinterleave_heads(d_ctx_heads.data(), d_context->data(), batch, seq_q, nh, hd);

            // --- Backward through context matmul: ctx = attn @ Vh ---
            // d_attn[bh] = d_ctx_heads[bh] @ Vh[bh]^T
            // d_Vh[bh] = attn[bh]^T @ d_ctx_heads[bh]
            auto d_attn = Tensor::create({batch, nh, seq_q, seq_k}, false);
            std::memset(d_attn->data(), 0, d_attn->size() * sizeof(float));
            std::vector<float> d_Vh(BH * seq_k * hd, 0.0f);

            for (size_t bh = 0; bh < BH; bh++) {
                // d_attn[bh] = d_ctx_heads[bh] @ Vh[bh]^T
                // Transpose Vh[seq_k, hd] → [hd, seq_k]
                if (tmp_T.size() < hd * seq_k) tmp_T.resize(hd * seq_k);
                const float* Vh_ptr = Vh.data() + bh * seq_k * hd;
                for (size_t i = 0; i < seq_k; i++)
                    for (size_t j = 0; j < hd; j++)
                        tmp_T[j * seq_k + i] = Vh_ptr[i * hd + j];
                matmul_blocked(d_attn->data() + bh * seq_q * seq_k,
                               d_ctx_heads.data() + bh * seq_q * hd,
                               tmp_T.data(), seq_q, hd, seq_k);

                // d_Vh[bh] = attn[bh]^T @ d_ctx_heads[bh]
                // Transpose attn[seq_q, seq_k] → [seq_k, seq_q]
                if (tmp_T.size() < seq_q * seq_k) tmp_T.resize(seq_q * seq_k);
                const float* attn_ptr = attn->data() + bh * seq_q * seq_k;
                for (size_t i = 0; i < seq_q; i++)
                    for (size_t j = 0; j < seq_k; j++)
                        tmp_T[j * seq_q + i] = attn_ptr[i * seq_k + j];
                matmul_blocked(d_Vh.data() + bh * seq_k * hd,
                               tmp_T.data(),
                               d_ctx_heads.data() + bh * seq_q * hd,
                               seq_k, seq_q, hd);
            }

            // --- Backward through softmax ---
            auto d_scores = Tensor::create({batch, nh, seq_q, seq_k}, false);
            for (size_t bh = 0; bh < BH; bh++) {
                for (size_t i = 0; i < seq_q; i++) {
                    const float* a_row = attn->data() + bh * seq_q * seq_k + i * seq_k;
                    const float* da_row = d_attn->data() + bh * seq_q * seq_k + i * seq_k;
                    float* ds_row = d_scores->data() + bh * seq_q * seq_k + i * seq_k;

                    float dot_sum = 0.0f;
                    for (size_t j = 0; j < seq_k; j++)
                        dot_sum += a_row[j] * da_row[j];
                    for (size_t j = 0; j < seq_k; j++)
                        ds_row[j] = a_row[j] * (da_row[j] - dot_sum) * scale;
                }
            }

            // --- Backward through scores matmul: scores = Qh @ Kh^T ---
            // d_Qh[bh] = d_scores[bh] @ Kh[bh]
            // d_Kh[bh] = d_scores[bh]^T @ Qh[bh]
            std::vector<float> d_Qh(BH * seq_q * hd, 0.0f);
            std::vector<float> d_Kh(BH * seq_k * hd, 0.0f);

            for (size_t bh = 0; bh < BH; bh++) {
                // d_Qh[bh] = d_scores[bh] @ Kh[bh]  (seq_q×seq_k @ seq_k×hd = seq_q×hd)
                matmul_blocked(d_Qh.data() + bh * seq_q * hd,
                               d_scores->data() + bh * seq_q * seq_k,
                               Kh.data() + bh * seq_k * hd,
                               seq_q, seq_k, hd);

                // d_Kh[bh] = d_scores[bh]^T @ Qh[bh]  (seq_k×seq_q @ seq_q×hd = seq_k×hd)
                if (tmp_T.size() < seq_q * seq_k) tmp_T.resize(seq_q * seq_k);
                const float* ds_ptr = d_scores->data() + bh * seq_q * seq_k;
                for (size_t i = 0; i < seq_q; i++)
                    for (size_t j = 0; j < seq_k; j++)
                        tmp_T[j * seq_q + i] = ds_ptr[i * seq_k + j];
                matmul_blocked(d_Kh.data() + bh * seq_k * hd,
                               tmp_T.data(),
                               Qh.data() + bh * seq_q * hd,
                               seq_k, seq_q, hd);
            }

            // --- Reshape head grads back to [batch, seq, embed] ---
            auto d_Q = Tensor::create({batch, seq_q, E}, false);
            auto d_K = Tensor::create({batch, seq_k, E}, false);
            auto d_V = Tensor::create({batch, seq_k, E}, false);
            std::memset(d_Q->data(), 0, d_Q->size() * sizeof(float));
            std::memset(d_K->data(), 0, d_K->size() * sizeof(float));
            std::memset(d_V->data(), 0, d_V->size() * sizeof(float));
            interleave_heads(d_Q->data(), d_Qh.data(), batch, seq_q, nh, hd);
            interleave_heads(d_K->data(), d_Kh.data(), batch, seq_k, nh, hd);
            interleave_heads(d_V->data(), d_Vh.data(), batch, seq_k, nh, hd);

            // --- Backward through Q/K/V projections via matmul ---
            // For each projection X (Q/K/V):
            //   d_input[b] += d_X[b] @ W        (input grad)
            //   d_W       += d_X[b]^T @ input[b] (weight grad)
            //   d_b       += colsum(d_X[b])       (bias grad)

            // Helper lambda for projection backward
            auto proj_backward = [&](
                const float* d_proj, size_t seq,
                const TensorPtr& input_ptr, const TensorPtr& W_ptr, const TensorPtr& b_ptr,
                bool input_needs_grad
            ) {
                for (size_t b = 0; b < batch; b++) {
                    const float* dX = d_proj + b * seq * E;
                    const float* inp = input_ptr->data() + b * seq * E;

                    // d_input[b] += dX[b] @ W  (seq×E @ E×E = seq×E)
                    if (input_needs_grad) {
                        thread_local std::vector<float> dinp_buf;
                        if (dinp_buf.size() < seq * E) dinp_buf.resize(seq * E);
                        matmul_blocked(dinp_buf.data(), dX, W_ptr->data(), seq, E, E);
                        float* grad_inp = input_ptr->grad() + b * seq * E;
                        for (size_t i = 0; i < seq * E; i++)
                            grad_inp[i] += dinp_buf[i];
                    }

                    // d_W += dX^T @ input  (E×seq @ seq×E = E×E)
                    // Transpose dX[seq,E] → [E,seq]
                    if (tmp_T.size() < seq * E) tmp_T.resize(seq * E);
                    for (size_t i = 0; i < seq; i++)
                        for (size_t j = 0; j < E; j++)
                            tmp_T[j * seq + i] = dX[i * E + j];
                    if (grad_tmp.size() < E * E) grad_tmp.resize(E * E);
                    matmul_blocked(grad_tmp.data(), tmp_T.data(), inp, E, seq, E);
                    for (size_t i = 0; i < E * E; i++)
                        W_ptr->grad()[i] += grad_tmp[i];

                    // d_b += colsum(dX)
                    for (size_t s = 0; s < seq; s++)
                        for (size_t d = 0; d < E; d++)
                            b_ptr->grad()[d] += dX[s * E + d];
                }
            };

            proj_backward(d_V->data(), seq_k, value_ptr, W_v_ptr, b_v_ptr, value_ptr->requires_grad);
            proj_backward(d_K->data(), seq_k, key_ptr, W_k_ptr, b_k_ptr, key_ptr->requires_grad);
            proj_backward(d_Q->data(), seq_q, query_ptr, W_q_ptr, b_q_ptr, query_ptr->requires_grad);
        };
    }

    return output;
}

std::vector<TensorPtr> MultiHeadAttention::parameters() {
    return {W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o};
}

TensorPtr MultiHeadAttention::causal_mask(size_t seq_len) {
    auto mask = Tensor::create({1, 1, seq_len, seq_len}, false);

    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            size_t idx = i * seq_len + j;
            mask->data()[idx] = (j > i) ? -1e9f : 0.0f;
        }
    }

    return mask;
}

std::string MultiHeadAttention::extra_repr() const {
    return "embed_dim=" + std::to_string(embed_dim) +
           ", num_heads=" + std::to_string(num_heads);
}
