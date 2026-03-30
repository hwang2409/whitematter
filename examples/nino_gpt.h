#ifndef NINO_GPT_H
#define NINO_GPT_H

/**
 * NinoGPT: A decoder-only transformer for character-level dialogue generation.
 *
 * Architecture (GPT-2 style, pre-norm):
 *   Token Embedding(256, 256) + Learned Position Embedding(512, 256)
 *   6x TransformerBlock:
 *       RMSNorm -> MultiHeadAttention(256, 8) + residual
 *       RMSNorm -> MLP(256 -> 1024 -> 256, SiLU) + residual
 *   RMSNorm -> Linear(256, 256)  (language model head)
 *
 * Conversation format:
 *   <|system|>...<|nino|>...<|user|>...<|nino|>...
 */

#include "tensor.h"
#include "layer.h"
#include "serialize.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <limits>

// =============================================================================
// Hyperparameters
// =============================================================================
static constexpr size_t EMBED_DIM   = 256;
static constexpr size_t NUM_HEADS   = 8;
static constexpr size_t NUM_LAYERS  = 6;
static constexpr size_t MAX_SEQ_LEN = 512;
static constexpr size_t VOCAB_SIZE  = 256;  // byte-level
static constexpr size_t MLP_DIM     = 4 * EMBED_DIM;  // 1024

// =============================================================================
// Conversation delimiters (byte sequences the model learns)
// =============================================================================
static const std::string SYSTEM_TAG = "<|system|>";
static const std::string USER_TAG   = "<|user|>";
static const std::string NINO_TAG   = "<|nino|>";

// =============================================================================
// Data loading
// =============================================================================
static std::vector<uint8_t> load_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::cerr << "ERROR: cannot open " << path << std::endl;
        exit(1);
    }
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

// =============================================================================
// Helper: apply a Linear layer to a 3-D tensor [batch, seq, in] -> [batch, seq, out]
// =============================================================================
static TensorPtr linear3d(const TensorPtr& input, Linear* layer) {
    size_t batch = input->shape[0];
    size_t seq   = input->shape[1];
    size_t in_f  = layer->in_features;
    size_t out_f = layer->out_features;

    bool track = input->requires_grad && GradMode::is_enabled();
    auto output = Tensor::create({batch, seq, out_f}, track);

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq; s++) {
            const float* x_ptr = input->data() + (b * seq + s) * in_f;
            float* o_ptr = output->data() + (b * seq + s) * out_f;
            for (size_t o = 0; o < out_f; o++) {
                float acc = layer->bias->data()[o];
                for (size_t i = 0; i < in_f; i++) {
                    acc += x_ptr[i] * layer->weight->data()[o * in_f + i];
                }
                o_ptr[o] = acc;
            }
        }
    }

    if (track) {
        auto inp = input;
        auto w   = layer->weight;
        auto bi  = layer->bias;
        output->parents = {inp, w, bi};
        output->grad_fn = [=]() {
            for (size_t b = 0; b < batch; b++) {
                for (size_t s = 0; s < seq; s++) {
                    const float* x_ptr = inp->data() + (b * seq + s) * in_f;
                    const float* dout  = output->grad() + (b * seq + s) * out_f;
                    float* dx = inp->grad() + (b * seq + s) * in_f;
                    for (size_t o = 0; o < out_f; o++) {
                        float d = dout[o];
                        bi->grad()[o] += d;
                        for (size_t i = 0; i < in_f; i++) {
                            w->grad()[o * in_f + i] += d * x_ptr[i];
                            dx[i] += d * w->data()[o * in_f + i];
                        }
                    }
                }
            }
        };
    }
    return output;
}

// =============================================================================
// Helper: element-wise SiLU on a 3-D tensor
// =============================================================================
static TensorPtr silu3d(const TensorPtr& input) {
    bool track = input->requires_grad && GradMode::is_enabled();
    auto output = Tensor::create(input->shape, track);
    size_t n = input->size();
    for (size_t i = 0; i < n; i++) {
        float x = input->data()[i];
        float sig = 1.0f / (1.0f + std::exp(-x));
        output->data()[i] = x * sig;
    }
    if (track) {
        auto inp = input;
        output->parents = {inp};
        output->grad_fn = [=]() {
            for (size_t i = 0; i < n; i++) {
                float x = inp->data()[i];
                float sig = 1.0f / (1.0f + std::exp(-x));
                float grad = sig + x * sig * (1.0f - sig);
                inp->grad()[i] += output->grad()[i] * grad;
            }
        };
    }
    return output;
}

// =============================================================================
// Helper: element-wise add of two 3-D tensors (residual connection)
// =============================================================================
static TensorPtr add3d(const TensorPtr& a, const TensorPtr& b) {
    assert(a->size() == b->size());
    bool track = (a->requires_grad || b->requires_grad) && GradMode::is_enabled();
    auto out = Tensor::create(a->shape, track);
    for (size_t i = 0; i < a->size(); i++) {
        out->data()[i] = a->data()[i] + b->data()[i];
    }
    if (track) {
        out->parents = {a, b};
        out->grad_fn = [a, b, out]() {
            for (size_t i = 0; i < a->size(); i++) {
                if (a->requires_grad) a->grad()[i] += out->grad()[i];
                if (b->requires_grad) b->grad()[i] += out->grad()[i];
            }
        };
    }
    return out;
}

// =============================================================================
// Transformer Block (pre-norm GPT-2 style)
// =============================================================================
struct TransformerBlock {
    RMSNorm*            ln1;
    MultiHeadAttention* attn;
    RMSNorm*            ln2;
    Linear*             fc1;
    Linear*             fc2;

    TransformerBlock() {
        ln1  = new RMSNorm(EMBED_DIM);
        attn = new MultiHeadAttention(EMBED_DIM, NUM_HEADS);
        ln2  = new RMSNorm(EMBED_DIM);
        fc1  = new Linear(EMBED_DIM, MLP_DIM);
        fc2  = new Linear(MLP_DIM, EMBED_DIM);
    }
    ~TransformerBlock() {
        delete ln1; delete attn; delete ln2; delete fc1; delete fc2;
    }

    TensorPtr forward(const TensorPtr& x, const TensorPtr& mask) {
        auto norm_x   = ln1->forward(x);
        auto attn_out = attn->forward(norm_x, norm_x, norm_x, mask);
        auto h        = add3d(x, attn_out);

        auto norm_h   = ln2->forward(h);
        auto mlp_up   = linear3d(norm_h, fc1);
        auto mlp_act  = silu3d(mlp_up);
        auto mlp_down = linear3d(mlp_act, fc2);
        return add3d(h, mlp_down);
    }

    std::vector<TensorPtr> parameters() {
        std::vector<TensorPtr> p;
        auto add = [&](const std::vector<TensorPtr>& v) {
            p.insert(p.end(), v.begin(), v.end());
        };
        add(ln1->parameters());
        add(attn->parameters());
        add(ln2->parameters());
        add(fc1->parameters());
        add(fc2->parameters());
        return p;
    }
};

// =============================================================================
// NinoGPT Model (inherits Module for save/load compatibility)
// =============================================================================
struct NinoGPT : public Module {
    Embedding*    tok_emb;
    Embedding*    pos_emb;
    std::vector<TransformerBlock*> blocks;
    RMSNorm*      final_norm;
    Linear*       lm_head;

    NinoGPT() {
        tok_emb    = new Embedding(VOCAB_SIZE, EMBED_DIM);
        pos_emb    = new Embedding(MAX_SEQ_LEN, EMBED_DIM);
        for (size_t i = 0; i < NUM_LAYERS; i++)
            blocks.push_back(new TransformerBlock());
        final_norm = new RMSNorm(EMBED_DIM);
        lm_head    = new Linear(EMBED_DIM, VOCAB_SIZE);
    }

    ~NinoGPT() {
        delete tok_emb; delete pos_emb; delete final_norm; delete lm_head;
        for (auto* b : blocks) delete b;
    }

    std::string name() const override { return "NinoGPT"; }

    TensorPtr forward(const TensorPtr& tokens) override {
        size_t batch   = tokens->shape[0];
        size_t seq_len = tokens->shape[1];

        auto te = tok_emb->forward(tokens);

        auto pos_idx = Tensor::create({seq_len}, false);
        for (size_t i = 0; i < seq_len; i++)
            pos_idx->data()[i] = static_cast<float>(i);

        auto pe = pos_emb->forward(pos_idx);

        bool track = te->requires_grad && GradMode::is_enabled();
        auto x = Tensor::create({batch, seq_len, EMBED_DIM}, track);
        for (size_t b = 0; b < batch; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t d = 0; d < EMBED_DIM; d++) {
                    x->data()[(b * seq_len + s) * EMBED_DIM + d] =
                        te->data()[(b * seq_len + s) * EMBED_DIM + d] +
                        pe->data()[s * EMBED_DIM + d];
                }
            }
        }
        if (track) {
            auto te_p = te;
            auto pe_p = pe;
            x->parents = {te_p, pe_p};
            x->grad_fn = [=]() {
                for (size_t b = 0; b < batch; b++) {
                    for (size_t s = 0; s < seq_len; s++) {
                        for (size_t d = 0; d < EMBED_DIM; d++) {
                            float g = x->grad()[(b * seq_len + s) * EMBED_DIM + d];
                            te_p->grad()[(b * seq_len + s) * EMBED_DIM + d] += g;
                            pe_p->grad()[s * EMBED_DIM + d] += g;
                        }
                    }
                }
            };
        }

        auto mask = Tensor::create({1, 1, seq_len, seq_len}, false);
        for (size_t i = 0; i < seq_len; i++)
            for (size_t j = 0; j < seq_len; j++)
                mask->data()[i * seq_len + j] = (j > i) ? -1e9f : 0.0f;

        for (auto* block : blocks)
            x = block->forward(x, mask);

        x = final_norm->forward(x);

        return linear3d(x, lm_head);
    }

    std::vector<TensorPtr> parameters() override {
        std::vector<TensorPtr> p;
        auto add = [&](const std::vector<TensorPtr>& v) {
            p.insert(p.end(), v.begin(), v.end());
        };
        add(tok_emb->parameters());
        add(pos_emb->parameters());
        for (auto* b : blocks)
            add(b->parameters());
        add(final_norm->parameters());
        add(lm_head->parameters());
        return p;
    }

    size_t count_params() {
        size_t n = 0;
        for (auto& p : parameters()) n += p->size();
        return n;
    }
};

// =============================================================================
// Cross-entropy loss for language modeling on 3-D logits
// logits: [batch, seq, vocab],  targets: [batch, seq] (integer labels)
// =============================================================================
static TensorPtr lm_cross_entropy(const TensorPtr& logits, const TensorPtr& targets) {
    size_t batch      = logits->shape[0];
    size_t seq_len    = logits->shape[1];
    size_t vocab_size = logits->shape[2];
    size_t N = batch * seq_len;

    auto loss = Tensor::create({1}, logits->requires_grad);
    loss->data()[0] = 0.0f;

    std::vector<float> log_probs(batch * seq_len * vocab_size);

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            size_t offset = (b * seq_len + s) * vocab_size;
            float mx = -std::numeric_limits<float>::max();
            for (size_t v = 0; v < vocab_size; v++)
                mx = std::max(mx, logits->data()[offset + v]);

            float sum_exp = 0.0f;
            for (size_t v = 0; v < vocab_size; v++)
                sum_exp += std::exp(logits->data()[offset + v] - mx);
            float log_Z = mx + std::log(sum_exp);

            for (size_t v = 0; v < vocab_size; v++)
                log_probs[offset + v] = logits->data()[offset + v] - log_Z;

            size_t label = static_cast<size_t>(targets->data()[b * seq_len + s]);
            loss->data()[0] -= log_probs[offset + label];
        }
    }
    loss->data()[0] /= static_cast<float>(N);

    if (loss->requires_grad) {
        auto lp = logits;
        loss->parents = {lp};
        loss->grad_fn = [=]() {
            float scale = loss->grad()[0] / static_cast<float>(N);
            for (size_t b = 0; b < batch; b++) {
                for (size_t s = 0; s < seq_len; s++) {
                    size_t offset = (b * seq_len + s) * vocab_size;
                    size_t label = static_cast<size_t>(targets->data()[b * seq_len + s]);
                    for (size_t v = 0; v < vocab_size; v++) {
                        float softmax_v = std::exp(log_probs[offset + v]);
                        float g = softmax_v;
                        if (v == label) g -= 1.0f;
                        lp->grad()[offset + v] += scale * g;
                    }
                }
            }
        };
    }
    return loss;
}

// =============================================================================
// Text generation with delimiter-based stopping
// =============================================================================
static std::string generate(NinoGPT& model, const std::string& context,
                            size_t max_tokens, float temperature = 0.8f,
                            const std::string& stop_token = USER_TAG) {
    NoGradGuard no_grad;

    std::mt19937 rng(static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count()));

    std::string result = context;

    for (size_t t = 0; t < max_tokens; t++) {
        size_t ctx_len = std::min(result.size(), MAX_SEQ_LEN);
        auto tokens = Tensor::create({1, ctx_len}, false);
        for (size_t i = 0; i < ctx_len; i++) {
            uint8_t byte = static_cast<uint8_t>(result[result.size() - ctx_len + i]);
            tokens->data()[i] = static_cast<float>(byte);
        }

        auto logits = model.forward(tokens);

        size_t last = ctx_len - 1;
        size_t offset = last * VOCAB_SIZE;

        float mx = -std::numeric_limits<float>::max();
        for (size_t v = 0; v < VOCAB_SIZE; v++)
            mx = std::max(mx, logits->data()[offset + v]);

        std::vector<float> probs(VOCAB_SIZE);
        float sum = 0.0f;
        for (size_t v = 0; v < VOCAB_SIZE; v++) {
            probs[v] = std::exp((logits->data()[offset + v] - mx) / temperature);
            sum += probs[v];
        }
        for (size_t v = 0; v < VOCAB_SIZE; v++)
            probs[v] /= sum;

        std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
        size_t sampled = dist(rng);
        result += static_cast<char>(sampled);

        // Check if we've generated the stop token
        if (result.size() >= stop_token.size()) {
            std::string tail = result.substr(result.size() - stop_token.size());
            if (tail == stop_token) {
                // Remove the stop token from result
                result = result.substr(0, result.size() - stop_token.size());
                break;
            }
        }
    }

    // Return only the newly generated text (after the input context)
    return result.substr(context.size());
}

#endif // NINO_GPT_H
