/**
 * Simple Transformer Language Model Example
 *
 * This example demonstrates building a small transformer decoder
 * that learns to predict the next character in simple patterns.
 *
 * Architecture:
 *   - Character Embedding
 *   - Sinusoidal Positional Encoding
 *   - N x Transformer Blocks:
 *       - Multi-Head Self-Attention (causal)
 *       - LayerNorm
 *       - Feed-Forward Network
 *       - LayerNorm
 *   - Output Linear projection
 *
 * Task: Learn simple character patterns like "abcabcabc..."
 */

#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <map>

// Sinusoidal positional encoding
TensorPtr positional_encoding(size_t seq_len, size_t embed_dim) {
    auto pe = Tensor::create({1, seq_len, embed_dim}, false);

    for (size_t pos = 0; pos < seq_len; pos++) {
        for (size_t i = 0; i < embed_dim; i++) {
            float angle = static_cast<float>(pos) / std::pow(10000.0f, (2.0f * (i / 2)) / embed_dim);
            if (i % 2 == 0) {
                pe->data[pos * embed_dim + i] = std::sin(angle);
            } else {
                pe->data[pos * embed_dim + i] = std::cos(angle);
            }
        }
    }

    return pe;
}

// Simple Transformer Block
class TransformerBlock {
public:
    MultiHeadAttention* attn;
    LayerNorm* ln1;
    LayerNorm* ln2;
    Linear* ff1;
    Linear* ff2;
    size_t embed_dim;

    TransformerBlock(size_t embed_dim, size_t num_heads, size_t ff_dim)
        : embed_dim(embed_dim) {
        attn = new MultiHeadAttention(embed_dim, num_heads);
        ln1 = new LayerNorm(embed_dim);
        ln2 = new LayerNorm(embed_dim);
        ff1 = new Linear(embed_dim, ff_dim);
        ff2 = new Linear(ff_dim, embed_dim);
    }

    ~TransformerBlock() {
        delete attn;
        delete ln1;
        delete ln2;
        delete ff1;
        delete ff2;
    }

    // Helper to apply Linear to 3D tensor [batch, seq, features]
    TensorPtr apply_linear_3d(const TensorPtr& input, Linear* linear) {
        size_t batch = input->shape[0];
        size_t seq_len = input->shape[1];
        size_t in_features = input->shape[2];
        size_t out_features = linear->out_features;

        bool track = input->requires_grad && GradMode::is_enabled();
        auto output = Tensor::create({batch, seq_len, out_features}, track);

        // Apply linear to each position
        for (size_t b = 0; b < batch; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t o = 0; o < out_features; o++) {
                    float sum = linear->bias->data[o];
                    for (size_t i = 0; i < in_features; i++) {
                        sum += input->data[b * seq_len * in_features + s * in_features + i] *
                               linear->weight->data[o * in_features + i];
                    }
                    output->data[b * seq_len * out_features + s * out_features + o] = sum;
                }
            }
        }

        if (track) {
            auto input_ptr = input;
            auto weight = linear->weight;
            auto bias = linear->bias;
            output->parents = {input_ptr, weight, bias};
            output->grad_fn = [=]() {
                for (size_t b = 0; b < batch; b++) {
                    for (size_t s = 0; s < seq_len; s++) {
                        for (size_t o = 0; o < out_features; o++) {
                            float dout = output->grad[b * seq_len * out_features + s * out_features + o];
                            bias->grad[o] += dout;
                            for (size_t i = 0; i < in_features; i++) {
                                weight->grad[o * in_features + i] += dout * input_ptr->data[b * seq_len * in_features + s * in_features + i];
                                input_ptr->grad[b * seq_len * in_features + s * in_features + i] += dout * weight->data[o * in_features + i];
                            }
                        }
                    }
                }
            };
        }

        return output;
    }

    TensorPtr forward(const TensorPtr& x, const TensorPtr& mask) {
        // Self-attention with residual
        auto attn_out = attn->forward(x, x, x, mask);

        // Residual connection: x + attn_out
        auto res1 = Tensor::create(x->shape, x->requires_grad && GradMode::is_enabled());
        for (size_t i = 0; i < x->data.size(); i++) {
            res1->data[i] = x->data[i] + attn_out->data[i];
        }
        if (res1->requires_grad) {
            res1->parents = {x, attn_out};
            res1->grad_fn = [x, attn_out, res1]() {
                for (size_t i = 0; i < x->data.size(); i++) {
                    x->grad[i] += res1->grad[i];
                    attn_out->grad[i] += res1->grad[i];
                }
            };
        }

        // LayerNorm
        auto norm1 = ln1->forward(res1);

        // Feed-forward network with ReLU (using helper for 3D tensors)
        auto ff_hidden = apply_linear_3d(norm1, ff1);

        // ReLU activation
        auto ff_relu = Tensor::create(ff_hidden->shape, ff_hidden->requires_grad && GradMode::is_enabled());
        for (size_t i = 0; i < ff_hidden->data.size(); i++) {
            ff_relu->data[i] = std::max(0.0f, ff_hidden->data[i]);
        }
        if (ff_relu->requires_grad) {
            ff_relu->parents = {ff_hidden};
            ff_relu->grad_fn = [ff_hidden, ff_relu]() {
                for (size_t i = 0; i < ff_hidden->data.size(); i++) {
                    if (ff_hidden->data[i] > 0) {
                        ff_hidden->grad[i] += ff_relu->grad[i];
                    }
                }
            };
        }

        auto ff_out = apply_linear_3d(ff_relu, ff2);

        // Residual connection: norm1 + ff_out
        auto res2 = Tensor::create(norm1->shape, norm1->requires_grad && GradMode::is_enabled());
        for (size_t i = 0; i < norm1->data.size(); i++) {
            res2->data[i] = norm1->data[i] + ff_out->data[i];
        }
        if (res2->requires_grad) {
            res2->parents = {norm1, ff_out};
            res2->grad_fn = [norm1, ff_out, res2]() {
                for (size_t i = 0; i < norm1->data.size(); i++) {
                    norm1->grad[i] += res2->grad[i];
                    ff_out->grad[i] += res2->grad[i];
                }
            };
        }

        // Final LayerNorm
        return ln2->forward(res2);
    }

    std::vector<TensorPtr> parameters() {
        std::vector<TensorPtr> params;
        auto attn_params = attn->parameters();
        auto ln1_params = ln1->parameters();
        auto ln2_params = ln2->parameters();
        auto ff1_params = ff1->parameters();
        auto ff2_params = ff2->parameters();

        params.insert(params.end(), attn_params.begin(), attn_params.end());
        params.insert(params.end(), ln1_params.begin(), ln1_params.end());
        params.insert(params.end(), ln2_params.begin(), ln2_params.end());
        params.insert(params.end(), ff1_params.begin(), ff1_params.end());
        params.insert(params.end(), ff2_params.begin(), ff2_params.end());

        return params;
    }
};

// Simple Transformer Language Model
class TransformerLM {
public:
    Embedding* token_embed;
    TensorPtr pos_encoding;
    std::vector<TransformerBlock*> blocks;
    Linear* output_proj;
    TensorPtr causal_mask;

    size_t vocab_size;
    size_t embed_dim;
    size_t max_seq_len;

    TransformerLM(size_t vocab_size, size_t embed_dim, size_t num_heads,
                  size_t num_layers, size_t ff_dim, size_t max_seq_len)
        : vocab_size(vocab_size), embed_dim(embed_dim), max_seq_len(max_seq_len) {

        token_embed = new Embedding(vocab_size, embed_dim);
        pos_encoding = positional_encoding(max_seq_len, embed_dim);

        for (size_t i = 0; i < num_layers; i++) {
            blocks.push_back(new TransformerBlock(embed_dim, num_heads, ff_dim));
        }

        output_proj = new Linear(embed_dim, vocab_size);
        causal_mask = MultiHeadAttention::causal_mask(max_seq_len);
    }

    ~TransformerLM() {
        delete token_embed;
        for (auto block : blocks) delete block;
        delete output_proj;
    }

    TensorPtr forward(const TensorPtr& tokens) {
        // tokens: [batch, seq_len] - integer indices
        size_t batch = tokens->shape[0];
        size_t seq_len = tokens->shape[1];

        // Get token embeddings: [batch, seq_len, embed_dim]
        auto embeds = token_embed->forward(tokens);

        // Add positional encoding
        auto x = Tensor::create({batch, seq_len, embed_dim}, embeds->requires_grad);
        for (size_t b = 0; b < batch; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t d = 0; d < embed_dim; d++) {
                    x->data[b * seq_len * embed_dim + s * embed_dim + d] =
                        embeds->data[b * seq_len * embed_dim + s * embed_dim + d] +
                        pos_encoding->data[s * embed_dim + d];
                }
            }
        }
        if (x->requires_grad) {
            x->parents = {embeds};
            x->grad_fn = [embeds, x]() {
                for (size_t i = 0; i < embeds->data.size(); i++) {
                    embeds->grad[i] += x->grad[i];
                }
            };
        }

        // Create mask for current sequence length
        auto mask = Tensor::create({1, 1, seq_len, seq_len}, false);
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < seq_len; j++) {
                mask->data[i * seq_len + j] = (j > i) ? -1e9f : 0.0f;
            }
        }

        // Pass through transformer blocks
        for (auto block : blocks) {
            x = block->forward(x, mask);
        }

        // Project to vocabulary: [batch, seq_len, vocab_size]
        // We need to manually apply linear to each position
        auto logits = Tensor::create({batch, seq_len, vocab_size}, x->requires_grad);

        for (size_t b = 0; b < batch; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t v = 0; v < vocab_size; v++) {
                    float sum = output_proj->bias->data[v];
                    for (size_t d = 0; d < embed_dim; d++) {
                        sum += x->data[b * seq_len * embed_dim + s * embed_dim + d] *
                               output_proj->weight->data[v * embed_dim + d];
                    }
                    logits->data[b * seq_len * vocab_size + s * vocab_size + v] = sum;
                }
            }
        }

        if (logits->requires_grad) {
            auto x_ptr = x;
            auto weight = output_proj->weight;
            auto bias = output_proj->bias;
            logits->parents = {x_ptr, weight, bias};
            logits->grad_fn = [=]() {
                for (size_t b = 0; b < batch; b++) {
                    for (size_t s = 0; s < seq_len; s++) {
                        for (size_t v = 0; v < vocab_size; v++) {
                            float dout = logits->grad[b * seq_len * vocab_size + s * vocab_size + v];
                            bias->grad[v] += dout;
                            for (size_t d = 0; d < embed_dim; d++) {
                                weight->grad[v * embed_dim + d] += dout * x_ptr->data[b * seq_len * embed_dim + s * embed_dim + d];
                                x_ptr->grad[b * seq_len * embed_dim + s * embed_dim + d] += dout * weight->data[v * embed_dim + d];
                            }
                        }
                    }
                }
            };
        }

        return logits;
    }

    std::vector<TensorPtr> parameters() {
        std::vector<TensorPtr> params;

        auto embed_params = token_embed->parameters();
        params.insert(params.end(), embed_params.begin(), embed_params.end());

        for (auto block : blocks) {
            auto block_params = block->parameters();
            params.insert(params.end(), block_params.begin(), block_params.end());
        }

        auto proj_params = output_proj->parameters();
        params.insert(params.end(), proj_params.begin(), proj_params.end());

        return params;
    }
};

// Cross-entropy loss for language modeling
TensorPtr lm_cross_entropy(const TensorPtr& logits, const TensorPtr& targets) {
    // logits: [batch, seq_len, vocab_size]
    // targets: [batch, seq_len] - integer class indices

    size_t batch = logits->shape[0];
    size_t seq_len = logits->shape[1];
    size_t vocab_size = logits->shape[2];

    auto loss = Tensor::create({1}, logits->requires_grad);
    loss->data[0] = 0.0f;

    // Compute log-softmax and NLL loss
    std::vector<float> log_probs(batch * seq_len * vocab_size);

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            // Find max for numerical stability
            float max_val = -std::numeric_limits<float>::max();
            for (size_t v = 0; v < vocab_size; v++) {
                max_val = std::max(max_val, logits->data[b * seq_len * vocab_size + s * vocab_size + v]);
            }

            // Compute log-softmax
            float sum_exp = 0.0f;
            for (size_t v = 0; v < vocab_size; v++) {
                sum_exp += std::exp(logits->data[b * seq_len * vocab_size + s * vocab_size + v] - max_val);
            }
            float log_sum_exp = max_val + std::log(sum_exp);

            for (size_t v = 0; v < vocab_size; v++) {
                log_probs[b * seq_len * vocab_size + s * vocab_size + v] =
                    logits->data[b * seq_len * vocab_size + s * vocab_size + v] - log_sum_exp;
            }

            // NLL loss
            size_t target_idx = static_cast<size_t>(targets->data[b * seq_len + s]);
            loss->data[0] -= log_probs[b * seq_len * vocab_size + s * vocab_size + target_idx];
        }
    }
    loss->data[0] /= static_cast<float>(batch * seq_len);

    if (loss->requires_grad) {
        auto logits_ptr = logits;
        loss->parents = {logits_ptr};
        loss->grad_fn = [=]() {
            float scale = loss->grad[0] / static_cast<float>(batch * seq_len);

            for (size_t b = 0; b < batch; b++) {
                for (size_t s = 0; s < seq_len; s++) {
                    size_t target_idx = static_cast<size_t>(targets->data[b * seq_len + s]);

                    // Gradient of cross-entropy w.r.t. logits is: softmax(logits) - one_hot(target)
                    for (size_t v = 0; v < vocab_size; v++) {
                        float softmax_v = std::exp(log_probs[b * seq_len * vocab_size + s * vocab_size + v]);
                        float grad = softmax_v;
                        if (v == target_idx) grad -= 1.0f;
                        logits_ptr->grad[b * seq_len * vocab_size + s * vocab_size + v] += scale * grad;
                    }
                }
            }
        };
    }

    return loss;
}

// Generate text from the model
std::string generate(TransformerLM& model, const std::string& prompt,
                     const std::vector<char>& vocab, size_t num_tokens) {
    std::string result = prompt;

    // Build char to index map
    std::map<char, int> char_to_idx;
    for (size_t i = 0; i < vocab.size(); i++) {
        char_to_idx[vocab[i]] = i;
    }

    NoGradGuard no_grad;

    for (size_t i = 0; i < num_tokens; i++) {
        // Prepare input tokens
        size_t seq_len = std::min(result.size(), model.max_seq_len);
        auto tokens = Tensor::create({1, seq_len}, false);

        for (size_t j = 0; j < seq_len; j++) {
            char c = result[result.size() - seq_len + j];
            tokens->data[j] = static_cast<float>(char_to_idx[c]);
        }

        // Forward pass
        auto logits = model.forward(tokens);

        // Get logits for last position
        size_t last_pos = seq_len - 1;

        // Greedy sampling: pick highest probability token
        float max_logit = -std::numeric_limits<float>::max();
        size_t best_idx = 0;
        for (size_t v = 0; v < model.vocab_size; v++) {
            float logit = logits->data[last_pos * model.vocab_size + v];
            if (logit > max_logit) {
                max_logit = logit;
                best_idx = v;
            }
        }

        result += vocab[best_idx];
    }

    return result;
}

int main() {
    std::cout << "=== Simple Transformer Language Model ===" << std::endl;
    std::cout << std::endl;

    // Create a simple vocabulary and training data
    // Pattern: "abcdefabcdefabcdef..."
    std::vector<char> vocab = {'a', 'b', 'c', 'd', 'e', 'f'};
    size_t vocab_size = vocab.size();

    std::map<char, int> char_to_idx;
    for (size_t i = 0; i < vocab.size(); i++) {
        char_to_idx[vocab[i]] = i;
    }

    // Training data: repeating pattern
    std::string pattern = "abcdef";
    std::string train_text;
    for (int i = 0; i < 100; i++) {
        train_text += pattern;
    }

    // Hyperparameters
    size_t embed_dim = 32;
    size_t num_heads = 4;
    size_t num_layers = 2;
    size_t ff_dim = 64;
    size_t max_seq_len = 16;
    size_t batch_size = 8;
    float learning_rate = 0.001f;
    int num_epochs = 100;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Vocabulary: ";
    for (char c : vocab) std::cout << c;
    std::cout << " (size=" << vocab_size << ")" << std::endl;
    std::cout << "  Pattern: " << pattern << std::endl;
    std::cout << "  Embed dim: " << embed_dim << std::endl;
    std::cout << "  Num heads: " << num_heads << std::endl;
    std::cout << "  Num layers: " << num_layers << std::endl;
    std::cout << "  FF dim: " << ff_dim << std::endl;
    std::cout << "  Max seq len: " << max_seq_len << std::endl;
    std::cout << std::endl;

    // Create model
    std::cout << "Creating model..." << std::endl;
    TransformerLM model(vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len);

    // Count parameters
    auto params = model.parameters();
    size_t total_params = 0;
    for (auto& p : params) {
        total_params += p->data.size();
    }
    std::cout << "Total parameters: " << total_params << std::endl;
    std::cout << std::endl;

    // Optimizer
    Adam optimizer(params, learning_rate);

    // Training loop
    std::cout << "Training..." << std::endl;
    std::mt19937 rng(42);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int num_batches = 0;

        // Create random batches
        for (size_t start = 0; start + max_seq_len + 1 < train_text.size(); start += max_seq_len) {
            // Prepare batch
            auto input_tokens = Tensor::create({batch_size, max_seq_len}, true);
            auto target_tokens = Tensor::create({batch_size, max_seq_len}, false);

            for (size_t b = 0; b < batch_size; b++) {
                // Random starting position
                size_t pos = rng() % (train_text.size() - max_seq_len - 1);

                for (size_t s = 0; s < max_seq_len; s++) {
                    input_tokens->data[b * max_seq_len + s] = static_cast<float>(char_to_idx[train_text[pos + s]]);
                    target_tokens->data[b * max_seq_len + s] = static_cast<float>(char_to_idx[train_text[pos + s + 1]]);
                }
            }

            // Forward pass
            optimizer.zero_grad();
            auto logits = model.forward(input_tokens);
            auto loss = lm_cross_entropy(logits, target_tokens);

            // Backward pass
            loss->backward();

            // Gradient clipping
            clip_grad_norm_(params, 1.0f);

            // Update
            optimizer.step();

            epoch_loss += loss->data[0];
            num_batches++;

            if (num_batches >= 10) break;  // Limit batches per epoch
        }

        epoch_loss /= num_batches;

        if (epoch % 10 == 0 || epoch == num_epochs - 1) {
            std::cout << "Epoch " << epoch + 1 << "/" << num_epochs
                      << " - Loss: " << epoch_loss << std::endl;

            // Generate sample
            std::string sample = generate(model, "abc", vocab, 20);
            std::cout << "  Sample: " << sample << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "=== Final Generation ===" << std::endl;

    // Generate from different prompts
    std::vector<std::string> prompts = {"a", "ab", "abc", "def", "f"};
    for (const auto& prompt : prompts) {
        std::string generated = generate(model, prompt, vocab, 30);
        std::cout << "Prompt '" << prompt << "' -> " << generated << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Expected pattern: abcdefabcdef..." << std::endl;
    std::cout << std::endl;

    return 0;
}
