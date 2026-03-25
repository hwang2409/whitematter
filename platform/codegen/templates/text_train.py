"""
Text model (Transformer language model) training code template.
"""

import re


def generate_text_training_code(
    layers_code: str,
    optimizer_code: str,
    scheduler_code: str,
    epochs: int,
    batch_size: int,
    dataset_config: dict
) -> str:
    """Generate text model training code for language modeling using Transformer."""

    has_scheduler = bool(scheduler_code)
    scheduler_step = "scheduler.step();" if has_scheduler else ""
    scheduler_include = scheduler_code if has_scheduler else ""

    vocab_size = dataset_config.get("vocab_size", dataset_config.get("num_classes", 100))
    seq_length = dataset_config.get("seq_length", 128)

    # Extract layer parameters for the custom model
    embed_dim = 128
    num_heads = 4
    num_layers = 4
    ff_dim = 256
    dropout_rate = 0.1

    # Parse layers_code to extract dimensions
    if "embedding_dim" in layers_code:
        match = re.search(r'embedding_dim.*?(\d+)', layers_code)
        if match:
            embed_dim = int(match.group(1))
    if "hidden_size" in layers_code:
        match = re.search(r'hidden_size.*?(\d+)', layers_code)
        if match:
            embed_dim = int(match.group(1))  # Use as embed_dim for transformer
            ff_dim = embed_dim * 2
    if "num_heads" in layers_code:
        match = re.search(r'num_heads.*?(\d+)', layers_code)
        if match:
            num_heads = int(match.group(1))
    if "num_layers" in layers_code:
        match = re.search(r'num_layers.*?(\d+)', layers_code)
        if match:
            num_layers = int(match.group(1))

    # Ensure embed_dim is divisible by num_heads
    if embed_dim % num_heads != 0:
        num_heads = 4
        while embed_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1

    code = f'''// Auto-generated Transformer language model training code
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "serialize.h"

// Binary tensor loading
struct TensorFile {{
    std::vector<size_t> shape;
    std::vector<float> data;
}};

TensorFile load_tensor_file(const std::string& path) {{
    TensorFile result;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x54454E53) throw std::runtime_error("Invalid tensor file");

    uint32_t ndim;
    f.read(reinterpret_cast<char*>(&ndim), 4);

    result.shape.resize(ndim);
    for (uint32_t i = 0; i < ndim; i++) {{
        uint64_t dim;
        f.read(reinterpret_cast<char*>(&dim), 8);
        result.shape[i] = dim;
    }}

    size_t total = 1;
    for (auto d : result.shape) total *= d;
    result.data.resize(total);
    f.read(reinterpret_cast<char*>(result.data.data()), total * sizeof(float));

    return result;
}}

// Transformer Block: Self-Attention + FFN with residual connections
class TransformerBlock : public Module {{
public:
    MultiHeadAttention* attn;
    LayerNorm* ln1;
    LayerNorm* ln2;
    Linear* ff1;
    Linear* ff2;
    Dropout* dropout1;
    Dropout* dropout2;

    size_t embed_dim_;
    size_t ff_dim_;

    TransformerBlock(size_t embed_dim, size_t num_heads, size_t ff_dim, float dropout = 0.1f)
        : embed_dim_(embed_dim), ff_dim_(ff_dim) {{
        attn = new MultiHeadAttention(embed_dim, num_heads);
        ln1 = new LayerNorm(embed_dim);
        ln2 = new LayerNorm(embed_dim);
        ff1 = new Linear(embed_dim, ff_dim);
        ff2 = new Linear(ff_dim, embed_dim);
        dropout1 = new Dropout(dropout);
        dropout2 = new Dropout(dropout);
    }}

    ~TransformerBlock() {{
        delete attn;
        delete ln1;
        delete ln2;
        delete ff1;
        delete ff2;
        delete dropout1;
        delete dropout2;
    }}

    // Forward with causal mask
    TensorPtr forward_with_mask(const TensorPtr& x, const TensorPtr& mask) {{
        // Self-attention with residual
        auto attn_out = attn->forward(x, x, x, mask);
        attn_out = dropout1->forward(attn_out);
        auto h = ln1->forward(x + attn_out);

        // FFN with residual: Linear -> ReLU -> Linear
        size_t batch = h->shape[0];
        size_t seq_len = h->shape[1];

        auto h_flat = h->reshape({{batch * seq_len, embed_dim_}});
        auto ff_out = ff1->forward(h_flat);
        ff_out = ff_out->relu();
        ff_out = ff2->forward(ff_out);
        ff_out = ff_out->reshape({{batch, seq_len, embed_dim_}});
        ff_out = dropout2->forward(ff_out);

        return ln2->forward(h + ff_out);
    }}

    TensorPtr forward(const TensorPtr& x) override {{
        // Create causal mask
        size_t seq_len = x->shape[1];
        auto mask = MultiHeadAttention::causal_mask(seq_len);
        return forward_with_mask(x, mask);
    }}

    std::vector<TensorPtr> parameters() override {{
        std::vector<TensorPtr> params;
        for (auto& p : attn->parameters()) params.push_back(p);
        for (auto& p : ln1->parameters()) params.push_back(p);
        for (auto& p : ln2->parameters()) params.push_back(p);
        for (auto& p : ff1->parameters()) params.push_back(p);
        for (auto& p : ff2->parameters()) params.push_back(p);
        return params;
    }}

    void train() {{
        dropout1->train();
        dropout2->train();
    }}

    void eval() {{
        dropout1->eval();
        dropout2->eval();
    }}

    std::string name() const override {{ return "TransformerBlock"; }}
}};

// Transformer Language Model
class TransformerLM : public Module {{
public:
    Embedding* token_emb;
    TensorPtr pos_emb;  // Learned positional embeddings
    std::vector<TransformerBlock*> blocks;
    LayerNorm* ln_final;
    Linear* head;
    Dropout* dropout;

    size_t vocab_size_;
    size_t embed_dim_;
    size_t max_seq_len_;

    TransformerLM(size_t vocab_size, size_t embed_dim, size_t num_heads,
                  size_t num_layers, size_t ff_dim, size_t max_seq_len, float dropout_rate = 0.1f)
        : vocab_size_(vocab_size), embed_dim_(embed_dim), max_seq_len_(max_seq_len) {{

        token_emb = new Embedding(vocab_size, embed_dim);

        // Learned positional embeddings
        pos_emb = Tensor::randn({{max_seq_len, embed_dim}}, true);
        float scale = 0.02f;
        for (auto& v : pos_emb->data) v *= scale;

        // Transformer blocks
        for (size_t i = 0; i < num_layers; i++) {{
            blocks.push_back(new TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate));
        }}

        ln_final = new LayerNorm(embed_dim);
        head = new Linear(embed_dim, vocab_size);
        dropout = new Dropout(dropout_rate);
    }}

    ~TransformerLM() {{
        delete token_emb;
        for (auto* b : blocks) delete b;
        delete ln_final;
        delete head;
        delete dropout;
    }}

    // Forward: [batch, seq_len] -> [batch, seq_len, vocab_size]
    TensorPtr forward(const TensorPtr& x) override {{
        size_t batch = x->shape[0];
        size_t seq_len = x->shape[1];

        // Token embeddings: [batch, seq_len] -> [batch, seq_len, embed_dim]
        auto h = token_emb->forward(x);

        // Add positional embeddings (broadcast across batch)
        // pos_emb is [max_seq_len, embed_dim], we need [seq_len, embed_dim]
        for (size_t b = 0; b < batch; b++) {{
            for (size_t t = 0; t < seq_len; t++) {{
                for (size_t d = 0; d < embed_dim_; d++) {{
                    h->data[(b * seq_len + t) * embed_dim_ + d] += pos_emb->data[t * embed_dim_ + d];
                }}
            }}
        }}

        h = dropout->forward(h);

        // Create causal mask once
        auto mask = MultiHeadAttention::causal_mask(seq_len);

        // Pass through transformer blocks
        for (auto* block : blocks) {{
            h = block->forward_with_mask(h, mask);
        }}

        h = ln_final->forward(h);

        // Project to vocab: [batch * seq_len, embed_dim] -> [batch * seq_len, vocab_size]
        auto h_flat = h->reshape({{batch * seq_len, embed_dim_}});
        auto logits_flat = head->forward(h_flat);
        auto logits = logits_flat->reshape({{batch, seq_len, vocab_size_}});

        return logits;
    }}

    std::vector<TensorPtr> parameters() override {{
        std::vector<TensorPtr> params;
        for (auto& p : token_emb->parameters()) params.push_back(p);
        params.push_back(pos_emb);  // Learnable positional embeddings
        for (auto* block : blocks) {{
            for (auto& p : block->parameters()) params.push_back(p);
        }}
        for (auto& p : ln_final->parameters()) params.push_back(p);
        for (auto& p : head->parameters()) params.push_back(p);
        return params;
    }}

    void train() {{
        dropout->train();
        for (auto* block : blocks) block->train();
    }}

    void eval() {{
        dropout->eval();
        for (auto* block : blocks) block->eval();
    }}

    std::string name() const override {{ return "TransformerLM"; }}
}};

// Alias for compatibility
using LanguageModel = TransformerLM;

struct TextDataset {{
    TensorPtr inputs;   // [num_sequences, seq_length]
    TensorPtr targets;  // [num_sequences, seq_length]
    size_t num_sequences;
    size_t seq_length;
}};

TextDataset load_text_dataset(const std::string& data_dir, bool train) {{
    std::string prefix = train ? "train" : "test";
    auto inputs_file = load_tensor_file(data_dir + "/" + prefix + "_inputs.bin");
    auto targets_file = load_tensor_file(data_dir + "/" + prefix + "_targets.bin");

    TextDataset ds;
    ds.inputs = Tensor::create(inputs_file.shape, false);
    ds.inputs->data = inputs_file.data;
    ds.targets = Tensor::create(targets_file.shape, false);
    ds.targets->data = targets_file.data;
    ds.num_sequences = inputs_file.shape[0];
    ds.seq_length = inputs_file.shape[1];
    return ds;
}}

class TextDataLoader {{
public:
    TextDataLoader(const TextDataset& dataset, size_t batch_size, bool shuffle)
        : dataset(dataset), batch_size(batch_size), shuffle(shuffle), current_idx(0) {{
        indices.resize(dataset.num_sequences);
        for (size_t i = 0; i < dataset.num_sequences; i++) indices[i] = i;
        if (shuffle) reset();
    }}

    void reset() {{
        current_idx = 0;
        if (shuffle) {{
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
        }}
    }}

    bool has_next() const {{ return current_idx < dataset.num_sequences; }}

    size_t num_batches() const {{
        return (dataset.num_sequences + batch_size - 1) / batch_size;
    }}

    std::pair<TensorPtr, TensorPtr> next_batch() {{
        size_t actual_batch = std::min(batch_size, dataset.num_sequences - current_idx);
        size_t seq_len = dataset.seq_length;

        // Create batch tensors [batch_size, seq_length]
        auto batch_inputs = Tensor::create({{actual_batch, seq_len}}, false);
        auto batch_targets = Tensor::create({{actual_batch, seq_len}}, false);

        for (size_t i = 0; i < actual_batch; i++) {{
            size_t idx = indices[current_idx + i];
            std::copy(
                dataset.inputs->data.begin() + idx * seq_len,
                dataset.inputs->data.begin() + (idx + 1) * seq_len,
                batch_inputs->data.begin() + i * seq_len
            );
            std::copy(
                dataset.targets->data.begin() + idx * seq_len,
                dataset.targets->data.begin() + (idx + 1) * seq_len,
                batch_targets->data.begin() + i * seq_len
            );
        }}

        current_idx += actual_batch;
        return {{batch_inputs, batch_targets}};
    }}

private:
    const TextDataset& dataset;
    size_t batch_size;
    bool shuffle;
    size_t current_idx;
    std::vector<size_t> indices;
}};

// Compute perplexity = exp(avg cross entropy loss)
float compute_perplexity(LanguageModel& model, TextDataLoader& loader, size_t vocab_size) {{
    NoGradGuard no_grad;
    model.eval();

    float total_loss = 0.0f;
    size_t total_tokens = 0;
    loader.reset();

    while (loader.has_next()) {{
        auto [inputs, targets] = loader.next_batch();
        auto output = model.forward(inputs);  // [batch, seq_len, vocab_size]

        size_t batch_size = inputs->shape[0];
        size_t seq_len = inputs->shape[1];

        // Compute cross-entropy loss for each position
        for (size_t b = 0; b < batch_size; b++) {{
            for (size_t t = 0; t < seq_len; t++) {{
                size_t target_idx = static_cast<size_t>(targets->data[b * seq_len + t]);
                if (target_idx >= vocab_size) continue;  // Skip padding

                // Softmax + log
                float max_val = output->data[(b * seq_len + t) * vocab_size];
                for (size_t v = 1; v < vocab_size; v++) {{
                    max_val = std::max(max_val, output->data[(b * seq_len + t) * vocab_size + v]);
                }}

                float sum_exp = 0.0f;
                for (size_t v = 0; v < vocab_size; v++) {{
                    sum_exp += std::exp(output->data[(b * seq_len + t) * vocab_size + v] - max_val);
                }}

                float log_prob = output->data[(b * seq_len + t) * vocab_size + target_idx] - max_val - std::log(sum_exp);
                total_loss -= log_prob;
                total_tokens++;
            }}
        }}
    }}

    model.train();
    float avg_loss = total_loss / total_tokens;
    return std::exp(avg_loss);
}}

int main(int argc, char* argv[]) {{
    if (argc < 3) {{
        printf("Usage: %s <data_dir> <output_model> [resume_weights] [start_epoch]\\n", argv[0]);
        return 1;
    }}

    std::string data_dir = argv[1];
    std::string output_path = argv[2];
    std::string resume_weights = (argc > 3) ? argv[3] : "";
    int start_epoch = (argc > 4) ? std::atoi(argv[4]) : 0;

    // Disable stdout buffering for real-time output
    setbuf(stdout, NULL);

    printf("Text Model Training (Language Model)\\n");
    printf("=====================================\\n\\n");

    printf("Loading dataset from '%s'...\\n", data_dir.c_str());
    auto train_data = load_text_dataset(data_dir, true);
    auto test_data = load_text_dataset(data_dir, false);

    printf("Train sequences: %zu\\n", train_data.num_sequences);
    printf("Test sequences: %zu\\n", test_data.num_sequences);
    printf("Sequence length: %zu\\n", train_data.seq_length);
    printf("Vocabulary size: {vocab_size}\\n");
    printf("Architecture: Transformer ({num_layers} layers, {num_heads} heads, dim={embed_dim})\\n\\n");

    // Build Transformer language model
    TransformerLM model({vocab_size}, {embed_dim}, {num_heads}, {num_layers}, {ff_dim}, train_data.seq_length, {dropout_rate}f);

    // Load weights if resuming
    if (!resume_weights.empty()) {{
        printf("Resuming from weights: %s (epoch %d)\\n", resume_weights.c_str(), start_epoch);
        load_model(&model, resume_weights);
    }}

    CrossEntropyLoss criterion;
    {optimizer_code}
    {scheduler_include}

    TextDataLoader train_loader(train_data, {batch_size}, true);
    TextDataLoader test_loader(test_data, {batch_size}, false);

    // Count parameters
    size_t total_params = 0;
    for (const auto& p : model.parameters()) total_params += p->size();
    printf("Total parameters: %zu\\n\\n", total_params);

    printf("Training for {epochs} epochs (starting from %d)...\\n", start_epoch);
    printf("------------------------------------------------------------------\\n");

    float best_ppl = 1e9f;

    for (int epoch = start_epoch; epoch < {epochs}; epoch++) {{
        model.train();
        train_loader.reset();
        float total_loss = 0.0f;
        size_t num_batches = 0;
        size_t total_tokens = 0;

        while (train_loader.has_next()) {{
            auto [inputs, targets] = train_loader.next_batch();

            optimizer.zero_grad();
            auto output = model.forward(inputs);

            // Reshape for cross-entropy: [batch * seq_len, vocab_size]
            size_t batch_size = inputs->shape[0];
            size_t seq_len = inputs->shape[1];
            size_t vocab_size = {vocab_size};

            // Use reshape to maintain gradient graph
            auto output_flat = output->reshape({{batch_size * seq_len, vocab_size}});

            auto targets_flat = targets->reshape({{batch_size * seq_len}});

            auto loss = criterion(output_flat, targets_flat);
            loss->backward();
            optimizer.step();

            total_loss += loss->data[0] * batch_size * seq_len;
            total_tokens += batch_size * seq_len;
            num_batches++;
        }}

        {scheduler_step}

        float avg_loss = total_loss / total_tokens;
        float train_ppl = std::exp(avg_loss);
        float test_ppl = compute_perplexity(model, test_loader, {vocab_size});
        best_ppl = std::min(best_ppl, test_ppl);

        // Output accuracy as inverse perplexity percentage for compatibility
        float acc = 100.0f / test_ppl;

        printf("Epoch %2d | Loss: %.4f | Test Acc: %.2f%% | PPL: %.2f\\n",
               epoch + 1, avg_loss, acc, test_ppl);

        // Save checkpoint after each epoch
        save_model(&model, output_path);
    }}

    printf("------------------------------------------------------------------\\n");
    printf("Training complete! Best perplexity: %.2f\\n\\n", best_ppl);

    printf("Saving model to '%s'...\\n", output_path.c_str());
    save_model(&model, output_path);

    return 0;
}}
'''
    return code
