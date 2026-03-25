import re


def generate_text_inference_code(
    layers_code: str,
    dataset_config: dict
) -> str:
    vocab_size = dataset_config.get("vocab_size", dataset_config.get("num_classes", 100))
    seq_length = dataset_config.get("seq_length", 128)

    # Extract layer parameters (same as training)
    embed_dim = 128
    num_heads = 4
    num_layers = 4
    ff_dim = 256
    dropout_rate = 0.1

    if "embedding_dim" in layers_code:
        match = re.search(r'embedding_dim.*?(\d+)', layers_code)
        if match:
            embed_dim = int(match.group(1))
    if "hidden_size" in layers_code:
        match = re.search(r'hidden_size.*?(\d+)', layers_code)
        if match:
            embed_dim = int(match.group(1))
            ff_dim = embed_dim * 2
    if "num_heads" in layers_code:
        match = re.search(r'num_heads.*?(\d+)', layers_code)
        if match:
            num_heads = int(match.group(1))
    if "num_layers" in layers_code:
        match = re.search(r'num_layers.*?(\d+)', layers_code)
        if match:
            num_layers = int(match.group(1))

    if embed_dim % num_heads != 0:
        num_heads = 4
        while embed_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1

    code = f'''// Auto-generated Transformer text generation inference code
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <sstream>
#include "tensor.h"
#include "layer.h"
#include "serialize.h"

// Transformer Block (must match training architecture)
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
        delete attn; delete ln1; delete ln2; delete ff1; delete ff2; delete dropout1; delete dropout2;
    }}

    TensorPtr forward_with_mask(const TensorPtr& x, const TensorPtr& mask) {{
        auto attn_out = attn->forward(x, x, x, mask);
        attn_out = dropout1->forward(attn_out);
        auto h = ln1->forward(x + attn_out);

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
        auto mask = MultiHeadAttention::causal_mask(x->shape[1]);
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

    void train() {{ dropout1->train(); dropout2->train(); }}
    void eval() {{ dropout1->eval(); dropout2->eval(); }}
    std::string name() const override {{ return "TransformerBlock"; }}
}};

// Transformer Language Model
class TransformerLM : public Module {{
public:
    Embedding* token_emb;
    TensorPtr pos_emb;
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
        pos_emb = Tensor::randn({{max_seq_len, embed_dim}}, true);
        float scale = 0.02f;
        for (auto& v : pos_emb->data) v *= scale;

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

    TensorPtr forward(const TensorPtr& x) override {{
        size_t batch = x->shape[0];
        size_t seq_len = x->shape[1];

        auto h = token_emb->forward(x);

        for (size_t b = 0; b < batch; b++) {{
            for (size_t t = 0; t < seq_len; t++) {{
                for (size_t d = 0; d < embed_dim_; d++) {{
                    h->data[(b * seq_len + t) * embed_dim_ + d] += pos_emb->data[t * embed_dim_ + d];
                }}
            }}
        }}

        h = dropout->forward(h);
        auto mask = MultiHeadAttention::causal_mask(seq_len);

        for (auto* block : blocks) {{
            h = block->forward_with_mask(h, mask);
        }}

        h = ln_final->forward(h);
        auto h_flat = h->reshape({{batch * seq_len, embed_dim_}});
        auto logits_flat = head->forward(h_flat);
        return logits_flat->reshape({{batch, seq_len, vocab_size_}});
    }}

    std::vector<TensorPtr> parameters() override {{
        std::vector<TensorPtr> params;
        for (auto& p : token_emb->parameters()) params.push_back(p);
        params.push_back(pos_emb);
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

using LanguageModel = TransformerLM;

// Load vocabulary from JSON
#include <map>
std::map<int, std::string> load_vocabulary(const std::string& path) {{
    std::map<int, std::string> idx_to_token;
    std::ifstream f(path);
    if (!f) return idx_to_token;

    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // Simple JSON parsing for idx_to_token
    size_t pos = content.find("\\"idx_to_token\\"");
    if (pos == std::string::npos) return idx_to_token;

    pos = content.find("{{", pos);
    size_t end = content.find("}}", pos);
    std::string inner = content.substr(pos + 1, end - pos - 1);

    // Parse "idx": "token" pairs
    size_t i = 0;
    while (i < inner.length()) {{
        size_t key_start = inner.find("\\"", i);
        if (key_start == std::string::npos) break;
        size_t key_end = inner.find("\\"", key_start + 1);
        std::string key = inner.substr(key_start + 1, key_end - key_start - 1);

        size_t val_start = inner.find("\\"", key_end + 1);
        if (val_start == std::string::npos) break;
        size_t val_end = inner.find("\\"", val_start + 1);
        std::string val = inner.substr(val_start + 1, val_end - val_start - 1);

        // Handle escape sequences
        std::string decoded;
        for (size_t j = 0; j < val.length(); j++) {{
            if (val[j] == '\\\\' && j + 1 < val.length()) {{
                if (val[j+1] == 'n') {{ decoded += '\\n'; j++; }}
                else if (val[j+1] == 't') {{ decoded += '\\t'; j++; }}
                else decoded += val[j];
            }} else decoded += val[j];
        }}

        idx_to_token[std::stoi(key)] = decoded;
        i = val_end + 1;
    }}

    return idx_to_token;
}}

std::map<std::string, int> load_token_to_idx(const std::string& path) {{
    auto idx_to_token = load_vocabulary(path);
    std::map<std::string, int> token_to_idx;
    for (auto& [idx, token] : idx_to_token) {{
        token_to_idx[token] = idx;
    }}
    return token_to_idx;
}}

int sample_token(const std::vector<float>& logits, float temperature, std::mt19937& rng) {{
    std::vector<float> probs(logits.size());
    float max_val = *std::max_element(logits.begin(), logits.end());

    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {{
        probs[i] = std::exp((logits[i] - max_val) / temperature);
        sum += probs[i];
    }}
    for (size_t i = 0; i < logits.size(); i++) {{
        probs[i] /= sum;
    }}

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}}

int main(int argc, char* argv[]) {{
    if (argc < 4) {{
        fprintf(stderr, "Usage: %s <model.bin> <vocab.json> <prompt> [max_tokens] [temperature]\\n", argv[0]);
        return 1;
    }}

    std::string model_path = argv[1];
    std::string vocab_path = argv[2];
    std::string prompt = argv[3];
    int max_tokens = argc > 4 ? std::stoi(argv[4]) : 100;
    float temperature = argc > 5 ? std::stof(argv[5]) : 0.8f;

    try {{
        // Load vocabulary
        auto idx_to_token = load_vocabulary(vocab_path);
        auto token_to_idx = load_token_to_idx(vocab_path);

        if (idx_to_token.empty()) {{
            fprintf(stderr, "Failed to load vocabulary\\n");
            return 1;
        }}

        // Build Transformer language model (must match training)
        TransformerLM model({vocab_size}, {embed_dim}, {num_heads}, {num_layers}, {ff_dim}, {seq_length}, {dropout_rate}f);

        if (!load_model(&model, model_path)) {{
            fprintf(stderr, "Failed to load model\\n");
            return 1;
        }}

        NoGradGuard no_grad;
        model.eval();

        // Encode prompt (character-level)
        std::vector<int> tokens;
        for (char c : prompt) {{
            std::string s(1, c);
            if (token_to_idx.count(s)) {{
                tokens.push_back(token_to_idx[s]);
            }} else {{
                tokens.push_back(1); // <unk>
            }}
        }}

        std::random_device rd;
        std::mt19937 rng(rd());

        // Generate tokens
        std::string output = prompt;
        size_t seq_length = {seq_length};
        size_t vocab_size = {vocab_size};

        for (int i = 0; i < max_tokens; i++) {{
            // Prepare input: last seq_length tokens
            std::vector<float> input_data(seq_length, 0);  // padding
            size_t start = tokens.size() > seq_length ? tokens.size() - seq_length : 0;
            size_t len = std::min(tokens.size(), seq_length);
            for (size_t j = 0; j < len; j++) {{
                input_data[seq_length - len + j] = static_cast<float>(tokens[start + j]);
            }}

            auto input = Tensor::create({{1, seq_length}}, false);
            input->data = input_data;

            // Forward pass
            auto logits = model.forward(input);  // [1, seq_length, vocab_size]

            // Get logits for last position
            std::vector<float> last_logits(vocab_size);
            for (size_t v = 0; v < vocab_size; v++) {{
                last_logits[v] = logits->data[(seq_length - 1) * vocab_size + v];
            }}

            // Sample next token
            int next_token = sample_token(last_logits, temperature, rng);
            tokens.push_back(next_token);

            // Decode and output
            if (idx_to_token.count(next_token)) {{
                output += idx_to_token[next_token];
            }}

            // Stop on EOS
            if (next_token == 2) break;  // <eos>
        }}

        printf("%s\\n", output.c_str());
        return 0;

    }} catch (const std::exception& e) {{
        fprintf(stderr, "Error: %s\\n", e.what());
        return 1;
    }}
}}
'''
    return code
