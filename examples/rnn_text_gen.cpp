// RNN Text Generation Example
// Character-level language model using LSTM for text generation

#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "logging.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>

// =============================================================================
// Sample text corpus (Shakespeare-inspired)
// =============================================================================
const std::string CORPUS = R"(
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wished. To die, to sleep;
To sleep, perchance to dream: ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life.

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts.

Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that lour'd upon our house
In the deep bosom of the ocean buried.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones.

But soft, what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief.

O Romeo, Romeo, wherefore art thou Romeo?
Deny thy father and refuse thy name.
Or if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.

If music be the food of love, play on,
Give me excess of it, that surfeiting,
The appetite may sicken, and so die.

We are such stuff as dreams are made on,
And our little life is rounded with a sleep.

The quality of mercy is not strained;
It droppeth as the gentle rain from heaven
Upon the place beneath. It is twice blest;
It blesseth him that gives and him that takes.

All that glitters is not gold;
Often have you heard that told.
Many a man his life hath sold
But my outside to behold.

Love all, trust a few, do wrong to none.
Be not afraid of greatness. Some are born great,
Some achieve greatness, and others have greatness thrust upon them.

There is nothing either good or bad, but thinking makes it so.
To thine own self be true.
Brevity is the soul of wit.
Though she be but little, she is fierce.
)";

// =============================================================================
// Character Vocabulary
// =============================================================================
class Vocabulary {
public:
    std::unordered_map<char, int> char_to_idx;
    std::unordered_map<int, char> idx_to_char;
    size_t vocab_size;

    Vocabulary(const std::string& text) {
        // Build vocabulary from text
        std::vector<char> chars;
        for (char c : text) {
            if (char_to_idx.find(c) == char_to_idx.end()) {
                char_to_idx[c] = chars.size();
                idx_to_char[chars.size()] = c;
                chars.push_back(c);
            }
        }
        vocab_size = chars.size();
    }

    int encode(char c) const {
        auto it = char_to_idx.find(c);
        return (it != char_to_idx.end()) ? it->second : 0;
    }

    char decode(int idx) const {
        auto it = idx_to_char.find(idx);
        return (it != idx_to_char.end()) ? it->second : '?';
    }

    std::vector<int> encode_string(const std::string& s) const {
        std::vector<int> result;
        for (char c : s) {
            result.push_back(encode(c));
        }
        return result;
    }

    std::string decode_indices(const std::vector<int>& indices) const {
        std::string result;
        for (int idx : indices) {
            result += decode(idx);
        }
        return result;
    }
};

// =============================================================================
// Character-Level RNN Language Model
// =============================================================================
class CharRNN : public Module {
public:
    Embedding* embedding;
    LSTM* lstm1;
    LSTM* lstm2;
    Dropout* dropout;
    Linear* fc;

    size_t vocab_size;
    size_t embed_dim;
    size_t hidden_size;

    CharRNN(size_t vocab_size, size_t embed_dim = 128, size_t hidden_size = 256)
        : vocab_size(vocab_size), embed_dim(embed_dim), hidden_size(hidden_size) {

        embedding = new Embedding(vocab_size, embed_dim);
        lstm1 = new LSTM(embed_dim, hidden_size, true);
        lstm2 = new LSTM(hidden_size, hidden_size, true);
        dropout = new Dropout(0.2f);
        fc = new Linear(hidden_size, vocab_size);
    }

    ~CharRNN() {
        delete embedding;
        delete lstm1;
        delete lstm2;
        delete dropout;
        delete fc;
    }

    // Forward pass: input shape [batch, seq_len] of character indices
    // Returns logits of shape [batch, seq_len, vocab_size]
    TensorPtr forward(const TensorPtr& x) override {
        // x: [batch, seq_len]
        size_t batch = x->shape[0];
        size_t seq_len = x->shape[1];

        // Embedding: [batch, seq_len] -> [batch, seq_len, embed_dim]
        auto embedded = embedding->forward(x);

        // LSTM layers
        auto h1 = lstm1->forward(embedded);  // [batch, seq_len, hidden_size]
        h1 = dropout->forward(h1);
        auto h2 = lstm2->forward(h1);        // [batch, seq_len, hidden_size]
        h2 = dropout->forward(h2);

        // Output layer: apply fc to each time step
        // Reshape to [batch * seq_len, hidden_size]
        auto h_flat = h2->reshape({batch * seq_len, hidden_size});
        auto logits_flat = fc->forward(h_flat);  // [batch * seq_len, vocab_size]

        // Reshape back to [batch, seq_len, vocab_size]
        auto logits = logits_flat->reshape({batch, seq_len, vocab_size});

        return logits;
    }

    std::vector<TensorPtr> parameters() override {
        std::vector<TensorPtr> params;
        for (auto& p : embedding->parameters()) params.push_back(p);
        for (auto& p : lstm1->parameters()) params.push_back(p);
        for (auto& p : lstm2->parameters()) params.push_back(p);
        for (auto& p : fc->parameters()) params.push_back(p);
        return params;
    }

    void train_mode() {
        dropout->train();
    }

    void eval_mode() {
        dropout->eval();
    }

    std::string name() const override { return "CharRNN"; }
};

// =============================================================================
// Data Preparation
// =============================================================================
struct TextBatch {
    TensorPtr input;   // [batch, seq_len]
    TensorPtr target;  // [batch, seq_len]
};

class TextDataLoader {
public:
    std::vector<int> encoded_text;
    size_t seq_len;
    size_t batch_size;
    size_t current_idx;
    std::mt19937 rng;

    TextDataLoader(const std::vector<int>& text, size_t seq_len, size_t batch_size)
        : encoded_text(text), seq_len(seq_len), batch_size(batch_size), current_idx(0), rng(42) {}

    void reset() {
        current_idx = 0;
    }

    bool has_next() const {
        return current_idx + (seq_len + 1) * batch_size <= encoded_text.size();
    }

    TextBatch next_batch() {
        auto input = Tensor::create({batch_size, seq_len}, false);
        auto target = Tensor::create({batch_size, seq_len}, false);

        for (size_t b = 0; b < batch_size; b++) {
            size_t start = current_idx + b * seq_len;
            for (size_t t = 0; t < seq_len; t++) {
                input->data[b * seq_len + t] = static_cast<float>(encoded_text[start + t]);
                target->data[b * seq_len + t] = static_cast<float>(encoded_text[start + t + 1]);
            }
        }

        current_idx += batch_size * seq_len;
        return {input, target};
    }

    size_t num_batches() const {
        return (encoded_text.size() - 1) / (seq_len * batch_size);
    }
};

// =============================================================================
// Text Generation
// =============================================================================
static std::mt19937 sample_rng(123);

int sample_from_logits(const TensorPtr& logits, float temperature = 1.0f) {
    // logits: [vocab_size]
    size_t vocab_size = logits->data.size();

    // Apply temperature
    std::vector<float> scaled(vocab_size);
    float max_val = *std::max_element(logits->data.begin(), logits->data.end());

    for (size_t i = 0; i < vocab_size; i++) {
        scaled[i] = (logits->data[i] - max_val) / temperature;
    }

    // Softmax
    float sum_exp = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        scaled[i] = std::exp(scaled[i]);
        sum_exp += scaled[i];
    }
    for (size_t i = 0; i < vocab_size; i++) {
        scaled[i] /= sum_exp;
    }

    // Sample from distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(sample_rng);
    float cumsum = 0.0f;

    for (size_t i = 0; i < vocab_size; i++) {
        cumsum += scaled[i];
        if (r < cumsum) {
            return static_cast<int>(i);
        }
    }

    return static_cast<int>(vocab_size - 1);
}

std::string generate_text(CharRNN& model, const Vocabulary& vocab,
                          const std::string& seed, size_t length, float temperature = 0.8f) {
    model.eval_mode();

    std::string result = seed;
    std::vector<int> context = vocab.encode_string(seed);

    for (size_t i = 0; i < length; i++) {
        // Create input tensor [1, context_len]
        size_t ctx_len = context.size();
        auto input = Tensor::create({1, ctx_len}, false);
        for (size_t j = 0; j < ctx_len; j++) {
            input->data[j] = static_cast<float>(context[j]);
        }

        // Forward pass
        TensorPtr logits;
        {
            NoGradGuard no_grad;
            logits = model.forward(input);  // [1, ctx_len, vocab_size]
        }

        // Get logits for last position
        size_t vocab_size = vocab.vocab_size;
        auto last_logits = Tensor::create({vocab_size}, false);
        for (size_t v = 0; v < vocab_size; v++) {
            last_logits->data[v] = logits->data[(ctx_len - 1) * vocab_size + v];
        }

        // Sample next character
        int next_char = sample_from_logits(last_logits, temperature);
        result += vocab.decode(next_char);

        // Update context (sliding window, keep last 50 chars for efficiency)
        context.push_back(next_char);
        if (context.size() > 50) {
            context.erase(context.begin());
        }
    }

    model.train_mode();
    return result;
}

// =============================================================================
// Cross-Entropy Loss for Sequences
// =============================================================================
TensorPtr sequence_cross_entropy(const TensorPtr& logits, const TensorPtr& targets) {
    // logits: [batch, seq_len, vocab_size]
    // targets: [batch, seq_len]

    size_t batch = logits->shape[0];
    size_t seq_len = logits->shape[1];
    size_t vocab_size = logits->shape[2];

    bool track = logits->requires_grad && GradMode::is_enabled();
    auto loss = Tensor::create({1}, track);
    loss->data[0] = 0.0f;

    // Compute cross-entropy for each position
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            // Get target class
            int target_class = static_cast<int>(targets->data[b * seq_len + t]);

            // Compute log-softmax for this position
            float max_val = -1e30f;
            for (size_t v = 0; v < vocab_size; v++) {
                size_t idx = b * seq_len * vocab_size + t * vocab_size + v;
                max_val = std::max(max_val, logits->data[idx]);
            }

            float sum_exp = 0.0f;
            for (size_t v = 0; v < vocab_size; v++) {
                size_t idx = b * seq_len * vocab_size + t * vocab_size + v;
                sum_exp += std::exp(logits->data[idx] - max_val);
            }

            size_t target_idx = b * seq_len * vocab_size + t * vocab_size + target_class;
            float log_softmax = logits->data[target_idx] - max_val - std::log(sum_exp);

            loss->data[0] -= log_softmax;
        }
    }

    loss->data[0] /= static_cast<float>(batch * seq_len);

    if (track) {
        auto logits_ptr = logits;
        auto targets_ptr = targets;
        loss->parents = {logits_ptr};

        loss->grad_fn = [logits_ptr, targets_ptr, loss, batch, seq_len, vocab_size]() {
            float scale = loss->grad[0] / static_cast<float>(batch * seq_len);

            for (size_t b = 0; b < batch; b++) {
                for (size_t t = 0; t < seq_len; t++) {
                    int target_class = static_cast<int>(targets_ptr->data[b * seq_len + t]);

                    // Compute softmax for gradient
                    float max_val = -1e30f;
                    for (size_t v = 0; v < vocab_size; v++) {
                        size_t idx = b * seq_len * vocab_size + t * vocab_size + v;
                        max_val = std::max(max_val, logits_ptr->data[idx]);
                    }

                    float sum_exp = 0.0f;
                    std::vector<float> softmax(vocab_size);
                    for (size_t v = 0; v < vocab_size; v++) {
                        size_t idx = b * seq_len * vocab_size + t * vocab_size + v;
                        softmax[v] = std::exp(logits_ptr->data[idx] - max_val);
                        sum_exp += softmax[v];
                    }

                    // Gradient: softmax - one_hot(target)
                    for (size_t v = 0; v < vocab_size; v++) {
                        size_t idx = b * seq_len * vocab_size + t * vocab_size + v;
                        float grad = softmax[v] / sum_exp;
                        if (static_cast<int>(v) == target_class) {
                            grad -= 1.0f;
                        }
                        logits_ptr->grad[idx] += scale * grad;
                    }
                }
            }
        };
    }

    return loss;
}

// =============================================================================
// Count Parameters
// =============================================================================
size_t count_params(Module* model) {
    size_t total = 0;
    for (auto& p : model->parameters()) {
        total += p->data.size();
    }
    return total;
}

// =============================================================================
// Main Training Loop
// =============================================================================
int main() {
    printf("=======================================================\n");
    printf("RNN Text Generation - Character-Level LSTM\n");
    printf("=======================================================\n\n");

    // Hyperparameters
    const size_t embed_dim = 128;
    const size_t hidden_size = 256;
    const size_t seq_len = 50;
    const size_t batch_size = 32;
    const size_t num_epochs = 50;
    const float learning_rate = 0.002f;
    const float clip_norm = 5.0f;

    // Build vocabulary
    Vocabulary vocab(CORPUS);
    printf("Vocabulary size: %zu characters\n", vocab.vocab_size);

    // Encode corpus
    std::vector<int> encoded = vocab.encode_string(CORPUS);
    printf("Corpus length: %zu characters\n\n", encoded.size());

    printf("Configuration:\n");
    printf("  Embedding dim: %zu\n", embed_dim);
    printf("  Hidden size: %zu\n", hidden_size);
    printf("  Sequence length: %zu\n", seq_len);
    printf("  Batch size: %zu\n", batch_size);
    printf("  Epochs: %zu\n", num_epochs);
    printf("  Learning rate: %.4f\n", learning_rate);
    printf("  Gradient clipping: %.1f\n\n", clip_norm);

    // Create model
    CharRNN model(vocab.vocab_size, embed_dim, hidden_size);
    printf("Model Parameters: %zu\n\n", count_params(&model));

    // Optimizer
    Adam optimizer(model.parameters(), learning_rate);

    // Data loader
    TextDataLoader loader(encoded, seq_len, batch_size);
    printf("Training batches per epoch: %zu\n\n", loader.num_batches());

    // Training logger
    TrainingLogger logger("logs", "rnn_text_gen");
    logger.set_total_epochs(num_epochs);

    printf("Training...\n");
    printf("-------------------------------------------------------\n");

    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        logger.new_epoch();
        loader.reset();

        float epoch_loss = 0.0f;
        size_t num_batches = 0;

        model.train_mode();

        while (loader.has_next()) {
            auto [input, target] = loader.next_batch();

            optimizer.zero_grad();

            // Forward pass
            auto logits = model.forward(input);  // [batch, seq_len, vocab_size]

            // Compute loss
            auto loss = sequence_cross_entropy(logits, target);

            // Backward pass
            loss->backward();

            // Gradient clipping
            auto params = model.parameters();
            clip_grad_norm_(params, clip_norm);

            optimizer.step();

            epoch_loss += loss->item();
            num_batches++;

            logger.log_batch("loss", loss->item());
        }

        epoch_loss /= num_batches;
        float perplexity = std::exp(epoch_loss);

        logger.log("loss", epoch_loss);
        logger.log("perplexity", perplexity);
        logger.step();

        printf("Epoch %2zu/%zu  Loss: %.4f  Perplexity: %.2f\n",
               epoch + 1, num_epochs, epoch_loss, perplexity);

        // Generate sample every 10 epochs
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            printf("\n--- Sample (temperature=0.8) ---\n");
            std::string sample = generate_text(model, vocab, "To be", 150, 0.8f);
            printf("%s\n", sample.c_str());
            printf("--------------------------------\n\n");
        }
    }

    printf("-------------------------------------------------------\n\n");

    // Final text generation with different temperatures
    printf("=======================================================\n");
    printf("Text Generation Examples\n");
    printf("=======================================================\n\n");

    std::vector<std::string> prompts = {"To be", "The ", "And ", "Love ", "What "};
    std::vector<float> temperatures = {0.5f, 0.8f, 1.0f, 1.2f};

    for (const auto& prompt : prompts) {
        printf("Prompt: \"%s\"\n", prompt.c_str());
        printf("-------------------------------------------\n");

        for (float temp : temperatures) {
            std::string generated = generate_text(model, vocab, prompt, 100, temp);
            printf("  temp=%.1f: %s...\n", temp, generated.substr(0, 80).c_str());
        }
        printf("\n");
    }

    // Generate longer text
    printf("=======================================================\n");
    printf("Extended Generation (temperature=0.7)\n");
    printf("=======================================================\n\n");

    std::string long_text = generate_text(model, vocab, "To be, or not to be", 300, 0.7f);

    // Word wrap at 70 characters
    size_t line_len = 0;
    for (char c : long_text) {
        printf("%c", c);
        line_len++;
        if (c == '\n') {
            line_len = 0;
        } else if (line_len > 70 && c == ' ') {
            printf("\n");
            line_len = 0;
        }
    }
    printf("\n\n");

    printf("=======================================================\n");

    // Interactive-style prompts demo
    printf("Various Prompts Demo:\n");
    printf("=======================================================\n\n");

    std::vector<std::pair<std::string, std::string>> demo_prompts = {
        {"All the world", " (about the stage)"},
        {"Friends, Romans", " (about Caesar)"},
        {"But soft, what", " (Romeo and Juliet)"},
        {"If music be", " (about love)"},
        {"The quality of", " (about mercy)"}
    };

    for (const auto& [prompt, context] : demo_prompts) {
        printf("Prompt: \"%s\"%s\n", prompt.c_str(), context.c_str());
        std::string text = generate_text(model, vocab, prompt, 120, 0.7f);
        printf("  -> %s\n\n", text.c_str());
    }

    printf("=======================================================\n");
    printf("Training complete!\n");
    printf("=======================================================\n");

    // Save logs
    logger.save_csv();
    logger.save_json();

    return 0;
}
