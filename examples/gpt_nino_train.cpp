/**
 * GPT-Nino Training: Train a character-level GPT on Nino Nakano dialogue.
 *
 * Prerequisite:
 *   1. Create data/nino/raw_dialogue.jsonl with dialogue data
 *   2. Run: python examples/preprocess_nino.py
 *
 * Build: make nino-train
 * Run:   ./build/gpt_nino_train [--resume]
 */

#include "nino_gpt.h"
#include "optimizer.h"
#include <iostream>
#include <cstring>

// Training hyperparameters
static constexpr size_t BATCH_SIZE   = 32;
static constexpr float  LR           = 3e-4f;
static constexpr size_t TOTAL_STEPS  = 10000;
static constexpr size_t WARMUP_STEPS = 200;
static constexpr size_t LOG_EVERY    = 100;
static constexpr size_t SAVE_EVERY   = 1000;
static constexpr size_t SAMPLE_EVERY = 500;

static const std::string MODEL_PATH = "data/nino/nino_gpt.wm";


int main(int argc, char* argv[]) {
    bool resume = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--resume") == 0) resume = true;
    }

    std::cout << "=== GPT-Nino Training ===" << std::endl;
    std::cout << "Training a Nino Nakano chatbot with WhiteMatter" << std::endl;
    std::cout << std::endl;

    // ---- Backend info ----
    std::cout << "Training on CPU (matmuls accelerated via BLAS)" << std::endl;
    std::cout << std::endl;

    // ---- Load data ----
    auto train_data = load_binary("data/nino/train.bin");
    auto val_data   = load_binary("data/nino/val.bin");

    if (train_data.size() < MAX_SEQ_LEN + 2) {
        std::cerr << "ERROR: Training data too small (" << train_data.size() << " bytes)." << std::endl;
        std::cerr << "Run: python examples/preprocess_nino.py" << std::endl;
        return 1;
    }

    std::cout << "Train tokens: " << train_data.size() << std::endl;
    std::cout << "Val tokens:   " << val_data.size()   << std::endl;
    std::cout << std::endl;

    // ---- Create model ----
    NinoGPT model;
    std::cout << "Model parameters: " << model.count_params() << std::endl;
    std::cout << "Architecture: " << NUM_LAYERS << " layers, "
              << EMBED_DIM << " dim, " << NUM_HEADS << " heads, "
              << MAX_SEQ_LEN << " seq_len" << std::endl;
    std::cout << std::endl;

    // ---- Resume from checkpoint ----
    if (resume) {
        std::cout << "Resuming from " << MODEL_PATH << "..." << std::endl;
        if (load_model(&model, MODEL_PATH)) {
            std::cout << "Checkpoint loaded successfully." << std::endl;
        } else {
            std::cerr << "WARNING: Could not load checkpoint, training from scratch." << std::endl;
        }
        std::cout << std::endl;
    }

    // ---- Optimizer & scheduler ----
    auto params = model.parameters();
    Adam optimizer(params, LR);
    LinearWarmupCosineDecay scheduler(&optimizer, WARMUP_STEPS, TOTAL_STEPS);

    // ---- Training ----
    std::mt19937 rng(42);
    size_t train_len = train_data.size();
    size_t val_len   = val_data.size();

    auto sample_batch = [&](const std::vector<uint8_t>& data, size_t data_len) {
        auto inputs  = Tensor::create({BATCH_SIZE, MAX_SEQ_LEN}, true);
        auto targets = Tensor::create({BATCH_SIZE, MAX_SEQ_LEN}, false);

        std::uniform_int_distribution<size_t> dist(0, data_len - MAX_SEQ_LEN - 2);
        for (size_t b = 0; b < BATCH_SIZE; b++) {
            size_t pos = dist(rng);
            for (size_t s = 0; s < MAX_SEQ_LEN; s++) {
                inputs->data()[b * MAX_SEQ_LEN + s]  = static_cast<float>(data[pos + s]);
                targets->data()[b * MAX_SEQ_LEN + s] = static_cast<float>(data[pos + s + 1]);
            }
        }
        return std::make_pair(inputs, targets);
    };

    std::cout << "Training for " << TOTAL_STEPS << " steps (batch_size="
              << BATCH_SIZE << ", seq_len=" << MAX_SEQ_LEN << ")" << std::endl;
    std::cout << "Optimizer: Adam (lr=" << LR << "), LinearWarmupCosineDecay"
              << " (warmup=" << WARMUP_STEPS << ")" << std::endl;
    std::cout << "Checkpoints saved to: " << MODEL_PATH << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    auto t_start = std::chrono::steady_clock::now();

    for (size_t step = 1; step <= TOTAL_STEPS; step++) {
        // -- Train step --
        optimizer.zero_grad();
        auto [inp, tgt] = sample_batch(train_data, train_len);
        auto logits = model.forward(inp);
        auto loss   = lm_cross_entropy(logits, tgt);
        loss->backward();
        clip_grad_norm_(params, 1.0f);
        optimizer.step();
        scheduler.step();

        // -- Logging --
        bool should_log = (step <= 10) || (step % LOG_EVERY == 0) || (step == TOTAL_STEPS);
        if (should_log) {
            float val_loss = 0.0f;
            size_t val_batches = 5;
            {
                NoGradGuard no_grad;
                for (size_t vb = 0; vb < val_batches; vb++) {
                    auto [vi, vt] = sample_batch(val_data, val_len);
                    auto vlogits = model.forward(vi);
                    auto vloss   = lm_cross_entropy(vlogits, vt);
                    val_loss += vloss->data()[0];
                }
            }
            val_loss /= static_cast<float>(val_batches);

            auto t_now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();

            printf("step %5zu/%zu | train_loss %.4f | val_loss %.4f | lr %.2e | %.1fs\n",
                   step, TOTAL_STEPS, loss->data()[0], val_loss,
                   optimizer.lr, elapsed);
            fflush(stdout);
        }

        // -- Generate sample --
        if (step % SAMPLE_EVERY == 0 || step == TOTAL_STEPS) {
            std::string prompt = SYSTEM_TAG +
                "You are Nino Nakano. You are tsundere and love cooking." +
                USER_TAG + "Hi Nino!" + NINO_TAG;
            std::string sample = generate(model, prompt, 200, 0.8f);
            std::cout << "\n--- Sample (step " << step << ") ---" << std::endl;
            std::cout << "Nino: " << sample << std::endl;
            std::cout << "---\n" << std::endl;
        }

        // -- Checkpoint --
        if (step % SAVE_EVERY == 0 || step == TOTAL_STEPS) {
            save_model(&model, MODEL_PATH);
            std::cout << "[Checkpoint saved: " << MODEL_PATH << "]" << std::endl;
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();
    printf("\nTraining complete in %.1f seconds.\n", total_time);
    printf("Model saved to: %s\n", MODEL_PATH.c_str());

    return 0;
}
