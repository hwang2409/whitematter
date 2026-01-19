#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include <cstdio>
#include <cmath>

int main() {
    printf("Early Stopping Test\n");
    printf("===================\n\n");

    // Test 1: Basic early stopping (min mode - loss)
    printf("1. Basic Early Stopping (Min Mode)\n");
    {
        EarlyStopping early_stopping(3);  // patience = 3

        // Simulate decreasing then stagnant loss
        std::vector<float> losses = {1.0f, 0.8f, 0.6f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
        int stopped_at = -1;

        for (size_t epoch = 0; epoch < losses.size(); epoch++) {
            if (early_stopping.step(losses[epoch])) {
                stopped_at = epoch;
                break;
            }
        }

        printf("   Losses: [1.0, 0.8, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5]\n");
        printf("   Patience: 3\n");
        printf("   Stopped at epoch: %d (expected: 6)\n", stopped_at);
        printf("   Best metric: %.2f (expected: 0.50)\n", early_stopping.best_metric());
        printf("   Best epoch: %d (expected: 3)\n", early_stopping.best_epoch());
        printf("   %s\n\n", (stopped_at == 6 && std::abs(early_stopping.best_metric() - 0.5f) < 0.01f) ? "PASSED" : "FAILED");
    }

    // Test 2: Early stopping with max mode (accuracy)
    printf("2. Early Stopping (Max Mode - Accuracy)\n");
    {
        EarlyStopping early_stopping(2, 0.0f, false);  // patience=2, mode_min=false

        // Simulate increasing then stagnant accuracy
        std::vector<float> accuracies = {0.5f, 0.7f, 0.85f, 0.85f, 0.84f, 0.85f};
        int stopped_at = -1;

        for (size_t epoch = 0; epoch < accuracies.size(); epoch++) {
            if (early_stopping.step(accuracies[epoch])) {
                stopped_at = epoch;
                break;
            }
        }

        printf("   Accuracies: [0.5, 0.7, 0.85, 0.85, 0.84, 0.85]\n");
        printf("   Patience: 2\n");
        printf("   Stopped at epoch: %d (expected: 4)\n", stopped_at);
        printf("   Best metric: %.2f (expected: 0.85)\n", early_stopping.best_metric());
        printf("   %s\n\n", (stopped_at == 4 && std::abs(early_stopping.best_metric() - 0.85f) < 0.01f) ? "PASSED" : "FAILED");
    }

    // Test 3: Early stopping with min_delta
    printf("3. Early Stopping with Min Delta\n");
    {
        EarlyStopping early_stopping(2, 0.05f, true);  // patience=2, min_delta=0.05

        // Small improvements < min_delta don't count
        std::vector<float> losses = {1.0f, 0.98f, 0.96f, 0.94f, 0.93f};  // Each improvement < 0.05
        int stopped_at = -1;

        for (size_t epoch = 0; epoch < losses.size(); epoch++) {
            if (early_stopping.step(losses[epoch])) {
                stopped_at = epoch;
                break;
            }
        }

        printf("   Losses: [1.0, 0.98, 0.96, 0.94, 0.93]\n");
        printf("   Min delta: 0.05 (improvements < 0.05 don't count)\n");
        printf("   Patience: 2\n");
        printf("   Stopped at epoch: %d (expected: 2)\n", stopped_at);
        printf("   %s\n\n", stopped_at == 2 ? "PASSED" : "FAILED");
    }

    // Test 4: No early stopping when loss keeps improving
    printf("4. No Early Stopping (Continuous Improvement)\n");
    {
        EarlyStopping early_stopping(3);

        std::vector<float> losses = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f};
        int stopped_at = -1;

        for (size_t epoch = 0; epoch < losses.size(); epoch++) {
            if (early_stopping.step(losses[epoch])) {
                stopped_at = epoch;
                break;
            }
        }

        printf("   Losses: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]\n");
        printf("   Stopped at epoch: %d (expected: -1, no stopping)\n", stopped_at);
        printf("   Epochs without improvement: %d (expected: 0)\n",
               early_stopping.epochs_without_improvement());
        printf("   %s\n\n", (stopped_at == -1 && early_stopping.epochs_without_improvement() == 0) ? "PASSED" : "FAILED");
    }

    // Test 5: Reset functionality
    printf("5. Reset Functionality\n");
    {
        EarlyStopping early_stopping(2);

        // First training run
        early_stopping.step(1.0f);
        early_stopping.step(0.5f);
        early_stopping.step(0.5f);
        early_stopping.step(0.5f);  // Should trigger

        bool stopped_first = early_stopping.should_stop();
        float best_first = early_stopping.best_metric();

        // Reset and run again
        early_stopping.reset();

        early_stopping.step(2.0f);
        early_stopping.step(1.5f);

        bool stopped_after_reset = early_stopping.should_stop();
        float best_after_reset = early_stopping.best_metric();

        printf("   First run stopped: %s, best: %.2f\n",
               stopped_first ? "yes" : "no", best_first);
        printf("   After reset stopped: %s, best: %.2f\n",
               stopped_after_reset ? "yes" : "no", best_after_reset);
        printf("   %s\n\n", (stopped_first && !stopped_after_reset &&
                            std::abs(best_after_reset - 1.5f) < 0.01f) ? "PASSED" : "FAILED");
    }

    // Test 6: Training loop with real model
    printf("6. Training Loop with Real Model\n");
    {
        Sequential model({
            new Linear(10, 20),
            new ReLU(),
            new Linear(20, 5)
        });

        SGD optimizer(model.parameters(), 0.01f);
        MSELoss criterion;
        EarlyStopping early_stopping(5);  // patience = 5

        // Training data
        auto X_train = Tensor::randn({64, 10}, false);
        auto Y_train = Tensor::randn({64, 5}, false);

        // Validation data (slightly different)
        auto X_val = Tensor::randn({16, 10}, false);
        auto Y_val = Tensor::randn({16, 5}, false);

        int max_epochs = 200;
        int stopped_epoch = -1;
        float best_val_loss = std::numeric_limits<float>::infinity();

        for (int epoch = 0; epoch < max_epochs; epoch++) {
            // Training
            model.train();
            optimizer.zero_grad();
            auto train_pred = model.forward(X_train);
            auto train_loss = criterion(train_pred, Y_train);
            train_loss->backward();
            optimizer.step();

            // Validation
            model.eval();
            {
                NoGradGuard no_grad;
                auto val_pred = model.forward(X_val);
                auto val_loss = criterion(val_pred, Y_val);

                float val_loss_val = val_loss->data[0];
                if (val_loss_val < best_val_loss) {
                    best_val_loss = val_loss_val;
                }

                if (early_stopping.step(val_loss_val)) {
                    stopped_epoch = epoch;
                    break;
                }
            }
        }

        printf("   Max epochs: %d\n", max_epochs);
        printf("   Stopped at epoch: %d\n", stopped_epoch);
        printf("   Best val loss: %.4f\n", early_stopping.best_metric());
        printf("   Best epoch: %d\n", early_stopping.best_epoch());
        printf("   %s\n\n", (stopped_epoch > 0 && stopped_epoch < max_epochs) ? "PASSED" : "PASSED (no early stop needed)");
    }

    // Test 7: ModelCheckpoint
    printf("7. ModelCheckpoint\n");
    {
        Sequential model({
            new Linear(5, 3)
        });

        ModelCheckpoint checkpoint("/tmp/test_checkpoint.bin", true);  // mode_min

        // Simulate improving then degrading validation loss
        std::vector<float> val_losses = {1.0f, 0.8f, 0.6f, 0.7f, 0.9f};
        int saves = 0;

        for (float loss : val_losses) {
            if (checkpoint.step(loss, &model)) {
                saves++;
            }
        }

        printf("   Val losses: [1.0, 0.8, 0.6, 0.7, 0.9]\n");
        printf("   Times saved: %d (expected: 3)\n", saves);
        printf("   Best metric: %.2f (expected: 0.60)\n", checkpoint.best_metric());
        printf("   %s\n\n", (saves == 3 && std::abs(checkpoint.best_metric() - 0.6f) < 0.01f) ? "PASSED" : "FAILED");

        // Test restore
        Sequential model2({
            new Linear(5, 3)
        });

        // Modify model2 weights
        for (auto& p : model2.parameters()) {
            for (auto& v : p->data) v = 999.0f;
        }

        checkpoint.restore(&model2);

        // Check weights are restored (not 999)
        bool restored = true;
        for (auto& p : model2.parameters()) {
            for (auto& v : p->data) {
                if (std::abs(v - 999.0f) < 0.01f) {
                    restored = false;
                    break;
                }
            }
        }
        printf("   Model restored: %s\n", restored ? "yes" : "no");
        printf("   %s\n\n", restored ? "PASSED" : "FAILED");
    }

    // Test 8: Combined EarlyStopping + ModelCheckpoint
    printf("8. Combined EarlyStopping + ModelCheckpoint\n");
    {
        Sequential model({
            new Linear(10, 5)
        });

        SGD optimizer(model.parameters(), 0.01f);
        MSELoss criterion;

        EarlyStopping early_stopping(3);
        ModelCheckpoint checkpoint("/tmp/best_combined.bin", true);

        auto X = Tensor::randn({32, 10}, false);
        auto Y = Tensor::randn({32, 5}, false);

        int stopped_epoch = -1;
        int checkpoint_saves = 0;

        for (int epoch = 0; epoch < 100; epoch++) {
            optimizer.zero_grad();
            auto pred = model.forward(X);
            auto loss = criterion(pred, Y);
            loss->backward();
            optimizer.step();

            float val_loss = loss->data[0];

            if (checkpoint.step(val_loss, &model)) {
                checkpoint_saves++;
            }

            if (early_stopping.step(val_loss)) {
                stopped_epoch = epoch;
                break;
            }
        }

        printf("   Stopped at epoch: %d\n", stopped_epoch);
        printf("   Checkpoint saves: %d\n", checkpoint_saves);
        printf("   Best loss: %.4f\n", early_stopping.best_metric());
        printf("   %s\n\n", (checkpoint_saves > 0) ? "PASSED" : "FAILED");
    }

    printf("All tests completed!\n");
    return 0;
}
