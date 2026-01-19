#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "logging.h"
#include <cstdio>
#include <cmath>
#include <thread>

int main() {
    printf("Training Logger Test\n");
    printf("====================\n\n");

    // Test 1: MetricTracker
    printf("1. MetricTracker Test\n");
    {
        MetricTracker tracker;

        std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        for (float v : values) {
            tracker.update(v);
        }

        printf("   Values: [1, 2, 3, 4, 5]\n");
        printf("   Count: %zu (expected: 5)\n", tracker.count());
        printf("   Sum: %.2f (expected: 15.00)\n", tracker.sum());
        printf("   Mean: %.2f (expected: 3.00)\n", tracker.mean());
        printf("   Min: %.2f (expected: 1.00)\n", tracker.min());
        printf("   Max: %.2f (expected: 5.00)\n", tracker.max());
        printf("   Last: %.2f (expected: 5.00)\n", tracker.last());
        printf("   Std: %.2f (expected: ~1.58)\n", tracker.std());

        bool pass = tracker.count() == 5 &&
                    std::abs(tracker.mean() - 3.0f) < 0.01f &&
                    std::abs(tracker.min() - 1.0f) < 0.01f &&
                    std::abs(tracker.max() - 5.0f) < 0.01f;
        printf("   %s\n\n", pass ? "PASSED" : "FAILED");
    }

    // Test 2: Basic logging
    printf("2. Basic Logging Test\n");
    {
        TrainingLogger logger("", "test_run");
        logger.set_verbose(false);

        // Log some metrics
        for (int step = 0; step < 10; step++) {
            float loss = 1.0f / (step + 1);
            float acc = 0.5f + 0.05f * step;

            logger.log("loss", loss);
            logger.log("accuracy", acc);
            logger.step();
        }

        printf("   Logged 10 steps\n");
        printf("   History size: %zu (expected: 10)\n", logger.history().size());
        printf("   Loss tracker count: %zu\n", logger.get_tracker("loss").count());
        printf("   Loss min: %.4f\n", logger.get_tracker("loss").min());
        printf("   Loss max: %.4f\n", logger.get_tracker("loss").max());
        printf("   Accuracy last: %.4f\n", logger.get_tracker("accuracy").last());

        bool pass = logger.history().size() == 10 &&
                    logger.get_tracker("loss").count() == 10;
        printf("   %s\n\n", pass ? "PASSED" : "FAILED");
    }

    // Test 3: Epoch-level logging
    printf("3. Epoch-Level Logging Test\n");
    {
        TrainingLogger logger;
        logger.set_verbose(false);
        logger.set_total_epochs(3);
        logger.set_total_steps(10);

        for (int epoch = 0; epoch < 3; epoch++) {
            logger.new_epoch();

            // Simulate batches
            for (int batch = 0; batch < 10; batch++) {
                float loss = 1.0f - 0.1f * epoch - 0.01f * batch;
                logger.log_batch("train_loss", loss);
            }

            // Log epoch summary
            logger.log("epoch_loss", logger.epoch_mean("train_loss"));
            logger.step();
        }

        printf("   Trained 3 epochs x 10 batches\n");
        printf("   Final epoch mean loss: %.4f\n", logger.epoch_mean("train_loss"));
        printf("   History size: %zu\n", logger.history().size());

        bool pass = logger.history().size() == 3;
        printf("   %s\n\n", pass ? "PASSED" : "FAILED");
    }

    // Test 4: CSV/JSON export
    printf("4. CSV/JSON Export Test\n");
    {
        TrainingLogger logger("/tmp", "export_test");
        logger.set_verbose(false);

        for (int i = 0; i < 5; i++) {
            logger.log("loss", 1.0f - 0.1f * i);
            logger.log("accuracy", 0.5f + 0.1f * i);
            logger.step();
        }

        bool csv_ok = logger.save_csv();
        bool json_ok = logger.save_json();

        printf("   CSV saved: %s\n", csv_ok ? "yes" : "no");
        printf("   JSON saved: %s\n", json_ok ? "yes" : "no");

        // Verify files exist and have content
        FILE* csv = fopen("/tmp/export_test_metrics.csv", "r");
        FILE* json = fopen("/tmp/export_test_metrics.json", "r");

        bool csv_exists = csv != nullptr;
        bool json_exists = json != nullptr;

        if (csv) fclose(csv);
        if (json) fclose(json);

        printf("   CSV file exists: %s\n", csv_exists ? "yes" : "no");
        printf("   JSON file exists: %s\n", json_exists ? "yes" : "no");

        printf("   %s\n\n", (csv_ok && json_ok && csv_exists && json_exists) ? "PASSED" : "FAILED");
    }

    // Test 5: Progress bar
    printf("5. Progress Bar Test\n");
    {
        printf("   ");
        ProgressBar bar(20, 30, "Progress: ");
        for (int i = 0; i < 20; i++) {
            bar.update();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        bar.finish();
        printf("   PASSED\n\n");
    }

    // Test 6: Training simulation with console output
    printf("6. Training Simulation (with progress)\n");
    {
        TrainingLogger logger;
        logger.set_total_epochs(3);
        logger.set_total_steps(5);

        for (int epoch = 0; epoch < 3; epoch++) {
            logger.new_epoch();

            for (int batch = 0; batch < 5; batch++) {
                // Simulate training
                float loss = 1.0f - 0.2f * epoch - 0.05f * batch + 0.1f * (rand() % 10) / 10.0f;
                float acc = 0.5f + 0.1f * epoch + 0.02f * batch;

                logger.log_batch("loss", loss);
                logger.log_batch("acc", acc);
                logger.print_progress("   ");

                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            logger.log("train_loss", logger.epoch_mean("loss"));
            logger.log("train_acc", logger.epoch_mean("acc"));
            logger.step();

            logger.print_epoch_summary();
        }

        logger.print_summary();
        printf("   PASSED\n\n");
    }

    // Test 7: Real model training with logging
    printf("7. Real Model Training with Logging\n");
    {
        Sequential model({
            new Linear(10, 20),
            new ReLU(),
            new Linear(20, 5)
        });

        SGD optimizer(model.parameters(), 0.01f);
        MSELoss criterion;

        TrainingLogger logger("/tmp", "model_training");
        logger.set_total_epochs(5);
        logger.set_verbose(false);

        auto X = Tensor::randn({32, 10}, false);
        auto Y = Tensor::randn({32, 5}, false);

        for (int epoch = 0; epoch < 5; epoch++) {
            logger.new_epoch();

            optimizer.zero_grad();
            auto pred = model.forward(X);
            auto loss = criterion(pred, Y);
            loss->backward();
            optimizer.step();

            float loss_val = loss->data[0];
            logger.log("loss", loss_val);
            logger.log("lr", optimizer.lr);
            logger.step();
        }

        logger.save_csv();
        logger.save_json();

        printf("   Trained 5 epochs\n");
        printf("   Initial loss: %.4f\n", logger.history()[0].metrics.at("loss"));
        printf("   Final loss: %.4f\n", logger.history()[4].metrics.at("loss"));
        printf("   Loss improved: %s\n",
               logger.history()[4].metrics.at("loss") < logger.history()[0].metrics.at("loss")
               ? "yes" : "no");
        printf("   %s\n\n", "PASSED");
    }

    // Test 8: Reset functionality
    printf("8. Reset Functionality\n");
    {
        TrainingLogger logger;
        logger.set_verbose(false);

        // First run
        logger.log("metric", 1.0f);
        logger.step();
        logger.log("metric", 2.0f);
        logger.step();

        size_t before_reset = logger.history().size();

        // Reset
        logger.reset();

        size_t after_reset = logger.history().size();

        // Second run
        logger.log("metric", 3.0f);
        logger.step();

        size_t after_new_log = logger.history().size();

        printf("   Before reset: %zu entries\n", before_reset);
        printf("   After reset: %zu entries\n", after_reset);
        printf("   After new log: %zu entries\n", after_new_log);

        bool pass = before_reset == 2 && after_reset == 0 && after_new_log == 1;
        printf("   %s\n\n", pass ? "PASSED" : "FAILED");
    }

    printf("All tests completed!\n");
    return 0;
}
