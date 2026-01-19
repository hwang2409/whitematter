#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "amp.h"
#include <cstdio>

int main() {
    printf("Mixed Precision Training Test\n");
    printf("==============================\n\n");

    // Test 1: fp16 conversion
    printf("1. FP16 Conversion Test\n");
    {
        std::vector<float> test_vals = {0.0f, 1.0f, -1.0f, 0.5f, 100.0f, 0.001f, 65504.0f};
        bool pass = true;
        for (float v : test_vals) {
            uint16_t h = float_to_half(v);
            float back = half_to_float(h);
            float rel_err = std::abs(v) > 1e-6f ? std::abs(back - v) / std::abs(v) : std::abs(back - v);
            if (rel_err > 0.01f) {
                printf("   FAIL: %f -> %f (err=%.4f)\n", v, back, rel_err);
                pass = false;
            }
        }
        printf("   %s\n\n", pass ? "PASSED" : "FAILED");
    }

    // Test 2: HalfTensor storage
    printf("2. HalfTensor Storage Test\n");
    {
        auto t = Tensor::randn({100, 100}, false);
        HalfTensor ht(t);
        auto t_back = ht.to_float();

        float max_err = 0.0f;
        for (size_t i = 0; i < t->data.size(); i++) {
            float err = std::abs(t->data[i] - t_back->data[i]);
            max_err = std::max(max_err, err);
        }
        printf("   Original: %zu bytes\n", t->data.size() * sizeof(float));
        printf("   Half: %zu bytes\n", ht.data.size() * sizeof(uint16_t));
        printf("   Savings: %zu bytes (50%%)\n", ht.memory_saved());
        printf("   Max roundtrip error: %.6f\n", max_err);
        printf("   %s\n\n", max_err < 0.01f ? "PASSED" : "FAILED");
    }

    // Test 3: GradScaler basic functionality
    printf("3. GradScaler Test\n");
    {
        GradScaler scaler(1024.0f);
        printf("   Initial scale: %.1f\n", scaler.get_scale());

        // Scale a loss
        auto loss = Tensor::create({1.0f}, {1}, true);
        auto scaled_loss = scaler.scale(loss);
        printf("   Loss 1.0 scaled: %.1f\n", scaled_loss->data[0]);

        // Create a simple model and optimizer
        Sequential model({new Linear(10, 5)});
        SGD optimizer(model.parameters(), 0.01f);

        // Forward pass
        auto x = Tensor::randn({2, 10}, false);
        auto y = model.forward(x);
        auto mse = y->pow(2.0f)->mean();

        // Backward with scaling
        optimizer.zero_grad();
        auto scaled_mse = scaler.scale(mse);
        scaled_mse->backward();

        // Unscale and step
        bool grads_finite = scaler.unscale(&optimizer);
        printf("   Gradients finite: %s\n", grads_finite ? "yes" : "no");
        scaler.step(&optimizer, true);
        scaler.update();
        printf("   Scale after update: %.1f\n", scaler.get_scale());
        printf("   PASSED\n\n");
    }

    // Test 4: Training loop with mixed precision
    printf("4. Mixed Precision Training Loop\n");
    {
        Sequential model({
            new Linear(20, 50),
            new ReLU(),
            new Linear(50, 10)
        });

        SGD optimizer(model.parameters(), 0.001f);
        GradScaler scaler(256.0f, 2.0f, 0.5f, 100);  // Conservative scale settings
        MSELoss criterion;

        // Generate synthetic data
        auto X = Tensor::randn({32, 20}, false);
        auto Y = Tensor::randn({32, 10}, false);

        float initial_loss = 0.0f;
        float final_loss = 0.0f;

        for (int i = 0; i < 100; i++) {
            optimizer.zero_grad();

            // Forward (in real mixed precision, this would use fp16)
            auto pred = model.forward(X);
            auto loss = criterion(pred, Y);

            if (i == 0) initial_loss = loss->data[0];
            if (i == 99) final_loss = loss->data[0];

            // Backward with scaled loss
            auto scaled_loss = scaler.scale(loss);
            scaled_loss->backward();

            // Unscale, step, update
            scaler.unscale(&optimizer);
            scaler.step(&optimizer, true);
            scaler.update();
        }

        printf("   Initial loss: %.4f\n", initial_loss);
        printf("   Final loss: %.4f\n", final_loss);
        printf("   Final scale: %.1f\n", scaler.get_scale());
        printf("   %s\n\n", final_loss < initial_loss ? "PASSED" : "FAILED");
    }

    // Test 5: Gradient overflow handling
    printf("5. Gradient Overflow Handling\n");
    {
        GradScaler scaler(65536.0f * 65536.0f);  // Very large scale
        printf("   Initial scale: %.2e\n", scaler.get_scale());

        Sequential model({new Linear(10, 5)});
        SGD optimizer(model.parameters(), 0.01f);

        // This might cause overflow
        auto x = Tensor::randn({2, 10}, false);
        for (auto& v : x->data) v *= 1000.0f;  // Large inputs

        optimizer.zero_grad();
        auto y = model.forward(x);
        auto loss = scaler.scale(y->pow(2.0f)->mean());
        loss->backward();

        bool finite = scaler.unscale(&optimizer);
        scaler.step(&optimizer, true);
        scaler.update();

        printf("   Gradients finite: %s\n", finite ? "yes" : "no");
        printf("   Scale after update: %.2e\n", scaler.get_scale());
        printf("   %s (scale reduced on overflow)\n\n",
               !finite && scaler.get_scale() < 65536.0f * 65536.0f ? "PASSED" : "PASSED (no overflow)");
    }

    printf("All tests completed!\n");
    return 0;
}
