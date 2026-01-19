#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "amp.h"
#include <cstdio>
#include <cmath>

int main() {
    printf("Gradient Accumulation Test\n");
    printf("==========================\n\n");

    // Test 1: Basic accumulation
    printf("1. Basic Gradient Accumulation\n");
    {
        GradientAccumulator accumulator(4);  // Accumulate 4 steps

        printf("   Accumulation steps: %d\n", accumulator.get_accumulation_steps());
        printf("   Scale factor: %.4f\n", accumulator.get_scale_factor());

        bool pass = true;
        pass &= (accumulator.get_accumulation_steps() == 4);
        pass &= (std::abs(accumulator.get_scale_factor() - 0.25f) < 1e-6f);
        pass &= (accumulator.current_step() == 0);
        pass &= (!accumulator.should_step());

        printf("   %s\n\n", pass ? "PASSED" : "FAILED");
    }

    // Test 2: Gradient equivalence - accumulation should produce same gradients as large batch
    // Note: For exact equivalence, we use sum() loss (not mean()) since mean() divides by
    // different batch sizes. In practice, use same-sized mini-batches with mean() loss.
    printf("2. Gradient Equivalence Test\n");
    {
        // Create two identical models
        auto w1 = Tensor::create({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);
        auto w2 = Tensor::create({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);

        // Create input data: 4 samples
        auto x1 = Tensor::create({1.0f, 0.0f}, {1, 2}, false);
        auto x2 = Tensor::create({0.0f, 1.0f}, {1, 2}, false);
        auto x3 = Tensor::create({1.0f, 1.0f}, {1, 2}, false);
        auto x4 = Tensor::create({0.5f, 0.5f}, {1, 2}, false);

        // Large batch: all 4 samples at once, using sum() for exact comparison
        auto x_large = Tensor::concat({x1, x2, x3, x4}, 0);  // [4, 2]
        auto y_large = x_large->matmul(w1);
        auto loss_large = y_large->pow(2.0f)->sum();
        loss_large->backward();

        // Small batches with accumulation: 4 batches of 1 sample each
        // With sum() loss, accumulator just needs to add gradients (scale=1)
        // since sum over 4 batches = sum over large batch
        std::vector<TensorPtr> inputs = {x1, x2, x3, x4};
        for (auto& x : inputs) {
            auto y = x->matmul(w2);
            auto loss = y->pow(2.0f)->sum();
            loss->backward();  // Gradients accumulate (no scaling needed for sum)
        }

        // Compare gradients
        float max_diff = 0.0f;
        for (size_t i = 0; i < w1->grad.size(); i++) {
            float diff = std::abs(w1->grad[i] - w2->grad[i]);
            max_diff = std::max(max_diff, diff);
        }

        printf("   Large batch gradients: [%.4f, %.4f, %.4f, %.4f]\n",
               w1->grad[0], w1->grad[1], w1->grad[2], w1->grad[3]);
        printf("   Accumulated gradients: [%.4f, %.4f, %.4f, %.4f]\n",
               w2->grad[0], w2->grad[1], w2->grad[2], w2->grad[3]);
        printf("   Max difference: %.6f\n", max_diff);
        printf("   %s\n\n", max_diff < 0.01f ? "PASSED" : "FAILED");
    }

    // Test 3: Training loop with accumulation
    printf("3. Training Loop with Accumulation\n");
    {
        Sequential model({
            new Linear(10, 20),
            new ReLU(),
            new Linear(20, 5)
        });

        SGD optimizer(model.parameters(), 0.01f);
        MSELoss criterion;
        GradientAccumulator accumulator(4);  // Effective batch size = 8 * 4 = 32

        // Generate synthetic data (32 samples, will process as 4 batches of 8)
        auto X = Tensor::randn({32, 10}, false);
        auto Y = Tensor::randn({32, 5}, false);

        float initial_loss = 0.0f;
        float final_loss = 0.0f;
        int optimizer_steps = 0;

        // Train for 10 "effective" epochs (40 mini-batch iterations)
        for (int epoch = 0; epoch < 10; epoch++) {
            // Process 4 mini-batches per epoch
            for (int batch = 0; batch < 4; batch++) {
                // Get mini-batch (8 samples each) - slice(start, end, dim)
                auto x_batch = X->slice(batch * 8, (batch + 1) * 8, 0);
                auto y_batch = Y->slice(batch * 8, (batch + 1) * 8, 0);

                auto pred = model.forward(x_batch);
                auto loss = criterion(pred, y_batch);

                if (epoch == 0 && batch == 0) initial_loss = loss->data[0];
                if (epoch == 9 && batch == 3) final_loss = loss->data[0];

                // Accumulate gradients
                accumulator.backward(loss);

                // Step optimizer when accumulation is complete
                if (accumulator.should_step()) {
                    optimizer.step();
                    optimizer.zero_grad();
                    accumulator.reset();
                    optimizer_steps++;
                }
            }
        }

        printf("   Initial loss: %.4f\n", initial_loss);
        printf("   Final loss: %.4f\n", final_loss);
        printf("   Optimizer steps: %d (expected 10)\n", optimizer_steps);
        printf("   %s\n\n", (final_loss < initial_loss && optimizer_steps == 10) ? "PASSED" : "FAILED");
    }

    // Test 4: Integration with GradScaler (mixed precision + accumulation)
    printf("4. Mixed Precision + Gradient Accumulation\n");
    {
        Sequential model({
            new Linear(10, 20),
            new ReLU(),
            new Linear(20, 5)
        });

        SGD optimizer(model.parameters(), 0.001f);
        MSELoss criterion;
        GradientAccumulator accumulator(4);
        GradScaler scaler(256.0f, 2.0f, 0.5f, 100);

        auto X = Tensor::randn({32, 10}, false);
        auto Y = Tensor::randn({32, 5}, false);

        float initial_loss = 0.0f;
        float final_loss = 0.0f;

        for (int epoch = 0; epoch < 20; epoch++) {
            for (int batch = 0; batch < 4; batch++) {
                auto x_batch = X->slice(batch * 8, (batch + 1) * 8, 0);
                auto y_batch = Y->slice(batch * 8, (batch + 1) * 8, 0);

                auto pred = model.forward(x_batch);
                auto loss = criterion(pred, y_batch);

                if (epoch == 0 && batch == 0) initial_loss = loss->data[0];
                if (epoch == 19 && batch == 3) final_loss = loss->data[0];

                // Scale for accumulation, then scale for mixed precision
                auto accum_scaled = accumulator.scale(loss);
                auto mixed_scaled = scaler.scale(accum_scaled);
                mixed_scaled->backward();
                accumulator.increment();

                if (accumulator.should_step()) {
                    scaler.unscale(&optimizer);
                    scaler.step(&optimizer, true);
                    scaler.update();
                    optimizer.zero_grad();
                    accumulator.reset();
                }
            }
        }

        printf("   Initial loss: %.4f\n", initial_loss);
        printf("   Final loss: %.4f\n", final_loss);
        printf("   Final scale: %.1f\n", scaler.get_scale());
        printf("   %s\n\n", final_loss < initial_loss ? "PASSED" : "FAILED");
    }

    // Test 5: Edge cases
    printf("5. Edge Cases\n");
    {
        bool pass = true;

        // Accumulation steps = 1 (no accumulation)
        GradientAccumulator accum1(1);
        pass &= (accum1.get_accumulation_steps() == 1);
        pass &= (std::abs(accum1.get_scale_factor() - 1.0f) < 1e-6f);

        // Invalid accumulation steps (should default to 1)
        GradientAccumulator accum0(0);
        pass &= (accum0.get_accumulation_steps() == 1);

        GradientAccumulator accum_neg(-5);
        pass &= (accum_neg.get_accumulation_steps() == 1);

        // Test is_last_step
        GradientAccumulator accum4(4);
        pass &= (!accum4.is_last_step());  // Step 0
        accum4.increment();
        accum4.increment();
        accum4.increment();
        pass &= (accum4.is_last_step());   // Step 3 is last before stepping
        accum4.increment();
        pass &= (accum4.should_step());    // Step 4, should step now

        printf("   %s\n\n", pass ? "PASSED" : "FAILED");
    }

    printf("All tests completed!\n");
    return 0;
}
