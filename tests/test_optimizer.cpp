#include "test_framework.h"
#include "../core/tensor.h"
#include "../core/layer.h"
#include "../core/loss.h"
#include "../core/optimizer.h"
#include <cmath>

// =============================================================================
// SGD Tests
// =============================================================================

void test_sgd_step() {
    Linear layer(10, 5);
    SGD optimizer(layer.parameters(), 0.1f);

    auto input = Tensor::randn({4, 10}, true);
    auto target = Tensor::randn({4, 5});
    MSELoss criterion;

    auto output = layer.forward(input);
    auto loss = criterion(output, target);
    loss->backward();

    // Store original weights
    float orig_w = layer.weight->data[0];

    optimizer.step();

    // Weights should have changed
    TEST_ASSERT(layer.weight->data[0] != orig_w);
}

void test_sgd_zero_grad() {
    Linear layer(10, 5);
    SGD optimizer(layer.parameters(), 0.1f);

    auto input = Tensor::randn({4, 10}, true);
    auto target = Tensor::randn({4, 5});
    MSELoss criterion;

    auto output = layer.forward(input);
    auto loss = criterion(output, target);
    loss->backward();

    // Gradients should be non-zero
    bool has_grad = false;
    for (float g : layer.weight->grad) {
        if (std::abs(g) > 1e-6f) has_grad = true;
    }
    TEST_ASSERT(has_grad);

    optimizer.zero_grad();

    // Gradients should be zero
    for (float g : layer.weight->grad) {
        TEST_ASSERT_NEAR(g, 0.0f, 1e-6f);
    }
}

void test_sgd_momentum() {
    Linear layer(5, 3);
    SGD optimizer(layer.parameters(), 0.1f, 0.9f);  // momentum = 0.9

    auto input = Tensor::randn({2, 5}, true);
    auto target = Tensor::randn({2, 3});
    MSELoss criterion;

    // Multiple steps to build up momentum
    for (int i = 0; i < 5; i++) {
        auto output = layer.forward(input);
        auto loss = criterion(output, target);
        optimizer.zero_grad();
        loss->backward();
        optimizer.step();
    }

    // Velocity should have been accumulated
    TEST_ASSERT_EQ(optimizer.velocity.size(), 2u);  // weight and bias
}

void test_sgd_loss_decreases() {
    Linear layer(10, 5);
    SGD optimizer(layer.parameters(), 0.01f);

    auto input = Tensor::randn({8, 10});
    auto target = Tensor::randn({8, 5});
    MSELoss criterion;

    // Get initial loss
    auto output0 = layer.forward(input);
    auto loss0 = criterion(output0, target);
    float initial_loss = loss0->data[0];

    // Train for several steps
    for (int i = 0; i < 50; i++) {
        optimizer.zero_grad();
        auto output = layer.forward(input);
        auto loss = criterion(output, target);
        loss->backward();
        optimizer.step();
    }

    // Get final loss
    auto output_final = layer.forward(input);
    auto loss_final = criterion(output_final, target);
    float final_loss = loss_final->data[0];

    TEST_ASSERT(final_loss < initial_loss);
}

// =============================================================================
// Adam Tests
// =============================================================================

void test_adam_step() {
    Linear layer(10, 5);
    Adam optimizer(layer.parameters(), 0.001f);

    auto input = Tensor::randn({4, 10}, true);
    auto target = Tensor::randn({4, 5});
    MSELoss criterion;

    auto output = layer.forward(input);
    auto loss = criterion(output, target);
    loss->backward();

    float orig_w = layer.weight->data[0];
    optimizer.step();

    TEST_ASSERT(layer.weight->data[0] != orig_w);
}

void test_adam_moments() {
    Linear layer(5, 3);
    Adam optimizer(layer.parameters(), 0.001f);

    auto input = Tensor::randn({2, 5}, true);
    auto target = Tensor::randn({2, 3});
    MSELoss criterion;

    // Multiple steps
    for (int i = 0; i < 3; i++) {
        optimizer.zero_grad();
        auto output = layer.forward(input);
        auto loss = criterion(output, target);
        loss->backward();
        optimizer.step();
    }

    // m and v should be populated
    TEST_ASSERT_EQ(optimizer.m.size(), 2u);
    TEST_ASSERT_EQ(optimizer.v.size(), 2u);
    TEST_ASSERT_EQ(optimizer.t, 3);
}

void test_adam_loss_decreases() {
    Linear layer(10, 5);
    Adam optimizer(layer.parameters(), 0.01f);

    auto input = Tensor::randn({8, 10});
    auto target = Tensor::randn({8, 5});
    MSELoss criterion;

    auto output0 = layer.forward(input);
    auto loss0 = criterion(output0, target);
    float initial_loss = loss0->data[0];

    for (int i = 0; i < 50; i++) {
        optimizer.zero_grad();
        auto output = layer.forward(input);
        auto loss = criterion(output, target);
        loss->backward();
        optimizer.step();
    }

    auto output_final = layer.forward(input);
    auto loss_final = criterion(output_final, target);
    float final_loss = loss_final->data[0];

    TEST_ASSERT(final_loss < initial_loss);
}

// =============================================================================
// AdamW Tests
// =============================================================================

void test_adamw_step() {
    Linear layer(10, 5);
    AdamW optimizer(layer.parameters(), 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);

    auto input = Tensor::randn({4, 10}, true);
    auto target = Tensor::randn({4, 5});
    MSELoss criterion;

    auto output = layer.forward(input);
    auto loss = criterion(output, target);
    loss->backward();

    float orig_w = layer.weight->data[0];
    optimizer.step();

    TEST_ASSERT(layer.weight->data[0] != orig_w);
}

void test_adamw_weight_decay() {
    Linear layer(10, 5);

    // High weight decay
    AdamW optimizer(layer.parameters(), 0.001f, 0.9f, 0.999f, 1e-8f, 0.5f);

    // Store initial weight magnitude
    float initial_norm = 0.0f;
    for (float w : layer.weight->data) {
        initial_norm += w * w;
    }
    initial_norm = std::sqrt(initial_norm);

    auto input = Tensor::randn({4, 10}, true);
    auto target = Tensor::zeros({4, 5});  // Zero target to focus on weight decay
    MSELoss criterion;

    for (int i = 0; i < 100; i++) {
        optimizer.zero_grad();
        auto output = layer.forward(input);
        auto loss = criterion(output, target);
        loss->backward();
        optimizer.step();
    }

    // Weight magnitude should decrease due to weight decay
    float final_norm = 0.0f;
    for (float w : layer.weight->data) {
        final_norm += w * w;
    }
    final_norm = std::sqrt(final_norm);

    TEST_ASSERT(final_norm < initial_norm);
}

// =============================================================================
// RMSprop Tests
// =============================================================================

void test_rmsprop_step() {
    Linear layer(10, 5);
    RMSprop optimizer(layer.parameters(), 0.01f);

    auto input = Tensor::randn({4, 10}, true);
    auto target = Tensor::randn({4, 5});
    MSELoss criterion;

    auto output = layer.forward(input);
    auto loss = criterion(output, target);
    loss->backward();

    float orig_w = layer.weight->data[0];
    optimizer.step();

    TEST_ASSERT(layer.weight->data[0] != orig_w);
}

// =============================================================================
// Gradient Clipping Tests
// =============================================================================

void test_clip_grad_norm() {
    auto params = std::vector<TensorPtr>{
        Tensor::create({1.0f, 2.0f, 3.0f}, {3}, true),
        Tensor::create({4.0f, 5.0f}, {2}, true)
    };

    // Set gradients
    params[0]->grad = {10.0f, 20.0f, 30.0f};
    params[1]->grad = {40.0f, 50.0f};

    float original_norm = get_grad_norm(params);
    clip_grad_norm_(params, 1.0f);
    float clipped_norm = get_grad_norm(params);

    TEST_ASSERT(clipped_norm <= 1.0f + 1e-5f);
    TEST_ASSERT(original_norm > clipped_norm);
}

void test_clip_grad_value() {
    auto params = std::vector<TensorPtr>{
        Tensor::create({1.0f, 2.0f, 3.0f}, {3}, true)
    };

    params[0]->grad = {10.0f, -20.0f, 5.0f};

    clip_grad_value_(params, 8.0f);

    TEST_ASSERT_NEAR(params[0]->grad[0], 8.0f, 1e-5f);   // Clipped from 10
    TEST_ASSERT_NEAR(params[0]->grad[1], -8.0f, 1e-5f); // Clipped from -20
    TEST_ASSERT_NEAR(params[0]->grad[2], 5.0f, 1e-5f);  // Unchanged
}

void test_get_grad_norm() {
    auto params = std::vector<TensorPtr>{
        Tensor::create({1.0f}, std::vector<size_t>{1}, true)
    };
    params[0]->grad = {3.0f};

    auto params2 = std::vector<TensorPtr>{
        Tensor::create({1.0f}, std::vector<size_t>{1}, true)
    };
    params2[0]->grad = {4.0f};

    float norm1 = get_grad_norm(params);
    float norm2 = get_grad_norm(params2);

    TEST_ASSERT_NEAR(norm1, 3.0f, 1e-5f);
    TEST_ASSERT_NEAR(norm2, 4.0f, 1e-5f);
}

// =============================================================================
// Gradient Accumulator Tests
// =============================================================================

void test_gradient_accumulator() {
    GradientAccumulator accumulator(4);

    TEST_ASSERT_EQ(accumulator.get_accumulation_steps(), 4);
    TEST_ASSERT_NEAR(accumulator.get_scale_factor(), 0.25f, 1e-6f);
}

void test_gradient_accumulator_should_step() {
    GradientAccumulator accumulator(3);

    TEST_ASSERT(!accumulator.should_step());

    accumulator.increment();
    TEST_ASSERT(!accumulator.should_step());

    accumulator.increment();
    TEST_ASSERT(!accumulator.should_step());

    accumulator.increment();
    TEST_ASSERT(accumulator.should_step());

    accumulator.reset();
    TEST_ASSERT(!accumulator.should_step());
}

void test_gradient_accumulator_scale() {
    GradientAccumulator accumulator(4);

    auto loss = Tensor::create({4.0f}, std::vector<size_t>{1});
    auto scaled = accumulator.scale(loss);

    TEST_ASSERT_NEAR(scaled->data[0], 1.0f, 1e-5f);  // 4 / 4 = 1
}

void test_gradient_accumulator_backward() {
    GradientAccumulator accumulator(2);

    auto x = Tensor::create({2.0f}, std::vector<size_t>{1}, true);

    // First accumulation step
    auto loss1 = x->mul(3.0f);  // loss = 3x, dloss/dx = 3
    accumulator.backward(loss1);

    // Second accumulation step
    auto loss2 = x->mul(3.0f);
    accumulator.backward(loss2);

    TEST_ASSERT(accumulator.should_step());
    // Gradients should have accumulated
    TEST_ASSERT(std::abs(x->grad[0]) > 0.0f);
}

// =============================================================================
// Learning Rate Scheduler Tests
// =============================================================================

void test_step_lr() {
    auto params = std::vector<TensorPtr>{Tensor::randn({10})};
    SGD optimizer(params, 0.1f);
    StepLR scheduler(&optimizer, 5, 0.1f);  // Decay every 5 epochs by 0.1

    float initial_lr = optimizer.lr;
    TEST_ASSERT_NEAR(initial_lr, 0.1f, 1e-6f);

    // Step through several epochs
    for (int i = 0; i < 6; i++) {
        scheduler.step();
    }

    // LR should have decayed at least once
    TEST_ASSERT(optimizer.lr < initial_lr);
}

void test_exponential_lr() {
    auto params = std::vector<TensorPtr>{Tensor::randn({10})};
    SGD optimizer(params, 1.0f);
    ExponentialLR scheduler(&optimizer, 0.9f);

    float initial_lr = optimizer.lr;

    // Step several times
    for (int i = 0; i < 3; i++) {
        scheduler.step();
    }
    float final_lr = optimizer.lr;

    // LR should have decayed after multiple steps
    TEST_ASSERT(final_lr < initial_lr);
}

void test_cosine_annealing_lr() {
    auto params = std::vector<TensorPtr>{Tensor::randn({10})};
    SGD optimizer(params, 1.0f);
    CosineAnnealingLR scheduler(&optimizer, 10, 0.0f);  // T_max=10, eta_min=0

    float initial_lr = optimizer.lr;
    TEST_ASSERT_NEAR(initial_lr, 1.0f, 1e-5f);

    // Step through the schedule
    for (int i = 0; i < 5; i++) {
        scheduler.step();
    }
    float mid_lr = optimizer.lr;

    for (int i = 0; i < 5; i++) {
        scheduler.step();
    }
    float end_lr = optimizer.lr;

    // LR should decrease over time with cosine annealing
    TEST_ASSERT(mid_lr < initial_lr);
    TEST_ASSERT(end_lr <= mid_lr);
}

void test_reduce_lr_on_plateau() {
    auto params = std::vector<TensorPtr>{Tensor::randn({10})};
    SGD optimizer(params, 0.1f);
    ReduceLROnPlateau scheduler(&optimizer, 0.5f, 3, 0.0f, true);  // factor=0.5, patience=3

    // Improving metrics (decreasing)
    scheduler.step(1.0f);
    scheduler.step(0.9f);
    scheduler.step(0.8f);

    // LR should not have changed
    TEST_ASSERT_NEAR(optimizer.lr, 0.1f, 1e-6f);

    // Non-improving metrics
    scheduler.step(0.8f);
    scheduler.step(0.8f);
    scheduler.step(0.8f);
    scheduler.step(0.8f);  // 4th non-improving, should trigger reduction

    // LR should have been reduced
    TEST_ASSERT_NEAR(optimizer.lr, 0.05f, 1e-5f);
}

// =============================================================================
// Early Stopping Tests
// =============================================================================

void test_early_stopping_improves() {
    EarlyStopping es(5, 0.0f, true);  // patience=5, mode_min

    // Improving metrics (decreasing loss)
    TEST_ASSERT(!es.step(1.0f));
    TEST_ASSERT(!es.step(0.9f));
    TEST_ASSERT(!es.step(0.8f));

    TEST_ASSERT(!es.should_stop());
    TEST_ASSERT_NEAR(es.best_metric(), 0.8f, 1e-5f);
}

void test_early_stopping_stops() {
    EarlyStopping es(3, 0.0f, true);  // patience=3

    es.step(1.0f);  // Initial

    // Non-improving
    es.step(1.1f);
    es.step(1.1f);
    bool should_stop = es.step(1.1f);  // 3rd non-improving

    TEST_ASSERT(should_stop);
    TEST_ASSERT(es.should_stop());
}

void test_early_stopping_mode_max() {
    EarlyStopping es(2, 0.0f, false);  // mode_max (higher is better, e.g., accuracy)

    es.step(0.7f);
    es.step(0.8f);  // Improvement
    TEST_ASSERT(!es.should_stop());

    es.step(0.75f);  // Not improvement
    es.step(0.75f);  // 2nd non-improving

    TEST_ASSERT(es.should_stop());
}

void test_early_stopping_min_delta() {
    EarlyStopping es(2, 0.05f, true);  // min_delta=0.05

    es.step(1.0f);
    es.step(0.98f);  // Only 0.02 improvement, less than delta
    es.step(0.97f);  // Still not enough improvement

    TEST_ASSERT(es.should_stop());
}

void test_early_stopping_reset() {
    EarlyStopping es(2);

    es.step(1.0f);
    es.step(1.1f);
    es.step(1.1f);

    TEST_ASSERT(es.should_stop());

    es.reset();

    TEST_ASSERT(!es.should_stop());
    // After reset, best_epoch should be -1 (reset state)
    TEST_ASSERT_EQ(es.best_epoch(), -1);
}

// =============================================================================
// Test Suite Registration
// =============================================================================

TestSuite* create_optimizer_tests() {
    auto* suite = new TestSuite("Optimizers");

    // SGD tests
    suite->add_test("sgd_step", test_sgd_step);
    suite->add_test("sgd_zero_grad", test_sgd_zero_grad);
    suite->add_test("sgd_momentum", test_sgd_momentum);
    suite->add_test("sgd_loss_decreases", test_sgd_loss_decreases);

    // Adam tests
    suite->add_test("adam_step", test_adam_step);
    suite->add_test("adam_moments", test_adam_moments);
    suite->add_test("adam_loss_decreases", test_adam_loss_decreases);

    // AdamW tests
    suite->add_test("adamw_step", test_adamw_step);
    suite->add_test("adamw_weight_decay", test_adamw_weight_decay);

    // RMSprop tests
    suite->add_test("rmsprop_step", test_rmsprop_step);

    // Gradient clipping tests
    suite->add_test("clip_grad_norm", test_clip_grad_norm);
    suite->add_test("clip_grad_value", test_clip_grad_value);
    suite->add_test("get_grad_norm", test_get_grad_norm);

    // Gradient accumulator tests
    suite->add_test("grad_accum_init", test_gradient_accumulator);
    suite->add_test("grad_accum_should_step", test_gradient_accumulator_should_step);
    suite->add_test("grad_accum_scale", test_gradient_accumulator_scale);
    suite->add_test("grad_accum_backward", test_gradient_accumulator_backward);

    // LR scheduler tests
    suite->add_test("step_lr", test_step_lr);
    suite->add_test("exponential_lr", test_exponential_lr);
    suite->add_test("cosine_annealing_lr", test_cosine_annealing_lr);
    suite->add_test("reduce_lr_on_plateau", test_reduce_lr_on_plateau);

    // Early stopping tests
    suite->add_test("early_stopping_improves", test_early_stopping_improves);
    suite->add_test("early_stopping_stops", test_early_stopping_stops);
    suite->add_test("early_stopping_mode_max", test_early_stopping_mode_max);
    suite->add_test("early_stopping_min_delta", test_early_stopping_min_delta);
    suite->add_test("early_stopping_reset", test_early_stopping_reset);

    return suite;
}
