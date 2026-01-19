#include "test_framework.h"
#include "../core/tensor.h"
#include "../core/loss.h"
#include <cmath>

// =============================================================================
// MSE Loss Tests
// =============================================================================

void test_mse_loss_zero() {
    MSELoss loss;
    auto pred = Tensor::create({1.0f, 2.0f, 3.0f}, {1, 3});
    auto target = Tensor::create({1.0f, 2.0f, 3.0f}, {1, 3});

    auto l = loss(pred, target);

    TEST_ASSERT_NEAR(l->data[0], 0.0f, 1e-6f);
}

void test_mse_loss_nonzero() {
    MSELoss loss;
    auto pred = Tensor::create({1.0f, 2.0f, 3.0f}, {1, 3});
    auto target = Tensor::create({2.0f, 3.0f, 4.0f}, {1, 3});

    auto l = loss(pred, target);

    // MSE = mean((1-2)^2 + (2-3)^2 + (3-4)^2) = mean(1 + 1 + 1) = 1
    TEST_ASSERT_NEAR(l->data[0], 1.0f, 1e-5f);
}

void test_mse_loss_gradient() {
    MSELoss loss;
    auto pred = Tensor::create({1.0f, 2.0f, 3.0f}, {1, 3}, true);
    auto target = Tensor::create({2.0f, 3.0f, 4.0f}, {1, 3});

    auto l = loss(pred, target);
    l->backward();

    // Verify gradients exist and point in the right direction
    // pred < target means gradient should be negative
    TEST_ASSERT_EQ(pred->grad.size(), 3u);
    TEST_ASSERT(pred->grad[0] < 0.0f);  // Should be negative since pred < target
}

// =============================================================================
// L1 Loss Tests
// =============================================================================

void test_l1_loss_zero() {
    L1Loss loss;
    auto pred = Tensor::create({1.0f, 2.0f, 3.0f}, {1, 3});
    auto target = Tensor::create({1.0f, 2.0f, 3.0f}, {1, 3});

    auto l = loss(pred, target);

    TEST_ASSERT_NEAR(l->data[0], 0.0f, 1e-6f);
}

void test_l1_loss_nonzero() {
    L1Loss loss;
    auto pred = Tensor::create({1.0f, 2.0f, 3.0f}, {1, 3});
    auto target = Tensor::create({2.0f, 4.0f, 6.0f}, {1, 3});

    auto l = loss(pred, target);

    // L1 = mean(|1-2| + |2-4| + |3-6|) = mean(1 + 2 + 3) = 2
    TEST_ASSERT_NEAR(l->data[0], 2.0f, 1e-5f);
}

// =============================================================================
// Smooth L1 Loss Tests
// =============================================================================

void test_smooth_l1_loss() {
    SmoothL1Loss loss(1.0f);
    auto pred = Tensor::create({0.5f, 2.0f}, {1, 2});
    auto target = Tensor::create({0.0f, 0.0f}, {1, 2});

    auto l = loss(pred, target);

    // For |diff| < beta (0.5 < 1): 0.5 * diff^2 / beta = 0.5 * 0.25 / 1 = 0.125
    // For |diff| >= beta (2 >= 1): |diff| - 0.5 * beta = 2 - 0.5 = 1.5
    // Mean: (0.125 + 1.5) / 2 = 0.8125
    TEST_ASSERT_NEAR(l->data[0], 0.8125f, 1e-4f);
}

// =============================================================================
// Cross Entropy Loss Tests
// =============================================================================

void test_cross_entropy_loss() {
    CrossEntropyLoss loss;
    // Logits for 2 samples, 3 classes
    auto pred = Tensor::create({
        1.0f, 2.0f, 3.0f,  // Sample 1
        1.0f, 2.0f, 3.0f   // Sample 2
    }, {2, 3});

    // Class labels
    auto target = Tensor::create({2.0f, 0.0f}, std::vector<size_t>{2});

    auto l = loss(pred, target);

    // CrossEntropy = -log(softmax(pred)[target])
    TEST_ASSERT(l->data[0] > 0.0f);  // Loss should be positive
}

void test_cross_entropy_perfect_prediction() {
    CrossEntropyLoss loss;
    // Very confident predictions
    auto pred = Tensor::create({
        -100.0f, -100.0f, 100.0f  // Class 2 should be selected
    }, {1, 3});

    auto target = Tensor::create({2.0f}, std::vector<size_t>{1});

    auto l = loss(pred, target);

    // With very confident prediction, loss should be very small
    TEST_ASSERT(l->data[0] < 0.1f);
}

void test_cross_entropy_gradient() {
    CrossEntropyLoss loss;
    auto pred = Tensor::create({1.0f, 2.0f, 3.0f}, {1, 3}, true);
    auto target = Tensor::create({1.0f}, std::vector<size_t>{1});

    auto l = loss(pred, target);
    l->backward();

    // Gradient should be softmax(pred) - one_hot(target)
    TEST_ASSERT_EQ(pred->grad.size(), 3u);
}

// =============================================================================
// NLL Loss Tests
// =============================================================================

void test_nll_loss() {
    NLLLoss loss;
    // Log probabilities
    auto pred = Tensor::create({
        std::log(0.1f), std::log(0.7f), std::log(0.2f)
    }, {1, 3});

    auto target = Tensor::create({1.0f}, std::vector<size_t>{1});

    auto l = loss(pred, target);

    // NLLLoss = -pred[target] = -log(0.7)
    TEST_ASSERT_NEAR(l->data[0], -std::log(0.7f), 1e-4f);
}

// =============================================================================
// BCE Loss Tests
// =============================================================================

void test_bce_loss() {
    BCELoss loss;
    // Predictions between 0 and 1
    auto pred = Tensor::create({0.8f, 0.2f}, std::vector<size_t>{2});
    auto target = Tensor::create({1.0f, 0.0f}, std::vector<size_t>{2});

    auto l = loss(pred, target);

    // BCE = -mean(y*log(p) + (1-y)*log(1-p))
    float expected = -(std::log(0.8f) + std::log(0.8f)) / 2.0f;
    TEST_ASSERT_NEAR(l->data[0], expected, 1e-4f);
}

void test_bce_loss_perfect() {
    BCELoss loss;
    auto pred = Tensor::create({1.0f, 0.0f}, std::vector<size_t>{2});
    auto target = Tensor::create({1.0f, 0.0f}, std::vector<size_t>{2});

    // Clamp prevents log(0), so loss should be very small
    auto l = loss(pred, target);
    TEST_ASSERT(l->data[0] < 0.1f);
}

// =============================================================================
// BCE With Logits Loss Tests
// =============================================================================

void test_bce_with_logits_loss() {
    BCEWithLogitsLoss loss;
    // Logits (not probabilities)
    auto pred = Tensor::create({2.0f, -2.0f}, std::vector<size_t>{2});
    auto target = Tensor::create({1.0f, 0.0f}, std::vector<size_t>{2});

    auto l = loss(pred, target);

    // Sigmoid of 2.0 ~= 0.88, -2.0 ~= 0.12
    TEST_ASSERT(l->data[0] > 0.0f);  // Loss should be positive
}

void test_bce_with_logits_gradient() {
    BCEWithLogitsLoss loss;
    auto pred = Tensor::create({0.0f}, std::vector<size_t>{1}, true);
    auto target = Tensor::create({1.0f}, std::vector<size_t>{1});

    auto l = loss(pred, target);
    l->backward();

    // At logit=0, sigmoid=0.5, grad = sigmoid - target = 0.5 - 1 = -0.5
    TEST_ASSERT_EQ(pred->grad.size(), 1u);
    TEST_ASSERT_NEAR(pred->grad[0], -0.5f, 1e-4f);
}

// =============================================================================
// KL Divergence Loss Tests
// =============================================================================

void test_kl_div_loss() {
    KLDivLoss loss;
    // Log predictions
    auto pred = Tensor::create({
        std::log(0.5f), std::log(0.3f), std::log(0.2f)
    }, {1, 3});

    // Target probabilities
    auto target = Tensor::create({0.4f, 0.4f, 0.2f}, {1, 3});

    auto l = loss(pred, target);

    // KL should be >= 0 (Gibbs inequality)
    TEST_ASSERT(l->data[0] >= -1e-5f);
}

void test_kl_div_loss_same() {
    KLDivLoss loss;
    auto p = Tensor::create({0.5f, 0.3f, 0.2f}, {1, 3});
    auto pred = p->log_();

    auto target = Tensor::create({0.5f, 0.3f, 0.2f}, {1, 3});

    auto l = loss(pred, target);

    // KL divergence with same distribution should be 0
    TEST_ASSERT_NEAR(l->data[0], 0.0f, 1e-4f);
}

// =============================================================================
// Focal Loss Tests
// =============================================================================

void test_focal_loss() {
    FocalLoss loss(2.0f, -1.0f);  // gamma=2, no alpha weighting
    auto pred = Tensor::create({1.0f, 2.0f, 3.0f}, {1, 3});
    auto target = Tensor::create({2.0f}, std::vector<size_t>{1});

    auto l = loss(pred, target);

    TEST_ASSERT(l->data[0] > 0.0f);  // Loss should be positive
}

void test_focal_loss_confident() {
    FocalLoss loss(2.0f);
    // Very confident correct prediction
    auto pred = Tensor::create({-100.0f, -100.0f, 100.0f}, {1, 3});
    auto target = Tensor::create({2.0f}, std::vector<size_t>{1});

    auto l = loss(pred, target);

    // Focal loss should be very small for confident correct predictions
    TEST_ASSERT(l->data[0] < 1e-5f);
}

// =============================================================================
// Binary Focal Loss Tests
// =============================================================================

void test_binary_focal_loss() {
    BinaryFocalLoss loss(2.0f, 0.25f);
    auto pred = Tensor::create({2.0f, -2.0f}, std::vector<size_t>{2});  // Logits
    auto target = Tensor::create({1.0f, 0.0f}, std::vector<size_t>{2});

    auto l = loss(pred, target);

    TEST_ASSERT(l->data[0] > 0.0f);  // Loss should be positive
}

void test_binary_focal_loss_hard_example() {
    BinaryFocalLoss loss(2.0f, 0.5f);

    // Hard example (wrong prediction)
    auto pred_hard = Tensor::create({-3.0f}, std::vector<size_t>{1});
    auto target = Tensor::create({1.0f}, std::vector<size_t>{1});
    auto l_hard = loss(pred_hard, target);

    // Easy example (correct prediction)
    auto pred_easy = Tensor::create({3.0f}, std::vector<size_t>{1});
    auto l_easy = loss(pred_easy, target);

    // Hard example should have higher loss
    TEST_ASSERT(l_hard->data[0] > l_easy->data[0]);
}

// =============================================================================
// Batch Tests
// =============================================================================

void test_mse_loss_batch() {
    MSELoss loss;
    auto pred = Tensor::randn({32, 10});
    auto target = Tensor::randn({32, 10});

    auto l = loss(pred, target);

    // Should produce scalar loss
    TEST_ASSERT_EQ(l->size(), 1u);
}

void test_cross_entropy_batch() {
    CrossEntropyLoss loss;
    auto pred = Tensor::randn({32, 10});
    auto target = Tensor::zeros({32});

    // Set random class labels
    for (size_t i = 0; i < 32; i++) {
        target->data[i] = static_cast<float>(i % 10);
    }

    auto l = loss(pred, target);

    // Should produce scalar loss
    TEST_ASSERT_EQ(l->size(), 1u);
}

// =============================================================================
// Test Suite Registration
// =============================================================================

TestSuite* create_loss_tests() {
    auto* suite = new TestSuite("Loss Functions");

    // MSE Loss
    suite->add_test("mse_loss_zero", test_mse_loss_zero);
    suite->add_test("mse_loss_nonzero", test_mse_loss_nonzero);
    suite->add_test("mse_loss_gradient", test_mse_loss_gradient);

    // L1 Loss
    suite->add_test("l1_loss_zero", test_l1_loss_zero);
    suite->add_test("l1_loss_nonzero", test_l1_loss_nonzero);

    // Smooth L1 Loss
    suite->add_test("smooth_l1_loss", test_smooth_l1_loss);

    // Cross Entropy Loss
    suite->add_test("cross_entropy_loss", test_cross_entropy_loss);
    suite->add_test("cross_entropy_perfect", test_cross_entropy_perfect_prediction);
    suite->add_test("cross_entropy_gradient", test_cross_entropy_gradient);

    // NLL Loss
    suite->add_test("nll_loss", test_nll_loss);

    // BCE Loss
    suite->add_test("bce_loss", test_bce_loss);
    suite->add_test("bce_loss_perfect", test_bce_loss_perfect);

    // BCE With Logits Loss
    suite->add_test("bce_with_logits_loss", test_bce_with_logits_loss);
    suite->add_test("bce_with_logits_gradient", test_bce_with_logits_gradient);

    // KL Divergence Loss
    suite->add_test("kl_div_loss", test_kl_div_loss);
    suite->add_test("kl_div_loss_same", test_kl_div_loss_same);

    // Focal Loss
    suite->add_test("focal_loss", test_focal_loss);
    suite->add_test("focal_loss_confident", test_focal_loss_confident);

    // Binary Focal Loss
    suite->add_test("binary_focal_loss", test_binary_focal_loss);
    suite->add_test("binary_focal_loss_hard", test_binary_focal_loss_hard_example);

    // Batch tests
    suite->add_test("mse_loss_batch", test_mse_loss_batch);
    suite->add_test("cross_entropy_batch", test_cross_entropy_batch);

    return suite;
}
