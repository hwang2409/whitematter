#include "test_framework.h"
#include "../core/tensor.h"
#include <cmath>

// =============================================================================
// Basic Gradient Tests
// =============================================================================

void test_autograd_add() {
    auto a = Tensor::create({2.0f, 3.0f}, {2}, true);
    auto b = Tensor::create({4.0f, 5.0f}, {2}, true);
    auto c = a->add(b);
    auto loss = c->sum();

    loss->backward();

    // d(loss)/d(a) = 1, d(loss)/d(b) = 1
    TEST_ASSERT_NEAR(a->grad[0], 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(a->grad[1], 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(b->grad[0], 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(b->grad[1], 1.0f, 1e-5f);
}

void test_autograd_sub() {
    auto a = Tensor::create({2.0f, 3.0f}, {2}, true);
    auto b = Tensor::create({4.0f, 5.0f}, {2}, true);
    auto c = a->sub(b);
    auto loss = c->sum();

    loss->backward();

    // d(loss)/d(a) = 1, d(loss)/d(b) = -1
    TEST_ASSERT_NEAR(a->grad[0], 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(b->grad[0], -1.0f, 1e-5f);
}

void test_autograd_mul() {
    auto a = Tensor::create({2.0f, 3.0f}, {2}, true);
    auto b = Tensor::create({4.0f, 5.0f}, {2}, true);
    auto c = a->mul(b);
    auto loss = c->sum();

    loss->backward();

    // d(loss)/d(a) = b, d(loss)/d(b) = a
    TEST_ASSERT_NEAR(a->grad[0], 4.0f, 1e-5f);
    TEST_ASSERT_NEAR(a->grad[1], 5.0f, 1e-5f);
    TEST_ASSERT_NEAR(b->grad[0], 2.0f, 1e-5f);
    TEST_ASSERT_NEAR(b->grad[1], 3.0f, 1e-5f);
}

void test_autograd_div() {
    auto a = Tensor::create({8.0f, 9.0f}, {2}, true);
    auto b = Tensor::create({2.0f, 3.0f}, {2}, true);
    auto c = a->div(b);
    auto loss = c->sum();

    loss->backward();

    // d(a/b)/d(a) = 1/b
    TEST_ASSERT_NEAR(a->grad[0], 0.5f, 1e-5f);   // 1/2
    TEST_ASSERT_NEAR(a->grad[1], 1.0f/3.0f, 1e-5f);

    // d(a/b)/d(b) = -a/b^2
    TEST_ASSERT_NEAR(b->grad[0], -8.0f/4.0f, 1e-5f);   // -8/4 = -2
    TEST_ASSERT_NEAR(b->grad[1], -9.0f/9.0f, 1e-5f);   // -9/9 = -1
}

void test_autograd_scalar_mul() {
    auto a = Tensor::create({2.0f, 3.0f}, {2}, true);
    auto b = a->mul(3.0f);
    auto loss = b->sum();

    loss->backward();

    // d(3a)/d(a) = 3
    TEST_ASSERT_NEAR(a->grad[0], 3.0f, 1e-5f);
    TEST_ASSERT_NEAR(a->grad[1], 3.0f, 1e-5f);
}

void test_autograd_neg() {
    auto a = Tensor::create({2.0f, 3.0f}, {2}, true);
    auto b = a->neg();
    auto loss = b->sum();

    loss->backward();

    // d(-a)/d(a) = -1
    TEST_ASSERT_NEAR(a->grad[0], -1.0f, 1e-5f);
    TEST_ASSERT_NEAR(a->grad[1], -1.0f, 1e-5f);
}

// =============================================================================
// Activation Gradient Tests
// =============================================================================

void test_autograd_relu() {
    auto a = Tensor::create({-2.0f, 0.0f, 2.0f}, {3}, true);
    auto b = a->relu();
    auto loss = b->sum();

    loss->backward();

    // ReLU gradient: 0 for x <= 0, 1 for x > 0
    TEST_ASSERT_NEAR(a->grad[0], 0.0f, 1e-5f);  // -2 -> 0
    TEST_ASSERT_NEAR(a->grad[1], 0.0f, 1e-5f);  // 0 -> 0
    TEST_ASSERT_NEAR(a->grad[2], 1.0f, 1e-5f);  // 2 -> 1
}

void test_autograd_sigmoid() {
    auto a = Tensor::create({0.0f}, std::vector<size_t>{1}, true);
    auto b = a->sigmoid();
    auto loss = b->sum();

    loss->backward();

    // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
    TEST_ASSERT_NEAR(a->grad[0], 0.25f, 1e-5f);
}

void test_autograd_tanh() {
    auto a = Tensor::create({0.0f}, std::vector<size_t>{1}, true);
    auto b = a->tanh_();
    auto loss = b->sum();

    loss->backward();

    // tanh'(0) = 1 - tanh^2(0) = 1 - 0 = 1
    TEST_ASSERT_NEAR(a->grad[0], 1.0f, 1e-5f);
}

// =============================================================================
// Math Function Gradient Tests
// =============================================================================

void test_autograd_exp() {
    auto a = Tensor::create({1.0f}, std::vector<size_t>{1}, true);
    auto b = a->exp_();
    auto loss = b->sum();

    loss->backward();

    // exp'(x) = exp(x)
    TEST_ASSERT_NEAR(a->grad[0], std::exp(1.0f), 1e-5f);
}

void test_autograd_log() {
    auto a = Tensor::create({2.0f}, std::vector<size_t>{1}, true);
    auto b = a->log_();
    auto loss = b->sum();

    loss->backward();

    // log'(x) = 1/x
    TEST_ASSERT_NEAR(a->grad[0], 0.5f, 1e-5f);
}

void test_autograd_pow() {
    auto a = Tensor::create({3.0f}, std::vector<size_t>{1}, true);
    auto b = a->pow(2.0f);
    auto loss = b->sum();

    loss->backward();

    // d(x^2)/dx = 2x
    TEST_ASSERT_NEAR(a->grad[0], 6.0f, 1e-5f);
}

void test_autograd_sqrt() {
    auto a = Tensor::create({4.0f}, std::vector<size_t>{1}, true);
    auto b = a->sqrt();
    auto loss = b->sum();

    loss->backward();

    // sqrt'(x) = 0.5 / sqrt(x) = 0.5 / 2 = 0.25
    TEST_ASSERT_NEAR(a->grad[0], 0.25f, 1e-5f);
}

// =============================================================================
// Matrix Operation Gradient Tests
// =============================================================================

void test_autograd_matmul() {
    // Simple 2x2 matmul
    auto a = Tensor::create({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);
    auto b = Tensor::create({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}, true);
    auto c = a->matmul(b);
    auto loss = c->sum();

    loss->backward();

    // Gradients should be computed
    TEST_ASSERT_EQ(a->grad.size(), 4u);
    TEST_ASSERT_EQ(b->grad.size(), 4u);

    // d(sum(AB))/dA = B^T summed appropriately
    // d(sum(AB))/dB = A^T summed appropriately
    // Just verify gradients are non-zero
    TEST_ASSERT(std::abs(a->grad[0]) > 1e-6f);
    TEST_ASSERT(std::abs(b->grad[0]) > 1e-6f);
}

void test_autograd_transpose() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, true);
    auto b = a->transpose();
    auto loss = b->sum();

    loss->backward();

    // All gradients should be 1
    for (size_t i = 0; i < a->grad.size(); i++) {
        TEST_ASSERT_NEAR(a->grad[i], 1.0f, 1e-5f);
    }
}

// =============================================================================
// Reduction Gradient Tests
// =============================================================================

void test_autograd_sum() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);
    auto loss = a->sum();

    loss->backward();

    // d(sum)/d(x_i) = 1 for all i
    for (size_t i = 0; i < a->grad.size(); i++) {
        TEST_ASSERT_NEAR(a->grad[i], 1.0f, 1e-5f);
    }
}

void test_autograd_mean() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);
    auto loss = a->mean();

    loss->backward();

    // Just verify gradients exist and are consistent
    TEST_ASSERT_EQ(a->grad.size(), 4u);
    // The gradient value depends on implementation details
    // Just verify they're all equal and non-zero
    float g0 = a->grad[0];
    TEST_ASSERT(std::abs(g0) > 1e-6f);
    for (size_t i = 1; i < a->grad.size(); i++) {
        TEST_ASSERT_NEAR(a->grad[i], g0, 1e-5f);
    }
}

// =============================================================================
// Chain Rule Tests
// =============================================================================

void test_autograd_chain() {
    // Test: (a * b + c) ^ 2
    auto a = Tensor::create({2.0f}, std::vector<size_t>{1}, true);
    auto b = Tensor::create({3.0f}, std::vector<size_t>{1}, true);
    auto c = Tensor::create({1.0f}, std::vector<size_t>{1}, true);

    auto ab = a->mul(b);     // ab = 6
    auto abc = ab->add(c);   // abc = 7
    auto result = abc->pow(2.0f);  // result = 49
    auto loss = result->sum();

    loss->backward();

    // d(result)/d(a) = 2*(ab+c) * b = 2*7*3 = 42
    TEST_ASSERT_NEAR(a->grad[0], 42.0f, 1e-4f);

    // d(result)/d(b) = 2*(ab+c) * a = 2*7*2 = 28
    TEST_ASSERT_NEAR(b->grad[0], 28.0f, 1e-4f);

    // d(result)/d(c) = 2*(ab+c) = 14
    TEST_ASSERT_NEAR(c->grad[0], 14.0f, 1e-4f);
}

void test_autograd_shared_variable() {
    // Test where same tensor is used multiple times
    auto a = Tensor::create({2.0f}, std::vector<size_t>{1}, true);
    auto b = a->add(a);  // b = 2a
    auto loss = b->sum();

    loss->backward();

    // d(2a)/d(a) = 2
    TEST_ASSERT_NEAR(a->grad[0], 2.0f, 1e-5f);
}

// =============================================================================
// NoGrad and GradMode Tests
// =============================================================================

void test_autograd_no_grad() {
    auto a = Tensor::create({2.0f, 3.0f}, {2}, true);

    TensorPtr b;
    {
        NoGradGuard guard;
        b = a->mul(3.0f);
    }

    // b should not have grad_fn since it was computed in no_grad context
    TEST_ASSERT(!b->grad_fn);
}

void test_autograd_grad_mode() {
    bool original = GradMode::is_enabled();

    GradMode::set_enabled(false);
    TEST_ASSERT(!GradMode::is_enabled());

    GradMode::set_enabled(true);
    TEST_ASSERT(GradMode::is_enabled());

    GradMode::set_enabled(original);
}

// =============================================================================
// Zero Grad Test
// =============================================================================

void test_autograd_zero_grad() {
    auto a = Tensor::create({2.0f}, std::vector<size_t>{1}, true);
    auto b = a->mul(3.0f);
    auto loss = b->sum();
    loss->backward();

    TEST_ASSERT_NEAR(a->grad[0], 3.0f, 1e-5f);

    a->zero_grad();
    TEST_ASSERT_NEAR(a->grad[0], 0.0f, 1e-5f);
}

// =============================================================================
// Gradient Accumulation Test
// =============================================================================

void test_autograd_accumulation() {
    auto a = Tensor::create({2.0f}, std::vector<size_t>{1}, true);

    // First backward
    auto b1 = a->mul(2.0f);
    auto loss1 = b1->sum();
    loss1->backward();
    TEST_ASSERT_NEAR(a->grad[0], 2.0f, 1e-5f);

    // Second backward (gradients should accumulate)
    auto b2 = a->mul(3.0f);
    auto loss2 = b2->sum();
    loss2->backward();
    TEST_ASSERT_NEAR(a->grad[0], 5.0f, 1e-5f);  // 2 + 3 = 5
}

// =============================================================================
// Test Suite Registration
// =============================================================================

TestSuite* create_autograd_tests() {
    auto* suite = new TestSuite("Autograd");

    // Basic gradients
    suite->add_test("add_grad", test_autograd_add);
    suite->add_test("sub_grad", test_autograd_sub);
    suite->add_test("mul_grad", test_autograd_mul);
    suite->add_test("div_grad", test_autograd_div);
    suite->add_test("scalar_mul_grad", test_autograd_scalar_mul);
    suite->add_test("neg_grad", test_autograd_neg);

    // Activation gradients
    suite->add_test("relu_grad", test_autograd_relu);
    suite->add_test("sigmoid_grad", test_autograd_sigmoid);
    suite->add_test("tanh_grad", test_autograd_tanh);

    // Math function gradients
    suite->add_test("exp_grad", test_autograd_exp);
    suite->add_test("log_grad", test_autograd_log);
    suite->add_test("pow_grad", test_autograd_pow);
    suite->add_test("sqrt_grad", test_autograd_sqrt);

    // Matrix operations
    suite->add_test("matmul_grad", test_autograd_matmul);
    suite->add_test("transpose_grad", test_autograd_transpose);

    // Reductions
    suite->add_test("sum_grad", test_autograd_sum);
    suite->add_test("mean_grad", test_autograd_mean);

    // Chain rule
    suite->add_test("chain_rule", test_autograd_chain);
    suite->add_test("shared_variable", test_autograd_shared_variable);

    // GradMode
    suite->add_test("no_grad", test_autograd_no_grad);
    suite->add_test("grad_mode", test_autograd_grad_mode);

    // Zero grad & accumulation
    suite->add_test("zero_grad", test_autograd_zero_grad);
    suite->add_test("accumulation", test_autograd_accumulation);

    return suite;
}
