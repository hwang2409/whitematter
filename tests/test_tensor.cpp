#include "test_framework.h"
#include "../core/tensor.h"
#include <cmath>

// =============================================================================
// Tensor Creation Tests
// =============================================================================

void test_tensor_create_zeros() {
    auto t = Tensor::zeros({2, 3});
    TEST_ASSERT_SHAPE(t, std::vector<size_t>({2, 3}));
    TEST_ASSERT_EQ(t->size(), 6u);
    for (size_t i = 0; i < t->size(); i++) {
        TEST_ASSERT_NEAR(t->data[i], 0.0f, 1e-6f);
    }
}

void test_tensor_create_ones() {
    auto t = Tensor::ones({3, 2});
    TEST_ASSERT_SHAPE(t, std::vector<size_t>({3, 2}));
    TEST_ASSERT_EQ(t->size(), 6u);
    for (size_t i = 0; i < t->size(); i++) {
        TEST_ASSERT_NEAR(t->data[i], 1.0f, 1e-6f);
    }
}

void test_tensor_create_randn() {
    auto t = Tensor::randn({100, 100});
    TEST_ASSERT_EQ(t->size(), 10000u);

    // Check mean is approximately 0 and std is approximately 1
    float sum = 0.0f;
    for (size_t i = 0; i < t->size(); i++) {
        sum += t->data[i];
    }
    float mean = sum / t->size();
    TEST_ASSERT_NEAR(mean, 0.0f, 0.1f); // Should be close to 0

    float var_sum = 0.0f;
    for (size_t i = 0; i < t->size(); i++) {
        var_sum += (t->data[i] - mean) * (t->data[i] - mean);
    }
    float std = std::sqrt(var_sum / t->size());
    TEST_ASSERT_NEAR(std, 1.0f, 0.1f); // Should be close to 1
}

void test_tensor_create_xavier() {
    auto t = Tensor::xavier(100, 100);
    TEST_ASSERT_EQ(t->requires_grad, true);

    // Xavier variance should be 2 / (fan_in + fan_out)
    float expected_std = std::sqrt(2.0f / 200.0f);

    float sum = 0.0f;
    for (size_t i = 0; i < t->size(); i++) {
        sum += t->data[i];
    }
    float mean = sum / t->size();

    float var_sum = 0.0f;
    for (size_t i = 0; i < t->size(); i++) {
        var_sum += (t->data[i] - mean) * (t->data[i] - mean);
    }
    float std = std::sqrt(var_sum / t->size());
    TEST_ASSERT_NEAR(std, expected_std, 0.05f);
}

void test_tensor_create_from_data() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t = Tensor::create(data, {2, 3});
    TEST_ASSERT_SHAPE(t, std::vector<size_t>({2, 3}));
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_NEAR(t->data[i], data[i], 1e-6f);
    }
}

// =============================================================================
// Basic Arithmetic Tests
// =============================================================================

void test_tensor_add() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f}, std::vector<size_t>{3});
    auto b = Tensor::create({4.0f, 5.0f, 6.0f}, std::vector<size_t>{3});
    auto c = a->add(b);

    TEST_ASSERT_NEAR(c->data[0], 5.0f, 1e-6f);
    TEST_ASSERT_NEAR(c->data[1], 7.0f, 1e-6f);
    TEST_ASSERT_NEAR(c->data[2], 9.0f, 1e-6f);
}

void test_tensor_sub() {
    auto a = Tensor::create({5.0f, 7.0f, 9.0f}, std::vector<size_t>{3});
    auto b = Tensor::create({1.0f, 2.0f, 3.0f}, std::vector<size_t>{3});
    auto c = a->sub(b);

    TEST_ASSERT_NEAR(c->data[0], 4.0f, 1e-6f);
    TEST_ASSERT_NEAR(c->data[1], 5.0f, 1e-6f);
    TEST_ASSERT_NEAR(c->data[2], 6.0f, 1e-6f);
}

void test_tensor_mul() {
    auto a = Tensor::create({2.0f, 3.0f, 4.0f}, std::vector<size_t>{3});
    auto b = Tensor::create({5.0f, 6.0f, 7.0f}, std::vector<size_t>{3});
    auto c = a->mul(b);

    TEST_ASSERT_NEAR(c->data[0], 10.0f, 1e-6f);
    TEST_ASSERT_NEAR(c->data[1], 18.0f, 1e-6f);
    TEST_ASSERT_NEAR(c->data[2], 28.0f, 1e-6f);
}

void test_tensor_div() {
    auto a = Tensor::create({10.0f, 20.0f, 30.0f}, std::vector<size_t>{3});
    auto b = Tensor::create({2.0f, 4.0f, 5.0f}, std::vector<size_t>{3});
    auto c = a->div(b);

    TEST_ASSERT_NEAR(c->data[0], 5.0f, 1e-6f);
    TEST_ASSERT_NEAR(c->data[1], 5.0f, 1e-6f);
    TEST_ASSERT_NEAR(c->data[2], 6.0f, 1e-6f);
}

void test_tensor_scalar_mul() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f}, std::vector<size_t>{3});
    auto b = a->mul(2.5f);

    TEST_ASSERT_NEAR(b->data[0], 2.5f, 1e-6f);
    TEST_ASSERT_NEAR(b->data[1], 5.0f, 1e-6f);
    TEST_ASSERT_NEAR(b->data[2], 7.5f, 1e-6f);
}

void test_tensor_scalar_div() {
    auto a = Tensor::create({10.0f, 20.0f, 30.0f}, std::vector<size_t>{3});
    auto b = a->div(2.0f);

    TEST_ASSERT_NEAR(b->data[0], 5.0f, 1e-6f);
    TEST_ASSERT_NEAR(b->data[1], 10.0f, 1e-6f);
    TEST_ASSERT_NEAR(b->data[2], 15.0f, 1e-6f);
}

void test_tensor_neg() {
    auto a = Tensor::create({1.0f, -2.0f, 3.0f}, std::vector<size_t>{3});
    auto b = a->neg();

    TEST_ASSERT_NEAR(b->data[0], -1.0f, 1e-6f);
    TEST_ASSERT_NEAR(b->data[1], 2.0f, 1e-6f);
    TEST_ASSERT_NEAR(b->data[2], -3.0f, 1e-6f);
}

// =============================================================================
// Matrix Operations Tests
// =============================================================================

void test_tensor_matmul_2d() {
    // [2, 3] x [3, 2] = [2, 2]
    auto a = Tensor::create({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    auto b = Tensor::create({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
    auto c = a->matmul(b);

    TEST_ASSERT_SHAPE(c, std::vector<size_t>({2, 2}));
    // Row 0: [1,2,3] dot [[1,2],[3,4],[5,6]] = [22, 28]
    // Row 1: [4,5,6] dot [[1,2],[3,4],[5,6]] = [49, 64]
    TEST_ASSERT_NEAR(c->data[0], 22.0f, 1e-5f);
    TEST_ASSERT_NEAR(c->data[1], 28.0f, 1e-5f);
    TEST_ASSERT_NEAR(c->data[2], 49.0f, 1e-5f);
    TEST_ASSERT_NEAR(c->data[3], 64.0f, 1e-5f);
}

void test_tensor_matmul_batch() {
    // Batch matrix multiplication: [2, 2, 3] x [2, 3, 2] = [2, 2, 2]
    auto a = Tensor::randn({2, 2, 3});
    auto b = Tensor::randn({2, 3, 2});
    auto c = a->bmm(b);  // Use bmm for batch matrix multiply

    TEST_ASSERT_SHAPE(c, std::vector<size_t>({2, 2, 2}));
}

void test_tensor_transpose() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    auto b = a->transpose();

    TEST_ASSERT_SHAPE(b, std::vector<size_t>({3, 2}));
    // Original: [[1,2,3], [4,5,6]]
    // Transposed: [[1,4], [2,5], [3,6]]
    TEST_ASSERT_NEAR(b->data[0], 1.0f, 1e-6f);
    TEST_ASSERT_NEAR(b->data[1], 4.0f, 1e-6f);
    TEST_ASSERT_NEAR(b->data[2], 2.0f, 1e-6f);
    TEST_ASSERT_NEAR(b->data[3], 5.0f, 1e-6f);
    TEST_ASSERT_NEAR(b->data[4], 3.0f, 1e-6f);
    TEST_ASSERT_NEAR(b->data[5], 6.0f, 1e-6f);
}

// =============================================================================
// Activation Functions Tests
// =============================================================================

void test_tensor_relu() {
    auto a = Tensor::create({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, std::vector<size_t>{5});
    auto b = a->relu();

    TEST_ASSERT_NEAR(b->data[0], 0.0f, 1e-6f);
    TEST_ASSERT_NEAR(b->data[1], 0.0f, 1e-6f);
    TEST_ASSERT_NEAR(b->data[2], 0.0f, 1e-6f);
    TEST_ASSERT_NEAR(b->data[3], 1.0f, 1e-6f);
    TEST_ASSERT_NEAR(b->data[4], 2.0f, 1e-6f);
}

void test_tensor_sigmoid() {
    auto a = Tensor::create({0.0f}, std::vector<size_t>{1});
    auto b = a->sigmoid();
    TEST_ASSERT_NEAR(b->data[0], 0.5f, 1e-6f);

    // Large positive -> ~1
    auto c = Tensor::create({10.0f}, std::vector<size_t>{1});
    auto d = c->sigmoid();
    TEST_ASSERT(d->data[0] > 0.999f);

    // Large negative -> ~0
    auto e = Tensor::create({-10.0f}, std::vector<size_t>{1});
    auto f = e->sigmoid();
    TEST_ASSERT(f->data[0] < 0.001f);
}

void test_tensor_tanh() {
    auto a = Tensor::create({0.0f}, std::vector<size_t>{1});
    auto b = a->tanh_();
    TEST_ASSERT_NEAR(b->data[0], 0.0f, 1e-6f);

    // Large positive -> ~1
    auto c = Tensor::create({10.0f}, std::vector<size_t>{1});
    auto d = c->tanh_();
    TEST_ASSERT(d->data[0] > 0.999f);

    // Large negative -> ~-1
    auto e = Tensor::create({-10.0f}, std::vector<size_t>{1});
    auto f = e->tanh_();
    TEST_ASSERT(f->data[0] < -0.999f);
}

void test_tensor_softmax() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f}, {1, 3});
    auto b = a->softmax(-1);

    // Softmax should sum to 1
    float sum = b->data[0] + b->data[1] + b->data[2];
    TEST_ASSERT_NEAR(sum, 1.0f, 1e-5f);

    // Values should be in increasing order
    TEST_ASSERT(b->data[0] < b->data[1]);
    TEST_ASSERT(b->data[1] < b->data[2]);
}

// =============================================================================
// Math Functions Tests
// =============================================================================

void test_tensor_exp() {
    auto a = Tensor::create({0.0f, 1.0f, 2.0f}, std::vector<size_t>{3});
    auto b = a->exp_();

    TEST_ASSERT_NEAR(b->data[0], 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(b->data[1], std::exp(1.0f), 1e-5f);
    TEST_ASSERT_NEAR(b->data[2], std::exp(2.0f), 1e-5f);
}

void test_tensor_log() {
    auto a = Tensor::create({1.0f, std::exp(1.0f), std::exp(2.0f)}, std::vector<size_t>{3});
    auto b = a->log_();

    TEST_ASSERT_NEAR(b->data[0], 0.0f, 1e-5f);
    TEST_ASSERT_NEAR(b->data[1], 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(b->data[2], 2.0f, 1e-5f);
}

void test_tensor_pow() {
    auto a = Tensor::create({2.0f, 3.0f, 4.0f}, std::vector<size_t>{3});
    auto b = a->pow(2.0f);

    TEST_ASSERT_NEAR(b->data[0], 4.0f, 1e-5f);
    TEST_ASSERT_NEAR(b->data[1], 9.0f, 1e-5f);
    TEST_ASSERT_NEAR(b->data[2], 16.0f, 1e-5f);
}

void test_tensor_sqrt() {
    auto a = Tensor::create({4.0f, 9.0f, 16.0f}, std::vector<size_t>{3});
    auto b = a->sqrt();

    TEST_ASSERT_NEAR(b->data[0], 2.0f, 1e-5f);
    TEST_ASSERT_NEAR(b->data[1], 3.0f, 1e-5f);
    TEST_ASSERT_NEAR(b->data[2], 4.0f, 1e-5f);
}

void test_tensor_abs() {
    auto a = Tensor::create({-2.0f, 0.0f, 3.0f}, std::vector<size_t>{3});
    auto b = a->abs();

    TEST_ASSERT_NEAR(b->data[0], 2.0f, 1e-5f);
    TEST_ASSERT_NEAR(b->data[1], 0.0f, 1e-5f);
    TEST_ASSERT_NEAR(b->data[2], 3.0f, 1e-5f);
}

void test_tensor_clamp() {
    auto a = Tensor::create({-5.0f, 0.0f, 5.0f, 10.0f}, std::vector<size_t>{4});
    auto b = a->clamp(0.0f, 6.0f);

    TEST_ASSERT_NEAR(b->data[0], 0.0f, 1e-5f);  // Clamped from -5
    TEST_ASSERT_NEAR(b->data[1], 0.0f, 1e-5f);  // Unchanged
    TEST_ASSERT_NEAR(b->data[2], 5.0f, 1e-5f);  // Unchanged
    TEST_ASSERT_NEAR(b->data[3], 6.0f, 1e-5f);  // Clamped from 10
}

// =============================================================================
// Reduction Tests
// =============================================================================

void test_tensor_sum() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});

    // Sum all
    auto sum_all = a->sum();
    TEST_ASSERT_NEAR(sum_all->data[0], 21.0f, 1e-5f);

    // Sum along dim 0
    auto sum_0 = a->sum(0);
    TEST_ASSERT_SHAPE(sum_0, std::vector<size_t>({3}));
    TEST_ASSERT_NEAR(sum_0->data[0], 5.0f, 1e-5f);  // 1+4
    TEST_ASSERT_NEAR(sum_0->data[1], 7.0f, 1e-5f);  // 2+5
    TEST_ASSERT_NEAR(sum_0->data[2], 9.0f, 1e-5f);  // 3+6

    // Sum along dim 1
    auto sum_1 = a->sum(1);
    TEST_ASSERT_SHAPE(sum_1, std::vector<size_t>({2}));
    TEST_ASSERT_NEAR(sum_1->data[0], 6.0f, 1e-5f);   // 1+2+3
    TEST_ASSERT_NEAR(sum_1->data[1], 15.0f, 1e-5f);  // 4+5+6
}

void test_tensor_mean() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});

    // Mean all
    auto mean_all = a->mean();
    TEST_ASSERT_NEAR(mean_all->data[0], 3.5f, 1e-5f);

    // Mean along dim 0
    auto mean_0 = a->mean(0);
    TEST_ASSERT_SHAPE(mean_0, std::vector<size_t>({3}));
    TEST_ASSERT_NEAR(mean_0->data[0], 2.5f, 1e-5f);  // (1+4)/2
    TEST_ASSERT_NEAR(mean_0->data[1], 3.5f, 1e-5f);  // (2+5)/2
    TEST_ASSERT_NEAR(mean_0->data[2], 4.5f, 1e-5f);  // (3+6)/2
}

void test_tensor_max() {
    auto a = Tensor::create({1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f}, {2, 3});

    // Max along dim 1
    auto max_1 = a->max(1);
    TEST_ASSERT_SHAPE(max_1, std::vector<size_t>({2}));
    TEST_ASSERT_NEAR(max_1->data[0], 5.0f, 1e-5f);  // max(1,5,3)
    TEST_ASSERT_NEAR(max_1->data[1], 6.0f, 1e-5f);  // max(4,2,6)
}

void test_tensor_min() {
    auto a = Tensor::create({1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f}, {2, 3});

    // Min along dim 1
    auto min_1 = a->min(1);
    TEST_ASSERT_SHAPE(min_1, std::vector<size_t>({2}));
    TEST_ASSERT_NEAR(min_1->data[0], 1.0f, 1e-5f);  // min(1,5,3)
    TEST_ASSERT_NEAR(min_1->data[1], 2.0f, 1e-5f);  // min(4,2,6)
}

void test_tensor_argmax() {
    auto a = Tensor::create({1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f}, {2, 3});

    auto argmax_1 = a->argmax(1);
    TEST_ASSERT_SHAPE(argmax_1, std::vector<size_t>({2}));
    TEST_ASSERT_NEAR(argmax_1->data[0], 1.0f, 1e-5f);  // index of 5
    TEST_ASSERT_NEAR(argmax_1->data[1], 2.0f, 1e-5f);  // index of 6
}

// =============================================================================
// Shape Manipulation Tests
// =============================================================================

void test_tensor_reshape() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    auto b = a->reshape({3, 2});

    TEST_ASSERT_SHAPE(b, std::vector<size_t>({3, 2}));
    TEST_ASSERT_EQ(b->size(), 6u);
    // Data should be preserved
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_NEAR(b->data[i], a->data[i], 1e-6f);
    }
}

void test_tensor_slice() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});

    // Slice first row
    auto row0 = a->slice(0, 1, 0);
    TEST_ASSERT_SHAPE(row0, std::vector<size_t>({1, 3}));
    TEST_ASSERT_NEAR(row0->data[0], 1.0f, 1e-6f);
    TEST_ASSERT_NEAR(row0->data[1], 2.0f, 1e-6f);
    TEST_ASSERT_NEAR(row0->data[2], 3.0f, 1e-6f);
}

void test_tensor_flatten() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    auto b = a->flatten(0);

    TEST_ASSERT_SHAPE(b, std::vector<size_t>({6}));
}

void test_tensor_squeeze() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f}, {1, 3, 1});
    auto b = a->squeeze();

    TEST_ASSERT_SHAPE(b, std::vector<size_t>({3}));
}

void test_tensor_unsqueeze() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f}, std::vector<size_t>{3});
    auto b = a->unsqueeze(0);

    TEST_ASSERT_SHAPE(b, std::vector<size_t>({1, 3}));
}

void test_tensor_permute() {
    auto a = Tensor::randn({2, 3, 4});
    auto b = a->permute({2, 0, 1});

    TEST_ASSERT_SHAPE(b, std::vector<size_t>({4, 2, 3}));
}

// =============================================================================
// Concatenation Tests
// =============================================================================

void test_tensor_concat() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f}, {1, 3});
    auto b = Tensor::create({4.0f, 5.0f, 6.0f}, {1, 3});
    auto c = Tensor::concat({a, b}, 0);

    TEST_ASSERT_SHAPE(c, std::vector<size_t>({2, 3}));
    TEST_ASSERT_NEAR(c->data[0], 1.0f, 1e-6f);
    TEST_ASSERT_NEAR(c->data[3], 4.0f, 1e-6f);
}

void test_tensor_stack() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f}, std::vector<size_t>{3});
    auto b = Tensor::create({4.0f, 5.0f, 6.0f}, std::vector<size_t>{3});
    auto c = Tensor::stack({a, b}, 0);

    TEST_ASSERT_SHAPE(c, std::vector<size_t>({2, 3}));
}

// =============================================================================
// Operator Tests
// =============================================================================

void test_tensor_operators() {
    auto a = Tensor::create({1.0f, 2.0f, 3.0f}, std::vector<size_t>{3});
    auto b = Tensor::create({4.0f, 5.0f, 6.0f}, std::vector<size_t>{3});

    // Operators are member functions on Tensor, so dereference TensorPtr
    auto c = (*a) + b;
    TEST_ASSERT_NEAR(c->data[0], 5.0f, 1e-6f);

    auto d = (*a) - b;
    TEST_ASSERT_NEAR(d->data[0], -3.0f, 1e-6f);

    auto e = (*a) * b;
    TEST_ASSERT_NEAR(e->data[0], 4.0f, 1e-6f);

    auto f = (*a) / b;
    TEST_ASSERT_NEAR(f->data[0], 0.25f, 1e-6f);

    auto g = -(*a);
    TEST_ASSERT_NEAR(g->data[0], -1.0f, 1e-6f);

    auto h = (*a) * 2.0f;
    TEST_ASSERT_NEAR(h->data[0], 2.0f, 1e-6f);
}

// =============================================================================
// Test Suite Registration
// =============================================================================

TestSuite* create_tensor_tests() {
    auto* suite = new TestSuite("Tensor Operations");

    // Creation tests
    suite->add_test("zeros", test_tensor_create_zeros);
    suite->add_test("ones", test_tensor_create_ones);
    suite->add_test("randn", test_tensor_create_randn);
    suite->add_test("xavier", test_tensor_create_xavier);
    suite->add_test("from_data", test_tensor_create_from_data);

    // Arithmetic tests
    suite->add_test("add", test_tensor_add);
    suite->add_test("sub", test_tensor_sub);
    suite->add_test("mul", test_tensor_mul);
    suite->add_test("div", test_tensor_div);
    suite->add_test("scalar_mul", test_tensor_scalar_mul);
    suite->add_test("scalar_div", test_tensor_scalar_div);
    suite->add_test("neg", test_tensor_neg);

    // Matrix operations
    suite->add_test("matmul_2d", test_tensor_matmul_2d);
    suite->add_test("matmul_batch", test_tensor_matmul_batch);
    suite->add_test("transpose", test_tensor_transpose);

    // Activation functions
    suite->add_test("relu", test_tensor_relu);
    suite->add_test("sigmoid", test_tensor_sigmoid);
    suite->add_test("tanh", test_tensor_tanh);
    suite->add_test("softmax", test_tensor_softmax);

    // Math functions
    suite->add_test("exp", test_tensor_exp);
    suite->add_test("log", test_tensor_log);
    suite->add_test("pow", test_tensor_pow);
    suite->add_test("sqrt", test_tensor_sqrt);
    suite->add_test("abs", test_tensor_abs);
    suite->add_test("clamp", test_tensor_clamp);

    // Reductions
    suite->add_test("sum", test_tensor_sum);
    suite->add_test("mean", test_tensor_mean);
    suite->add_test("max", test_tensor_max);
    suite->add_test("min", test_tensor_min);
    suite->add_test("argmax", test_tensor_argmax);

    // Shape manipulation
    suite->add_test("reshape", test_tensor_reshape);
    suite->add_test("slice", test_tensor_slice);
    suite->add_test("flatten", test_tensor_flatten);
    suite->add_test("squeeze", test_tensor_squeeze);
    suite->add_test("unsqueeze", test_tensor_unsqueeze);
    suite->add_test("permute", test_tensor_permute);

    // Concatenation
    suite->add_test("concat", test_tensor_concat);
    suite->add_test("stack", test_tensor_stack);

    // Operators
    suite->add_test("operators", test_tensor_operators);

    return suite;
}
