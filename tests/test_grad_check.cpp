#include "test_framework.h"
#include "../core/tensor.h"
#include "../core/layer.h"
#include "../core/loss.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <functional>

// =============================================================================
// Numerical Gradient Checking Utility
// =============================================================================

// Compute relative error between analytical and numerical gradient.
// Uses a floor of 1e-5 to avoid blow-up when both values are near zero.
// rel_error = |analytical - numerical| / max(|analytical|, |numerical|, 1e-5)
static float relative_error(float analytical, float numerical) {
    float denom = std::max(std::max(std::abs(analytical), std::abs(numerical)), 1e-5f);
    return std::abs(analytical - numerical) / denom;
}

// Numerically check gradients of a parameter tensor by finite differences.
//
// loss_fn: a callable that returns a scalar loss value (forward pass only).
//          It will be called many times with perturbed parameter values.
// param:   the parameter tensor whose gradients to check.
// analytical_grad: the gradient buffer from backward() for this parameter.
// eps:     perturbation size for finite differences.
// tol:     maximum acceptable relative error.
// atol:    absolute tolerance -- skip elements where both |analytical| and
//          |numerical| are below this threshold (avoids noise-vs-noise comparison).
// param_name: descriptive name for diagnostics.
//
// Returns true if all elements pass, false otherwise.
static bool check_grad_param(
    std::function<float()> loss_fn,
    TensorPtr param,
    const float* analytical_grad,
    float eps,
    float tol,
    const std::string& param_name,
    float atol = 1e-2f)
{
    float max_rel_err = 0.0f;
    size_t worst_idx = 0;
    float worst_analytical = 0.0f;
    float worst_numerical = 0.0f;

    for (size_t i = 0; i < param->size(); i++) {
        float original = param->data()[i];

        // f(x + eps)
        param->data()[i] = original + eps;
        float loss_plus;
        {
            NoGradGuard guard;
            loss_plus = loss_fn();
        }

        // f(x - eps)
        param->data()[i] = original - eps;
        float loss_minus;
        {
            NoGradGuard guard;
            loss_minus = loss_fn();
        }

        // Restore original value
        param->data()[i] = original;

        // Use double for the division to reduce rounding error
        double numerical_d = (static_cast<double>(loss_plus) - static_cast<double>(loss_minus))
                           / (2.0 * static_cast<double>(eps));
        float numerical = static_cast<float>(numerical_d);
        float analytical = analytical_grad[i];

        // Skip elements where both gradients are tiny (below absolute tolerance).
        // These are numerically indistinguishable from zero and comparing them
        // produces misleading relative errors from floating-point noise.
        if (std::abs(analytical) < atol && std::abs(numerical) < atol) {
            continue;
        }

        float rel_err = relative_error(analytical, numerical);

        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
            worst_idx = i;
            worst_analytical = analytical;
            worst_numerical = numerical;
        }
    }

    if (max_rel_err > tol) {
        printf("         GRAD CHECK FAILED for %s: max_rel_err=%.6e at idx=%zu "
               "(analytical=%.6e, numerical=%.6e)\n",
               param_name.c_str(), max_rel_err, worst_idx,
               worst_analytical, worst_numerical);
        return false;
    }
    return true;
}

// Helper: compute sum of all elements as a scalar float (double-precision accumulation).
// Used in numerical loss_fn to avoid the autograd sum() path and reduce rounding error.
static float sum_all(const TensorPtr& t) {
    double s = 0.0;
    for (size_t i = 0; i < t->size(); i++) {
        s += static_cast<double>(t->data()[i]);
    }
    return static_cast<float>(s);
}

// =============================================================================
// 1. Linear Layer Gradient Check
// =============================================================================

void test_gradcheck_linear() {
    const float eps = 1e-4f;
    const float tol = 5e-3f;

    Linear layer(4, 3);
    auto input = Tensor::create({0.5f, -0.3f, 0.8f, -0.1f,
                                  0.2f, 0.6f, -0.4f, 0.7f}, {2, 4}, true);

    // Analytical backward
    for (auto& p : layer.parameters()) p->zero_grad();
    auto output = layer.forward(input);
    auto loss = output->sum();
    loss->backward();

    std::vector<float> weight_grad_analytical(layer.weight->grad(),
        layer.weight->grad() + layer.weight->grad_size());
    std::vector<float> bias_grad_analytical(layer.bias->grad(),
        layer.bias->grad() + layer.bias->grad_size());

    // Numerical: use sum_all to avoid autograd sum() in the numerical pass
    auto loss_fn = [&]() -> float {
        auto out = layer.forward(input);
        return sum_all(out);
    };

    TEST_ASSERT_MSG(
        check_grad_param(loss_fn, layer.weight, weight_grad_analytical.data(), eps, tol, "Linear.weight"),
        "Linear weight gradient check failed");
    TEST_ASSERT_MSG(
        check_grad_param(loss_fn, layer.bias, bias_grad_analytical.data(), eps, tol, "Linear.bias"),
        "Linear bias gradient check failed");
}

// =============================================================================
// 2. Conv2d Layer Gradient Check
// =============================================================================

void test_gradcheck_conv2d() {
    const float eps = 1e-4f;
    const float tol = 5e-2f;  // Conv uses im2col + BLAS; needs wider tolerance

    Conv2d layer(1, 2, 3, 1, 1);  // in=1, out=2, kernel=3, stride=1, pad=1
    // Use deterministic small input with non-zero sum (avoids near-zero gradients
    // at the center kernel position that occur with zero-mean symmetric inputs)
    std::vector<float> input_data(25);
    for (int i = 0; i < 25; i++) input_data[i] = 0.1f * static_cast<float>(i) + 0.05f;
    auto input = Tensor::create(input_data, {1, 1, 5, 5}, true);

    // Analytical backward
    for (auto& p : layer.parameters()) p->zero_grad();
    auto output = layer.forward(input);
    auto loss = output->sum();
    loss->backward();

    std::vector<float> weight_grad_analytical(layer.weight->grad(),
        layer.weight->grad() + layer.weight->grad_size());
    std::vector<float> bias_grad_analytical(layer.bias->grad(),
        layer.bias->grad() + layer.bias->grad_size());

    auto loss_fn = [&]() -> float {
        auto out = layer.forward(input);
        return sum_all(out);
    };

    TEST_ASSERT_MSG(
        check_grad_param(loss_fn, layer.weight, weight_grad_analytical.data(), eps, tol, "Conv2d.weight"),
        "Conv2d weight gradient check failed");
    TEST_ASSERT_MSG(
        check_grad_param(loss_fn, layer.bias, bias_grad_analytical.data(), eps, tol, "Conv2d.bias"),
        "Conv2d bias gradient check failed");
}

// =============================================================================
// 3. BatchNorm2d Gradient Check
// =============================================================================

void test_gradcheck_batchnorm2d() {
    const float eps = 1e-3f;   // Larger eps for batchnorm stability
    const float tol = 5e-2f;   // Batchnorm backward involves statistics; needs tolerance

    BatchNorm2d layer(2);
    layer.train();
    // Use deterministic input with sufficient variance
    std::vector<float> input_data(2 * 2 * 3 * 3);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = 0.1f * static_cast<float>(i) - 1.8f;
    }
    auto input = Tensor::create(input_data, {2, 2, 3, 3}, true);

    // Analytical backward
    for (auto& p : layer.parameters()) p->zero_grad();
    auto output = layer.forward(input);
    auto loss = output->sum();
    loss->backward();

    std::vector<float> gamma_grad_analytical(layer.gamma->grad(),
        layer.gamma->grad() + layer.gamma->grad_size());
    std::vector<float> beta_grad_analytical(layer.beta->grad(),
        layer.beta->grad() + layer.beta->grad_size());

    // Save running stats to restore before each numerical forward
    std::vector<float> saved_running_mean(layer.running_mean->data(),
        layer.running_mean->data() + layer.running_mean->size());
    std::vector<float> saved_running_var(layer.running_var->data(),
        layer.running_var->data() + layer.running_var->size());

    auto loss_fn = [&]() -> float {
        std::copy(saved_running_mean.begin(), saved_running_mean.end(), layer.running_mean->data());
        std::copy(saved_running_var.begin(), saved_running_var.end(), layer.running_var->data());
        auto out = layer.forward(input);
        return sum_all(out);
    };

    TEST_ASSERT_MSG(
        check_grad_param(loss_fn, layer.gamma, gamma_grad_analytical.data(), eps, tol, "BatchNorm2d.gamma"),
        "BatchNorm2d gamma gradient check failed");
    TEST_ASSERT_MSG(
        check_grad_param(loss_fn, layer.beta, beta_grad_analytical.data(), eps, tol, "BatchNorm2d.beta"),
        "BatchNorm2d beta gradient check failed");
}

// =============================================================================
// 4. Cross-Entropy Loss Gradient Check (w.r.t. logits)
// =============================================================================

void test_gradcheck_cross_entropy() {
    const float eps = 1e-4f;
    const float tol = 5e-3f;

    CrossEntropyLoss loss_fn_obj;
    auto logits = Tensor::create({0.5f, -0.3f, 0.8f, -0.1f,
                                   0.2f, 0.6f, -0.4f, 0.7f}, {2, 4}, true);
    auto target = Tensor::create(std::vector<float>{1.0f, 3.0f}, std::vector<size_t>{2});

    // Analytical backward
    logits->zero_grad();
    auto loss = loss_fn_obj(logits, target);
    loss->backward();

    std::vector<float> logits_grad_analytical(logits->grad(),
        logits->grad() + logits->grad_size());

    auto loss_fn = [&]() -> float {
        auto l = loss_fn_obj(logits, target);
        return l->data()[0];
    };

    TEST_ASSERT_MSG(
        check_grad_param(loss_fn, logits, logits_grad_analytical.data(), eps, tol, "CrossEntropy.logits"),
        "Cross-entropy logits gradient check failed");
}

// =============================================================================
// 5. MSE Loss Gradient Check (w.r.t. predictions)
// =============================================================================

void test_gradcheck_mse_loss() {
    const float eps = 1e-4f;
    const float tol = 5e-2f;  // Relaxed for float32 with -ffast-math

    MSELoss loss_fn_obj;
    auto pred = Tensor::create({0.5f, -0.3f, 0.8f, -0.1f, 0.2f, 0.6f}, {2, 3}, true);
    auto target = Tensor::create({0.1f, 0.4f, -0.2f, 0.3f, -0.5f, 0.9f}, {2, 3});

    pred->zero_grad();
    auto loss = loss_fn_obj(pred, target);
    loss->backward();

    std::vector<float> pred_grad_analytical(pred->grad(),
        pred->grad() + pred->grad_size());

    auto loss_fn = [&]() -> float {
        auto l = loss_fn_obj(pred, target);
        return l->data()[0];
    };

    TEST_ASSERT_MSG(
        check_grad_param(loss_fn, pred, pred_grad_analytical.data(), eps, tol, "MSELoss.pred"),
        "MSE loss prediction gradient check failed");
}

// =============================================================================
// 6. Matmul Gradient Check
// =============================================================================

void test_gradcheck_matmul() {
    const float eps = 1e-4f;
    const float tol = 5e-3f;

    auto A = Tensor::create({0.5f, -0.3f, 0.8f,
                              0.2f, 0.6f, -0.4f}, {2, 3}, true);
    auto B = Tensor::create({0.1f, -0.5f,
                              0.3f, 0.7f,
                             -0.2f, 0.4f}, {3, 2}, true);

    // Analytical backward
    A->zero_grad();
    B->zero_grad();
    auto C = A->matmul(B);
    auto loss = C->sum();
    loss->backward();

    std::vector<float> A_grad_analytical(A->grad(), A->grad() + A->grad_size());
    std::vector<float> B_grad_analytical(B->grad(), B->grad() + B->grad_size());

    auto loss_fn = [&]() -> float {
        auto c = A->matmul(B);
        return sum_all(c);
    };

    TEST_ASSERT_MSG(
        check_grad_param(loss_fn, A, A_grad_analytical.data(), eps, tol, "Matmul.A"),
        "Matmul A gradient check failed");
    TEST_ASSERT_MSG(
        check_grad_param(loss_fn, B, B_grad_analytical.data(), eps, tol, "Matmul.B"),
        "Matmul B gradient check failed");
}

// =============================================================================
// 7a. ReLU Gradient Check
// =============================================================================

void test_gradcheck_relu() {
    const float eps = 1e-4f;
    const float tol = 5e-3f;

    // Avoid values near zero where ReLU is non-differentiable
    auto input = Tensor::create({-1.5f, 0.8f, -0.3f, 1.2f, 2.0f, -0.7f}, {2, 3}, true);

    input->zero_grad();
    auto output = input->relu();
    auto loss = output->sum();
    loss->backward();

    std::vector<float> input_grad_analytical(input->grad(),
        input->grad() + input->grad_size());

    auto loss_fn = [&]() -> float {
        auto out = input->relu();
        return sum_all(out);
    };

    TEST_ASSERT_MSG(
        check_grad_param(loss_fn, input, input_grad_analytical.data(), eps, tol, "ReLU.input"),
        "ReLU input gradient check failed");
}

// =============================================================================
// 7b. Sigmoid Gradient Check
// =============================================================================

void test_gradcheck_sigmoid() {
    const float eps = 1e-4f;
    const float tol = 5e-3f;

    auto input = Tensor::create({0.5f, -0.3f, 0.8f, -0.1f, 0.2f, 0.6f}, {2, 3}, true);

    input->zero_grad();
    auto output = input->sigmoid();
    auto loss = output->sum();
    loss->backward();

    std::vector<float> input_grad_analytical(input->grad(),
        input->grad() + input->grad_size());

    auto loss_fn = [&]() -> float {
        auto out = input->sigmoid();
        return sum_all(out);
    };

    TEST_ASSERT_MSG(
        check_grad_param(loss_fn, input, input_grad_analytical.data(), eps, tol, "Sigmoid.input"),
        "Sigmoid input gradient check failed");
}

// =============================================================================
// 7c. Tanh Gradient Check
// =============================================================================

void test_gradcheck_tanh() {
    const float eps = 1e-4f;
    const float tol = 5e-3f;

    auto input = Tensor::create({0.5f, -0.3f, 0.8f, -0.1f, 0.2f, 0.6f}, {2, 3}, true);

    input->zero_grad();
    auto output = input->tanh_();
    auto loss = output->sum();
    loss->backward();

    std::vector<float> input_grad_analytical(input->grad(),
        input->grad() + input->grad_size());

    auto loss_fn = [&]() -> float {
        auto out = input->tanh_();
        return sum_all(out);
    };

    TEST_ASSERT_MSG(
        check_grad_param(loss_fn, input, input_grad_analytical.data(), eps, tol, "Tanh.input"),
        "Tanh input gradient check failed");
}

// =============================================================================
// 8. MultiHeadAttention Gradient Check (W_q only for speed)
// =============================================================================

void test_gradcheck_mha() {
    const float eps = 1e-4f;
    const float tol = 5e-2f;  // MHA chains many ops; wider tolerance needed

    MultiHeadAttention mha(8, 2);  // embed_dim=8, num_heads=2
    // Use small deterministic input
    std::vector<float> input_data(1 * 4 * 8);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = 0.05f * static_cast<float>(i) - 0.8f;
    }
    auto input = Tensor::create(input_data, {1, 4, 8}, true);

    // Analytical backward
    for (auto& p : mha.parameters()) p->zero_grad();
    auto output = mha.forward(input);
    auto loss = output->sum();
    loss->backward();

    std::vector<float> wq_grad_analytical(mha.W_q->grad(),
        mha.W_q->grad() + mha.W_q->grad_size());

    auto loss_fn = [&]() -> float {
        auto out = mha.forward(input);
        return sum_all(out);
    };

    // Use larger atol (0.05) to skip near-zero gradient elements where numerical
    // noise from the many chained BLAS operations dominates the comparison.
    TEST_ASSERT_MSG(
        check_grad_param(loss_fn, mha.W_q, wq_grad_analytical.data(), eps, tol, "MHA.W_q", 5e-2f),
        "MultiHeadAttention W_q gradient check failed");
}

// =============================================================================
// Test Suite Registration
// =============================================================================

TestSuite* create_grad_check_tests() {
    auto* suite = new TestSuite("Gradient Checking");

    suite->add_test("gradcheck_linear", test_gradcheck_linear);
    suite->add_test("gradcheck_conv2d", test_gradcheck_conv2d);
    suite->add_test("gradcheck_batchnorm2d", test_gradcheck_batchnorm2d);
    suite->add_test("gradcheck_cross_entropy", test_gradcheck_cross_entropy);
    suite->add_test("gradcheck_mse_loss", test_gradcheck_mse_loss);
    suite->add_test("gradcheck_matmul", test_gradcheck_matmul);
    suite->add_test("gradcheck_relu", test_gradcheck_relu);
    suite->add_test("gradcheck_sigmoid", test_gradcheck_sigmoid);
    suite->add_test("gradcheck_tanh", test_gradcheck_tanh);
    suite->add_test("gradcheck_mha", test_gradcheck_mha);

    return suite;
}
