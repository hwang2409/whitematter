#include "tensor.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstring>

// SIMD headers
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
    #define USE_SIMD 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define USE_NEON 1
#endif

// ============================================================================
// GradMode implementation
// ============================================================================

bool GradMode::enabled_ = true;

bool GradMode::is_enabled() {
    return enabled_;
}

void GradMode::set_enabled(bool enabled) {
    enabled_ = enabled;
}

NoGradGuard::NoGradGuard() : prev_mode_(GradMode::is_enabled()) {
    GradMode::set_enabled(false);
}

NoGradGuard::~NoGradGuard() {
    GradMode::set_enabled(prev_mode_);
}

// ============================================================================
// SIMD helper functions
// ============================================================================

#ifdef USE_SIMD

// SIMD vector addition: dst = a + b
static void simd_add(float* dst, const float* a, const float* b, size_t n) {
    size_t i = 0;
    #ifdef __AVX__
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(dst + i, vc);
    }
    #endif
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_add_ps(va, vb);
        _mm_storeu_ps(dst + i, vc);
    }
    for (; i < n; i++) {
        dst[i] = a[i] + b[i];
    }
}

// SIMD vector subtraction: dst = a - b
static void simd_sub(float* dst, const float* a, const float* b, size_t n) {
    size_t i = 0;
    #ifdef __AVX__
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(dst + i, vc);
    }
    #endif
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_sub_ps(va, vb);
        _mm_storeu_ps(dst + i, vc);
    }
    for (; i < n; i++) {
        dst[i] = a[i] - b[i];
    }
}

// SIMD vector multiplication: dst = a * b
static void simd_mul(float* dst, const float* a, const float* b, size_t n) {
    size_t i = 0;
    #ifdef __AVX__
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(dst + i, vc);
    }
    #endif
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_mul_ps(va, vb);
        _mm_storeu_ps(dst + i, vc);
    }
    for (; i < n; i++) {
        dst[i] = a[i] * b[i];
    }
}

// SIMD scalar multiplication: dst = a * scalar
static void simd_scale(float* dst, const float* a, float scalar, size_t n) {
    size_t i = 0;
    #ifdef __AVX__
    __m256 vs = _mm256_set1_ps(scalar);
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vc = _mm256_mul_ps(va, vs);
        _mm256_storeu_ps(dst + i, vc);
    }
    #endif
    __m128 vs4 = _mm_set1_ps(scalar);
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vc = _mm_mul_ps(va, vs4);
        _mm_storeu_ps(dst + i, vc);
    }
    for (; i < n; i++) {
        dst[i] = a[i] * scalar;
    }
}

// SIMD ReLU: dst = max(0, a)
static void simd_relu(float* dst, const float* a, size_t n) {
    size_t i = 0;
    #ifdef __AVX__
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vc = _mm256_max_ps(va, zero);
        _mm256_storeu_ps(dst + i, vc);
    }
    #endif
    __m128 zero4 = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vc = _mm_max_ps(va, zero4);
        _mm_storeu_ps(dst + i, vc);
    }
    for (; i < n; i++) {
        dst[i] = a[i] > 0 ? a[i] : 0;
    }
}

// SIMD dot product
static float simd_dot(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    size_t i = 0;
    #ifdef __AVX__
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb));
    }
    float temp[8];
    _mm256_storeu_ps(temp, vsum);
    sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    #endif
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

#elif defined(USE_NEON)

static void simd_add(float* dst, const float* a, const float* b, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(dst + i, vaddq_f32(va, vb));
    }
    for (; i < n; i++) dst[i] = a[i] + b[i];
}

static void simd_sub(float* dst, const float* a, const float* b, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(dst + i, vsubq_f32(va, vb));
    }
    for (; i < n; i++) dst[i] = a[i] - b[i];
}

static void simd_mul(float* dst, const float* a, const float* b, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(dst + i, vmulq_f32(va, vb));
    }
    for (; i < n; i++) dst[i] = a[i] * b[i];
}

static void simd_scale(float* dst, const float* a, float scalar, size_t n) {
    size_t i = 0;
    float32x4_t vs = vdupq_n_f32(scalar);
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vst1q_f32(dst + i, vmulq_f32(va, vs));
    }
    for (; i < n; i++) dst[i] = a[i] * scalar;
}

static void simd_relu(float* dst, const float* a, size_t n) {
    size_t i = 0;
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vst1q_f32(dst + i, vmaxq_f32(va, zero));
    }
    for (; i < n; i++) dst[i] = a[i] > 0 ? a[i] : 0;
}

static float simd_dot(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    size_t i = 0;
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vsum = vmlaq_f32(vsum, va, vb);
    }
    float temp[4];
    vst1q_f32(temp, vsum);
    sum = temp[0] + temp[1] + temp[2] + temp[3];
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

#else

// Fallback non-SIMD implementations
static void simd_add(float* dst, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = a[i] + b[i];
}
static void simd_sub(float* dst, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = a[i] - b[i];
}
static void simd_mul(float* dst, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = a[i] * b[i];
}
static void simd_scale(float* dst, const float* a, float scalar, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = a[i] * scalar;
}
static void simd_relu(float* dst, const float* a, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = a[i] > 0 ? a[i] : 0;
}
static float simd_dot(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

#endif

// ============================================================================
// Blocked matrix multiplication (cache-friendly)
// ============================================================================

static constexpr size_t BLOCK_SIZE = 32;

static void matmul_blocked(float* C, const float* A, const float* B,
                           size_t M, size_t K, size_t N) {
    std::memset(C, 0, M * N * sizeof(float));

    for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        size_t imax = std::min(i0 + BLOCK_SIZE, M);
        for (size_t k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
            size_t kmax = std::min(k0 + BLOCK_SIZE, K);
            for (size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
                size_t jmax = std::min(j0 + BLOCK_SIZE, N);

                for (size_t i = i0; i < imax; i++) {
                    for (size_t k = k0; k < kmax; k++) {
                        float a_ik = A[i * K + k];
                        #if defined(USE_SIMD) && defined(__AVX__)
                        __m256 va = _mm256_set1_ps(a_ik);
                        size_t j = j0;
                        for (; j + 8 <= jmax; j += 8) {
                            __m256 vb = _mm256_loadu_ps(&B[k * N + j]);
                            __m256 vc = _mm256_loadu_ps(&C[i * N + j]);
                            vc = _mm256_add_ps(vc, _mm256_mul_ps(va, vb));
                            _mm256_storeu_ps(&C[i * N + j], vc);
                        }
                        for (; j < jmax; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                        #elif defined(USE_NEON)
                        float32x4_t va = vdupq_n_f32(a_ik);
                        size_t j = j0;
                        for (; j + 4 <= jmax; j += 4) {
                            float32x4_t vb = vld1q_f32(&B[k * N + j]);
                            float32x4_t vc = vld1q_f32(&C[i * N + j]);
                            vc = vmlaq_f32(vc, va, vb);
                            vst1q_f32(&C[i * N + j], vc);
                        }
                        for (; j < jmax; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                        #else
                        for (size_t j = j0; j < jmax; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                        #endif
                    }
                }
            }
        }
    }
}

// ============================================================================
// Tensor implementation
// ============================================================================

static std::mt19937 rng(42);

Tensor::Tensor() : requires_grad(false) {}

Tensor::Tensor(const std::vector<size_t>& shape, bool requires_grad)
    : shape(shape), requires_grad(requires_grad) {
    size_t total = 1;
    for (auto s : shape) total *= s;
    data.resize(total, 0.0f);
    if (requires_grad) grad.resize(total, 0.0f);
}

Tensor::Tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad)
    : data(data), shape(shape), requires_grad(requires_grad) {
    if (requires_grad) grad.resize(data.size(), 0.0f);
}

TensorPtr Tensor::create(const std::vector<size_t>& shape, bool requires_grad) {
    return std::make_shared<Tensor>(shape, requires_grad);
}

TensorPtr Tensor::create(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad) {
    return std::make_shared<Tensor>(data, shape, requires_grad);
}

TensorPtr Tensor::zeros(const std::vector<size_t>& shape, bool requires_grad) {
    auto t = create(shape, requires_grad);
    std::fill(t->data.begin(), t->data.end(), 0.0f);
    return t;
}

TensorPtr Tensor::ones(const std::vector<size_t>& shape, bool requires_grad) {
    auto t = create(shape, requires_grad);
    std::fill(t->data.begin(), t->data.end(), 1.0f);
    return t;
}

TensorPtr Tensor::randn(const std::vector<size_t>& shape, bool requires_grad) {
    auto t = create(shape, requires_grad);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : t->data) v = dist(rng);
    return t;
}

TensorPtr Tensor::xavier(size_t fan_in, size_t fan_out, bool requires_grad) {
    auto t = create({fan_in, fan_out}, requires_grad);
    float std = std::sqrt(2.0f / (fan_in + fan_out));
    std::normal_distribution<float> dist(0.0f, std);
    for (auto& v : t->data) v = dist(rng);
    return t;
}

size_t Tensor::size() const { return data.size(); }
size_t Tensor::ndim() const { return shape.size(); }

float Tensor::item() const {
    assert(data.size() == 1);
    return data[0];
}

void Tensor::zero_grad() {
    std::fill(grad.begin(), grad.end(), 0.0f);
}

float& Tensor::operator[](size_t idx) { return data[idx]; }
float Tensor::operator[](size_t idx) const { return data[idx]; }

float& Tensor::at(const std::vector<size_t>& indices) {
    size_t idx = 0, stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        idx += indices[i] * stride;
        stride *= shape[i];
    }
    return data[idx];
}

float Tensor::at(const std::vector<size_t>& indices) const {
    size_t idx = 0, stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        idx += indices[i] * stride;
        stride *= shape[i];
    }
    return data[idx];
}

bool Tensor::should_track_grad() const {
    return requires_grad && GradMode::is_enabled();
}

void Tensor::build_topo(std::vector<Tensor*>& topo, std::vector<Tensor*>& visited) {
    if (std::find(visited.begin(), visited.end(), this) != visited.end()) return;
    visited.push_back(this);
    for (auto& p : parents) {
        p->build_topo(topo, visited);
    }
    topo.push_back(this);
}

void Tensor::backward() {
    assert(data.size() == 1);
    if (grad.empty()) grad.resize(1, 0.0f);
    grad[0] = 1.0f;

    std::vector<Tensor*> topo, visited;
    build_topo(topo, visited);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->grad_fn) (*it)->grad_fn();
    }
}

TensorPtr Tensor::matmul(const TensorPtr& other) const {
    assert(shape.size() == 2 && other->shape.size() == 2);
    assert(shape[1] == other->shape[0]);

    size_t m = shape[0], k = shape[1], n = other->shape[1];
    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();
    auto result = create({m, n}, track);

    matmul_blocked(result->data.data(), data.data(), other->data.data(), m, k, n);

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, m, k, n]() {
            if (self_ptr->requires_grad) {
                // dL/dA = dL/dC @ B^T
                for (size_t i = 0; i < m; i++) {
                    for (size_t l = 0; l < k; l++) {
                        self_ptr->grad[i * k + l] += simd_dot(
                            &result->grad[i * n], &other_ptr->data[l * n], n);
                    }
                }
            }
            if (other_ptr->requires_grad) {
                // dL/dB = A^T @ dL/dC
                for (size_t l = 0; l < k; l++) {
                    for (size_t j = 0; j < n; j++) {
                        float sum = 0.0f;
                        for (size_t i = 0; i < m; i++) {
                            sum += self_ptr->data[i * k + l] * result->grad[i * n + j];
                        }
                        other_ptr->grad[l * n + j] += sum;
                    }
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::add(const TensorPtr& other) const {
    assert(data.size() == other->data.size() || other->data.size() == shape.back());

    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();
    auto result = create(shape, track);

    if (data.size() == other->data.size()) {
        simd_add(result->data.data(), data.data(), other->data.data(), data.size());
    } else {
        size_t cols = shape.back();
        for (size_t i = 0; i < data.size(); i += cols) {
            simd_add(&result->data[i], &data[i], other->data.data(), cols);
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result]() {
            if (self_ptr->requires_grad) {
                simd_add(self_ptr->grad.data(), self_ptr->grad.data(),
                         result->grad.data(), self_ptr->data.size());
            }
            if (other_ptr->requires_grad) {
                if (other_ptr->data.size() == result->data.size()) {
                    simd_add(other_ptr->grad.data(), other_ptr->grad.data(),
                             result->grad.data(), other_ptr->data.size());
                } else {
                    size_t cols = other_ptr->data.size();
                    for (size_t i = 0; i < result->data.size(); i++) {
                        other_ptr->grad[i % cols] += result->grad[i];
                    }
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::sub(const TensorPtr& other) const {
    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();
    auto result = create(shape, track);

    simd_sub(result->data.data(), data.data(), other->data.data(), data.size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result]() {
            if (self_ptr->requires_grad) {
                simd_add(self_ptr->grad.data(), self_ptr->grad.data(),
                         result->grad.data(), self_ptr->data.size());
            }
            if (other_ptr->requires_grad) {
                for (size_t i = 0; i < other_ptr->data.size(); i++) {
                    other_ptr->grad[i] -= result->grad[i];
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::mul(const TensorPtr& other) const {
    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();
    auto result = create(shape, track);

    simd_mul(result->data.data(), data.data(), other->data.data(), data.size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result]() {
            if (self_ptr->requires_grad) {
                for (size_t i = 0; i < self_ptr->data.size(); i++) {
                    self_ptr->grad[i] += result->grad[i] * other_ptr->data[i];
                }
            }
            if (other_ptr->requires_grad) {
                for (size_t i = 0; i < other_ptr->data.size(); i++) {
                    other_ptr->grad[i] += result->grad[i] * self_ptr->data[i];
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::mul(float scalar) const {
    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    simd_scale(result->data.data(), data.data(), scalar, data.size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, scalar]() {
            for (size_t i = 0; i < self_ptr->data.size(); i++) {
                self_ptr->grad[i] += result->grad[i] * scalar;
            }
        };
    }
    return result;
}

TensorPtr Tensor::div(float scalar) const {
    return mul(1.0f / scalar);
}

TensorPtr Tensor::neg() const {
    return mul(-1.0f);
}

TensorPtr Tensor::relu() const {
    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    simd_relu(result->data.data(), data.data(), data.size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            for (size_t i = 0; i < self_ptr->data.size(); i++) {
                self_ptr->grad[i] += result->grad[i] * (self_ptr->data[i] > 0 ? 1.0f : 0.0f);
            }
        };
    }
    return result;
}

TensorPtr Tensor::sigmoid() const {
    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    for (size_t i = 0; i < data.size(); i++) {
        result->data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            for (size_t i = 0; i < self_ptr->data.size(); i++) {
                float s = result->data[i];
                self_ptr->grad[i] += result->grad[i] * s * (1.0f - s);
            }
        };
    }
    return result;
}

TensorPtr Tensor::tanh_() const {
    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    for (size_t i = 0; i < data.size(); i++) {
        result->data[i] = std::tanh(data[i]);
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            for (size_t i = 0; i < self_ptr->data.size(); i++) {
                float t = result->data[i];
                self_ptr->grad[i] += result->grad[i] * (1.0f - t * t);
            }
        };
    }
    return result;
}

TensorPtr Tensor::log_() const {
    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    for (size_t i = 0; i < data.size(); i++) {
        result->data[i] = std::log(data[i] + 1e-8f);
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            for (size_t i = 0; i < self_ptr->data.size(); i++) {
                self_ptr->grad[i] += result->grad[i] / (self_ptr->data[i] + 1e-8f);
            }
        };
    }
    return result;
}

TensorPtr Tensor::exp_() const {
    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    for (size_t i = 0; i < data.size(); i++) {
        result->data[i] = std::exp(data[i]);
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            for (size_t i = 0; i < self_ptr->data.size(); i++) {
                self_ptr->grad[i] += result->grad[i] * result->data[i];
            }
        };
    }
    return result;
}

TensorPtr Tensor::softmax(int dim) const {
    assert(shape.size() == 2);
    if (dim < 0) dim = shape.size() + dim;

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);
    size_t rows = shape[0], cols = shape[1];

    for (size_t i = 0; i < rows; i++) {
        float max_val = data[i * cols];
        for (size_t j = 1; j < cols; j++) {
            max_val = std::max(max_val, data[i * cols + j]);
        }

        float sum = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            result->data[i * cols + j] = std::exp(data[i * cols + j] - max_val);
            sum += result->data[i * cols + j];
        }

        float inv_sum = 1.0f / sum;
        for (size_t j = 0; j < cols; j++) {
            result->data[i * cols + j] *= inv_sum;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, rows, cols]() {
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    float sj = result->data[i * cols + j];
                    for (size_t k = 0; k < cols; k++) {
                        float sk = result->data[i * cols + k];
                        float grad = (j == k) ? sj * (1.0f - sj) : -sj * sk;
                        self_ptr->grad[i * cols + j] += result->grad[i * cols + k] * grad;
                    }
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::log_softmax(int dim) const {
    assert(shape.size() == 2);
    if (dim < 0) dim = shape.size() + dim;

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);
    size_t rows = shape[0], cols = shape[1];

    for (size_t i = 0; i < rows; i++) {
        float max_val = data[i * cols];
        for (size_t j = 1; j < cols; j++) {
            max_val = std::max(max_val, data[i * cols + j]);
        }

        float sum = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            sum += std::exp(data[i * cols + j] - max_val);
        }
        float log_sum = max_val + std::log(sum);

        for (size_t j = 0; j < cols; j++) {
            result->data[i * cols + j] = data[i * cols + j] - log_sum;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, rows, cols]() {
            for (size_t i = 0; i < rows; i++) {
                float grad_sum = 0.0f;
                for (size_t j = 0; j < cols; j++) {
                    grad_sum += result->grad[i * cols + j];
                }
                for (size_t j = 0; j < cols; j++) {
                    float softmax_j = std::exp(result->data[i * cols + j]);
                    self_ptr->grad[i * cols + j] += result->grad[i * cols + j] - softmax_j * grad_sum;
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::sum(int dim, bool keepdim) const {
    if (dim == -1) {
        bool track = requires_grad && GradMode::is_enabled();
        auto result = create({1}, track);
        result->data[0] = 0.0f;
        for (size_t i = 0; i < data.size(); i++) {
            result->data[0] += data[i];
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result]() {
                for (size_t i = 0; i < self_ptr->data.size(); i++) {
                    self_ptr->grad[i] += result->grad[0];
                }
            };
        }
        return result;
    }

    assert(shape.size() == 2);
    size_t rows = shape[0], cols = shape[1];
    bool track = requires_grad && GradMode::is_enabled();

    if (dim == 0) {
        auto result = keepdim ? create({1, cols}, track) : create({cols}, track);
        for (size_t j = 0; j < cols; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < rows; i++) {
                sum += data[i * cols + j];
            }
            result->data[j] = sum;
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result, rows, cols]() {
                for (size_t i = 0; i < rows; i++) {
                    for (size_t j = 0; j < cols; j++) {
                        self_ptr->grad[i * cols + j] += result->grad[j];
                    }
                }
            };
        }
        return result;
    } else {
        auto result = keepdim ? create({rows, 1}, track) : create({rows}, track);
        for (size_t i = 0; i < rows; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < cols; j++) {
                sum += data[i * cols + j];
            }
            result->data[i] = sum;
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result, rows, cols]() {
                for (size_t i = 0; i < rows; i++) {
                    for (size_t j = 0; j < cols; j++) {
                        self_ptr->grad[i * cols + j] += result->grad[i];
                    }
                }
            };
        }
        return result;
    }
}

TensorPtr Tensor::mean(int dim, bool keepdim) const {
    if (dim == -1) {
        auto result = sum(-1, keepdim);
        float n = static_cast<float>(data.size());
        result->data[0] /= n;
        return result;
    }

    auto result = sum(dim, keepdim);
    float n = static_cast<float>(dim == 0 ? shape[0] : shape[1]);
    for (auto& v : result->data) v /= n;
    return result;
}

TensorPtr Tensor::transpose() const {
    assert(shape.size() == 2);
    size_t rows = shape[0], cols = shape[1];

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create({cols, rows}, track);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result->data[j * rows + i] = data[i * cols + j];
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, rows, cols]() {
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    self_ptr->grad[i * cols + j] += result->grad[j * rows + i];
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t total = 1;
    for (auto s : new_shape) total *= s;
    assert(total == data.size());

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(data, new_shape, track);
    if (track) result->grad = grad;

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            simd_add(self_ptr->grad.data(), self_ptr->grad.data(),
                     result->grad.data(), self_ptr->data.size());
        };
    }
    return result;
}

TensorPtr Tensor::slice(size_t start, size_t end, int dim) const {
    assert(shape.size() == 2);
    assert(dim == 0);

    size_t cols = shape[1];
    size_t num_rows = end - start;

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create({num_rows, cols}, track);

    std::memcpy(result->data.data(), &data[start * cols], num_rows * cols * sizeof(float));

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, start, cols, num_rows]() {
            for (size_t i = 0; i < num_rows * cols; i++) {
                self_ptr->grad[start * cols + i] += result->grad[i];
            }
        };
    }
    return result;
}

void Tensor::print(const char* name) const {
    if (name) printf("%s: ", name);
    printf("Tensor(shape=[");
    for (size_t i = 0; i < shape.size(); i++) {
        printf("%zu%s", shape[i], i < shape.size() - 1 ? ", " : "");
    }
    printf("]");

    if (data.size() <= 10) {
        printf(", data=[");
        for (size_t i = 0; i < data.size(); i++) {
            printf("%.4f%s", data[i], i < data.size() - 1 ? ", " : "");
        }
        printf("]");
    }
    printf(")\n");
}

TensorPtr operator*(float scalar, const TensorPtr& t) {
    return t->mul(scalar);
}
