#include "tensor.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <limits>

// OpenMP for parallelization
#ifdef _OPENMP
#include <omp.h>
#endif

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

    // Parallelize over row blocks - each thread handles different output rows
    #pragma omp parallel for schedule(static)
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
// Broadcasting utilities
// ============================================================================

// Compute broadcast shape from two input shapes
static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a,
                                           const std::vector<size_t>& b) {
    size_t ndim = std::max(a.size(), b.size());
    std::vector<size_t> result(ndim);

    for (size_t i = 0; i < ndim; i++) {
        size_t dim_a = (i < ndim - a.size()) ? 1 : a[i - (ndim - a.size())];
        size_t dim_b = (i < ndim - b.size()) ? 1 : b[i - (ndim - b.size())];

        if (dim_a == dim_b) {
            result[i] = dim_a;
        } else if (dim_a == 1) {
            result[i] = dim_b;
        } else if (dim_b == 1) {
            result[i] = dim_a;
        } else {
            assert(false && "Shapes are not broadcastable");
        }
    }
    return result;
}

// Check if two shapes are broadcastable
static bool is_broadcastable(const std::vector<size_t>& a,
                             const std::vector<size_t>& b) {
    size_t ndim = std::max(a.size(), b.size());
    for (size_t i = 0; i < ndim; i++) {
        size_t dim_a = (i < ndim - a.size()) ? 1 : a[i - (ndim - a.size())];
        size_t dim_b = (i < ndim - b.size()) ? 1 : b[i - (ndim - b.size())];
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }
    return true;
}

// Compute linear index in source tensor given index in broadcast result
static size_t broadcast_index(const std::vector<size_t>& idx,
                              const std::vector<size_t>& src_shape,
                              const std::vector<size_t>& src_strides,
                              size_t ndim) {
    size_t src_ndim = src_shape.size();
    size_t linear = 0;
    for (size_t i = 0; i < src_ndim; i++) {
        size_t broadcast_dim = ndim - src_ndim + i;
        size_t src_idx = (src_shape[i] == 1) ? 0 : idx[broadcast_dim];
        linear += src_idx * src_strides[i];
    }
    return linear;
}

// Compute strides for a shape
static std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
    if (shape.empty()) return {};
    std::vector<size_t> strides(shape.size());
    strides[shape.size() - 1] = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
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

TensorPtr Tensor::concat(const std::vector<TensorPtr>& tensors, int dim) {
    assert(!tensors.empty() && "concat requires at least one tensor");

    // Handle negative dimension
    int ndim = static_cast<int>(tensors[0]->shape.size());
    if (dim < 0) dim += ndim;
    assert(dim >= 0 && dim < ndim && "Invalid dimension for concat");

    // Verify all tensors have compatible shapes
    std::vector<size_t> result_shape = tensors[0]->shape;
    size_t concat_size = tensors[0]->shape[dim];

    for (size_t i = 1; i < tensors.size(); i++) {
        assert(tensors[i]->shape.size() == tensors[0]->shape.size() &&
               "All tensors must have same number of dimensions");
        for (int d = 0; d < ndim; d++) {
            if (d == dim) {
                concat_size += tensors[i]->shape[d];
            } else {
                assert(tensors[i]->shape[d] == tensors[0]->shape[d] &&
                       "Tensor shapes must match except in concat dimension");
            }
        }
    }
    result_shape[dim] = concat_size;

    // Check if any input requires grad
    bool track = false;
    for (const auto& t : tensors) {
        if (t->requires_grad) track = true;
    }
    track = track && GradMode::is_enabled();

    auto result = create(result_shape, track);

    // Compute strides for the result tensor
    std::vector<size_t> strides(ndim);
    strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--) {
        strides[d] = strides[d + 1] * result_shape[d + 1];
    }

    // Copy data from each tensor
    size_t offset_in_dim = 0;
    for (size_t t_idx = 0; t_idx < tensors.size(); t_idx++) {
        const auto& t = tensors[t_idx];
        size_t t_dim_size = t->shape[dim];

        // Compute strides for input tensor
        std::vector<size_t> t_strides(ndim);
        t_strides[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; d--) {
            t_strides[d] = t_strides[d + 1] * t->shape[d + 1];
        }

        // Copy elements
        for (size_t i = 0; i < t->data.size(); i++) {
            // Convert linear index to multi-dimensional index
            std::vector<size_t> idx(ndim);
            size_t tmp = i;
            for (int d = 0; d < ndim; d++) {
                idx[d] = tmp / t_strides[d];
                tmp %= t_strides[d];
            }

            // Adjust index in concat dimension
            idx[dim] += offset_in_dim;

            // Convert back to linear index in result
            size_t result_idx = 0;
            for (int d = 0; d < ndim; d++) {
                result_idx += idx[d] * strides[d];
            }

            result->data[result_idx] = t->data[i];
        }

        offset_in_dim += t_dim_size;
    }

    if (track) {
        result->parents = tensors;
        std::vector<size_t> dim_offsets;
        size_t off = 0;
        for (const auto& t : tensors) {
            dim_offsets.push_back(off);
            off += t->shape[dim];
        }

        result->grad_fn = [tensors, result, dim, ndim, strides, dim_offsets]() {
            for (size_t t_idx = 0; t_idx < tensors.size(); t_idx++) {
                const auto& t = tensors[t_idx];
                if (!t->requires_grad) continue;

                size_t offset_in_dim = dim_offsets[t_idx];

                // Compute strides for input tensor
                std::vector<size_t> t_strides(ndim);
                t_strides[ndim - 1] = 1;
                for (int d = ndim - 2; d >= 0; d--) {
                    t_strides[d] = t_strides[d + 1] * t->shape[d + 1];
                }

                // Copy gradients back
                for (size_t i = 0; i < t->data.size(); i++) {
                    // Convert linear index to multi-dimensional index
                    std::vector<size_t> idx(ndim);
                    size_t tmp = i;
                    for (int d = 0; d < ndim; d++) {
                        idx[d] = tmp / t_strides[d];
                        tmp %= t_strides[d];
                    }

                    // Adjust index in concat dimension
                    idx[dim] += offset_in_dim;

                    // Convert back to linear index in result
                    size_t result_idx = 0;
                    for (int d = 0; d < ndim; d++) {
                        result_idx += idx[d] * strides[d];
                    }

                    t->grad[i] += result->grad[result_idx];
                }
            }
        };
    }

    return result;
}

TensorPtr Tensor::stack(const std::vector<TensorPtr>& tensors, int dim) {
    assert(!tensors.empty() && "stack requires at least one tensor");

    // Handle negative dimension
    int ndim = static_cast<int>(tensors[0]->shape.size());
    if (dim < 0) dim += ndim + 1;
    assert(dim >= 0 && dim <= ndim && "Invalid dimension for stack");

    // Verify all tensors have the same shape
    for (size_t i = 1; i < tensors.size(); i++) {
        assert(tensors[i]->shape == tensors[0]->shape &&
               "All tensors must have the same shape for stack");
    }

    // Create result shape with new dimension
    std::vector<size_t> result_shape;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) {
            result_shape.push_back(tensors.size());
        }
        result_shape.push_back(tensors[0]->shape[d]);
    }
    if (dim == ndim) {
        result_shape.push_back(tensors.size());
    }

    // Check if any input requires grad
    bool track = false;
    for (const auto& t : tensors) {
        if (t->requires_grad) track = true;
    }
    track = track && GradMode::is_enabled();

    auto result = create(result_shape, track);

    int new_ndim = ndim + 1;

    // Compute strides for result tensor
    std::vector<size_t> strides(new_ndim);
    strides[new_ndim - 1] = 1;
    for (int d = new_ndim - 2; d >= 0; d--) {
        strides[d] = strides[d + 1] * result_shape[d + 1];
    }

    // Compute strides for input tensors
    std::vector<size_t> t_strides(ndim);
    if (ndim > 0) {
        t_strides[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; d--) {
            t_strides[d] = t_strides[d + 1] * tensors[0]->shape[d + 1];
        }
    }

    // Copy data from each tensor
    for (size_t t_idx = 0; t_idx < tensors.size(); t_idx++) {
        const auto& t = tensors[t_idx];

        for (size_t i = 0; i < t->data.size(); i++) {
            // Convert linear index to multi-dimensional index in input tensor
            std::vector<size_t> t_idx_vec(ndim);
            size_t tmp = i;
            for (int d = 0; d < ndim; d++) {
                t_idx_vec[d] = tmp / t_strides[d];
                tmp %= t_strides[d];
            }

            // Create index for result tensor (insert stack dimension)
            std::vector<size_t> result_idx_vec(new_ndim);
            int t_d = 0;
            for (int d = 0; d < new_ndim; d++) {
                if (d == dim) {
                    result_idx_vec[d] = t_idx;
                } else {
                    result_idx_vec[d] = t_idx_vec[t_d++];
                }
            }

            // Convert to linear index in result
            size_t result_idx = 0;
            for (int d = 0; d < new_ndim; d++) {
                result_idx += result_idx_vec[d] * strides[d];
            }

            result->data[result_idx] = t->data[i];
        }
    }

    if (track) {
        result->parents = tensors;

        result->grad_fn = [tensors, result, dim, ndim, new_ndim, strides, t_strides]() {
            for (size_t t_idx = 0; t_idx < tensors.size(); t_idx++) {
                const auto& t = tensors[t_idx];
                if (!t->requires_grad) continue;

                for (size_t i = 0; i < t->data.size(); i++) {
                    // Convert linear index to multi-dimensional index in input tensor
                    std::vector<size_t> t_idx_vec(ndim);
                    size_t tmp = i;
                    for (int d = 0; d < ndim; d++) {
                        t_idx_vec[d] = tmp / t_strides[d];
                        tmp %= t_strides[d];
                    }

                    // Create index for result tensor (insert stack dimension)
                    std::vector<size_t> result_idx_vec(new_ndim);
                    int t_d = 0;
                    for (int d = 0; d < new_ndim; d++) {
                        if (d == dim) {
                            result_idx_vec[d] = t_idx;
                        } else {
                            result_idx_vec[d] = t_idx_vec[t_d++];
                        }
                    }

                    // Convert to linear index in result
                    size_t result_idx = 0;
                    for (int d = 0; d < new_ndim; d++) {
                        result_idx += result_idx_vec[d] * strides[d];
                    }

                    t->grad[i] += result->grad[result_idx];
                }
            }
        };
    }

    return result;
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

    // Clear computation graph to free memory (like PyTorch's default behavior)
    // This releases references to intermediate tensors
    for (auto* t : topo) {
        t->grad_fn = nullptr;
        t->parents.clear();
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

TensorPtr Tensor::bmm(const TensorPtr& other) const {
    // Batch matrix multiplication: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
    assert(shape.size() == 3 && other->shape.size() == 3 && "bmm requires 3D tensors");
    assert(shape[0] == other->shape[0] && "Batch sizes must match");
    assert(shape[2] == other->shape[1] && "Inner dimensions must match");

    size_t batch = shape[0];
    size_t m = shape[1];
    size_t k = shape[2];
    size_t n = other->shape[2];

    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();
    auto result = create({batch, m, n}, track);

    // Perform batched matmul
    size_t a_batch_stride = m * k;
    size_t b_batch_stride = k * n;
    size_t c_batch_stride = m * n;

    for (size_t b_idx = 0; b_idx < batch; b_idx++) {
        const float* a_ptr = &data[b_idx * a_batch_stride];
        const float* b_ptr = &other->data[b_idx * b_batch_stride];
        float* c_ptr = &result->data[b_idx * c_batch_stride];

        // Use blocked matmul for each batch
        matmul_blocked(c_ptr, a_ptr, b_ptr, m, k, n);
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, batch, m, k, n,
                          a_batch_stride, b_batch_stride, c_batch_stride]() {
            for (size_t b_idx = 0; b_idx < batch; b_idx++) {
                if (self_ptr->requires_grad) {
                    // dL/dA = dL/dC @ B^T
                    float* a_grad = &self_ptr->grad[b_idx * a_batch_stride];
                    const float* c_grad = &result->grad[b_idx * c_batch_stride];
                    const float* b_data = &other_ptr->data[b_idx * b_batch_stride];

                    for (size_t i = 0; i < m; i++) {
                        for (size_t l = 0; l < k; l++) {
                            float sum = 0.0f;
                            for (size_t j = 0; j < n; j++) {
                                sum += c_grad[i * n + j] * b_data[l * n + j];
                            }
                            a_grad[i * k + l] += sum;
                        }
                    }
                }
                if (other_ptr->requires_grad) {
                    // dL/dB = A^T @ dL/dC
                    const float* a_data = &self_ptr->data[b_idx * a_batch_stride];
                    const float* c_grad = &result->grad[b_idx * c_batch_stride];
                    float* b_grad = &other_ptr->grad[b_idx * b_batch_stride];

                    for (size_t l = 0; l < k; l++) {
                        for (size_t j = 0; j < n; j++) {
                            float sum = 0.0f;
                            for (size_t i = 0; i < m; i++) {
                                sum += a_data[i * k + l] * c_grad[i * n + j];
                            }
                            b_grad[l * n + j] += sum;
                        }
                    }
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::add(const TensorPtr& other) const {
    assert(is_broadcastable(shape, other->shape) && "Shapes are not broadcastable");

    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();

    // Fast path: same shape
    if (shape == other->shape) {
        auto result = create(shape, track);
        simd_add(result->data.data(), data.data(), other->data.data(), data.size());

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
                    simd_add(other_ptr->grad.data(), other_ptr->grad.data(),
                             result->grad.data(), other_ptr->data.size());
                }
            };
        }
        return result;
    }

    // Broadcasting path
    auto out_shape = broadcast_shape(shape, other->shape);
    auto result = create(out_shape, track);

    auto a_strides = compute_strides(shape);
    auto b_strides = compute_strides(other->shape);
    auto out_strides = compute_strides(out_shape);
    size_t ndim = out_shape.size();

    // Compute result with broadcasting
    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < result->data.size(); i++) {
        size_t a_idx = broadcast_index(idx, shape, a_strides, ndim);
        size_t b_idx = broadcast_index(idx, other->shape, b_strides, ndim);
        result->data[i] = data[a_idx] + other->data[b_idx];

        // Increment multi-dimensional index
        for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        auto self_shape = shape;
        auto other_shape = other->shape;

        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, self_shape, other_shape,
                          out_shape, a_strides, b_strides, out_strides, ndim]() {
            std::vector<size_t> idx(ndim, 0);
            for (size_t i = 0; i < result->data.size(); i++) {
                if (self_ptr->requires_grad) {
                    size_t a_idx = broadcast_index(idx, self_shape, a_strides, ndim);
                    self_ptr->grad[a_idx] += result->grad[i];
                }
                if (other_ptr->requires_grad) {
                    size_t b_idx = broadcast_index(idx, other_shape, b_strides, ndim);
                    other_ptr->grad[b_idx] += result->grad[i];
                }

                // Increment multi-dimensional index
                for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < out_shape[d]) break;
                    idx[d] = 0;
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::sub(const TensorPtr& other) const {
    assert(is_broadcastable(shape, other->shape) && "Shapes are not broadcastable");

    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();

    // Fast path: same shape
    if (shape == other->shape) {
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

    // Broadcasting path
    auto out_shape = broadcast_shape(shape, other->shape);
    auto result = create(out_shape, track);

    auto a_strides = compute_strides(shape);
    auto b_strides = compute_strides(other->shape);
    auto out_strides = compute_strides(out_shape);
    size_t ndim = out_shape.size();

    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < result->data.size(); i++) {
        size_t a_idx = broadcast_index(idx, shape, a_strides, ndim);
        size_t b_idx = broadcast_index(idx, other->shape, b_strides, ndim);
        result->data[i] = data[a_idx] - other->data[b_idx];

        for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        auto self_shape = shape;
        auto other_shape = other->shape;

        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, self_shape, other_shape,
                          out_shape, a_strides, b_strides, out_strides, ndim]() {
            std::vector<size_t> idx(ndim, 0);
            for (size_t i = 0; i < result->data.size(); i++) {
                if (self_ptr->requires_grad) {
                    size_t a_idx = broadcast_index(idx, self_shape, a_strides, ndim);
                    self_ptr->grad[a_idx] += result->grad[i];
                }
                if (other_ptr->requires_grad) {
                    size_t b_idx = broadcast_index(idx, other_shape, b_strides, ndim);
                    other_ptr->grad[b_idx] -= result->grad[i];
                }

                for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < out_shape[d]) break;
                    idx[d] = 0;
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::mul(const TensorPtr& other) const {
    assert(is_broadcastable(shape, other->shape) && "Shapes are not broadcastable");

    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();

    // Fast path: same shape
    if (shape == other->shape) {
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

    // Broadcasting path
    auto out_shape = broadcast_shape(shape, other->shape);
    auto result = create(out_shape, track);

    auto a_strides = compute_strides(shape);
    auto b_strides = compute_strides(other->shape);
    auto out_strides = compute_strides(out_shape);
    size_t ndim = out_shape.size();

    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < result->data.size(); i++) {
        size_t a_idx = broadcast_index(idx, shape, a_strides, ndim);
        size_t b_idx = broadcast_index(idx, other->shape, b_strides, ndim);
        result->data[i] = data[a_idx] * other->data[b_idx];

        for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        auto self_shape = shape;
        auto other_shape = other->shape;

        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, self_shape, other_shape,
                          out_shape, a_strides, b_strides, out_strides, ndim]() {
            std::vector<size_t> idx(ndim, 0);
            for (size_t i = 0; i < result->data.size(); i++) {
                size_t a_idx = broadcast_index(idx, self_shape, a_strides, ndim);
                size_t b_idx = broadcast_index(idx, other_shape, b_strides, ndim);

                if (self_ptr->requires_grad) {
                    self_ptr->grad[a_idx] += result->grad[i] * other_ptr->data[b_idx];
                }
                if (other_ptr->requires_grad) {
                    other_ptr->grad[b_idx] += result->grad[i] * self_ptr->data[a_idx];
                }

                for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < out_shape[d]) break;
                    idx[d] = 0;
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::div(const TensorPtr& other) const {
    assert(is_broadcastable(shape, other->shape) && "Shapes are not broadcastable");

    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();

    // Fast path: same shape
    if (shape == other->shape) {
        auto result = create(shape, track);
        for (size_t i = 0; i < data.size(); i++) {
            result->data[i] = data[i] / other->data[i];
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            auto other_ptr = other;
            result->parents = {self_ptr, other_ptr};
            result->grad_fn = [self_ptr, other_ptr, result]() {
                for (size_t i = 0; i < self_ptr->data.size(); i++) {
                    // d(a/b)/da = 1/b
                    if (self_ptr->requires_grad) {
                        self_ptr->grad[i] += result->grad[i] / other_ptr->data[i];
                    }
                    // d(a/b)/db = -a/b^2
                    if (other_ptr->requires_grad) {
                        other_ptr->grad[i] -= result->grad[i] * self_ptr->data[i] /
                                              (other_ptr->data[i] * other_ptr->data[i]);
                    }
                }
            };
        }
        return result;
    }

    // Broadcasting path
    auto out_shape = broadcast_shape(shape, other->shape);
    auto result = create(out_shape, track);

    auto a_strides = compute_strides(shape);
    auto b_strides = compute_strides(other->shape);
    auto out_strides = compute_strides(out_shape);
    size_t ndim = out_shape.size();

    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < result->data.size(); i++) {
        size_t a_idx = broadcast_index(idx, shape, a_strides, ndim);
        size_t b_idx = broadcast_index(idx, other->shape, b_strides, ndim);
        result->data[i] = data[a_idx] / other->data[b_idx];

        for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        auto self_shape = shape;
        auto other_shape = other->shape;

        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, self_shape, other_shape,
                          out_shape, a_strides, b_strides, out_strides, ndim]() {
            std::vector<size_t> idx(ndim, 0);
            for (size_t i = 0; i < result->data.size(); i++) {
                size_t a_idx = broadcast_index(idx, self_shape, a_strides, ndim);
                size_t b_idx = broadcast_index(idx, other_shape, b_strides, ndim);

                // d(a/b)/da = 1/b
                if (self_ptr->requires_grad) {
                    self_ptr->grad[a_idx] += result->grad[i] / other_ptr->data[b_idx];
                }
                // d(a/b)/db = -a/b^2
                if (other_ptr->requires_grad) {
                    other_ptr->grad[b_idx] -= result->grad[i] * self_ptr->data[a_idx] /
                                              (other_ptr->data[b_idx] * other_ptr->data[b_idx]);
                }

                for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < out_shape[d]) break;
                    idx[d] = 0;
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

TensorPtr Tensor::pow(float exponent) const {
    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    for (size_t i = 0; i < data.size(); i++) {
        result->data[i] = std::pow(data[i], exponent);
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, exponent]() {
            // d(x^n)/dx = n * x^(n-1)
            for (size_t i = 0; i < self_ptr->data.size(); i++) {
                self_ptr->grad[i] += result->grad[i] * exponent *
                                     std::pow(self_ptr->data[i], exponent - 1.0f);
            }
        };
    }
    return result;
}

TensorPtr Tensor::pow(const TensorPtr& exponent) const {
    assert(is_broadcastable(shape, exponent->shape) && "Shapes are not broadcastable");

    bool track = (requires_grad || exponent->requires_grad) && GradMode::is_enabled();

    // Fast path: same shape
    if (shape == exponent->shape) {
        auto result = create(shape, track);
        for (size_t i = 0; i < data.size(); i++) {
            result->data[i] = std::pow(data[i], exponent->data[i]);
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            auto exp_ptr = exponent;
            result->parents = {self_ptr, exp_ptr};
            result->grad_fn = [self_ptr, exp_ptr, result]() {
                for (size_t i = 0; i < self_ptr->data.size(); i++) {
                    // d(x^y)/dx = y * x^(y-1)
                    if (self_ptr->requires_grad) {
                        self_ptr->grad[i] += result->grad[i] * exp_ptr->data[i] *
                                             std::pow(self_ptr->data[i], exp_ptr->data[i] - 1.0f);
                    }
                    // d(x^y)/dy = x^y * ln(x)
                    if (exp_ptr->requires_grad) {
                        exp_ptr->grad[i] += result->grad[i] * result->data[i] *
                                            std::log(self_ptr->data[i]);
                    }
                }
            };
        }
        return result;
    }

    // Broadcasting path
    auto out_shape = broadcast_shape(shape, exponent->shape);
    auto result = create(out_shape, track);

    auto a_strides = compute_strides(shape);
    auto b_strides = compute_strides(exponent->shape);
    auto out_strides = compute_strides(out_shape);
    size_t ndim = out_shape.size();

    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < result->data.size(); i++) {
        size_t a_idx = broadcast_index(idx, shape, a_strides, ndim);
        size_t b_idx = broadcast_index(idx, exponent->shape, b_strides, ndim);
        result->data[i] = std::pow(data[a_idx], exponent->data[b_idx]);

        for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto exp_ptr = exponent;
        auto self_shape = shape;
        auto exp_shape = exponent->shape;

        result->parents = {self_ptr, exp_ptr};
        result->grad_fn = [self_ptr, exp_ptr, result, self_shape, exp_shape,
                          out_shape, a_strides, b_strides, out_strides, ndim]() {
            std::vector<size_t> idx(ndim, 0);
            for (size_t i = 0; i < result->data.size(); i++) {
                size_t a_idx = broadcast_index(idx, self_shape, a_strides, ndim);
                size_t b_idx = broadcast_index(idx, exp_shape, b_strides, ndim);

                // d(x^y)/dx = y * x^(y-1)
                if (self_ptr->requires_grad) {
                    self_ptr->grad[a_idx] += result->grad[i] * exp_ptr->data[b_idx] *
                                             std::pow(self_ptr->data[a_idx], exp_ptr->data[b_idx] - 1.0f);
                }
                // d(x^y)/dy = x^y * ln(x)
                if (exp_ptr->requires_grad) {
                    exp_ptr->grad[b_idx] += result->grad[i] * result->data[i] *
                                            std::log(self_ptr->data[a_idx]);
                }

                for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < out_shape[d]) break;
                    idx[d] = 0;
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::sqrt() const {
    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    for (size_t i = 0; i < data.size(); i++) {
        result->data[i] = std::sqrt(data[i]);
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            // d(sqrt(x))/dx = 1 / (2 * sqrt(x))
            for (size_t i = 0; i < self_ptr->data.size(); i++) {
                self_ptr->grad[i] += result->grad[i] / (2.0f * result->data[i]);
            }
        };
    }
    return result;
}

TensorPtr Tensor::abs() const {
    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    for (size_t i = 0; i < data.size(); i++) {
        result->data[i] = std::abs(data[i]);
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            // d|x|/dx = sign(x) = x / |x| (undefined at 0, we use 0)
            for (size_t i = 0; i < self_ptr->data.size(); i++) {
                if (self_ptr->data[i] > 0) {
                    self_ptr->grad[i] += result->grad[i];
                } else if (self_ptr->data[i] < 0) {
                    self_ptr->grad[i] -= result->grad[i];
                }
                // gradient is 0 when x == 0 (subgradient convention)
            }
        };
    }
    return result;
}

TensorPtr Tensor::clamp(float min_val, float max_val) const {
    bool track = requires_grad && GradMode::is_enabled();
    auto result = create(shape, track);

    for (size_t i = 0; i < data.size(); i++) {
        result->data[i] = std::max(min_val, std::min(max_val, data[i]));
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, min_val, max_val]() {
            // Gradient passes through only where value was not clamped
            for (size_t i = 0; i < self_ptr->data.size(); i++) {
                if (self_ptr->data[i] > min_val && self_ptr->data[i] < max_val) {
                    self_ptr->grad[i] += result->grad[i];
                }
                // gradient is 0 where clamped
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

TensorPtr Tensor::max(int dim, bool keepdim) const {
    bool track = requires_grad && GradMode::is_enabled();

    // Global max (dim == -1)
    if (dim == -1) {
        auto result = keepdim ? create(std::vector<size_t>(shape.size(), 1), track)
                              : create({1}, track);
        size_t max_idx = 0;
        float max_val = data[0];
        for (size_t i = 1; i < data.size(); i++) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
        result->data[0] = max_val;

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result, max_idx]() {
                self_ptr->grad[max_idx] += result->grad[0];
            };
        }
        return result;
    }

    assert(shape.size() == 2);
    size_t rows = shape[0], cols = shape[1];

    if (dim == 0) {
        // Max along rows, result shape: [cols] or [1, cols]
        auto result = keepdim ? create({1, cols}, track) : create({cols}, track);
        std::vector<size_t> max_indices(cols);

        for (size_t j = 0; j < cols; j++) {
            size_t max_i = 0;
            float max_val = data[j];
            for (size_t i = 1; i < rows; i++) {
                if (data[i * cols + j] > max_val) {
                    max_val = data[i * cols + j];
                    max_i = i;
                }
            }
            result->data[j] = max_val;
            max_indices[j] = max_i;
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result, cols, max_indices]() {
                for (size_t j = 0; j < cols; j++) {
                    self_ptr->grad[max_indices[j] * cols + j] += result->grad[j];
                }
            };
        }
        return result;
    } else {
        // Max along cols (dim == 1), result shape: [rows] or [rows, 1]
        auto result = keepdim ? create({rows, 1}, track) : create({rows}, track);
        std::vector<size_t> max_indices(rows);

        for (size_t i = 0; i < rows; i++) {
            size_t max_j = 0;
            float max_val = data[i * cols];
            for (size_t j = 1; j < cols; j++) {
                if (data[i * cols + j] > max_val) {
                    max_val = data[i * cols + j];
                    max_j = j;
                }
            }
            result->data[i] = max_val;
            max_indices[i] = max_j;
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result, rows, cols, max_indices]() {
                for (size_t i = 0; i < rows; i++) {
                    self_ptr->grad[i * cols + max_indices[i]] += result->grad[i];
                }
            };
        }
        return result;
    }
}

TensorPtr Tensor::min(int dim, bool keepdim) const {
    bool track = requires_grad && GradMode::is_enabled();

    // Global min (dim == -1)
    if (dim == -1) {
        auto result = keepdim ? create(std::vector<size_t>(shape.size(), 1), track)
                              : create({1}, track);
        size_t min_idx = 0;
        float min_val = data[0];
        for (size_t i = 1; i < data.size(); i++) {
            if (data[i] < min_val) {
                min_val = data[i];
                min_idx = i;
            }
        }
        result->data[0] = min_val;

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result, min_idx]() {
                self_ptr->grad[min_idx] += result->grad[0];
            };
        }
        return result;
    }

    assert(shape.size() == 2);
    size_t rows = shape[0], cols = shape[1];

    if (dim == 0) {
        // Min along rows, result shape: [cols] or [1, cols]
        auto result = keepdim ? create({1, cols}, track) : create({cols}, track);
        std::vector<size_t> min_indices(cols);

        for (size_t j = 0; j < cols; j++) {
            size_t min_i = 0;
            float min_val = data[j];
            for (size_t i = 1; i < rows; i++) {
                if (data[i * cols + j] < min_val) {
                    min_val = data[i * cols + j];
                    min_i = i;
                }
            }
            result->data[j] = min_val;
            min_indices[j] = min_i;
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result, cols, min_indices]() {
                for (size_t j = 0; j < cols; j++) {
                    self_ptr->grad[min_indices[j] * cols + j] += result->grad[j];
                }
            };
        }
        return result;
    } else {
        // Min along cols (dim == 1), result shape: [rows] or [rows, 1]
        auto result = keepdim ? create({rows, 1}, track) : create({rows}, track);
        std::vector<size_t> min_indices(rows);

        for (size_t i = 0; i < rows; i++) {
            size_t min_j = 0;
            float min_val = data[i * cols];
            for (size_t j = 1; j < cols; j++) {
                if (data[i * cols + j] < min_val) {
                    min_val = data[i * cols + j];
                    min_j = j;
                }
            }
            result->data[i] = min_val;
            min_indices[i] = min_j;
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            result->parents = {self_ptr};
            result->grad_fn = [self_ptr, result, rows, cols, min_indices]() {
                for (size_t i = 0; i < rows; i++) {
                    self_ptr->grad[i * cols + min_indices[i]] += result->grad[i];
                }
            };
        }
        return result;
    }
}

TensorPtr Tensor::max(const TensorPtr& other) const {
    assert(is_broadcastable(shape, other->shape) && "Shapes are not broadcastable");

    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();

    // Fast path: same shape
    if (shape == other->shape) {
        auto result = create(shape, track);
        for (size_t i = 0; i < data.size(); i++) {
            result->data[i] = std::max(data[i], other->data[i]);
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            auto other_ptr = other;
            result->parents = {self_ptr, other_ptr};
            result->grad_fn = [self_ptr, other_ptr, result]() {
                for (size_t i = 0; i < self_ptr->data.size(); i++) {
                    // Gradient goes to whichever was larger
                    if (self_ptr->data[i] >= other_ptr->data[i]) {
                        if (self_ptr->requires_grad)
                            self_ptr->grad[i] += result->grad[i];
                    } else {
                        if (other_ptr->requires_grad)
                            other_ptr->grad[i] += result->grad[i];
                    }
                }
            };
        }
        return result;
    }

    // Broadcasting path
    auto out_shape = broadcast_shape(shape, other->shape);
    auto result = create(out_shape, track);

    auto a_strides = compute_strides(shape);
    auto b_strides = compute_strides(other->shape);
    auto out_strides = compute_strides(out_shape);
    size_t ndim = out_shape.size();

    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < result->data.size(); i++) {
        size_t a_idx = broadcast_index(idx, shape, a_strides, ndim);
        size_t b_idx = broadcast_index(idx, other->shape, b_strides, ndim);
        result->data[i] = std::max(data[a_idx], other->data[b_idx]);

        for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        auto self_shape = shape;
        auto other_shape = other->shape;

        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, self_shape, other_shape,
                          out_shape, a_strides, b_strides, out_strides, ndim]() {
            std::vector<size_t> idx(ndim, 0);
            for (size_t i = 0; i < result->data.size(); i++) {
                size_t a_idx = broadcast_index(idx, self_shape, a_strides, ndim);
                size_t b_idx = broadcast_index(idx, other_shape, b_strides, ndim);

                if (self_ptr->data[a_idx] >= other_ptr->data[b_idx]) {
                    if (self_ptr->requires_grad)
                        self_ptr->grad[a_idx] += result->grad[i];
                } else {
                    if (other_ptr->requires_grad)
                        other_ptr->grad[b_idx] += result->grad[i];
                }

                for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < out_shape[d]) break;
                    idx[d] = 0;
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::min(const TensorPtr& other) const {
    assert(is_broadcastable(shape, other->shape) && "Shapes are not broadcastable");

    bool track = (requires_grad || other->requires_grad) && GradMode::is_enabled();

    // Fast path: same shape
    if (shape == other->shape) {
        auto result = create(shape, track);
        for (size_t i = 0; i < data.size(); i++) {
            result->data[i] = std::min(data[i], other->data[i]);
        }

        if (track) {
            auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
            auto other_ptr = other;
            result->parents = {self_ptr, other_ptr};
            result->grad_fn = [self_ptr, other_ptr, result]() {
                for (size_t i = 0; i < self_ptr->data.size(); i++) {
                    // Gradient goes to whichever was smaller
                    if (self_ptr->data[i] <= other_ptr->data[i]) {
                        if (self_ptr->requires_grad)
                            self_ptr->grad[i] += result->grad[i];
                    } else {
                        if (other_ptr->requires_grad)
                            other_ptr->grad[i] += result->grad[i];
                    }
                }
            };
        }
        return result;
    }

    // Broadcasting path
    auto out_shape = broadcast_shape(shape, other->shape);
    auto result = create(out_shape, track);

    auto a_strides = compute_strides(shape);
    auto b_strides = compute_strides(other->shape);
    auto out_strides = compute_strides(out_shape);
    size_t ndim = out_shape.size();

    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < result->data.size(); i++) {
        size_t a_idx = broadcast_index(idx, shape, a_strides, ndim);
        size_t b_idx = broadcast_index(idx, other->shape, b_strides, ndim);
        result->data[i] = std::min(data[a_idx], other->data[b_idx]);

        for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto other_ptr = other;
        auto self_shape = shape;
        auto other_shape = other->shape;

        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, self_shape, other_shape,
                          out_shape, a_strides, b_strides, out_strides, ndim]() {
            std::vector<size_t> idx(ndim, 0);
            for (size_t i = 0; i < result->data.size(); i++) {
                size_t a_idx = broadcast_index(idx, self_shape, a_strides, ndim);
                size_t b_idx = broadcast_index(idx, other_shape, b_strides, ndim);

                if (self_ptr->data[a_idx] <= other_ptr->data[b_idx]) {
                    if (self_ptr->requires_grad)
                        self_ptr->grad[a_idx] += result->grad[i];
                } else {
                    if (other_ptr->requires_grad)
                        other_ptr->grad[b_idx] += result->grad[i];
                }

                for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < out_shape[d]) break;
                    idx[d] = 0;
                }
            }
        };
    }
    return result;
}

TensorPtr Tensor::argmax(int dim, bool keepdim) const {
    // Handle negative dim
    if (dim < 0) dim = static_cast<int>(shape.size()) + dim;
    assert(dim >= 0 && dim < static_cast<int>(shape.size()));

    size_t ndims = shape.size();

    // Compute output shape
    std::vector<size_t> out_shape;
    for (size_t i = 0; i < ndims; i++) {
        if (static_cast<int>(i) == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    // argmax returns indices, no gradients needed
    auto result = create(out_shape, false);

    // Compute strides
    std::vector<size_t> strides(ndims);
    strides[ndims - 1] = 1;
    for (int i = static_cast<int>(ndims) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    size_t dim_size = shape[dim];
    size_t dim_stride = strides[dim];

    // Number of slices (product of all dims except dim)
    size_t num_slices = 1;
    for (size_t i = 0; i < ndims; i++) {
        if (static_cast<int>(i) != dim) num_slices *= shape[i];
    }

    // Iterate over all slices
    size_t out_idx = 0;
    std::vector<size_t> idx(ndims, 0);

    for (size_t slice = 0; slice < num_slices; slice++) {
        // Find base index for this slice
        size_t base_idx = 0;
        for (size_t i = 0; i < ndims; i++) {
            if (static_cast<int>(i) != dim) {
                base_idx += idx[i] * strides[i];
            }
        }

        // Find argmax along dim
        float max_val = data[base_idx];
        size_t max_idx = 0;
        for (size_t d = 1; d < dim_size; d++) {
            float val = data[base_idx + d * dim_stride];
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }
        result->data[out_idx++] = static_cast<float>(max_idx);

        // Increment indices (skip dim dimension)
        for (int i = static_cast<int>(ndims) - 1; i >= 0; i--) {
            if (i == dim) continue;
            idx[i]++;
            if (idx[i] < shape[i]) break;
            idx[i] = 0;
        }
    }

    return result;
}

TensorPtr Tensor::argmin(int dim, bool keepdim) const {
    // Handle negative dim
    if (dim < 0) dim = static_cast<int>(shape.size()) + dim;
    assert(dim >= 0 && dim < static_cast<int>(shape.size()));

    size_t ndims = shape.size();

    // Compute output shape
    std::vector<size_t> out_shape;
    for (size_t i = 0; i < ndims; i++) {
        if (static_cast<int>(i) == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    // argmin returns indices, no gradients needed
    auto result = create(out_shape, false);

    // Compute strides
    std::vector<size_t> strides(ndims);
    strides[ndims - 1] = 1;
    for (int i = static_cast<int>(ndims) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    size_t dim_size = shape[dim];
    size_t dim_stride = strides[dim];

    // Number of slices (product of all dims except dim)
    size_t num_slices = 1;
    for (size_t i = 0; i < ndims; i++) {
        if (static_cast<int>(i) != dim) num_slices *= shape[i];
    }

    // Iterate over all slices
    size_t out_idx = 0;
    std::vector<size_t> idx(ndims, 0);

    for (size_t slice = 0; slice < num_slices; slice++) {
        // Find base index for this slice
        size_t base_idx = 0;
        for (size_t i = 0; i < ndims; i++) {
            if (static_cast<int>(i) != dim) {
                base_idx += idx[i] * strides[i];
            }
        }

        // Find argmin along dim
        float min_val = data[base_idx];
        size_t min_idx = 0;
        for (size_t d = 1; d < dim_size; d++) {
            float val = data[base_idx + d * dim_stride];
            if (val < min_val) {
                min_val = val;
                min_idx = d;
            }
        }
        result->data[out_idx++] = static_cast<float>(min_idx);

        // Increment indices (skip dim dimension)
        for (int i = static_cast<int>(ndims) - 1; i >= 0; i--) {
            if (i == dim) continue;
            idx[i]++;
            if (idx[i] < shape[i]) break;
            idx[i] = 0;
        }
    }

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

// ============================================================================
// im2col / col2im for optimized convolution
// ============================================================================

// im2col: Transform input patches into columns for matrix multiplication
// Input: [in_channels, in_h, in_w] (single image)
// Output: [in_channels * kernel_h * kernel_w, out_h * out_w]
static void im2col(const float* input, float* col,
                   size_t in_channels, size_t in_h, size_t in_w,
                   size_t kernel_h, size_t kernel_w,
                   size_t out_h, size_t out_w,
                   size_t stride, size_t padding) {
    size_t col_w = out_h * out_w;
    size_t kernel_size = kernel_h * kernel_w;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t c = 0; c < in_channels; c++) {
        for (size_t k = 0; k < kernel_size; k++) {
            size_t kh = k / kernel_w;
            size_t kw = k % kernel_w;
            size_t row = c * kernel_size + k;
            for (size_t oh = 0; oh < out_h; oh++) {
                int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
                for (size_t ow = 0; ow < out_w; ow++) {
                    int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);

                    size_t col_idx = row * col_w + oh * out_w + ow;
                    if (ih >= 0 && ih < static_cast<int>(in_h) &&
                        iw >= 0 && iw < static_cast<int>(in_w)) {
                        col[col_idx] = input[c * in_h * in_w + ih * in_w + iw];
                    } else {
                        col[col_idx] = 0.0f;  // padding
                    }
                }
            }
        }
    }
}

// col2im: Transform columns back to image (accumulating gradients)
// Input: [in_channels * kernel_h * kernel_w, out_h * out_w]
// Output: [in_channels, in_h, in_w] (accumulated)
static void col2im(const float* col, float* input,
                   size_t in_channels, size_t in_h, size_t in_w,
                   size_t kernel_h, size_t kernel_w,
                   size_t out_h, size_t out_w,
                   size_t stride, size_t padding) {
    size_t col_w = out_h * out_w;
    size_t kernel_size = kernel_h * kernel_w;

    // Parallelize over channels (each channel writes to independent memory)
    #pragma omp parallel for schedule(static)
    for (size_t c = 0; c < in_channels; c++) {
        for (size_t k = 0; k < kernel_size; k++) {
            size_t kh = k / kernel_w;
            size_t kw = k % kernel_w;
            size_t row = c * kernel_size + k;
            for (size_t oh = 0; oh < out_h; oh++) {
                int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
                if (ih < 0 || ih >= static_cast<int>(in_h)) continue;
                for (size_t ow = 0; ow < out_w; ow++) {
                    int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);

                    if (iw >= 0 && iw < static_cast<int>(in_w)) {
                        size_t col_idx = row * col_w + oh * out_w + ow;
                        input[c * in_h * in_w + ih * in_w + iw] += col[col_idx];
                    }
                }
            }
        }
    }
}

TensorPtr Tensor::conv2d(const TensorPtr& weight, const TensorPtr& bias,
                          size_t stride, size_t padding) const {
    // Input shape: (batch, in_channels, height, width)
    // Weight shape: (out_channels, in_channels, kernel_h, kernel_w)
    // Bias shape: (out_channels) or nullptr
    assert(shape.size() == 4);
    assert(weight->shape.size() == 4);
    assert(shape[1] == weight->shape[1]); // in_channels match

    size_t batch = shape[0];
    size_t in_channels = shape[1];
    size_t in_h = shape[2];
    size_t in_w = shape[3];

    size_t out_channels = weight->shape[0];
    size_t kernel_h = weight->shape[2];
    size_t kernel_w = weight->shape[3];

    size_t out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    size_t out_w = (in_w + 2 * padding - kernel_w) / stride + 1;

    bool track = (requires_grad || weight->requires_grad || (bias && bias->requires_grad))
                 && GradMode::is_enabled();
    auto result = create({batch, out_channels, out_h, out_w}, track);

    // im2col + GEMM forward pass
    // Weight reshaped: [out_channels, in_channels * kernel_h * kernel_w]
    // im2col output: [in_channels * kernel_h * kernel_w, out_h * out_w]
    // Result: weight @ im2col = [out_channels, out_h * out_w]

    size_t col_h = in_channels * kernel_h * kernel_w;
    size_t col_w = out_h * out_w;
    std::vector<float> col_buffer(col_h * col_w);

    for (size_t b = 0; b < batch; b++) {
        const float* input_ptr = data.data() + b * in_channels * in_h * in_w;
        float* output_ptr = result->data.data() + b * out_channels * out_h * out_w;

        // Convert input patches to columns
        im2col(input_ptr, col_buffer.data(),
               in_channels, in_h, in_w,
               kernel_h, kernel_w,
               out_h, out_w, stride, padding);

        // GEMM: weight [out_channels, col_h] @ col [col_h, col_w] -> output [out_channels, col_w]
        matmul_blocked(output_ptr, weight->data.data(), col_buffer.data(),
                       out_channels, col_h, col_w);

        // Add bias
        if (bias) {
            for (size_t oc = 0; oc < out_channels; oc++) {
                for (size_t i = 0; i < col_w; i++) {
                    output_ptr[oc * col_w + i] += bias->data[oc];
                }
            }
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto weight_ptr = weight;
        auto bias_ptr = bias;
        result->parents = {self_ptr, weight_ptr};
        if (bias_ptr) result->parents.push_back(bias_ptr);

        result->grad_fn = [self_ptr, weight_ptr, bias_ptr, result,
                           batch, in_channels, in_h, in_w,
                           out_channels, out_h, out_w,
                           kernel_h, kernel_w, stride, padding]() {

            size_t col_h = in_channels * kernel_h * kernel_w;
            size_t col_w = out_h * out_w;

            // Gradient w.r.t. input using col2im
            if (self_ptr->requires_grad) {
                // grad_input = col2im(weight^T @ grad_output)
                // weight^T: [col_h, out_channels]
                // grad_output: [out_channels, col_w]
                // result: [col_h, col_w]

                std::vector<float> weight_T(col_h * out_channels);
                // Transpose weight: [out_channels, col_h] -> [col_h, out_channels]
                for (size_t oc = 0; oc < out_channels; oc++) {
                    for (size_t k = 0; k < col_h; k++) {
                        weight_T[k * out_channels + oc] = weight_ptr->data[oc * col_h + k];
                    }
                }

                std::vector<float> col_grad(col_h * col_w);

                for (size_t b = 0; b < batch; b++) {
                    const float* grad_out = result->grad.data() + b * out_channels * col_w;
                    float* grad_in = self_ptr->grad.data() + b * in_channels * in_h * in_w;

                    // GEMM: weight_T [col_h, out_channels] @ grad_out [out_channels, col_w] -> col_grad [col_h, col_w]
                    matmul_blocked(col_grad.data(), weight_T.data(), grad_out,
                                   col_h, out_channels, col_w);

                    // col2im to accumulate gradients
                    col2im(col_grad.data(), grad_in,
                           in_channels, in_h, in_w,
                           kernel_h, kernel_w,
                           out_h, out_w, stride, padding);
                }
            }

            // Gradient w.r.t. weight
            if (weight_ptr->requires_grad) {
                // grad_weight = sum over batch of (grad_output @ im2col^T)
                // grad_output: [out_channels, col_w]
                // im2col^T: [col_w, col_h]
                // result: [out_channels, col_h]

                std::vector<float> col_buffer(col_h * col_w);
                std::vector<float> col_T(col_w * col_h);

                for (size_t b = 0; b < batch; b++) {
                    const float* input_ptr = self_ptr->data.data() + b * in_channels * in_h * in_w;
                    const float* grad_out = result->grad.data() + b * out_channels * col_w;

                    // im2col for this batch
                    im2col(input_ptr, col_buffer.data(),
                           in_channels, in_h, in_w,
                           kernel_h, kernel_w,
                           out_h, out_w, stride, padding);

                    // Transpose col: [col_h, col_w] -> [col_w, col_h]
                    for (size_t i = 0; i < col_h; i++) {
                        for (size_t j = 0; j < col_w; j++) {
                            col_T[j * col_h + i] = col_buffer[i * col_w + j];
                        }
                    }

                    // GEMM: grad_out [out_channels, col_w] @ col_T [col_w, col_h] -> grad_weight [out_channels, col_h]
                    // Accumulate into weight gradient
                    std::vector<float> grad_w_batch(out_channels * col_h);
                    matmul_blocked(grad_w_batch.data(), grad_out, col_T.data(),
                                   out_channels, col_w, col_h);

                    for (size_t i = 0; i < out_channels * col_h; i++) {
                        weight_ptr->grad[i] += grad_w_batch[i];
                    }
                }
            }

            // Gradient w.r.t. bias
            if (bias_ptr && bias_ptr->requires_grad) {
                for (size_t oc = 0; oc < out_channels; oc++) {
                    float grad_sum = 0.0f;
                    for (size_t b = 0; b < batch; b++) {
                        for (size_t i = 0; i < col_w; i++) {
                            size_t out_idx = b * out_channels * col_w + oc * col_w + i;
                            grad_sum += result->grad[out_idx];
                        }
                    }
                    bias_ptr->grad[oc] += grad_sum;
                }
            }
        };
    }

    return result;
}

TensorPtr Tensor::conv_transpose2d(const TensorPtr& weight, const TensorPtr& bias,
                                    size_t stride, size_t padding,
                                    size_t output_padding) const {
    // Transposed convolution (deconvolution) for upsampling
    // Input shape: (batch, in_channels, height, width)
    // Weight shape: (in_channels, out_channels, kernel_h, kernel_w)
    // Note: weight shape is transposed compared to Conv2d
    // Bias shape: (out_channels) or nullptr
    // Output shape: (batch, out_channels, out_h, out_w)
    // where out_h = (in_h - 1) * stride - 2 * padding + kernel_h + output_padding

    assert(shape.size() == 4);
    assert(weight->shape.size() == 4);
    assert(shape[1] == weight->shape[0]); // in_channels match
    assert(output_padding < stride); // output_padding must be smaller than stride

    size_t batch = shape[0];
    size_t in_channels = shape[1];
    size_t in_h = shape[2];
    size_t in_w = shape[3];

    size_t out_channels = weight->shape[1];
    size_t kernel_h = weight->shape[2];
    size_t kernel_w = weight->shape[3];

    // Output size formula for transposed convolution
    size_t out_h = (in_h - 1) * stride - 2 * padding + kernel_h + output_padding;
    size_t out_w = (in_w - 1) * stride - 2 * padding + kernel_w + output_padding;

    bool track = (requires_grad || weight->requires_grad || (bias && bias->requires_grad))
                 && GradMode::is_enabled();
    auto result = create({batch, out_channels, out_h, out_w}, track);

    // Initialize output to zeros
    std::fill(result->data.begin(), result->data.end(), 0.0f);

    // Forward pass: for each input position, scatter the kernel values to output
    // This is equivalent to the backward pass of a regular convolution w.r.t. input
    for (size_t b = 0; b < batch; b++) {
        for (size_t ic = 0; ic < in_channels; ic++) {
            for (size_t ih = 0; ih < in_h; ih++) {
                for (size_t iw = 0; iw < in_w; iw++) {
                    // Get input value
                    size_t in_idx = b * in_channels * in_h * in_w +
                                    ic * in_h * in_w +
                                    ih * in_w + iw;
                    float in_val = data[in_idx];

                    // Scatter to output for each output channel and kernel position
                    for (size_t oc = 0; oc < out_channels; oc++) {
                        for (size_t kh = 0; kh < kernel_h; kh++) {
                            for (size_t kw = 0; kw < kernel_w; kw++) {
                                // Calculate output position
                                int oh = static_cast<int>(ih * stride + kh) - static_cast<int>(padding);
                                int ow = static_cast<int>(iw * stride + kw) - static_cast<int>(padding);

                                // Check bounds
                                if (oh >= 0 && oh < static_cast<int>(out_h) &&
                                    ow >= 0 && ow < static_cast<int>(out_w)) {
                                    // Weight index: [in_channels, out_channels, kernel_h, kernel_w]
                                    size_t w_idx = ic * out_channels * kernel_h * kernel_w +
                                                   oc * kernel_h * kernel_w +
                                                   kh * kernel_w + kw;
                                    size_t out_idx = b * out_channels * out_h * out_w +
                                                     oc * out_h * out_w +
                                                     static_cast<size_t>(oh) * out_w +
                                                     static_cast<size_t>(ow);
                                    result->data[out_idx] += in_val * weight->data[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias
    if (bias) {
        for (size_t b = 0; b < batch; b++) {
            for (size_t oc = 0; oc < out_channels; oc++) {
                for (size_t oh = 0; oh < out_h; oh++) {
                    for (size_t ow = 0; ow < out_w; ow++) {
                        size_t idx = b * out_channels * out_h * out_w +
                                     oc * out_h * out_w +
                                     oh * out_w + ow;
                        result->data[idx] += bias->data[oc];
                    }
                }
            }
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto weight_ptr = weight;
        auto bias_ptr = bias;
        result->parents = {self_ptr, weight_ptr};
        if (bias_ptr) result->parents.push_back(bias_ptr);

        result->grad_fn = [self_ptr, weight_ptr, bias_ptr, result,
                           batch, in_channels, in_h, in_w,
                           out_channels, out_h, out_w,
                           kernel_h, kernel_w, stride, padding]() {

            // Gradient w.r.t. input: conv2d of grad_output with weight
            if (self_ptr->requires_grad) {
                for (size_t b = 0; b < batch; b++) {
                    for (size_t ic = 0; ic < in_channels; ic++) {
                        for (size_t ih = 0; ih < in_h; ih++) {
                            for (size_t iw = 0; iw < in_w; iw++) {
                                float grad_sum = 0.0f;

                                for (size_t oc = 0; oc < out_channels; oc++) {
                                    for (size_t kh = 0; kh < kernel_h; kh++) {
                                        for (size_t kw = 0; kw < kernel_w; kw++) {
                                            int oh = static_cast<int>(ih * stride + kh) - static_cast<int>(padding);
                                            int ow = static_cast<int>(iw * stride + kw) - static_cast<int>(padding);

                                            if (oh >= 0 && oh < static_cast<int>(out_h) &&
                                                ow >= 0 && ow < static_cast<int>(out_w)) {
                                                size_t w_idx = ic * out_channels * kernel_h * kernel_w +
                                                               oc * kernel_h * kernel_w +
                                                               kh * kernel_w + kw;
                                                size_t out_idx = b * out_channels * out_h * out_w +
                                                                 oc * out_h * out_w +
                                                                 static_cast<size_t>(oh) * out_w +
                                                                 static_cast<size_t>(ow);
                                                grad_sum += result->grad[out_idx] * weight_ptr->data[w_idx];
                                            }
                                        }
                                    }
                                }

                                size_t in_idx = b * in_channels * in_h * in_w +
                                                ic * in_h * in_w +
                                                ih * in_w + iw;
                                self_ptr->grad[in_idx] += grad_sum;
                            }
                        }
                    }
                }
            }

            // Gradient w.r.t. weight
            if (weight_ptr->requires_grad) {
                for (size_t b = 0; b < batch; b++) {
                    for (size_t ic = 0; ic < in_channels; ic++) {
                        for (size_t ih = 0; ih < in_h; ih++) {
                            for (size_t iw = 0; iw < in_w; iw++) {
                                size_t in_idx = b * in_channels * in_h * in_w +
                                                ic * in_h * in_w +
                                                ih * in_w + iw;
                                float in_val = self_ptr->data[in_idx];

                                for (size_t oc = 0; oc < out_channels; oc++) {
                                    for (size_t kh = 0; kh < kernel_h; kh++) {
                                        for (size_t kw = 0; kw < kernel_w; kw++) {
                                            int oh = static_cast<int>(ih * stride + kh) - static_cast<int>(padding);
                                            int ow = static_cast<int>(iw * stride + kw) - static_cast<int>(padding);

                                            if (oh >= 0 && oh < static_cast<int>(out_h) &&
                                                ow >= 0 && ow < static_cast<int>(out_w)) {
                                                size_t w_idx = ic * out_channels * kernel_h * kernel_w +
                                                               oc * kernel_h * kernel_w +
                                                               kh * kernel_w + kw;
                                                size_t out_idx = b * out_channels * out_h * out_w +
                                                                 oc * out_h * out_w +
                                                                 static_cast<size_t>(oh) * out_w +
                                                                 static_cast<size_t>(ow);
                                                weight_ptr->grad[w_idx] += in_val * result->grad[out_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Gradient w.r.t. bias
            if (bias_ptr && bias_ptr->requires_grad) {
                for (size_t oc = 0; oc < out_channels; oc++) {
                    float grad_sum = 0.0f;
                    for (size_t b = 0; b < batch; b++) {
                        for (size_t oh = 0; oh < out_h; oh++) {
                            for (size_t ow = 0; ow < out_w; ow++) {
                                size_t idx = b * out_channels * out_h * out_w +
                                             oc * out_h * out_w +
                                             oh * out_w + ow;
                                grad_sum += result->grad[idx];
                            }
                        }
                    }
                    bias_ptr->grad[oc] += grad_sum;
                }
            }
        };
    }

    return result;
}

TensorPtr Tensor::maxpool2d(size_t kernel_size, size_t stride) const {
    // Input shape: (batch, channels, height, width)
    assert(shape.size() == 4);

    if (stride == 0) stride = kernel_size;

    size_t batch = shape[0];
    size_t channels = shape[1];
    size_t in_h = shape[2];
    size_t in_w = shape[3];

    size_t out_h = (in_h - kernel_size) / stride + 1;
    size_t out_w = (in_w - kernel_size) / stride + 1;

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create({batch, channels, out_h, out_w}, track);

    // Store indices for backward pass
    std::vector<size_t> max_indices(result->data.size());

    // Forward pass
    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    float max_val = -std::numeric_limits<float>::max();
                    size_t max_idx = 0;

                    for (size_t kh = 0; kh < kernel_size; kh++) {
                        for (size_t kw = 0; kw < kernel_size; kw++) {
                            size_t ih = oh * stride + kh;
                            size_t iw = ow * stride + kw;
                            size_t input_idx = b * (channels * in_h * in_w) +
                                               c * (in_h * in_w) +
                                               ih * in_w + iw;
                            if (data[input_idx] > max_val) {
                                max_val = data[input_idx];
                                max_idx = input_idx;
                            }
                        }
                    }

                    size_t output_idx = b * (channels * out_h * out_w) +
                                        c * (out_h * out_w) +
                                        oh * out_w + ow;
                    result->data[output_idx] = max_val;
                    max_indices[output_idx] = max_idx;
                }
            }
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, max_indices]() {
            for (size_t i = 0; i < result->data.size(); i++) {
                self_ptr->grad[max_indices[i]] += result->grad[i];
            }
        };
    }

    return result;
}

TensorPtr Tensor::avgpool2d(size_t kernel_size, size_t stride) const {
    // Input shape: (batch, channels, height, width)
    assert(shape.size() == 4);

    if (stride == 0) stride = kernel_size;

    size_t batch = shape[0];
    size_t channels = shape[1];
    size_t in_h = shape[2];
    size_t in_w = shape[3];

    size_t out_h = (in_h - kernel_size) / stride + 1;
    size_t out_w = (in_w - kernel_size) / stride + 1;

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create({batch, channels, out_h, out_w}, track);

    float pool_size = static_cast<float>(kernel_size * kernel_size);

    // Forward pass
    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    float sum = 0.0f;

                    for (size_t kh = 0; kh < kernel_size; kh++) {
                        for (size_t kw = 0; kw < kernel_size; kw++) {
                            size_t ih = oh * stride + kh;
                            size_t iw = ow * stride + kw;
                            size_t input_idx = b * (channels * in_h * in_w) +
                                               c * (in_h * in_w) +
                                               ih * in_w + iw;
                            sum += data[input_idx];
                        }
                    }

                    size_t output_idx = b * (channels * out_h * out_w) +
                                        c * (out_h * out_w) +
                                        oh * out_w + ow;
                    result->data[output_idx] = sum / pool_size;
                }
            }
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, batch, channels, in_h, in_w,
                           out_h, out_w, kernel_size, stride, pool_size]() {
            for (size_t b = 0; b < batch; b++) {
                for (size_t c = 0; c < channels; c++) {
                    for (size_t oh = 0; oh < out_h; oh++) {
                        for (size_t ow = 0; ow < out_w; ow++) {
                            size_t out_idx = b * (channels * out_h * out_w) +
                                             c * (out_h * out_w) +
                                             oh * out_w + ow;
                            float grad_val = result->grad[out_idx] / pool_size;

                            for (size_t kh = 0; kh < kernel_size; kh++) {
                                for (size_t kw = 0; kw < kernel_size; kw++) {
                                    size_t ih = oh * stride + kh;
                                    size_t iw = ow * stride + kw;
                                    size_t input_idx = b * (channels * in_h * in_w) +
                                                       c * (in_h * in_w) +
                                                       ih * in_w + iw;
                                    self_ptr->grad[input_idx] += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        };
    }

    return result;
}

TensorPtr Tensor::flatten(size_t start_dim) const {
    // Flatten dimensions from start_dim onwards
    assert(start_dim < shape.size());

    std::vector<size_t> new_shape;
    size_t flat_size = 1;

    for (size_t i = 0; i < start_dim; i++) {
        new_shape.push_back(shape[i]);
    }
    for (size_t i = start_dim; i < shape.size(); i++) {
        flat_size *= shape[i];
    }
    new_shape.push_back(flat_size);

    return reshape(new_shape);
}

TensorPtr Tensor::squeeze(int dim) const {
    std::vector<size_t> new_shape;
    auto self = const_cast<Tensor*>(this)->shared_from_this();

    if (dim == -1) {
        // Remove all dimensions of size 1
        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] != 1) {
                new_shape.push_back(shape[i]);
            }
        }
        // If all dimensions were 1, keep at least one
        if (new_shape.empty()) {
            new_shape.push_back(1);
        }
    } else {
        // Handle negative dim
        int ndims = static_cast<int>(shape.size());
        if (dim < 0) dim += ndims;
        assert(dim >= 0 && dim < ndims && "Dimension out of range");

        for (int i = 0; i < ndims; i++) {
            if (i == dim) {
                // Only remove if size is 1
                if (shape[i] != 1) {
                    new_shape.push_back(shape[i]);
                }
            } else {
                new_shape.push_back(shape[i]);
            }
        }
    }

    // If shape unchanged, return self (no-op)
    if (new_shape == shape) {
        auto result = Tensor::create(data, shape, requires_grad);
        if (should_track_grad()) {
            result->parents = {self};
            result->grad_fn = [self, result]() {
                for (size_t i = 0; i < self->data.size(); i++) {
                    self->grad[i] += result->grad[i];
                }
            };
        }
        return result;
    }

    auto result = Tensor::create(data, new_shape, requires_grad);

    if (should_track_grad()) {
        result->parents = {self};
        auto orig_shape = shape;
        result->grad_fn = [self, result, orig_shape]() {
            // Gradient is just passed through unchanged (reshape)
            for (size_t i = 0; i < self->data.size(); i++) {
                self->grad[i] += result->grad[i];
            }
        };
    }

    return result;
}

TensorPtr Tensor::unsqueeze(int dim) const {
    auto self = const_cast<Tensor*>(this)->shared_from_this();
    int ndims = static_cast<int>(shape.size());

    // Handle negative dim (can insert at ndims, so range is [-ndims-1, ndims])
    if (dim < 0) dim += ndims + 1;
    assert(dim >= 0 && dim <= ndims && "Dimension out of range");

    // Build new shape with size-1 dimension inserted
    std::vector<size_t> new_shape;
    for (int i = 0; i < ndims + 1; i++) {
        if (i == dim) {
            new_shape.push_back(1);
        }
        if (i < ndims) {
            new_shape.push_back(shape[i]);
        }
    }

    auto result = Tensor::create(data, new_shape, requires_grad);

    if (should_track_grad()) {
        result->parents = {self};
        result->grad_fn = [self, result]() {
            // Gradient is just passed through unchanged (reshape)
            for (size_t i = 0; i < self->data.size(); i++) {
                self->grad[i] += result->grad[i];
            }
        };
    }

    return result;
}

TensorPtr Tensor::permute(const std::vector<int>& dims) const {
    auto self = const_cast<Tensor*>(this)->shared_from_this();
    int ndims = static_cast<int>(shape.size());

    // Validate dims
    assert(static_cast<int>(dims.size()) == ndims && "Permute dims must match tensor ndim");

    // Convert negative dims and validate
    std::vector<int> perm(ndims);
    std::vector<bool> seen(ndims, false);
    for (int i = 0; i < ndims; i++) {
        int d = dims[i];
        if (d < 0) d += ndims;
        assert(d >= 0 && d < ndims && "Permute dimension out of range");
        assert(!seen[d] && "Duplicate dimension in permute");
        seen[d] = true;
        perm[i] = d;
    }

    // Build new shape
    std::vector<size_t> new_shape(ndims);
    for (int i = 0; i < ndims; i++) {
        new_shape[i] = shape[perm[i]];
    }

    // Compute strides for original tensor
    std::vector<size_t> old_strides(ndims);
    old_strides[ndims - 1] = 1;
    for (int i = ndims - 2; i >= 0; i--) {
        old_strides[i] = old_strides[i + 1] * shape[i + 1];
    }

    // Compute strides for new tensor
    std::vector<size_t> new_strides(ndims);
    new_strides[ndims - 1] = 1;
    for (int i = ndims - 2; i >= 0; i--) {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    auto result = Tensor::create(new_shape, requires_grad);

    // Permute data
    for (size_t i = 0; i < data.size(); i++) {
        // Convert linear index to multi-dim index in new tensor
        std::vector<size_t> new_idx(ndims);
        size_t tmp = i;
        for (int d = 0; d < ndims; d++) {
            new_idx[d] = tmp / new_strides[d];
            tmp %= new_strides[d];
        }

        // Map to old tensor index
        size_t old_linear = 0;
        for (int d = 0; d < ndims; d++) {
            old_linear += new_idx[d] * old_strides[perm[d]];
        }

        result->data[i] = data[old_linear];
    }

    if (should_track_grad()) {
        result->parents = {self};
        auto orig_shape = shape;

        // Compute inverse permutation for backward pass
        std::vector<int> inv_perm(ndims);
        for (int i = 0; i < ndims; i++) {
            inv_perm[perm[i]] = i;
        }

        result->grad_fn = [self, result, perm, inv_perm, old_strides, new_strides, ndims]() {
            // Permute gradients back using inverse permutation
            for (size_t i = 0; i < result->data.size(); i++) {
                // Convert linear index to multi-dim index in result (new) tensor
                std::vector<size_t> new_idx(ndims);
                size_t tmp = i;
                for (int d = 0; d < ndims; d++) {
                    new_idx[d] = tmp / new_strides[d];
                    tmp %= new_strides[d];
                }

                // Map to old tensor index using permutation
                size_t old_linear = 0;
                for (int d = 0; d < ndims; d++) {
                    old_linear += new_idx[d] * old_strides[perm[d]];
                }

                self->grad[old_linear] += result->grad[i];
            }
        };
    }

    return result;
}

// Data augmentation operations for images
// Assumes format [N, C, H, W] or [C, H, W]

TensorPtr Tensor::flip_horizontal() const {
    // Flip image(s) horizontally (left-right)
    assert(shape.size() == 3 || shape.size() == 4);

    size_t batch = 1, channels, height, width;
    if (shape.size() == 4) {
        batch = shape[0];
        channels = shape[1];
        height = shape[2];
        width = shape[3];
    } else {
        channels = shape[0];
        height = shape[1];
        width = shape[2];
    }

    auto result = Tensor::create(shape, false);  // No gradient for augmentation

    for (size_t n = 0; n < batch; n++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    size_t src_idx, dst_idx;
                    if (shape.size() == 4) {
                        src_idx = n * channels * height * width + c * height * width + h * width + w;
                        dst_idx = n * channels * height * width + c * height * width + h * width + (width - 1 - w);
                    } else {
                        src_idx = c * height * width + h * width + w;
                        dst_idx = c * height * width + h * width + (width - 1 - w);
                    }
                    result->data[dst_idx] = data[src_idx];
                }
            }
        }
    }

    return result;
}

TensorPtr Tensor::random_flip_horizontal(float p) const {
    // Randomly flip with probability p
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if (shape.size() == 4) {
        // Batch mode: flip each image independently
        size_t batch = shape[0];
        size_t channels = shape[1];
        size_t height = shape[2];
        size_t width = shape[3];
        size_t img_size = channels * height * width;

        auto result = Tensor::create(shape, false);

        for (size_t n = 0; n < batch; n++) {
            bool do_flip = dist(gen) < p;
            for (size_t c = 0; c < channels; c++) {
                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        size_t src_idx = n * img_size + c * height * width + h * width + w;
                        size_t dst_w = do_flip ? (width - 1 - w) : w;
                        size_t dst_idx = n * img_size + c * height * width + h * width + dst_w;
                        result->data[dst_idx] = data[src_idx];
                    }
                }
            }
        }
        return result;
    } else {
        // Single image
        if (dist(gen) < p) {
            return flip_horizontal();
        } else {
            auto result = Tensor::create(shape, false);
            std::copy(data.begin(), data.end(), result->data.begin());
            return result;
        }
    }
}

TensorPtr Tensor::pad2d(size_t padding) const {
    // Zero-pad height and width dimensions
    assert(shape.size() == 3 || shape.size() == 4);

    size_t batch = 1, channels, height, width;
    std::vector<size_t> new_shape;

    if (shape.size() == 4) {
        batch = shape[0];
        channels = shape[1];
        height = shape[2];
        width = shape[3];
        new_shape = {batch, channels, height + 2 * padding, width + 2 * padding};
    } else {
        channels = shape[0];
        height = shape[1];
        width = shape[2];
        new_shape = {channels, height + 2 * padding, width + 2 * padding};
    }

    size_t new_height = height + 2 * padding;
    size_t new_width = width + 2 * padding;

    auto result = Tensor::zeros(new_shape, false);

    for (size_t n = 0; n < batch; n++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    size_t src_idx, dst_idx;
                    if (shape.size() == 4) {
                        src_idx = n * channels * height * width + c * height * width + h * width + w;
                        dst_idx = n * channels * new_height * new_width + c * new_height * new_width +
                                  (h + padding) * new_width + (w + padding);
                    } else {
                        src_idx = c * height * width + h * width + w;
                        dst_idx = c * new_height * new_width + (h + padding) * new_width + (w + padding);
                    }
                    result->data[dst_idx] = data[src_idx];
                }
            }
        }
    }

    return result;
}

TensorPtr Tensor::crop(size_t top, size_t left, size_t crop_height, size_t crop_width) const {
    // Crop a region from the image
    assert(shape.size() == 3 || shape.size() == 4);

    size_t batch = 1, channels, height, width;
    std::vector<size_t> new_shape;

    if (shape.size() == 4) {
        batch = shape[0];
        channels = shape[1];
        height = shape[2];
        width = shape[3];
        new_shape = {batch, channels, crop_height, crop_width};
    } else {
        channels = shape[0];
        height = shape[1];
        width = shape[2];
        new_shape = {channels, crop_height, crop_width};
    }

    assert(top + crop_height <= height && "Crop exceeds image height");
    assert(left + crop_width <= width && "Crop exceeds image width");

    auto result = Tensor::create(new_shape, false);

    for (size_t n = 0; n < batch; n++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < crop_height; h++) {
                for (size_t w = 0; w < crop_width; w++) {
                    size_t src_idx, dst_idx;
                    if (shape.size() == 4) {
                        src_idx = n * channels * height * width + c * height * width +
                                  (top + h) * width + (left + w);
                        dst_idx = n * channels * crop_height * crop_width + c * crop_height * crop_width +
                                  h * crop_width + w;
                    } else {
                        src_idx = c * height * width + (top + h) * width + (left + w);
                        dst_idx = c * crop_height * crop_width + h * crop_width + w;
                    }
                    result->data[dst_idx] = data[src_idx];
                }
            }
        }
    }

    return result;
}

TensorPtr Tensor::random_crop(size_t crop_height, size_t crop_width) const {
    // Random crop from the image (typically used after padding)
    assert(shape.size() == 3 || shape.size() == 4);

    static thread_local std::mt19937 gen(std::random_device{}());

    size_t batch = 1, channels, height, width;
    std::vector<size_t> new_shape;

    if (shape.size() == 4) {
        batch = shape[0];
        channels = shape[1];
        height = shape[2];
        width = shape[3];
        new_shape = {batch, channels, crop_height, crop_width};
    } else {
        channels = shape[0];
        height = shape[1];
        width = shape[2];
        new_shape = {channels, crop_height, crop_width};
    }

    assert(crop_height <= height && "Crop height exceeds image height");
    assert(crop_width <= width && "Crop width exceeds image width");

    auto result = Tensor::create(new_shape, false);

    std::uniform_int_distribution<size_t> top_dist(0, height - crop_height);
    std::uniform_int_distribution<size_t> left_dist(0, width - crop_width);

    for (size_t n = 0; n < batch; n++) {
        // Each image in batch gets its own random crop position
        size_t top = top_dist(gen);
        size_t left = left_dist(gen);

        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < crop_height; h++) {
                for (size_t w = 0; w < crop_width; w++) {
                    size_t src_idx, dst_idx;
                    if (shape.size() == 4) {
                        src_idx = n * channels * height * width + c * height * width +
                                  (top + h) * width + (left + w);
                        dst_idx = n * channels * crop_height * crop_width + c * crop_height * crop_width +
                                  h * crop_width + w;
                    } else {
                        src_idx = c * height * width + (top + h) * width + (left + w);
                        dst_idx = c * crop_height * crop_width + h * crop_width + w;
                    }
                    result->data[dst_idx] = data[src_idx];
                }
            }
        }
    }

    return result;
}

TensorPtr operator*(float scalar, const TensorPtr& t) {
    return t->mul(scalar);
}
