// Stub implementations when CUDA is not built (default or no nvcc).
// Provides all symbols referenced by other translation units so the
// non-CUDA build links cleanly.
#include "device.h"

namespace whitematter {

bool cuda_backend_available() { return false; }

}  // namespace whitematter

// ---------------------------------------------------------------------------
// CUDABackend stubs — included only when NOT building with CUDA=1.
// The header is always visible (it is a plain C++ header with no CUDA
// dependencies), so other code can reference CUDABackend without #ifdefs
// as long as these stubs supply the definitions.
// ---------------------------------------------------------------------------
#include "cuda/cuda_backend.h"

namespace whitematter {

CUDABackend& CUDABackend::instance() {
    static CUDABackend inst;
    return inst;
}

void CUDABackend::init() {}

bool CUDABackend::is_available() const { return false; }

// BLAS
void CUDABackend::matmul(const float*, const float*, float*, int, int, int) {}
void CUDABackend::bmm(const float*, const float*, float*, int, int, int, int) {}

// Element-wise
void CUDABackend::elementwise_add(const float*, const float*, float*, size_t) {}
void CUDABackend::elementwise_sub(const float*, const float*, float*, size_t) {}
void CUDABackend::elementwise_mul(const float*, const float*, float*, size_t) {}
void CUDABackend::elementwise_div(const float*, const float*, float*, size_t) {}
void CUDABackend::scalar_mul(const float*, float, float*, size_t) {}
void CUDABackend::fill(float*, float, size_t) {}

// Activations
void CUDABackend::relu_forward(const float*, float*, size_t) {}
void CUDABackend::relu_backward(float*, const float*, const float*, size_t) {}
void CUDABackend::sigmoid_forward(const float*, float*, size_t) {}
void CUDABackend::sigmoid_backward(float*, const float*, const float*, size_t) {}
void CUDABackend::tanh_forward(const float*, float*, size_t) {}
void CUDABackend::tanh_backward(float*, const float*, const float*, size_t) {}
void CUDABackend::silu_forward(const float*, float*, size_t) {}
void CUDABackend::silu_backward(float*, const float*, const float*, size_t) {}
void CUDABackend::gelu_forward(const float*, float*, size_t) {}
void CUDABackend::gelu_backward(float*, const float*, const float*, size_t) {}

// Reductions
void CUDABackend::sum(const float*, float*, size_t) {}
void CUDABackend::softmax_forward(const float*, float*, size_t, size_t) {}
void CUDABackend::log_softmax_forward(const float*, float*, size_t, size_t) {}

// Conv2d
void CUDABackend::conv2d_forward(const float*, const float*, const float*,
                                 float*, size_t, size_t, size_t,
                                 size_t, size_t, size_t, size_t,
                                 size_t, size_t, size_t) {}
void CUDABackend::conv2d_backward(const float*, const float*,
                                  const float*, float*,
                                  float*, float*,
                                  size_t, size_t, size_t,
                                  size_t, size_t, size_t, size_t,
                                  size_t, size_t, size_t) {}

// BatchNorm
void CUDABackend::batchnorm_forward(const float*, float*,
                                    const float*, const float*,
                                    float*, float*,
                                    float*, float*,
                                    size_t, size_t, size_t,
                                    float, float, bool) {}
void CUDABackend::batchnorm_backward(const float*, const float*,
                                     float*, float*, float*,
                                     const float*, const float*,
                                     const float*,
                                     size_t, size_t, size_t, float) {}

// Pooling
void CUDABackend::maxpool2d_forward(const float*, float*, int*,
                                    size_t, size_t, size_t, size_t,
                                    size_t, size_t) {}
void CUDABackend::avgpool2d_forward(const float*, float*,
                                    size_t, size_t, size_t, size_t,
                                    size_t, size_t) {}

// Loss
void CUDABackend::cross_entropy_forward(const float*, const float*,
                                        float*, float*,
                                        size_t, size_t) {}

// Optimizer
void CUDABackend::sgd_step(float*, const float*, float*,
                           float, float, size_t) {}
void CUDABackend::adam_step(float*, const float*, float*, float*,
                            float, float, float, float,
                            float, float, size_t) {}

// Weight cache (no-op without CUDA)
void CUDABackend::invalidate_weight_cache() {}

// Memory transfers (no-op without CUDA)
void CUDABackend::memcpy_h2d(float*, const float*, size_t) {}
void CUDABackend::memcpy_d2h(float*, const float*, size_t) {}
void CUDABackend::memcpy_d2d(float*, const float*, size_t) {}
void CUDABackend::memset_zero(float*, size_t) {}

}  // namespace whitematter

// ---------------------------------------------------------------------------
// CUDAMemoryPool stubs
// ---------------------------------------------------------------------------
#include "cuda/cuda_memory.h"

namespace whitematter {

CUDAMemoryPool& CUDAMemoryPool::instance() {
    static CUDAMemoryPool pool;
    return pool;
}

size_t CUDAMemoryPool::size_class(size_t n) {
    if (n <= 1) return 1;
    size_t v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

float* CUDAMemoryPool::acquire(size_t) { return nullptr; }
std::shared_ptr<float> CUDAMemoryPool::acquire_shared(size_t) { return nullptr; }
void CUDAMemoryPool::release(float*, size_t) {}
CUDAMemoryPool::~CUDAMemoryPool() {}

}  // namespace whitematter
