#include "cuda_backend.h"
#include "cuda_check.h"
#include "../device.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstring>
#include <cstdio>

namespace whitematter {

// ---------------------------------------------------------------------------
// Singleton & initialization
// ---------------------------------------------------------------------------

CUDABackend& CUDABackend::instance() {
    static CUDABackend inst;
    return inst;
}

void CUDABackend::init() {
    if (initialized_) return;
    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0)
        return;

    // Create persistent cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cublas_handle_ = static_cast<void*>(handle);

    // Create CUDA stream
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);

    // Set cuBLAS to use our stream
    CUBLAS_CHECK(cublasSetStream(handle, s));

    // Allocate 8 MB workspace for cuDNN etc. (kept small for WSL2/low-VRAM GPUs)
    workspace_size_ = 8 * 1024 * 1024;
    CUDA_CHECK(cudaMallocManaged(&workspace_, workspace_size_));

    // cuDNN handle would be created here if cudnn is linked:
    // cudnnHandle_t dnn; cudnnCreate(&dnn); cudnn_handle_ = dnn;
    cudnn_handle_ = nullptr;

    initialized_ = true;
}

bool CUDABackend::is_available() const {
    const_cast<CUDABackend*>(this)->init();
    return initialized_;
}

// ---------------------------------------------------------------------------
// BLAS: matmul  C[M,N] = A[M,K] * B[K,N]  (row-major)
// ---------------------------------------------------------------------------

void CUDABackend::matmul(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    init();
    if (!initialized_) return;

    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle_);
    float alpha = 1.0f, beta = 0.0f;

    // Row-major A[M,K] * B[K,N] = C[M,N].
    // cuBLAS is column-major. Treat row-major A as col-major A^T (K x M), etc.
    // C^T = B^T * A^T => col-major: C(N,M) = B(N,K) * A(K,M)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B, N,
                             d_A, K,
                             &beta,
                             d_C, N));
}

// ---------------------------------------------------------------------------
// BLAS: batched matmul
// ---------------------------------------------------------------------------

void CUDABackend::bmm(const float* d_A, const float* d_B, float* d_C,
                      int batch, int M, int K, int N) {
    init();
    if (!initialized_) return;

    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle_);
    float alpha = 1.0f, beta = 0.0f;
    long long int strideA = (long long int)M * K;
    long long int strideB = (long long int)K * N;
    long long int strideC = (long long int)M * N;

    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                           N, M, K,
                                           &alpha,
                                           d_B, N, strideB,
                                           d_A, K, strideA,
                                           &beta,
                                           d_C, N, strideC,
                                           batch));
}

// ---------------------------------------------------------------------------
// Memory transfers
// ---------------------------------------------------------------------------

void CUDABackend::memcpy_h2d(float* d_dst, const float* h_src, size_t n_floats) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, n_floats * sizeof(float), cudaMemcpyHostToDevice));
}

void CUDABackend::memcpy_d2h(float* h_dst, const float* d_src, size_t n_floats) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, n_floats * sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDABackend::memcpy_d2d(float* d_dst, const float* d_src, size_t n_floats) {
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, n_floats * sizeof(float), cudaMemcpyDeviceToDevice));
}

void CUDABackend::memset_zero(float* d_ptr, size_t n_floats) {
    CUDA_CHECK(cudaMemset(d_ptr, 0, n_floats * sizeof(float)));
}

// ---------------------------------------------------------------------------
// Element-wise stubs (to be implemented in separate kernel files)
// ---------------------------------------------------------------------------

void CUDABackend::elementwise_add(const float*, const float*, float*, size_t) {
    // TODO: implement CUDA kernel
}

void CUDABackend::elementwise_sub(const float*, const float*, float*, size_t) {
    // TODO: implement CUDA kernel
}

void CUDABackend::elementwise_mul(const float*, const float*, float*, size_t) {
    // TODO: implement CUDA kernel
}

void CUDABackend::elementwise_div(const float*, const float*, float*, size_t) {
    // TODO: implement CUDA kernel
}

void CUDABackend::scalar_mul(const float*, float, float*, size_t) {
    // TODO: implement CUDA kernel
}

void CUDABackend::fill(float*, float, size_t) {
    // TODO: implement CUDA kernel
}

// ---------------------------------------------------------------------------
// Activation stubs
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Reduction stubs
// ---------------------------------------------------------------------------

void CUDABackend::sum(const float*, float*, size_t) {}
void CUDABackend::softmax_forward(const float*, float*, size_t, size_t) {}
void CUDABackend::log_softmax_forward(const float*, float*, size_t, size_t) {}

// ---------------------------------------------------------------------------
// Conv2d stubs (cuDNN)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// BatchNorm stubs (cuDNN)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Pooling stubs
// ---------------------------------------------------------------------------

void CUDABackend::maxpool2d_forward(const float*, float*, int*,
                                    size_t, size_t, size_t, size_t,
                                    size_t, size_t) {}

void CUDABackend::avgpool2d_forward(const float*, float*,
                                    size_t, size_t, size_t, size_t,
                                    size_t, size_t) {}

// ---------------------------------------------------------------------------
// Loss stubs
// ---------------------------------------------------------------------------

void CUDABackend::cross_entropy_forward(const float*, const float*,
                                        float*, float*,
                                        size_t, size_t) {}

// ---------------------------------------------------------------------------
// Optimizer stubs
// ---------------------------------------------------------------------------

void CUDABackend::sgd_step(float*, const float*, float*,
                           float, float, size_t) {}

void CUDABackend::adam_step(float*, const float*, float*, float*,
                            float, float, float, float,
                            float, float, size_t) {}

void CUDABackend::adamw_step(float*, const float*, float*, float*,
                              float, float, float, float,
                              float, float, float, size_t) {}

void CUDABackend::rmsprop_step(float*, const float*, float*,
                                float, float, float,
                                float, float*, float, size_t) {}

}  // namespace whitematter

// Provide cuda_backend_available when CUDA backend is linked.
namespace whitematter {
bool cuda_backend_available() {
    return CUDABackend::instance().is_available();
}
}  // namespace whitematter
