#include "cuda_backend.h"
#include "../device.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace whitematter {

CUDABackend& CUDABackend::instance() {
    static CUDABackend inst;
    return inst;
}

void CUDABackend::init() {
    if (available_) return;
    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0)
        return;
    available_ = true;
}

bool CUDABackend::is_available() const {
    const_cast<CUDABackend*>(this)->init();
    return available_;
}

void CUDABackend::matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    init();
    if (!available_) return;

    size_t aBytes = (size_t)M * K * sizeof(float);
    size_t bBytes = (size_t)K * N * sizeof(float);
    size_t cBytes = (size_t)M * N * sizeof(float);
    size_t cTBytes = (size_t)N * M * sizeof(float);  // (A*B)^T

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    if (cudaMalloc(&d_A, aBytes) != cudaSuccess) return;
    if (cudaMalloc(&d_B, bBytes) != cudaSuccess) { cudaFree(d_A); return; }
    if (cudaMalloc(&d_C, cTBytes) != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); return; }

    if (cudaMemcpy(d_A, A, aBytes, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(d_B, B, bBytes, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;

    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) goto cleanup;

    float alpha = 1.0f;
    float beta = 0.0f;
    // cuBLAS is column-major. (A*B)^T = B^T * A^T. Pass B (KxN) ldb=N -> NxK, A (MxK) lda=K -> KxM.
    cublasStatus_t st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    N, M, K,
                                    &alpha,
                                    d_B, N,
                                    d_A, K,
                                    &beta,
                                    d_C, N);

    cublasDestroy(handle);
    if (st != CUBLAS_STATUS_SUCCESS) goto cleanup;

    // d_C holds (A*B)^T column-major NxM. Transpose to row-major C (MxN).
    std::vector<float> temp((size_t)N * M);
    if (cudaMemcpy(temp.data(), d_C, cTBytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto cleanup;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] = temp[j * N + i];

cleanup:
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void CUDABackend::bmm(const float* A, const float* B, float* C, int batch, int M, int K, int N) {
    init();
    if (!available_) return;

    size_t aBytes = (size_t)batch * M * K * sizeof(float);
    size_t bBytes = (size_t)batch * K * N * sizeof(float);
    size_t cTBytes = (size_t)batch * N * M * sizeof(float);  // per-batch (A*B)^T

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    if (cudaMalloc(&d_A, aBytes) != cudaSuccess) return;
    if (cudaMalloc(&d_B, bBytes) != cudaSuccess) { cudaFree(d_A); return; }
    if (cudaMalloc(&d_C, cTBytes) != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); return; }

    if (cudaMemcpy(d_A, A, aBytes, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup_bmm;
    if (cudaMemcpy(d_B, B, bBytes, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup_bmm;

    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) goto cleanup_bmm;

    float alpha = 1.0f;
    float beta = 0.0f;
    long long int strideA = (long long int)M * K;
    long long int strideB = (long long int)K * N;
    long long int strideC = (long long int)N * M;

    cublasStatus_t st = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                  N, M, K,
                                                  &alpha,
                                                  d_B, N, strideB,
                                                  d_A, K, strideA,
                                                  &beta,
                                                  d_C, N, strideC,
                                                  batch);

    cublasDestroy(handle);
    if (st != CUBLAS_STATUS_SUCCESS) goto cleanup_bmm;

    size_t cBytes = (size_t)batch * M * N * sizeof(float);
    std::vector<float> temp((size_t)batch * N * M);
    if (cudaMemcpy(temp.data(), d_C, cTBytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto cleanup_bmm;
    for (int b = 0; b < batch; b++) {
        const float* src = temp.data() + (size_t)b * N * M;
        float* dst = C + (size_t)b * M * N;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                dst[i * N + j] = src[j * N + i];
    }

cleanup_bmm:
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

}  // namespace whitematter

// Provide cuda_backend_available when CUDA backend is linked.
namespace whitematter {
bool cuda_backend_available() {
    return CUDABackend::instance().is_available();
}
}  // namespace whitematter
