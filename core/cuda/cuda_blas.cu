// cuBLAS wrapper kernels for whitematter.
// Matmul and batched matmul using cublasSgemm / cublasSgemmStridedBatched.

#include "cuda_check.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace whitematter {

// ---------------------------------------------------------------------------
// Matmul: C = A @ B (row-major)
// A is [M, K], B is [K, N], C is [M, N]
//
// cuBLAS uses column-major. Row-major A*B can be computed as:
//   In column-major terms, the row-major matrices look transposed.
//   C_rm = A_rm * B_rm  <==>  C_cm^T = A_cm^T * B_cm^T
//   So C_cm = (A_cm^T * B_cm^T)^T = B_cm * A_cm
//   We call cublasSgemm(N, N, N, M, K, alpha, B, N, A, K, beta, C, N)
// ---------------------------------------------------------------------------

void cuda_matmul(void* cublas_handle, const float* d_A, const float* d_B, float* d_C,
                 int M, int N, int K) {
    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle);
    float alpha = 1.0f, beta = 0.0f;
    // Row-major trick: compute B^T @ A^T = (A @ B)^T in column-major = A @ B in row-major
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B, N,
                             d_A, K,
                             &beta,
                             d_C, N));
}

// ---------------------------------------------------------------------------
// Matmul with transpose options
// C = op(A) @ op(B), where op is identity or transpose
// transA/transB: false = no transpose, true = transpose
// If transA=false: A is [M, K];  if transA=true: A is [K, M] (transposed to [M, K])
// If transB=false: B is [K, N];  if transB=true: B is [N, K] (transposed to [K, N])
// C is always [M, N]
// ---------------------------------------------------------------------------

void cuda_matmul_ex(void* cublas_handle, const float* d_A, const float* d_B, float* d_C,
                    int M, int N, int K, bool transA, bool transB,
                    float alpha, float beta) {
    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle);

    // In column-major cuBLAS we reverse operand order:
    // C_rm(M,N) = opA(A) * opB(B) => in column-major: C_cm = opB(B_cm) * opA(A_cm)
    // where B_cm is B viewed as column-major (i.e. B transposed), etc.

    // For row-major A[M,K]: column-major sees it as [K,M], so opA_cm = T if !transA, N if transA
    // For row-major B[K,N]: column-major sees it as [N,K], so opB_cm = T if !transB, N if transB
    cublasOperation_t cuTransB = transB ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransA = transA ? CUBLAS_OP_N : CUBLAS_OP_T;

    // Leading dimensions in memory (row-major):
    // A is stored as [rows_A, cols_A] contiguously.
    // If transA=false: A is [M,K], lda_rm = K; col-major sees [K,M], lda_cm = K
    // If transA=true:  A is [K,M], lda_rm = M; col-major sees [M,K], lda_cm = M
    int ldA = transA ? M : K;
    int ldB = transB ? K : N;
    int ldC = N;

    CUBLAS_CHECK(cublasSgemm(handle, cuTransB, cuTransA,
                             N, M, K,
                             &alpha,
                             d_B, ldB,
                             d_A, ldA,
                             &beta,
                             d_C, ldC));
}

// ---------------------------------------------------------------------------
// Batched matmul: C[i] = A[i] @ B[i]
// A[i] is [M, K], B[i] is [K, N], C[i] is [M, N]
// Matrices are contiguous: A_total is [batch, M, K], etc.
// ---------------------------------------------------------------------------

void cuda_bmm(void* cublas_handle, const float* d_A, const float* d_B, float* d_C,
              int batch, int M, int K, int N) {
    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle);
    float alpha = 1.0f, beta = 0.0f;
    long long strideA = (long long)M * K;
    long long strideB = (long long)K * N;
    long long strideC = (long long)M * N;

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
// GEMV: y = alpha * A @ x + beta * y
// A is [M, N] row-major, x is [N], y is [M]
// ---------------------------------------------------------------------------

void cuda_gemv(void* cublas_handle, const float* d_A, const float* d_x, float* d_y,
               int M, int N, float alpha, float beta) {
    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle);

    // Row-major A[M,N] in column-major is A^T[N,M].
    // y = A_rm * x => y = A_cm^T * x
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T,
                             N, M,  // A_cm is [N,M]
                             &alpha,
                             d_A, N,  // lda = N (row stride in row-major)
                             d_x, 1,
                             &beta,
                             d_y, 1));
}

// ---------------------------------------------------------------------------
// axpy: y = alpha * x + y
// ---------------------------------------------------------------------------

void cuda_axpy(void* cublas_handle, const float* d_x, float* d_y, float alpha, int n) {
    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle);
    CUBLAS_CHECK(cublasSaxpy(handle, n, &alpha, d_x, 1, d_y, 1));
}

} // namespace whitematter
