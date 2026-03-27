// Element-wise CUDA kernels for whitematter.
// 256 threads per block, grid computed from element count.

#include "cuda_check.h"
#include <cuda_runtime.h>

namespace whitematter {

// ---------------------------------------------------------------------------
// Forward kernels
// ---------------------------------------------------------------------------

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void sub_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];
}

__global__ void mul_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

__global__ void div_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] / b[i];
}

__global__ void scalar_mul_kernel(const float* a, float s, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * s;
}

__global__ void fill_kernel(float* a, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = val;
}

// ---------------------------------------------------------------------------
// Backward kernels
// ---------------------------------------------------------------------------

// add_backward: grad_a = grad_out, grad_b = grad_out (passthrough)
__global__ void add_backward_kernel(const float* grad_out, float* grad_a, float* grad_b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_a[i] = grad_out[i];
        grad_b[i] = grad_out[i];
    }
}

// sub_backward: grad_a = grad_out, grad_b = -grad_out
__global__ void sub_backward_kernel(const float* grad_out, float* grad_a, float* grad_b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_a[i] = grad_out[i];
        grad_b[i] = -grad_out[i];
    }
}

// mul_backward: grad_a = grad_out * b, grad_b = grad_out * a
__global__ void mul_backward_kernel(const float* grad_out, const float* a, const float* b,
                                    float* grad_a, float* grad_b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_a[i] = grad_out[i] * b[i];
        grad_b[i] = grad_out[i] * a[i];
    }
}

// div_backward: grad_a = grad_out / b, grad_b = -grad_out * a / (b * b)
__global__ void div_backward_kernel(const float* grad_out, const float* a, const float* b,
                                    float* grad_a, float* grad_b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_a[i] = grad_out[i] / b[i];
        grad_b[i] = -grad_out[i] * a[i] / (b[i] * b[i]);
    }
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

static constexpr int kBlockSize = 256;

static int grid(int n) { return (n + kBlockSize - 1) / kBlockSize; }

void cuda_elementwise_add(const float* a, const float* b, float* c, size_t n) {
    add_kernel<<<grid((int)n), kBlockSize>>>(a, b, c, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_elementwise_sub(const float* a, const float* b, float* c, size_t n) {
    sub_kernel<<<grid((int)n), kBlockSize>>>(a, b, c, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_elementwise_mul(const float* a, const float* b, float* c, size_t n) {
    mul_kernel<<<grid((int)n), kBlockSize>>>(a, b, c, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_elementwise_div(const float* a, const float* b, float* c, size_t n) {
    div_kernel<<<grid((int)n), kBlockSize>>>(a, b, c, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_scalar_mul(const float* a, float s, float* c, size_t n) {
    scalar_mul_kernel<<<grid((int)n), kBlockSize>>>(a, s, c, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_fill(float* a, float val, size_t n) {
    fill_kernel<<<grid((int)n), kBlockSize>>>(a, val, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_add_backward(const float* grad_out, float* grad_a, float* grad_b, size_t n) {
    add_backward_kernel<<<grid((int)n), kBlockSize>>>(grad_out, grad_a, grad_b, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_sub_backward(const float* grad_out, float* grad_a, float* grad_b, size_t n) {
    sub_backward_kernel<<<grid((int)n), kBlockSize>>>(grad_out, grad_a, grad_b, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_mul_backward(const float* grad_out, const float* a, const float* b,
                       float* grad_a, float* grad_b, size_t n) {
    mul_backward_kernel<<<grid((int)n), kBlockSize>>>(grad_out, a, b, grad_a, grad_b, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_div_backward(const float* grad_out, const float* a, const float* b,
                       float* grad_a, float* grad_b, size_t n) {
    div_backward_kernel<<<grid((int)n), kBlockSize>>>(grad_out, a, b, grad_a, grad_b, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace whitematter
