// Activation function CUDA kernels for whitematter.
// Forward and backward for ReLU, Sigmoid, Tanh, SiLU, GELU.

#include "cuda_check.h"
#include <cuda_runtime.h>
#include <cmath>

namespace whitematter {

static constexpr int kBlockSize = 256;
static int grid(int n) { return (n + kBlockSize - 1) / kBlockSize; }

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr float kSqrt2     = 1.4142135623730951f;   // sqrt(2)
static constexpr float kSqrt2Pi   = 2.5066282746310002f;   // sqrt(2*pi)
static constexpr float kInvSqrt2  = 0.7071067811865476f;   // 1/sqrt(2)

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

__global__ void relu_forward_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(0.0f, x[i]);
}

__global__ void relu_backward_kernel(const float* grad_out, const float* input, float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] = grad_out[i] * (input[i] > 0.0f ? 1.0f : 0.0f);
}

// ---------------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------------

__global__ void sigmoid_forward_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i]));
}

__global__ void sigmoid_backward_kernel(const float* grad_out, const float* output, float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = output[i];
        grad_in[i] = grad_out[i] * s * (1.0f - s);
    }
}

// ---------------------------------------------------------------------------
// Tanh
// ---------------------------------------------------------------------------

__global__ void tanh_forward_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanhf(x[i]);
}

__global__ void tanh_backward_kernel(const float* grad_out, const float* output, float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float t = output[i];
        grad_in[i] = grad_out[i] * (1.0f - t * t);
    }
}

// ---------------------------------------------------------------------------
// SiLU (Swish): x * sigmoid(x)
// ---------------------------------------------------------------------------

__global__ void silu_forward_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sig = 1.0f / (1.0f + expf(-x[i]));
        y[i] = x[i] * sig;
    }
}

__global__ void silu_backward_kernel(const float* grad_out, const float* input, float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = input[i];
        float sig = 1.0f / (1.0f + expf(-xi));
        // d/dx [x * sig(x)] = sig + x * sig * (1 - sig)
        grad_in[i] = grad_out[i] * (sig + xi * sig * (1.0f - sig));
    }
}

// ---------------------------------------------------------------------------
// GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
// ---------------------------------------------------------------------------

__global__ void gelu_forward_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float cdf = 0.5f * (1.0f + erff(xi * kInvSqrt2));
        y[i] = xi * cdf;
    }
}

__global__ void gelu_backward_kernel(const float* grad_out, const float* input, float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = input[i];
        float cdf = 0.5f * (1.0f + erff(xi * kInvSqrt2));
        float pdf = expf(-0.5f * xi * xi) / kSqrt2Pi;
        grad_in[i] = grad_out[i] * (cdf + xi * pdf);
    }
}

// ---------------------------------------------------------------------------
// Launch wrappers
// ---------------------------------------------------------------------------

// ReLU
void cuda_relu_forward(const float* x, float* y, size_t n) {
    relu_forward_kernel<<<grid((int)n), kBlockSize>>>(x, y, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_relu_backward(const float* grad_out, const float* input, float* grad_in, size_t n) {
    relu_backward_kernel<<<grid((int)n), kBlockSize>>>(grad_out, input, grad_in, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

// Sigmoid
void cuda_sigmoid_forward(const float* x, float* y, size_t n) {
    sigmoid_forward_kernel<<<grid((int)n), kBlockSize>>>(x, y, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_sigmoid_backward(const float* grad_out, const float* output, float* grad_in, size_t n) {
    sigmoid_backward_kernel<<<grid((int)n), kBlockSize>>>(grad_out, output, grad_in, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

// Tanh
void cuda_tanh_forward(const float* x, float* y, size_t n) {
    tanh_forward_kernel<<<grid((int)n), kBlockSize>>>(x, y, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_tanh_backward(const float* grad_out, const float* output, float* grad_in, size_t n) {
    tanh_backward_kernel<<<grid((int)n), kBlockSize>>>(grad_out, output, grad_in, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

// SiLU
void cuda_silu_forward(const float* x, float* y, size_t n) {
    silu_forward_kernel<<<grid((int)n), kBlockSize>>>(x, y, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_silu_backward(const float* grad_out, const float* input, float* grad_in, size_t n) {
    silu_backward_kernel<<<grid((int)n), kBlockSize>>>(grad_out, input, grad_in, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

// GELU
void cuda_gelu_forward(const float* x, float* y, size_t n) {
    gelu_forward_kernel<<<grid((int)n), kBlockSize>>>(x, y, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_gelu_backward(const float* grad_out, const float* input, float* grad_in, size_t n) {
    gelu_backward_kernel<<<grid((int)n), kBlockSize>>>(grad_out, input, grad_in, (int)n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace whitematter
