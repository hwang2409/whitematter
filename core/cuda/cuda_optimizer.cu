// Optimizer step CUDA kernels for whitematter.
// SGD (with momentum), Adam, AdamW, and RMSprop.

#include "cuda_check.h"
#include <cuda_runtime.h>
#include <cmath>

namespace whitematter {

static constexpr int kBlockSize = 256;
static int grid(int n) { return (n + kBlockSize - 1) / kBlockSize; }

// ---------------------------------------------------------------------------
// SGD with momentum
// v = momentum * v + grad
// param -= lr * v
// If momentum == 0, simplifies to param -= lr * grad.
// ---------------------------------------------------------------------------

__global__ void sgd_step_kernel(float* params, const float* grads, float* velocity,
                                 float lr, float momentum, float weight_decay, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grads[i];
    // L2 regularization (weight decay)
    if (weight_decay != 0.0f) {
        g += weight_decay * params[i];
    }

    float v = momentum * velocity[i] + g;
    velocity[i] = v;
    params[i] -= lr * v;
}

__global__ void sgd_step_no_momentum_kernel(float* params, const float* grads,
                                             float lr, float weight_decay, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grads[i];
    if (weight_decay != 0.0f) {
        g += weight_decay * params[i];
    }
    params[i] -= lr * g;
}

void cuda_sgd_step(float* params, const float* grads, float* velocity,
                   float lr, float momentum, float weight_decay, int n) {
    if (momentum == 0.0f && velocity == nullptr) {
        sgd_step_no_momentum_kernel<<<grid(n), kBlockSize>>>(params, grads, lr, weight_decay, n);
    } else {
        sgd_step_kernel<<<grid(n), kBlockSize>>>(params, grads, velocity, lr, momentum, weight_decay, n);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Adam optimizer
// m = beta1 * m + (1 - beta1) * g
// v = beta2 * v + (1 - beta2) * g^2
// m_hat = m / (1 - beta1^t)    [bc1 = 1 - beta1^t]
// v_hat = v / (1 - beta2^t)    [bc2 = 1 - beta2^t]
// param -= lr * m_hat / (sqrt(v_hat) + eps)
// ---------------------------------------------------------------------------

__global__ void adam_step_kernel(float* params, const float* grads,
                                  float* m, float* v,
                                  float lr, float beta1, float beta2, float eps,
                                  float bc1, float bc2, float weight_decay, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grads[i];
    // L2 regularization
    if (weight_decay != 0.0f) {
        g += weight_decay * params[i];
    }

    // Update biased first moment estimate
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    // Update biased second raw moment estimate
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

    // Bias-corrected estimates
    float m_hat = m[i] / bc1;
    float v_hat = v[i] / bc2;

    // Update parameters
    params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

void cuda_adam_step(float* params, const float* grads,
                    float* m, float* v,
                    float lr, float beta1, float beta2, float eps,
                    float bc1, float bc2, float weight_decay, int n) {
    adam_step_kernel<<<grid(n), kBlockSize>>>(params, grads, m, v,
                                               lr, beta1, beta2, eps,
                                               bc1, bc2, weight_decay, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// AdamW optimizer (decoupled weight decay)
// Like Adam but weight decay is applied directly to params, not to gradient.
// param -= lr * weight_decay * param    (decoupled decay step)
// Then standard Adam update without weight_decay in gradient.
// ---------------------------------------------------------------------------

__global__ void adamw_step_kernel(float* params, const float* grads,
                                   float* m, float* v,
                                   float lr, float beta1, float beta2, float eps,
                                   float bc1, float bc2, float weight_decay, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Decoupled weight decay
    if (weight_decay != 0.0f) {
        params[i] -= lr * weight_decay * params[i];
    }

    float g = grads[i];

    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

    float m_hat = m[i] / bc1;
    float v_hat = v[i] / bc2;

    params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

void cuda_adamw_step(float* params, const float* grads,
                     float* m, float* v,
                     float lr, float beta1, float beta2, float eps,
                     float bc1, float bc2, float weight_decay, int n) {
    adamw_step_kernel<<<grid(n), kBlockSize>>>(params, grads, m, v,
                                                lr, beta1, beta2, eps,
                                                bc1, bc2, weight_decay, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// RMSprop optimizer
// v = rho * v + (1 - rho) * g^2
// param -= lr * g / (sqrt(v) + eps)
// ---------------------------------------------------------------------------

__global__ void rmsprop_step_kernel(float* params, const float* grads, float* v,
                                     float lr, float rho, float eps,
                                     float weight_decay, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grads[i];
    if (weight_decay != 0.0f) {
        g += weight_decay * params[i];
    }

    v[i] = rho * v[i] + (1.0f - rho) * g * g;
    params[i] -= lr * g / (sqrtf(v[i]) + eps);
}

void cuda_rmsprop_step(float* params, const float* grads, float* v,
                       float lr, float rho, float eps,
                       float weight_decay, int n) {
    rmsprop_step_kernel<<<grid(n), kBlockSize>>>(params, grads, v,
                                                  lr, rho, eps,
                                                  weight_decay, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Gradient clipping by norm
// Computes global norm, then scales all gradients if norm > max_norm.
// Two-phase: first compute norm (reduction), then clip.
// ---------------------------------------------------------------------------

__global__ void grad_norm_partial_kernel(const float* grads, float* partials, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float val = 0.0f;
    if (i < n) val = grads[i] * grads[i];
    if (i + blockDim.x < n) val += grads[i + blockDim.x] * grads[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

__global__ void grad_clip_kernel(float* grads, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    grads[i] *= scale;
}

void cuda_clip_grad_norm(float* grads, float max_norm, int n) {
    // Phase 1: compute squared norm
    int num_blocks = (n + kBlockSize * 2 - 1) / (kBlockSize * 2);
    float* d_partials = nullptr;
    float* d_norm_sq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partials, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm_sq, sizeof(float)));

    grad_norm_partial_kernel<<<num_blocks, kBlockSize, kBlockSize * sizeof(float)>>>(grads, d_partials, n);
    CUDA_CHECK(cudaGetLastError());

    // Reduce partials (reuse sum_final-like logic)
    // For simplicity, copy partials to host
    float* h_partials = new float[num_blocks];
    CUDA_CHECK(cudaMemcpy(h_partials, d_partials, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));

    float norm_sq = 0.0f;
    for (int i = 0; i < num_blocks; i++) norm_sq += h_partials[i];
    delete[] h_partials;

    float norm = sqrtf(norm_sq);

    // Phase 2: clip if needed
    if (norm > max_norm) {
        float scale = max_norm / norm;
        grad_clip_kernel<<<grid(n), kBlockSize>>>(grads, scale, n);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(d_partials));
    CUDA_CHECK(cudaFree(d_norm_sq));
}

// ---------------------------------------------------------------------------
// Zero gradients (memset to 0)
// ---------------------------------------------------------------------------

void cuda_zero_grad(float* grads, int n) {
    CUDA_CHECK(cudaMemset(grads, 0, n * sizeof(float)));
}

} // namespace whitematter
