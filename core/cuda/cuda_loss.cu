// Loss function CUDA kernels for whitematter.
// Fused cross-entropy (softmax + NLL) and MSE loss.

#include "cuda_check.h"
#include <cuda_runtime.h>

namespace whitematter {

static constexpr int kBlockSize = 256;

// ---------------------------------------------------------------------------
// Fused cross-entropy loss
// Forward: computes softmax, then NLL loss, stores grad for backward
// Input: logits [batch, num_classes], targets [batch] (class indices as float)
// Output: loss (scalar on device), grad_logits [batch, num_classes]
//
// Each thread handles one sample in the batch.
// For large num_classes this is fine since the inner loops are sequential per sample.
// ---------------------------------------------------------------------------

__global__ void cross_entropy_forward_kernel(const float* logits, const float* targets,
                                              float* loss, float* grad_logits,
                                              int batch, int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    int target = (int)targets[b];
    const float* row = logits + b * num_classes;
    float* grad_row = grad_logits + b * num_classes;

    // Find max for numerical stability
    float max_val = row[0];
    for (int j = 1; j < num_classes; j++) {
        max_val = fmaxf(max_val, row[j]);
    }

    // Compute sum of exp(logit - max)
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; j++) {
        sum_exp += expf(row[j] - max_val);
    }
    float log_sum = logf(sum_exp);

    // Loss contribution for this sample: -log(softmax[target])
    // = -(row[target] - max_val - log_sum)
    float sample_loss = -(row[target] - max_val - log_sum);
    atomicAdd(loss, sample_loss / (float)batch);

    // Gradient: softmax(j) - 1{j == target}, divided by batch for mean reduction
    float inv_batch = 1.0f / (float)batch;
    for (int j = 0; j < num_classes; j++) {
        float softmax_j = expf(row[j] - max_val) / sum_exp;
        grad_row[j] = (softmax_j - (j == target ? 1.0f : 0.0f)) * inv_batch;
    }
}

void cuda_cross_entropy_forward(const float* d_logits, const float* d_targets,
                                float* d_loss, float* d_grad_logits,
                                int batch, int num_classes) {
    // Zero the loss scalar before accumulating
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));

    int grid = (batch + kBlockSize - 1) / kBlockSize;
    cross_entropy_forward_kernel<<<grid, kBlockSize>>>(d_logits, d_targets,
                                                        d_loss, d_grad_logits,
                                                        batch, num_classes);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Cross-entropy backward: gradient was already computed in forward.
// This kernel scales pre-computed gradients by an additional factor if needed.
// For standard use, the gradient is already in d_grad_logits after forward.
// This function applies upstream gradient scaling: grad_in *= upstream_grad.
// ---------------------------------------------------------------------------

__global__ void cross_entropy_backward_scale_kernel(float* grad_logits, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    grad_logits[i] *= scale;
}

void cuda_cross_entropy_backward(float* d_grad_logits, float upstream_grad,
                                 int batch, int num_classes) {
    if (upstream_grad == 1.0f) return;  // No scaling needed
    int n = batch * num_classes;
    int grid = (n + kBlockSize - 1) / kBlockSize;
    cross_entropy_backward_scale_kernel<<<grid, kBlockSize>>>(d_grad_logits, upstream_grad, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// MSE loss: L = (1/N) * sum((pred - target)^2)
// grad = 2 * (pred - target) / N
// ---------------------------------------------------------------------------

__global__ void mse_forward_kernel(const float* pred, const float* target,
                                    float* loss, float* grad,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float diff = pred[i] - target[i];
    atomicAdd(loss, diff * diff / (float)n);
    grad[i] = 2.0f * diff / (float)n;
}

void cuda_mse_forward(const float* d_pred, const float* d_target,
                      float* d_loss, float* d_grad,
                      int n) {
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    int grid = (n + kBlockSize - 1) / kBlockSize;
    mse_forward_kernel<<<grid, kBlockSize>>>(d_pred, d_target, d_loss, d_grad, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Binary cross-entropy loss
// L = -(1/N) * sum(target * log(pred) + (1 - target) * log(1 - pred))
// grad = (1/N) * (pred - target) / (pred * (1 - pred))
// Clamped for numerical stability.
// ---------------------------------------------------------------------------

__global__ void bce_forward_kernel(const float* pred, const float* target,
                                    float* loss, float* grad,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float p = fminf(fmaxf(pred[i], 1e-7f), 1.0f - 1e-7f);
    float t = target[i];
    float sample_loss = -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
    atomicAdd(loss, sample_loss / (float)n);
    grad[i] = ((p - t) / (p * (1.0f - p))) / (float)n;
}

void cuda_bce_forward(const float* d_pred, const float* d_target,
                      float* d_loss, float* d_grad,
                      int n) {
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    int grid = (n + kBlockSize - 1) / kBlockSize;
    bce_forward_kernel<<<grid, kBlockSize>>>(d_pred, d_target, d_loss, d_grad, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// NLL loss (assumes log-probabilities as input)
// L = -(1/N) * sum(input[i, target[i]])
// grad_input[i, j] = -1/N if j == target[i], else 0
// ---------------------------------------------------------------------------

__global__ void nll_forward_kernel(const float* log_probs, const float* targets,
                                    float* loss, float* grad,
                                    int batch, int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    int target = (int)targets[b];
    float* grad_row = grad + b * num_classes;

    // Loss contribution
    atomicAdd(loss, -log_probs[b * num_classes + target] / (float)batch);

    // Gradient
    float inv_batch = 1.0f / (float)batch;
    for (int j = 0; j < num_classes; j++) {
        grad_row[j] = (j == target) ? -inv_batch : 0.0f;
    }
}

void cuda_nll_forward(const float* d_log_probs, const float* d_targets,
                      float* d_loss, float* d_grad,
                      int batch, int num_classes) {
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    // Zero gradient
    CUDA_CHECK(cudaMemset(d_grad, 0, batch * num_classes * sizeof(float)));
    int grid = (batch + kBlockSize - 1) / kBlockSize;
    nll_forward_kernel<<<grid, kBlockSize>>>(d_log_probs, d_targets,
                                              d_loss, d_grad,
                                              batch, num_classes);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace whitematter
