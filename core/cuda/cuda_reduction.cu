// Reduction CUDA kernels for whitematter.
// sum, softmax_forward, log_softmax_forward using shared memory.

#include "cuda_check.h"
#include <cuda_runtime.h>
#include <cfloat>

namespace whitematter {

// ---------------------------------------------------------------------------
// Sum reduction
// ---------------------------------------------------------------------------

// Phase 1: Each block reduces a chunk to a partial sum stored in partials[blockIdx.x].
__global__ void sum_partial_kernel(const float* x, float* partials, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements per thread for first-level reduction
    float val = 0.0f;
    if (i < n) val = x[i];
    if (i + blockDim.x < n) val += x[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    // Tree-based reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

// Phase 2: Reduce partial sums to a single value.
// Called with a single block.
__global__ void sum_final_kernel(const float* partials, float* result, int num_partials) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    float val = 0.0f;
    // Each thread accumulates multiple partials if num_partials > blockDim.x
    for (int i = tid; i < num_partials; i += blockDim.x) {
        val += partials[i];
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) *result = sdata[0];
}

void cuda_sum(const float* x, float* result, size_t n) {
    const int block_size = 256;
    // Each block handles 2 * block_size elements
    int num_blocks = ((int)n + block_size * 2 - 1) / (block_size * 2);

    float* d_partials = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partials, num_blocks * sizeof(float)));

    // Phase 1: partial sums
    sum_partial_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(x, d_partials, (int)n);
    CUDA_CHECK(cudaGetLastError());

    // Phase 2: reduce partials
    int final_threads = (num_blocks < 256) ? num_blocks : 256;
    // Round up to next power of 2 for clean reduction
    int p = 1;
    while (p < final_threads) p <<= 1;
    final_threads = p;
    if (final_threads > 256) final_threads = 256;

    sum_final_kernel<<<1, final_threads, final_threads * sizeof(float)>>>(d_partials, result, num_blocks);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_partials));
}

// ---------------------------------------------------------------------------
// Softmax forward — per-row for [rows, cols] matrix
// One block per row. Works for cols <= 1024.
// For cols > 1024 a multi-block path is used.
// ---------------------------------------------------------------------------

// Single-block-per-row softmax (cols <= 1024)
__global__ void softmax_forward_kernel(const float* x, float* y, int rows, int cols) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    const float* x_row = x + row * cols;
    float* y_row = y + row * cols;

    // Step 1: find row max
    float max_val = -FLT_MAX;
    for (int j = tid; j < cols; j += blockDim.x) {
        max_val = fmaxf(max_val, x_row[j]);
    }
    sdata[tid] = max_val;
    __syncthreads();

    // Reduce to find max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Step 2: compute exp(x - max) and sum
    float sum_exp = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float e = expf(x_row[j] - row_max);
        y_row[j] = e;
        sum_exp += e;
    }
    sdata[tid] = sum_exp;
    __syncthreads();

    // Reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float total = sdata[0];
    __syncthreads();

    // Step 3: normalize
    for (int j = tid; j < cols; j += blockDim.x) {
        y_row[j] /= total;
    }
}

// Multi-block softmax for cols > 1024:
// Phase 1: find per-row max using atomicMax-style approach
// Phase 2: compute exp and sum
// Phase 3: normalize
// We use a simpler 3-kernel approach with per-row max/sum stored in scratch arrays.

__global__ void softmax_find_max_kernel(const float* x, float* row_max, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    const float* x_row = x + row * cols;

    extern __shared__ float sdata[];
    float max_val = -FLT_MAX;
    for (int j = tid; j < cols; j += blockDim.x) {
        max_val = fmaxf(max_val, x_row[j]);
    }
    sdata[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) row_max[row] = sdata[0];
}

__global__ void softmax_exp_sum_kernel(const float* x, float* y, const float* row_max,
                                       float* row_sum, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    const float* x_row = x + row * cols;
    float* y_row = y + row * cols;
    float mx = row_max[row];

    extern __shared__ float sdata[];
    float sum_val = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float e = expf(x_row[j] - mx);
        y_row[j] = e;
        sum_val += e;
    }
    sdata[tid] = sum_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) row_sum[row] = sdata[0];
}

__global__ void softmax_normalize_kernel(float* y, const float* row_sum, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    float* y_row = y + row * cols;
    float inv_sum = 1.0f / row_sum[row];

    for (int j = tid; j < cols; j += blockDim.x) {
        y_row[j] *= inv_sum;
    }
}

void cuda_softmax_forward(const float* x, float* y, int rows, int cols) {
    if (cols <= 1024) {
        // Single-block-per-row path
        int threads = 1;
        while (threads < cols && threads < 1024) threads <<= 1;
        // Ensure power of 2 for clean reduction
        softmax_forward_kernel<<<rows, threads, threads * sizeof(float)>>>(x, y, rows, cols);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Multi-block path: 3 kernels, 1024 threads per row-block
        const int threads = 1024;
        float* d_row_max = nullptr;
        float* d_row_sum = nullptr;
        CUDA_CHECK(cudaMalloc(&d_row_max, rows * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_row_sum, rows * sizeof(float)));

        softmax_find_max_kernel<<<rows, threads, threads * sizeof(float)>>>(x, d_row_max, rows, cols);
        CUDA_CHECK(cudaGetLastError());

        softmax_exp_sum_kernel<<<rows, threads, threads * sizeof(float)>>>(x, y, d_row_max, d_row_sum, rows, cols);
        CUDA_CHECK(cudaGetLastError());

        softmax_normalize_kernel<<<rows, threads>>>(y, d_row_sum, rows, cols);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(d_row_max));
        CUDA_CHECK(cudaFree(d_row_sum));
    }
}

// ---------------------------------------------------------------------------
// Softmax backward
// grad_in[i,j] = y[i,j] * (grad_out[i,j] - dot(grad_out[i,:], y[i,:]))
// ---------------------------------------------------------------------------

__global__ void softmax_backward_kernel(const float* grad_out, const float* output,
                                        float* grad_in, int rows, int cols) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    const float* go_row = grad_out + row * cols;
    const float* y_row = output + row * cols;
    float* gi_row = grad_in + row * cols;

    // Compute dot(grad_out, y) for this row
    float dot = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        dot += go_row[j] * y_row[j];
    }
    sdata[tid] = dot;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_dot = sdata[0];
    __syncthreads();

    for (int j = tid; j < cols; j += blockDim.x) {
        gi_row[j] = y_row[j] * (go_row[j] - row_dot);
    }
}

void cuda_softmax_backward(const float* grad_out, const float* output,
                            float* grad_in, int rows, int cols) {
    int threads = 1;
    while (threads < cols && threads < 1024) threads <<= 1;
    softmax_backward_kernel<<<rows, threads, threads * sizeof(float)>>>(grad_out, output, grad_in, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Log-softmax forward: output = x - max - log(sum(exp(x - max)))
// ---------------------------------------------------------------------------

__global__ void log_softmax_forward_kernel(const float* x, float* y, int rows, int cols) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    const float* x_row = x + row * cols;
    float* y_row = y + row * cols;

    // Step 1: find row max
    float max_val = -FLT_MAX;
    for (int j = tid; j < cols; j += blockDim.x) {
        max_val = fmaxf(max_val, x_row[j]);
    }
    sdata[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Step 2: compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        sum_exp += expf(x_row[j] - row_max);
    }
    sdata[tid] = sum_exp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float log_sum = logf(sdata[0]);
    __syncthreads();

    // Step 3: output = x - max - log(sum)
    for (int j = tid; j < cols; j += blockDim.x) {
        y_row[j] = x_row[j] - row_max - log_sum;
    }
}

void cuda_log_softmax_forward(const float* x, float* y, int rows, int cols) {
    int threads = 1;
    while (threads < cols && threads < 1024) threads <<= 1;
    log_softmax_forward_kernel<<<rows, threads, threads * sizeof(float)>>>(x, y, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Log-softmax backward
// grad_in[i,j] = grad_out[i,j] - softmax[i,j] * sum_j(grad_out[i,j])
// where softmax[i,j] = exp(log_softmax_output[i,j])
// ---------------------------------------------------------------------------

__global__ void log_softmax_backward_kernel(const float* grad_out, const float* output,
                                            float* grad_in, int rows, int cols) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    const float* go_row = grad_out + row * cols;
    const float* y_row = output + row * cols;
    float* gi_row = grad_in + row * cols;

    // Compute sum of grad_out for this row
    float sum_go = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        sum_go += go_row[j];
    }
    sdata[tid] = sum_go;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float total_go = sdata[0];
    __syncthreads();

    for (int j = tid; j < cols; j += blockDim.x) {
        gi_row[j] = go_row[j] - expf(y_row[j]) * total_go;
    }
}

void cuda_log_softmax_backward(const float* grad_out, const float* output,
                                float* grad_in, int rows, int cols) {
    int threads = 1;
    while (threads < cols && threads < 1024) threads <<= 1;
    log_softmax_backward_kernel<<<rows, threads, threads * sizeof(float)>>>(grad_out, output, grad_in, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace whitematter
