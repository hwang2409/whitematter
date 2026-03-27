#include "cuda_backend.h"
#include "cuda_memory.h"
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

    // Create cuDNN handle
    cudnnHandle_t dnn;
    CUDNN_CHECK(cudnnCreate(&dnn));
    CUDNN_CHECK(cudnnSetStream(dnn, s));
    cudnn_handle_ = static_cast<void*>(dnn);

    initialized_ = true;
}

bool CUDABackend::is_available() const {
    const_cast<CUDABackend*>(this)->init();
    return initialized_;
}

// ---------------------------------------------------------------------------
// BLAS: matmul  C[M,N] = A[M,K] * B[K,N]  (row-major)
// ---------------------------------------------------------------------------

void CUDABackend::matmul(const float* h_A, const float* h_B, float* h_C, int M, int N, int K) {
    init();
    if (!initialized_) return;

    // Transparent offload: host pointers in, GPU compute, host pointer out.
    size_t sA = (size_t)M * K, sB = (size_t)K * N, sC = (size_t)M * N;

    // Allocate device buffers from pool
    auto& pool = CUDAMemoryPool::instance();
    float* d_A = pool.acquire(sA);
    float* d_B = pool.acquire(sB);
    float* d_C = pool.acquire(sC);
    if (!d_A || !d_B || !d_C) {
        // OOM fallback — let CPU handle it
        if (d_A) pool.release(d_A, sA);
        if (d_B) pool.release(d_B, sB);
        if (d_C) pool.release(d_C, sC);
        return;  // Caller's CPU path will handle
    }

    // H2D
    cudaMemcpy(d_A, h_A, sA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sB * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS SGEMM (row-major via column-major trick)
    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle_);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);

    // D2H
    cudaMemcpy(h_C, d_C, sC * sizeof(float), cudaMemcpyDeviceToHost);

    // Return buffers to pool
    pool.release(d_A, sA);
    pool.release(d_B, sB);
    pool.release(d_C, sC);
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
    if (n_floats == 0) return;
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, n_floats * sizeof(float), cudaMemcpyDefault));
}

void CUDABackend::memcpy_d2h(float* h_dst, const float* d_src, size_t n_floats) {
    if (n_floats == 0) return;
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, n_floats * sizeof(float), cudaMemcpyDefault));
}

void CUDABackend::memcpy_d2d(float* d_dst, const float* d_src, size_t n_floats) {
    if (n_floats == 0) return;
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, n_floats * sizeof(float), cudaMemcpyDefault));
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

void CUDABackend::conv2d_forward(const float* h_input, const float* h_weight, const float* h_bias,
                                 float* h_output, size_t batch, size_t in_ch, size_t out_ch,
                                 size_t in_h, size_t in_w, size_t kernel_h, size_t kernel_w,
                                 size_t stride, size_t padding, size_t groups) {
    init();
    if (!initialized_ || !cudnn_handle_) {
        fprintf(stderr, "cuDNN conv2d_forward: backend not ready\n");
        return;
    }

    size_t out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    size_t out_w = (in_w + 2 * padding - kernel_w) / stride + 1;

    // Buffer sizes (in floats)
    size_t input_size  = batch * in_ch * in_h * in_w;
    size_t weight_size = out_ch * (in_ch / groups) * kernel_h * kernel_w;
    size_t output_size = batch * out_ch * out_h * out_w;

    // Allocate managed temp buffers from pool
    auto& pool = CUDAMemoryPool::instance();
    float* d_input  = pool.acquire(input_size);
    float* d_weight = pool.acquire(weight_size);
    float* d_output = pool.acquire(output_size);
    if (!d_input || !d_weight || !d_output) {
        if (d_input)  pool.release(d_input, input_size);
        if (d_weight) pool.release(d_weight, weight_size);
        if (d_output) pool.release(d_output, output_size);
        return;
    }

    // Copy data to managed buffers
    if (!h_input || !h_weight || !h_output) {
        fprintf(stderr, "cuDNN conv2d_forward: null host pointer (input=%p weight=%p output=%p)\n",
                (void*)h_input, (void*)h_weight, (void*)h_output);
        pool.release(d_input, input_size);
        pool.release(d_weight, weight_size);
        pool.release(d_output, output_size);
        return;
    }
    memcpy(d_input,  h_input,  input_size  * sizeof(float));
    memcpy(d_weight, h_weight, weight_size * sizeof(float));

    // Prefetch to GPU for performance
    int device;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(d_input,  input_size  * sizeof(float), device, 0);
    cudaMemPrefetchAsync(d_weight, weight_size * sizeof(float), device, 0);
    cudaMemPrefetchAsync(d_output, output_size * sizeof(float), device, 0);

    // Create cuDNN descriptors
    cudnnHandle_t dnn = static_cast<cudnnHandle_t>(cudnn_handle_);

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t weight_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&weight_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           (int)batch, (int)in_ch, (int)in_h, (int)in_w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           (int)batch, (int)out_ch, (int)out_h, (int)out_w));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(weight_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           (int)out_ch, (int)(in_ch / groups),
                                           (int)kernel_h, (int)kernel_w));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                (int)padding, (int)padding,
                                                (int)stride, (int)stride,
                                                1, 1,  // dilation
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    if (groups > 1) {
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, (int)groups));
    }

    // Find best forward algorithm
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    int returned_algo_count = 0;
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(dnn, input_desc, weight_desc, conv_desc,
                                                     output_desc, 1, &returned_algo_count, &algo_perf));
    cudnnConvolutionFwdAlgo_t algo = algo_perf.algo;

    // Get workspace size
    size_t ws_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(dnn, input_desc, weight_desc, conv_desc,
                                                        output_desc, algo, &ws_size));

    // Use pre-allocated workspace if large enough, otherwise allocate from pool
    void* ws = workspace_;
    float* ws_pool = nullptr;
    if (ws_size > workspace_size_) {
        size_t ws_floats = ws_size / sizeof(float) + 1;
        ws_pool = pool.acquire(ws_floats);
        ws = ws_pool;
    }

    // Forward convolution
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(dnn, &alpha, input_desc, d_input, weight_desc, d_weight,
                                        conv_desc, algo, ws, ws_size,
                                        &beta, output_desc, d_output));

    // Add bias if present
    float* d_bias = nullptr;
    if (h_bias) {
        cudnnTensorDescriptor_t bias_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               1, (int)out_ch, 1, 1));

        d_bias = pool.acquire(out_ch);
        memcpy(d_bias, h_bias, out_ch * sizeof(float));
        cudaMemPrefetchAsync(d_bias, out_ch * sizeof(float), device, 0);

        float bias_alpha = 1.0f, bias_beta = 1.0f;  // add bias to existing output
        CUDNN_CHECK(cudnnAddTensor(dnn, &bias_alpha, bias_desc, d_bias,
                                   &bias_beta, output_desc, d_output));

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    }

    cudaDeviceSynchronize();

    // Copy result back to CPU
    memcpy(h_output, d_output, output_size * sizeof(float));

    // Cleanup descriptors
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(weight_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));

    // Return pool buffers
    if (ws_pool) pool.release(ws_pool, ws_size / sizeof(float) + 1);
    if (d_bias)  pool.release(d_bias, out_ch);
    pool.release(d_input,  input_size);
    pool.release(d_weight, weight_size);
    pool.release(d_output, output_size);
}

void CUDABackend::conv2d_backward(const float* h_input, const float* h_weight,
                                  const float* h_grad_output, float* h_grad_input,
                                  float* h_grad_weight, float* h_grad_bias,
                                  size_t batch, size_t in_ch, size_t out_ch,
                                  size_t in_h, size_t in_w, size_t kernel_h, size_t kernel_w,
                                  size_t stride, size_t padding, size_t groups) {
    init();
    if (!initialized_ || !cudnn_handle_) return;

    size_t out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    size_t out_w = (in_w + 2 * padding - kernel_w) / stride + 1;

    // Buffer sizes
    size_t input_size      = batch * in_ch * in_h * in_w;
    size_t weight_size     = out_ch * (in_ch / groups) * kernel_h * kernel_w;
    size_t output_size     = batch * out_ch * out_h * out_w;

    // Allocate managed temp buffers
    auto& pool = CUDAMemoryPool::instance();
    float* d_input       = pool.acquire(input_size);
    float* d_weight      = pool.acquire(weight_size);
    float* d_grad_output = pool.acquire(output_size);
    float* d_grad_input  = h_grad_input  ? pool.acquire(input_size)  : nullptr;
    float* d_grad_weight = h_grad_weight ? pool.acquire(weight_size) : nullptr;
    float* d_grad_bias   = h_grad_bias   ? pool.acquire(out_ch)     : nullptr;

    // Copy inputs to managed buffers
    memcpy(d_input,       h_input,       input_size  * sizeof(float));
    memcpy(d_weight,      h_weight,      weight_size * sizeof(float));
    memcpy(d_grad_output, h_grad_output, output_size * sizeof(float));

    // Zero the gradient outputs (cuDNN backward with beta=0 overwrites, we accumulate in CPU later)
    if (d_grad_input)  memset(d_grad_input,  0, input_size  * sizeof(float));
    if (d_grad_weight) memset(d_grad_weight, 0, weight_size * sizeof(float));
    if (d_grad_bias)   memset(d_grad_bias,   0, out_ch      * sizeof(float));

    // Prefetch to GPU
    int device;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(d_input,       input_size  * sizeof(float), device, 0);
    cudaMemPrefetchAsync(d_weight,      weight_size * sizeof(float), device, 0);
    cudaMemPrefetchAsync(d_grad_output, output_size * sizeof(float), device, 0);
    if (d_grad_input)  cudaMemPrefetchAsync(d_grad_input,  input_size  * sizeof(float), device, 0);
    if (d_grad_weight) cudaMemPrefetchAsync(d_grad_weight, weight_size * sizeof(float), device, 0);
    if (d_grad_bias)   cudaMemPrefetchAsync(d_grad_bias,   out_ch      * sizeof(float), device, 0);

    // Create cuDNN descriptors
    cudnnHandle_t dnn = static_cast<cudnnHandle_t>(cudnn_handle_);

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t weight_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&weight_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           (int)batch, (int)in_ch, (int)in_h, (int)in_w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           (int)batch, (int)out_ch, (int)out_h, (int)out_w));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(weight_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           (int)out_ch, (int)(in_ch / groups),
                                           (int)kernel_h, (int)kernel_w));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                (int)padding, (int)padding,
                                                (int)stride, (int)stride,
                                                1, 1,  // dilation
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    if (groups > 1) {
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, (int)groups));
    }

    float alpha = 1.0f, beta = 0.0f;

    // --- Backward data: gradient w.r.t. input ---
    if (d_grad_input) {
        cudnnConvolutionBwdDataAlgoPerf_t data_algo_perf;
        int data_algo_count = 0;
        CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(dnn, weight_desc, output_desc,
                                                              conv_desc, input_desc,
                                                              1, &data_algo_count, &data_algo_perf));
        cudnnConvolutionBwdDataAlgo_t data_algo = data_algo_perf.algo;

        size_t data_ws_size = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(dnn, weight_desc, output_desc,
                                                                  conv_desc, input_desc,
                                                                  data_algo, &data_ws_size));

        void* data_ws = workspace_;
        float* data_ws_pool = nullptr;
        if (data_ws_size > workspace_size_) {
            size_t ws_floats = data_ws_size / sizeof(float) + 1;
            data_ws_pool = pool.acquire(ws_floats);
            data_ws = data_ws_pool;
        }

        CUDNN_CHECK(cudnnConvolutionBackwardData(dnn, &alpha, weight_desc, d_weight,
                                                  output_desc, d_grad_output,
                                                  conv_desc, data_algo, data_ws, data_ws_size,
                                                  &beta, input_desc, d_grad_input));

        if (data_ws_pool) pool.release(data_ws_pool, data_ws_size / sizeof(float) + 1);
    }

    // --- Backward filter: gradient w.r.t. weight ---
    if (d_grad_weight) {
        cudnnConvolutionBwdFilterAlgoPerf_t filter_algo_perf;
        int filter_algo_count = 0;
        CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(dnn, input_desc, output_desc,
                                                                 conv_desc, weight_desc,
                                                                 1, &filter_algo_count, &filter_algo_perf));
        cudnnConvolutionBwdFilterAlgo_t filter_algo = filter_algo_perf.algo;

        size_t filter_ws_size = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(dnn, input_desc, output_desc,
                                                                    conv_desc, weight_desc,
                                                                    filter_algo, &filter_ws_size));

        void* filter_ws = workspace_;
        float* filter_ws_pool = nullptr;
        if (filter_ws_size > workspace_size_) {
            size_t ws_floats = filter_ws_size / sizeof(float) + 1;
            filter_ws_pool = pool.acquire(ws_floats);
            filter_ws = filter_ws_pool;
        }

        CUDNN_CHECK(cudnnConvolutionBackwardFilter(dnn, &alpha, input_desc, d_input,
                                                    output_desc, d_grad_output,
                                                    conv_desc, filter_algo, filter_ws, filter_ws_size,
                                                    &beta, weight_desc, d_grad_weight));

        if (filter_ws_pool) pool.release(filter_ws_pool, filter_ws_size / sizeof(float) + 1);
    }

    // --- Backward bias: gradient w.r.t. bias ---
    if (d_grad_bias) {
        cudnnTensorDescriptor_t bias_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               1, (int)out_ch, 1, 1));

        CUDNN_CHECK(cudnnConvolutionBackwardBias(dnn, &alpha, output_desc, d_grad_output,
                                                  &beta, bias_desc, d_grad_bias));

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    }

    cudaDeviceSynchronize();

    // Copy gradients back to CPU and accumulate (add to existing grad, don't overwrite)
    if (h_grad_input && d_grad_input) {
        for (size_t i = 0; i < input_size; i++) {
            h_grad_input[i] += d_grad_input[i];
        }
    }
    if (h_grad_weight && d_grad_weight) {
        for (size_t i = 0; i < weight_size; i++) {
            h_grad_weight[i] += d_grad_weight[i];
        }
    }
    if (h_grad_bias && d_grad_bias) {
        for (size_t i = 0; i < out_ch; i++) {
            h_grad_bias[i] += d_grad_bias[i];
        }
    }

    // Cleanup descriptors
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(weight_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));

    // Return pool buffers
    if (d_grad_bias)   pool.release(d_grad_bias,   out_ch);
    if (d_grad_weight) pool.release(d_grad_weight, weight_size);
    if (d_grad_input)  pool.release(d_grad_input,  input_size);
    pool.release(d_grad_output, output_size);
    pool.release(d_weight,      weight_size);
    pool.release(d_input,       input_size);
}

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
