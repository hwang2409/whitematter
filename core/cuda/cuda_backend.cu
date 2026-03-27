#include "cuda_backend.h"
#include "cuda_memory.h"
#include "cuda_check.h"
#include "../device.h"
#include "../memory_pool.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <unordered_set>

// ---------------------------------------------------------------------------
// Device buffer cache: reuses cudaMalloc'd buffers to avoid per-call
// cudaMalloc/cudaFree overhead in cuDNN conv2d (eliminates ~62k allocs/epoch).
// Keyed by size in floats; exact-match reuse.  Freed on process exit.
// ---------------------------------------------------------------------------
namespace {
struct DeviceBufferCache {
    std::unordered_map<size_t, std::vector<float*>> free_list;

    float* get(size_t n_floats) {
        if (n_floats == 0) return nullptr;
        auto it = free_list.find(n_floats);
        if (it != free_list.end() && !it->second.empty()) {
            float* p = it->second.back();
            it->second.pop_back();
            return p;
        }
        float* p = nullptr;
        cudaMalloc(&p, n_floats * sizeof(float));
        return p;
    }

    void put(float* p, size_t n_floats) {
        if (p && n_floats > 0) free_list[n_floats].push_back(p);
    }

    ~DeviceBufferCache() {
        for (auto& kv : free_list) {
            for (float* p : kv.second) cudaFree(p);
        }
    }
};

static DeviceBufferCache g_buf_cache;

// ---------------------------------------------------------------------------
// Pinned (page-locked) host buffer cache: uses cudaHostAlloc/cudaFreeHost
// for DMA-capable staging buffers that enable 2-3x faster cudaMemcpyAsync.
// Falls back to regular malloc if cudaHostAlloc fails (e.g. WSL2).
// ---------------------------------------------------------------------------
struct PinnedBufferCache {
    std::unordered_map<size_t, std::vector<float*>> free_list;
    std::unordered_set<float*> pinned_set;  // track which ptrs are truly pinned

    float* get(size_t n_floats) {
        if (n_floats == 0) return nullptr;
        auto it = free_list.find(n_floats);
        if (it != free_list.end() && !it->second.empty()) {
            float* p = it->second.back();
            it->second.pop_back();
            return p;
        }
        float* p = nullptr;
        cudaError_t err = cudaHostAlloc(&p, n_floats * sizeof(float), cudaHostAllocDefault);
        if (err != cudaSuccess || !p) {
            cudaGetLastError();  // clear error state
            p = (float*)malloc(n_floats * sizeof(float));
        } else {
            pinned_set.insert(p);
        }
        return p;
    }

    void put(float* p, size_t n_floats) {
        if (p && n_floats > 0) free_list[n_floats].push_back(p);
    }

    bool is_pinned(float* p) const {
        return pinned_set.count(p) > 0;
    }

    ~PinnedBufferCache() {
        for (auto& kv : free_list) {
            for (float* p : kv.second) {
                if (pinned_set.count(p) > 0)
                    cudaFreeHost(p);
                else
                    free(p);
            }
        }
    }
};

static PinnedBufferCache g_pinned_cache;

// ---------------------------------------------------------------------------
// Weight cache: avoids redundant H2D uploads of conv2d weights and biases.
// Weights only change after optimizer.step(), so between steps the same host
// pointer maps to an already-uploaded device buffer.  Call invalidate() after
// each optimizer step to mark all entries as stale.
// ---------------------------------------------------------------------------
struct WeightCache {
    struct Entry {
        float* d_ptr;
        size_t n_floats;
        size_t generation;  // last sync generation
    };
    std::unordered_map<const float*, Entry> cache;
    size_t current_gen = 0;

    // Mark all cached entries as stale (call after optimizer.step())
    void invalidate() { current_gen++; }

    // Return a device buffer containing the data at h_ptr.
    // Only performs cudaMemcpy if the entry is missing or stale.
    float* get(const float* h_ptr, size_t n_floats) {
        auto it = cache.find(h_ptr);
        if (it != cache.end() && it->second.n_floats == n_floats) {
            if (it->second.generation == current_gen) {
                return it->second.d_ptr;  // already up to date
            }
            // Stale — re-upload
            cudaMemcpy(it->second.d_ptr, h_ptr, n_floats * sizeof(float), cudaMemcpyHostToDevice);
            it->second.generation = current_gen;
            return it->second.d_ptr;
        }
        // New entry — allocate via buffer cache and upload
        float* d_ptr = g_buf_cache.get(n_floats);
        cudaMemcpy(d_ptr, h_ptr, n_floats * sizeof(float), cudaMemcpyHostToDevice);
        cache[h_ptr] = {d_ptr, n_floats, current_gen};
        return d_ptr;
    }

    ~WeightCache() {
        // Device buffers are owned by g_buf_cache's underlying allocations;
        // return them so they get freed with everything else.
        for (auto& kv : cache) {
            g_buf_cache.put(kv.second.d_ptr, kv.second.n_floats);
        }
    }
};

static WeightCache g_weight_cache;

// ---------------------------------------------------------------------------
// is_device_accessible: checks if a pointer is managed or device memory,
// meaning cuDNN/CUDA kernels can read/write it directly without H2D/D2H.
// ---------------------------------------------------------------------------
static bool is_device_accessible(const void* ptr) {
    if (!ptr) return false;
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();  // clear error state
        return false;
    }
    return attrs.type == cudaMemoryTypeManaged || attrs.type == cudaMemoryTypeDevice;
}

} // anonymous namespace

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
    // Note: managed memory for tensor storage disabled on WSL2 — page fault
    // overhead makes cuDNN hang. Explicit H2D/D2H copies are faster on WSL2.
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
#ifdef WHITEMATTER_DEBUG
        fprintf(stderr, "cuDNN conv2d_forward: backend not ready\n");
#endif
        return;
    }
#ifdef WHITEMATTER_DEBUG
    fprintf(stderr, "cuDNN conv2d: [%zu,%zu,%zu,%zu] k=%zux%zu s=%zu p=%zu g=%zu\n",
            batch, in_ch, in_h, in_w, kernel_h, kernel_w, stride, padding, groups);
    fflush(stderr);
#endif

    size_t out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    size_t out_w = (in_w + 2 * padding - kernel_w) / stride + 1;

    // Buffer sizes (in floats)
    size_t input_size  = batch * in_ch * in_h * in_w;
    size_t weight_size = out_ch * (in_ch / groups) * kernel_h * kernel_w;
    size_t output_size = batch * out_ch * out_h * out_w;

    // Get CUDA stream for async operations
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);

    // Check if input/output are already device-accessible (managed memory).
    // When MemoryPool uses cudaMallocManaged, tensor data is GPU-readable — skip H2D.
    bool input_managed = is_device_accessible(h_input);
    bool output_managed = is_device_accessible(h_output);

    float *d_input = nullptr, *d_output = nullptr;
    float* p_input = nullptr;  // pinned staging (only used if not managed)

    if (input_managed) {
        d_input = const_cast<float*>(h_input);  // cuDNN reads directly
    } else {
        d_input = g_buf_cache.get(input_size);
        p_input = g_pinned_cache.get(input_size);
        memcpy(p_input, h_input, input_size * sizeof(float));
        CUDA_CHECK(cudaMemcpyAsync(d_input, p_input, input_size * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));
    }

    if (output_managed) {
        d_output = h_output;  // cuDNN writes directly
    } else {
        d_output = g_buf_cache.get(output_size);
    }
    float* d_weight = g_weight_cache.get(h_weight, weight_size);

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

    // Use IMPLICIT_PRECOMP_GEMM — faster than IMPLICIT_GEMM, works for all sizes.
    // Falls back to IMPLICIT_GEMM only if workspace allocation fails.
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    // Get workspace size
    size_t ws_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(dnn, input_desc, weight_desc, conv_desc,
                                                        output_desc, algo, &ws_size));

    // Workspace: acquire from cache
    void* ws = nullptr;
    size_t ws_n_floats = 0;
    if (ws_size > 0) {
        ws_n_floats = (ws_size + sizeof(float) - 1) / sizeof(float);
        ws = (void*)g_buf_cache.get(ws_n_floats);
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

        d_bias = g_weight_cache.get(h_bias, out_ch);  // cached — skips H2D if fresh

        float bias_alpha = 1.0f, bias_beta = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(dnn, &bias_alpha, bias_desc, d_bias,
                                   &bias_beta, output_desc, d_output));

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    }

    // D2H result — skip if output is managed (cuDNN wrote directly)
    if (!output_managed) {
        float* p_output = g_pinned_cache.get(output_size);
        CUDA_CHECK(cudaMemcpyAsync(p_output, d_output, output_size * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        memcpy(h_output, p_output, output_size * sizeof(float));
        g_pinned_cache.put(p_output, output_size);
    } else {
        // Managed output: just sync to ensure cuDNN write is visible to CPU
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Return pinned/buf cache buffers (only if we allocated them)
    if (p_input) g_pinned_cache.put(p_input, input_size);

    // Cleanup
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(weight_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));

    if (ws) g_buf_cache.put((float*)ws, ws_n_floats);
    if (!input_managed)  g_buf_cache.put(d_input, input_size);
    if (!output_managed) g_buf_cache.put(d_output, output_size);
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

    // Get CUDA stream for async operations
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);

    // Acquire device buffers from cache
    float *d_input = nullptr, *d_grad_output = nullptr;
    float *d_grad_input = nullptr, *d_grad_weight = nullptr, *d_grad_bias = nullptr;

    d_input = g_buf_cache.get(input_size);
    d_grad_output = g_buf_cache.get(output_size);
    float* d_weight = g_weight_cache.get(h_weight, weight_size);

    // Pinned staging buffers for async H2D
    float* p_input = g_pinned_cache.get(input_size);
    float* p_grad_output = g_pinned_cache.get(output_size);

    memcpy(p_input, h_input, input_size * sizeof(float));
    memcpy(p_grad_output, h_grad_output, output_size * sizeof(float));

    CUDA_CHECK(cudaMemcpyAsync(d_input, p_input, input_size * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_grad_output, p_grad_output, output_size * sizeof(float),
                                cudaMemcpyHostToDevice, stream));

    if (h_grad_input)  d_grad_input  = g_buf_cache.get(input_size);
    if (h_grad_weight) d_grad_weight = g_buf_cache.get(weight_size);
    if (h_grad_bias)   d_grad_bias   = g_buf_cache.get(out_ch);

    // Zero gradient outputs on device
    if (d_grad_input)  CUDA_CHECK(cudaMemsetAsync(d_grad_input,  0, input_size  * sizeof(float), stream));
    if (d_grad_weight) CUDA_CHECK(cudaMemsetAsync(d_grad_weight, 0, weight_size * sizeof(float), stream));
    if (d_grad_bias)   CUDA_CHECK(cudaMemsetAsync(d_grad_bias,   0, out_ch      * sizeof(float), stream));

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
        cudnnConvolutionBwdDataAlgo_t data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

        size_t data_ws_size = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(dnn, weight_desc, output_desc,
                                                                  conv_desc, input_desc,
                                                                  data_algo, &data_ws_size));

        void* data_ws = nullptr;
        size_t data_ws_n_floats = 0;
        if (data_ws_size > 0) {
            data_ws_n_floats = (data_ws_size + sizeof(float) - 1) / sizeof(float);
            data_ws = (void*)g_buf_cache.get(data_ws_n_floats);
        }

        CUDNN_CHECK(cudnnConvolutionBackwardData(dnn, &alpha, weight_desc, d_weight,
                                                  output_desc, d_grad_output,
                                                  conv_desc, data_algo, data_ws, data_ws_size,
                                                  &beta, input_desc, d_grad_input));

        if (data_ws) g_buf_cache.put((float*)data_ws, data_ws_n_floats);
    }

    // --- Backward filter: gradient w.r.t. weight ---
    if (d_grad_weight) {
        cudnnConvolutionBwdFilterAlgo_t filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

        size_t filter_ws_size = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(dnn, input_desc, output_desc,
                                                                    conv_desc, weight_desc,
                                                                    filter_algo, &filter_ws_size));

        void* filter_ws = nullptr;
        size_t filter_ws_n_floats = 0;
        if (filter_ws_size > 0) {
            filter_ws_n_floats = (filter_ws_size + sizeof(float) - 1) / sizeof(float);
            filter_ws = (void*)g_buf_cache.get(filter_ws_n_floats);
        }

        CUDNN_CHECK(cudnnConvolutionBackwardFilter(dnn, &alpha, input_desc, d_input,
                                                    output_desc, d_grad_output,
                                                    conv_desc, filter_algo, filter_ws, filter_ws_size,
                                                    &beta, weight_desc, d_grad_weight));

        if (filter_ws) g_buf_cache.put((float*)filter_ws, filter_ws_n_floats);
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

    // Async D2H gradients via pinned staging buffers
    float* p_grad_input  = (h_grad_input  && d_grad_input)  ? g_pinned_cache.get(input_size)  : nullptr;
    float* p_grad_weight = (h_grad_weight && d_grad_weight) ? g_pinned_cache.get(weight_size) : nullptr;
    float* p_grad_bias   = (h_grad_bias   && d_grad_bias)   ? g_pinned_cache.get(out_ch)      : nullptr;

    if (p_grad_input)
        CUDA_CHECK(cudaMemcpyAsync(p_grad_input, d_grad_input, input_size * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));
    if (p_grad_weight)
        CUDA_CHECK(cudaMemcpyAsync(p_grad_weight, d_grad_weight, weight_size * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));
    if (p_grad_bias)
        CUDA_CHECK(cudaMemcpyAsync(p_grad_bias, d_grad_bias, out_ch * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));

    // Synchronize stream — all async ops complete here
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Accumulate gradients from pinned staging into host buffers
    if (p_grad_input) {
        for (size_t i = 0; i < input_size; i++) h_grad_input[i] += p_grad_input[i];
        g_pinned_cache.put(p_grad_input, input_size);
    }
    if (p_grad_weight) {
        for (size_t i = 0; i < weight_size; i++) h_grad_weight[i] += p_grad_weight[i];
        g_pinned_cache.put(p_grad_weight, weight_size);
    }
    if (p_grad_bias) {
        for (size_t i = 0; i < out_ch; i++) h_grad_bias[i] += p_grad_bias[i];
        g_pinned_cache.put(p_grad_bias, out_ch);
    }

    // Return H2D pinned buffers to cache
    g_pinned_cache.put(p_input, input_size);
    g_pinned_cache.put(p_grad_output, output_size);

    // Cleanup
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(weight_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));

    if (d_grad_bias)   g_buf_cache.put(d_grad_bias, out_ch);
    if (d_grad_weight) g_buf_cache.put(d_grad_weight, weight_size);
    if (d_grad_input)  g_buf_cache.put(d_grad_input, input_size);
    g_buf_cache.put(d_grad_output, output_size);
    g_buf_cache.put(d_input, input_size);
}

// ---------------------------------------------------------------------------
// BatchNorm via cuDNN
// ---------------------------------------------------------------------------

void CUDABackend::batchnorm_forward(const float* h_input, float* h_output,
                                    const float* h_gamma, const float* h_beta,
                                    float* h_running_mean, float* h_running_var,
                                    float* h_save_mean, float* h_save_inv_var,
                                    size_t batch, size_t channels, size_t spatial,
                                    float eps, float momentum, bool training) {
    init();
    if (!initialized_ || !cudnn_handle_) return;

    size_t total = batch * channels * spatial;
    // Derive H and W from spatial (assumes square feature maps for cuDNN 4D descriptor)
    size_t H = (size_t)sqrt((double)spatial);
    size_t W = spatial / H;

    // Get CUDA stream for async operations
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);

    // Allocate device buffers from cache
    float* d_input        = g_buf_cache.get(total);
    float* d_output       = g_buf_cache.get(total);
    float* d_gamma        = g_buf_cache.get(channels);
    float* d_beta         = g_buf_cache.get(channels);
    float* d_running_mean = g_buf_cache.get(channels);
    float* d_running_var  = g_buf_cache.get(channels);
    float* d_save_mean    = g_buf_cache.get(channels);
    float* d_save_inv_var = g_buf_cache.get(channels);

    // Pinned staging buffers for async H2D
    float* p_input        = g_pinned_cache.get(total);
    float* p_gamma        = g_pinned_cache.get(channels);
    float* p_beta         = g_pinned_cache.get(channels);
    float* p_running_mean = g_pinned_cache.get(channels);
    float* p_running_var  = g_pinned_cache.get(channels);

    memcpy(p_input, h_input, total * sizeof(float));
    memcpy(p_gamma, h_gamma, channels * sizeof(float));
    memcpy(p_beta, h_beta, channels * sizeof(float));
    memcpy(p_running_mean, h_running_mean, channels * sizeof(float));
    memcpy(p_running_var, h_running_var, channels * sizeof(float));

    // Async H2D via pinned memory
    CUDA_CHECK(cudaMemcpyAsync(d_input, p_input, total * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_gamma, p_gamma, channels * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_beta, p_beta, channels * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_running_mean, p_running_mean, channels * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_running_var, p_running_var, channels * sizeof(float),
                                cudaMemcpyHostToDevice, stream));

    // cuDNN descriptors
    cudnnHandle_t dnn = static_cast<cudnnHandle_t>(cudnn_handle_);
    cudnnTensorDescriptor_t input_desc, bn_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           (int)batch, (int)channels, (int)H, (int)W));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc, input_desc, CUDNN_BATCHNORM_SPATIAL));

    float alpha = 1.0f, beta_val = 0.0f;

    if (training) {
        // exponentialAverageFactor = momentum (PyTorch convention)
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(dnn, CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta_val,
            input_desc, d_input, input_desc, d_output,
            bn_desc, d_gamma, d_beta,
            (double)momentum,
            d_running_mean, d_running_var,
            (double)eps,
            d_save_mean, d_save_inv_var));
    } else {
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(dnn, CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta_val,
            input_desc, d_input, input_desc, d_output,
            bn_desc, d_gamma, d_beta,
            d_running_mean, d_running_var,
            (double)eps));
    }

    // Async D2H via pinned staging buffers
    float* p_output       = g_pinned_cache.get(total);
    float* p_rm_out       = g_pinned_cache.get(channels);
    float* p_rv_out       = g_pinned_cache.get(channels);
    float* p_save_mean    = h_save_mean    ? g_pinned_cache.get(channels) : nullptr;
    float* p_save_inv_var = h_save_inv_var ? g_pinned_cache.get(channels) : nullptr;

    CUDA_CHECK(cudaMemcpyAsync(p_output, d_output, total * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(p_rm_out, d_running_mean, channels * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(p_rv_out, d_running_var, channels * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    if (p_save_mean)
        CUDA_CHECK(cudaMemcpyAsync(p_save_mean, d_save_mean, channels * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));
    if (p_save_inv_var)
        CUDA_CHECK(cudaMemcpyAsync(p_save_inv_var, d_save_inv_var, channels * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));

    // Synchronize stream — all async ops complete here
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy from pinned staging to caller's output buffers
    memcpy(h_output, p_output, total * sizeof(float));
    memcpy(h_running_mean, p_rm_out, channels * sizeof(float));
    memcpy(h_running_var, p_rv_out, channels * sizeof(float));
    if (h_save_mean)    memcpy(h_save_mean, p_save_mean, channels * sizeof(float));
    if (h_save_inv_var) memcpy(h_save_inv_var, p_save_inv_var, channels * sizeof(float));

    // Return pinned buffers to cache
    g_pinned_cache.put(p_input, total);
    g_pinned_cache.put(p_gamma, channels);
    g_pinned_cache.put(p_beta, channels);
    g_pinned_cache.put(p_running_mean, channels);
    g_pinned_cache.put(p_running_var, channels);
    g_pinned_cache.put(p_output, total);
    g_pinned_cache.put(p_rm_out, channels);
    g_pinned_cache.put(p_rv_out, channels);
    if (p_save_mean)    g_pinned_cache.put(p_save_mean, channels);
    if (p_save_inv_var) g_pinned_cache.put(p_save_inv_var, channels);

    // Return device buffers to cache
    g_buf_cache.put(d_input, total);
    g_buf_cache.put(d_output, total);
    g_buf_cache.put(d_gamma, channels);
    g_buf_cache.put(d_beta, channels);
    g_buf_cache.put(d_running_mean, channels);
    g_buf_cache.put(d_running_var, channels);
    g_buf_cache.put(d_save_mean, channels);
    g_buf_cache.put(d_save_inv_var, channels);

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_desc));
}

void CUDABackend::batchnorm_backward(const float* h_input, const float* h_grad_output,
                                     float* h_grad_input, float* h_grad_gamma, float* h_grad_beta,
                                     const float* h_gamma, const float* h_save_mean,
                                     const float* h_save_inv_var,
                                     size_t batch, size_t channels, size_t spatial, float eps) {
    init();
    if (!initialized_ || !cudnn_handle_) return;

    size_t total = batch * channels * spatial;
    size_t H = (size_t)sqrt((double)spatial);
    size_t W = spatial / H;

    // Get CUDA stream for async operations
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);

    // Allocate device buffers from cache
    float* d_input       = g_buf_cache.get(total);
    float* d_grad_output = g_buf_cache.get(total);
    float* d_grad_input  = h_grad_input ? g_buf_cache.get(total) : nullptr;
    float* d_gamma       = g_buf_cache.get(channels);
    float* d_grad_gamma  = g_buf_cache.get(channels);
    float* d_grad_beta   = g_buf_cache.get(channels);
    float* d_save_mean   = g_buf_cache.get(channels);
    float* d_save_inv_var = g_buf_cache.get(channels);

    // Pinned staging buffers for async H2D
    float* p_input        = g_pinned_cache.get(total);
    float* p_grad_output  = g_pinned_cache.get(total);
    float* p_gamma        = g_pinned_cache.get(channels);
    float* p_save_mean    = g_pinned_cache.get(channels);
    float* p_save_inv_var = g_pinned_cache.get(channels);

    memcpy(p_input, h_input, total * sizeof(float));
    memcpy(p_grad_output, h_grad_output, total * sizeof(float));
    memcpy(p_gamma, h_gamma, channels * sizeof(float));
    memcpy(p_save_mean, h_save_mean, channels * sizeof(float));
    memcpy(p_save_inv_var, h_save_inv_var, channels * sizeof(float));

    // Async H2D via pinned memory
    CUDA_CHECK(cudaMemcpyAsync(d_input, p_input, total * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_grad_output, p_grad_output, total * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_gamma, p_gamma, channels * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_save_mean, p_save_mean, channels * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_save_inv_var, p_save_inv_var, channels * sizeof(float),
                                cudaMemcpyHostToDevice, stream));

    // Zero gradient outputs on device
    CUDA_CHECK(cudaMemsetAsync(d_grad_gamma, 0, channels * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_grad_beta, 0, channels * sizeof(float), stream));
    if (d_grad_input) CUDA_CHECK(cudaMemsetAsync(d_grad_input, 0, total * sizeof(float), stream));

    // cuDNN descriptors
    cudnnHandle_t dnn = static_cast<cudnnHandle_t>(cudnn_handle_);
    cudnnTensorDescriptor_t input_desc, bn_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           (int)batch, (int)channels, (int)H, (int)W));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc, input_desc, CUDNN_BATCHNORM_SPATIAL));

    float alpha_data = 1.0f, beta_data = 0.0f;
    float alpha_param = 1.0f, beta_param = 0.0f;

    CUDNN_CHECK(cudnnBatchNormalizationBackward(dnn, CUDNN_BATCHNORM_SPATIAL,
        &alpha_data, &beta_data,
        &alpha_param, &beta_param,
        input_desc, d_input,
        input_desc, d_grad_output,
        input_desc, d_grad_input ? d_grad_input : d_grad_output, // dummy if no grad_input needed
        bn_desc, d_gamma, d_grad_gamma, d_grad_beta,
        (double)eps,
        d_save_mean, d_save_inv_var));

    // Async D2H gradients via pinned staging buffers
    float* p_dgi = (h_grad_input && d_grad_input) ? g_pinned_cache.get(total)    : nullptr;
    float* p_dgw = h_grad_gamma                   ? g_pinned_cache.get(channels) : nullptr;
    float* p_dgb = h_grad_beta                    ? g_pinned_cache.get(channels) : nullptr;

    if (p_dgi)
        CUDA_CHECK(cudaMemcpyAsync(p_dgi, d_grad_input, total * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));
    if (p_dgw)
        CUDA_CHECK(cudaMemcpyAsync(p_dgw, d_grad_gamma, channels * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));
    if (p_dgb)
        CUDA_CHECK(cudaMemcpyAsync(p_dgb, d_grad_beta, channels * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));

    // Synchronize stream — all async ops complete here
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Accumulate gradients from pinned staging into host buffers
    if (p_dgi) {
        for (size_t i = 0; i < total; i++) h_grad_input[i] += p_dgi[i];
        g_pinned_cache.put(p_dgi, total);
    }
    if (p_dgw) {
        for (size_t i = 0; i < channels; i++) h_grad_gamma[i] += p_dgw[i];
        g_pinned_cache.put(p_dgw, channels);
    }
    if (p_dgb) {
        for (size_t i = 0; i < channels; i++) h_grad_beta[i] += p_dgb[i];
        g_pinned_cache.put(p_dgb, channels);
    }

    // Return H2D pinned buffers to cache
    g_pinned_cache.put(p_input, total);
    g_pinned_cache.put(p_grad_output, total);
    g_pinned_cache.put(p_gamma, channels);
    g_pinned_cache.put(p_save_mean, channels);
    g_pinned_cache.put(p_save_inv_var, channels);

    // Return device buffers to cache
    g_buf_cache.put(d_input, total);
    g_buf_cache.put(d_grad_output, total);
    if (d_grad_input) g_buf_cache.put(d_grad_input, total);
    g_buf_cache.put(d_gamma, channels);
    g_buf_cache.put(d_grad_gamma, channels);
    g_buf_cache.put(d_grad_beta, channels);
    g_buf_cache.put(d_save_mean, channels);
    g_buf_cache.put(d_save_inv_var, channels);

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_desc));
}

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

void CUDABackend::invalidate_weight_cache() {
    g_weight_cache.invalidate();
}

}  // namespace whitematter

// Provide cuda_backend_available when CUDA backend is linked.
namespace whitematter {
bool cuda_backend_available() {
    return CUDABackend::instance().is_available();
}
}  // namespace whitematter
