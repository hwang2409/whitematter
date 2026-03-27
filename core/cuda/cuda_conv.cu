// Convolution, batch normalization, and pooling CUDA kernels for whitematter.
// Uses cuDNN for conv2d and batchnorm; hand-rolled kernels for pooling.

#include "cuda_check.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cfloat>
#include <cstdio>

namespace whitematter {

// ---------------------------------------------------------------------------
// cuDNN error checking
// ---------------------------------------------------------------------------

#define CUDNN_CHECK(call) do { \
    cudnnStatus_t status = (call); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Conv2d forward
// Input:  [N, C_in, H, W]
// Weight: [C_out, C_in, kH, kW]
// Bias:   [C_out] (may be nullptr)
// Output: [N, C_out, H_out, W_out]
// All NCHW format, float32.
// workspace: pre-allocated GPU buffer, workspace_bytes: its size.
// ---------------------------------------------------------------------------

void cuda_conv2d_forward(void* cudnn_handle,
                         const float* d_input, const float* d_weight, const float* d_bias,
                         float* d_output,
                         int batch, int C_in, int H, int W,
                         int C_out, int kH, int kW,
                         int padH, int padW, int strideH, int strideW,
                         int dilH, int dilW,
                         void* workspace, size_t workspace_bytes) {
    cudnnHandle_t handle = static_cast<cudnnHandle_t>(cudnn_handle);

    // Input descriptor
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch, C_in, H, W));

    // Filter descriptor
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           C_out, C_in, kH, kW));

    // Convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                padH, padW, strideH, strideW, dilH, dilW,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Output dimensions
    int out_n, out_c, out_h, out_w;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc,
                                                      &out_n, &out_c, &out_h, &out_w));

    // Output descriptor
    cudnnTensorDescriptor_t output_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           out_n, out_c, out_h, out_w));

    // Algorithm
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    // Check workspace requirement
    size_t ws_needed = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc,
                                                        conv_desc, output_desc, algo, &ws_needed));
    if (ws_needed > workspace_bytes) {
        // Fall back to a no-workspace algorithm
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        ws_needed = 0;
    }

    // Forward convolution
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha,
                                        input_desc, d_input,
                                        filter_desc, d_weight,
                                        conv_desc, algo,
                                        workspace, ws_needed,
                                        &beta,
                                        output_desc, d_output));

    // Add bias if present
    if (d_bias != nullptr) {
        cudnnTensorDescriptor_t bias_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               1, C_out, 1, 1));
        float alpha_bias = 1.0f, beta_bias = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(handle, &alpha_bias, bias_desc, d_bias,
                                   &beta_bias, output_desc, d_output));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    }

    // Cleanup
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
}

// ---------------------------------------------------------------------------
// Conv2d backward
// Computes grad_input, grad_weight, and grad_bias from grad_output.
// ---------------------------------------------------------------------------

void cuda_conv2d_backward(void* cudnn_handle,
                          const float* d_input, const float* d_weight,
                          const float* d_grad_output,
                          float* d_grad_input, float* d_grad_weight, float* d_grad_bias,
                          int batch, int C_in, int H, int W,
                          int C_out, int kH, int kW,
                          int padH, int padW, int strideH, int strideW,
                          int dilH, int dilW,
                          void* workspace, size_t workspace_bytes) {
    cudnnHandle_t handle = static_cast<cudnnHandle_t>(cudnn_handle);

    // Descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch, C_in, H, W));

    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           C_out, C_in, kH, kW));

    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                padH, padW, strideH, strideW, dilH, dilW,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    int out_n, out_c, out_h, out_w;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc,
                                                      &out_n, &out_c, &out_h, &out_w));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           out_n, out_c, out_h, out_w));

    float alpha = 1.0f, beta = 0.0f;

    // Gradient w.r.t. input
    if (d_grad_input != nullptr) {
        cudnnConvolutionBwdDataAlgo_t data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        size_t ws_needed = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filter_desc, output_desc,
                                                                  conv_desc, input_desc,
                                                                  data_algo, &ws_needed));
        if (ws_needed > workspace_bytes) {
            data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
            ws_needed = 0;
        }
        CUDNN_CHECK(cudnnConvolutionBackwardData(handle, &alpha,
                                                  filter_desc, d_weight,
                                                  output_desc, d_grad_output,
                                                  conv_desc, data_algo,
                                                  workspace, ws_needed,
                                                  &beta,
                                                  input_desc, d_grad_input));
    }

    // Gradient w.r.t. weights
    if (d_grad_weight != nullptr) {
        cudnnConvolutionBwdFilterAlgo_t filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
        size_t ws_needed = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, input_desc, output_desc,
                                                                    conv_desc, filter_desc,
                                                                    filter_algo, &ws_needed));
        if (ws_needed > workspace_bytes) {
            filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
            ws_needed = 0;
        }
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle, &alpha,
                                                    input_desc, d_input,
                                                    output_desc, d_grad_output,
                                                    conv_desc, filter_algo,
                                                    workspace, ws_needed,
                                                    &beta,
                                                    filter_desc, d_grad_weight));
    }

    // Gradient w.r.t. bias
    if (d_grad_bias != nullptr) {
        cudnnTensorDescriptor_t bias_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               1, C_out, 1, 1));
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle, &alpha,
                                                  output_desc, d_grad_output,
                                                  &beta,
                                                  bias_desc, d_grad_bias));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    }

    // Cleanup
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
}

// ---------------------------------------------------------------------------
// Batch normalization forward (training)
// Input/Output:  [N, C, H, W]  NCHW float32
// scale, bias:   [C]
// running_mean, running_var: [C]  (updated in place)
// save_mean, save_inv_var:  [C]  (saved for backward pass)
// ---------------------------------------------------------------------------

void cuda_batchnorm_forward_training(void* cudnn_handle,
                                     const float* d_input, float* d_output,
                                     const float* d_scale, const float* d_bias,
                                     float* d_running_mean, float* d_running_var,
                                     float* d_save_mean, float* d_save_inv_var,
                                     int batch, int C, int H, int W,
                                     double momentum, double epsilon) {
    cudnnHandle_t handle = static_cast<cudnnHandle_t>(cudnn_handle);

    cudnnTensorDescriptor_t data_desc, bn_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&data_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(data_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch, C, H, W));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc, data_desc, CUDNN_BATCHNORM_SPATIAL));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
        handle, CUDNN_BATCHNORM_SPATIAL,
        &alpha, &beta,
        data_desc, d_input,
        data_desc, d_output,
        bn_desc,
        d_scale, d_bias,
        momentum,
        d_running_mean, d_running_var,
        epsilon,
        d_save_mean, d_save_inv_var));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(data_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_desc));
}

// ---------------------------------------------------------------------------
// Batch normalization forward (inference)
// ---------------------------------------------------------------------------

void cuda_batchnorm_forward_inference(void* cudnn_handle,
                                      const float* d_input, float* d_output,
                                      const float* d_scale, const float* d_bias,
                                      const float* d_running_mean, const float* d_running_var,
                                      int batch, int C, int H, int W,
                                      double epsilon) {
    cudnnHandle_t handle = static_cast<cudnnHandle_t>(cudnn_handle);

    cudnnTensorDescriptor_t data_desc, bn_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&data_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(data_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch, C, H, W));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc, data_desc, CUDNN_BATCHNORM_SPATIAL));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
        handle, CUDNN_BATCHNORM_SPATIAL,
        &alpha, &beta,
        data_desc, d_input,
        data_desc, d_output,
        bn_desc,
        d_scale, d_bias,
        d_running_mean, d_running_var,
        epsilon));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(data_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_desc));
}

// ---------------------------------------------------------------------------
// Batch normalization backward
// ---------------------------------------------------------------------------

void cuda_batchnorm_backward(void* cudnn_handle,
                             const float* d_input, const float* d_grad_output,
                             float* d_grad_input,
                             const float* d_scale, float* d_grad_scale, float* d_grad_bias,
                             const float* d_save_mean, const float* d_save_inv_var,
                             int batch, int C, int H, int W,
                             double epsilon) {
    cudnnHandle_t handle = static_cast<cudnnHandle_t>(cudnn_handle);

    cudnnTensorDescriptor_t data_desc, bn_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&data_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(data_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch, C, H, W));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc, data_desc, CUDNN_BATCHNORM_SPATIAL));

    float alpha_data = 1.0f, beta_data = 0.0f;
    float alpha_param = 1.0f, beta_param = 0.0f;

    CUDNN_CHECK(cudnnBatchNormalizationBackward(
        handle, CUDNN_BATCHNORM_SPATIAL,
        &alpha_data, &beta_data,
        &alpha_param, &beta_param,
        data_desc, d_input,
        data_desc, d_grad_output,
        data_desc, d_grad_input,
        bn_desc,
        d_scale, d_grad_scale, d_grad_bias,
        epsilon,
        d_save_mean, d_save_inv_var));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(data_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_desc));
}

// ---------------------------------------------------------------------------
// MaxPool2d forward (hand-rolled kernel)
// Input:  [N, C, H, W]
// Output: [N, C, H_out, W_out]
// Indices: [N, C, H_out, W_out] (int, stores index within pool window for backward)
// ---------------------------------------------------------------------------

__global__ void maxpool2d_forward_kernel(const float* input, float* output, int* indices,
                                         int N, int C, int H, int W,
                                         int kH, int kW, int padH, int padW,
                                         int strideH, int strideW,
                                         int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    // Decompose linear index
    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;

    int h_start = h_out * strideH - padH;
    int w_start = w_out * strideW - padW;

    float max_val = -FLT_MAX;
    int max_idx = 0;

    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int h_in = h_start + kh;
            int w_in = w_start + kw;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                int in_idx = ((n * C + c) * H + h_in) * W + w_in;
                float val = input[in_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = in_idx;
                }
            }
        }
    }

    output[idx] = max_val;
    if (indices != nullptr) indices[idx] = max_idx;
}

void cuda_maxpool2d_forward(const float* d_input, float* d_output, int* d_indices,
                            int N, int C, int H, int W,
                            int kH, int kW, int padH, int padW,
                            int strideH, int strideW) {
    int H_out = (H + 2 * padH - kH) / strideH + 1;
    int W_out = (W + 2 * padW - kW) / strideW + 1;
    int total = N * C * H_out * W_out;
    int block = 256;
    int grid = (total + block - 1) / block;

    maxpool2d_forward_kernel<<<grid, block>>>(d_input, d_output, d_indices,
                                               N, C, H, W,
                                               kH, kW, padH, padW,
                                               strideH, strideW,
                                               H_out, W_out);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// MaxPool2d backward
// Uses saved indices to scatter gradients.
// ---------------------------------------------------------------------------

__global__ void maxpool2d_backward_kernel(const float* grad_output, const int* indices,
                                           float* grad_input,
                                           int total_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    atomicAdd(&grad_input[indices[idx]], grad_output[idx]);
}

void cuda_maxpool2d_backward(const float* d_grad_output, const int* d_indices,
                             float* d_grad_input,
                             int N, int C, int H, int W,
                             int kH, int kW, int padH, int padW,
                             int strideH, int strideW) {
    int H_out = (H + 2 * padH - kH) / strideH + 1;
    int W_out = (W + 2 * padW - kW) / strideW + 1;
    int total_out = N * C * H_out * W_out;
    int total_in = N * C * H * W;

    // Zero grad_input first
    CUDA_CHECK(cudaMemset(d_grad_input, 0, total_in * sizeof(float)));

    int block = 256;
    int grid = (total_out + block - 1) / block;
    maxpool2d_backward_kernel<<<grid, block>>>(d_grad_output, d_indices, d_grad_input, total_out);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// AvgPool2d forward (hand-rolled kernel)
// ---------------------------------------------------------------------------

__global__ void avgpool2d_forward_kernel(const float* input, float* output,
                                          int N, int C, int H, int W,
                                          int kH, int kW, int padH, int padW,
                                          int strideH, int strideW,
                                          int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;

    int h_start = h_out * strideH - padH;
    int w_start = w_out * strideW - padW;

    float sum = 0.0f;
    int count = 0;

    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int h_in = h_start + kh;
            int w_in = w_start + kw;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                sum += input[((n * C + c) * H + h_in) * W + w_in];
                count++;
            }
        }
    }

    output[idx] = (count > 0) ? sum / (float)count : 0.0f;
}

void cuda_avgpool2d_forward(const float* d_input, float* d_output,
                            int N, int C, int H, int W,
                            int kH, int kW, int padH, int padW,
                            int strideH, int strideW) {
    int H_out = (H + 2 * padH - kH) / strideH + 1;
    int W_out = (W + 2 * padW - kW) / strideW + 1;
    int total = N * C * H_out * W_out;
    int block = 256;
    int grid = (total + block - 1) / block;

    avgpool2d_forward_kernel<<<grid, block>>>(d_input, d_output,
                                               N, C, H, W,
                                               kH, kW, padH, padW,
                                               strideH, strideW,
                                               H_out, W_out);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// AvgPool2d backward
// Distributes gradient equally among the pooling window elements.
// ---------------------------------------------------------------------------

__global__ void avgpool2d_backward_kernel(const float* grad_output, float* grad_input,
                                           int N, int C, int H, int W,
                                           int kH, int kW, int padH, int padW,
                                           int strideH, int strideW,
                                           int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;

    int h_start = h_out * strideH - padH;
    int w_start = w_out * strideW - padW;

    // Count valid elements in this pool window
    int count = 0;
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int h_in = h_start + kh;
            int w_in = w_start + kw;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) count++;
        }
    }

    if (count == 0) return;
    float grad_val = grad_output[idx] / (float)count;

    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int h_in = h_start + kh;
            int w_in = w_start + kw;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                int in_idx = ((n * C + c) * H + h_in) * W + w_in;
                atomicAdd(&grad_input[in_idx], grad_val);
            }
        }
    }
}

void cuda_avgpool2d_backward(const float* d_grad_output, float* d_grad_input,
                             int N, int C, int H, int W,
                             int kH, int kW, int padH, int padW,
                             int strideH, int strideW) {
    int H_out = (H + 2 * padH - kH) / strideH + 1;
    int W_out = (W + 2 * padW - kW) / strideW + 1;
    int total_out = N * C * H_out * W_out;
    int total_in = N * C * H * W;

    // Zero grad_input first
    CUDA_CHECK(cudaMemset(d_grad_input, 0, total_in * sizeof(float)));

    int block = 256;
    int grid = (total_out + block - 1) / block;
    avgpool2d_backward_kernel<<<grid, block>>>(d_grad_output, d_grad_input,
                                                N, C, H, W,
                                                kH, kW, padH, padW,
                                                strideH, strideW,
                                                H_out, W_out);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Global average pooling: [N, C, H, W] -> [N, C, 1, 1]
// ---------------------------------------------------------------------------

__global__ void global_avgpool2d_forward_kernel(const float* input, float* output,
                                                 int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx >= total) return;

    int n = idx / C;
    int c = idx % C;
    int spatial = H * W;

    const float* in_ptr = input + (n * C + c) * spatial;
    float sum = 0.0f;
    for (int i = 0; i < spatial; i++) {
        sum += in_ptr[i];
    }
    output[idx] = sum / (float)spatial;
}

void cuda_global_avgpool2d_forward(const float* d_input, float* d_output,
                                   int N, int C, int H, int W) {
    int total = N * C;
    int block = 256;
    int grid = (total + block - 1) / block;
    global_avgpool2d_forward_kernel<<<grid, block>>>(d_input, d_output, N, C, H, W);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void global_avgpool2d_backward_kernel(const float* grad_output, float* grad_input,
                                                  int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int spatial = H * W;
    int nc = idx / spatial;  // which (n, c) pair
    float grad_val = grad_output[nc] / (float)spatial;
    grad_input[idx] = grad_val;
}

void cuda_global_avgpool2d_backward(const float* d_grad_output, float* d_grad_input,
                                    int N, int C, int H, int W) {
    int total = N * C * H * W;
    int block = 256;
    int grid = (total + block - 1) / block;
    global_avgpool2d_backward_kernel<<<grid, block>>>(d_grad_output, d_grad_input, N, C, H, W);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace whitematter
