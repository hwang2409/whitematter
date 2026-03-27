#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

#include <cstddef>

namespace whitematter {

// CUDA backend singleton. Only built when CUDA=1.
// All pointer arguments (d_*) are device pointers unless otherwise noted.
class CUDABackend {
public:
    static CUDABackend& instance();
    bool is_available() const;
    void init();

    // --- BLAS (device pointers) ---
    // matmul: C = A @ B, A is [M,K], B is [K,N], C is [M,N], row-major.
    void matmul(const float* d_A, const float* d_B, float* d_C, int M, int N, int K);
    // bmm: batched matmul, A [batch,M,K], B [batch,K,N], C [batch,M,N]
    void bmm(const float* d_A, const float* d_B, float* d_C, int batch, int M, int K, int N);

    // --- Element-wise ops (device pointers) ---
    void elementwise_add(const float* d_a, const float* d_b, float* d_c, size_t n);
    void elementwise_sub(const float* d_a, const float* d_b, float* d_c, size_t n);
    void elementwise_mul(const float* d_a, const float* d_b, float* d_c, size_t n);
    void elementwise_div(const float* d_a, const float* d_b, float* d_c, size_t n);
    void scalar_mul(const float* d_a, float scalar, float* d_c, size_t n);
    void fill(float* d_ptr, float val, size_t n);

    // --- Activations (device pointers) ---
    void relu_forward(const float* d_x, float* d_y, size_t n);
    void relu_backward(float* d_grad_in, const float* d_grad_out, const float* d_input, size_t n);
    void sigmoid_forward(const float* d_x, float* d_y, size_t n);
    void sigmoid_backward(float* d_grad_in, const float* d_grad_out, const float* d_output, size_t n);
    void tanh_forward(const float* d_x, float* d_y, size_t n);
    void tanh_backward(float* d_grad_in, const float* d_grad_out, const float* d_output, size_t n);
    void silu_forward(const float* d_x, float* d_y, size_t n);
    void silu_backward(float* d_grad_in, const float* d_grad_out, const float* d_input, size_t n);
    void gelu_forward(const float* d_x, float* d_y, size_t n);
    void gelu_backward(float* d_grad_in, const float* d_grad_out, const float* d_input, size_t n);

    // --- Reductions (device pointers) ---
    void sum(const float* d_x, float* d_out, size_t n);
    void softmax_forward(const float* d_x, float* d_y, size_t rows, size_t cols);
    void log_softmax_forward(const float* d_x, float* d_y, size_t rows, size_t cols);

    // --- Conv2d via cuDNN ---
    void conv2d_forward(const float* d_input, const float* d_weight, const float* d_bias,
                        float* d_output, size_t batch, size_t in_ch, size_t out_ch,
                        size_t in_h, size_t in_w, size_t kernel_h, size_t kernel_w,
                        size_t stride, size_t padding, size_t groups);
    void conv2d_backward(const float* d_input, const float* d_weight,
                         const float* d_grad_output, float* d_grad_input,
                         float* d_grad_weight, float* d_grad_bias,
                         size_t batch, size_t in_ch, size_t out_ch,
                         size_t in_h, size_t in_w, size_t kernel_h, size_t kernel_w,
                         size_t stride, size_t padding, size_t groups);

    // --- BatchNorm via cuDNN ---
    void batchnorm_forward(const float* d_input, float* d_output,
                           const float* d_gamma, const float* d_beta,
                           float* d_running_mean, float* d_running_var,
                           float* d_save_mean, float* d_save_inv_var,
                           size_t batch, size_t channels, size_t spatial,
                           float eps, float momentum, bool training);
    void batchnorm_backward(const float* d_input, const float* d_grad_output,
                            float* d_grad_input, float* d_grad_gamma, float* d_grad_beta,
                            const float* d_gamma, const float* d_save_mean,
                            const float* d_save_inv_var,
                            size_t batch, size_t channels, size_t spatial, float eps);

    // --- Pooling ---
    void maxpool2d_forward(const float* d_input, float* d_output, int* d_indices,
                           size_t batch, size_t channels, size_t in_h, size_t in_w,
                           size_t kernel, size_t stride);
    void avgpool2d_forward(const float* d_input, float* d_output,
                           size_t batch, size_t channels, size_t in_h, size_t in_w,
                           size_t kernel, size_t stride);

    // --- Loss ---
    void cross_entropy_forward(const float* d_logits, const float* d_targets,
                               float* d_loss, float* d_grad_logits,
                               size_t batch, size_t num_classes);

    // --- Optimizer ---
    void sgd_step(float* d_params, const float* d_grads, float* d_velocity,
                  float lr, float momentum, size_t n);
    void adam_step(float* d_params, const float* d_grads, float* d_m, float* d_v,
                   float lr, float beta1, float beta2, float eps,
                   float bc1, float bc2, size_t n);
    void adamw_step(float* d_params, const float* d_grads, float* d_m, float* d_v,
                    float lr, float beta1, float beta2, float eps,
                    float bc1, float bc2, float weight_decay, size_t n);
    void rmsprop_step(float* d_params, const float* d_grads, float* d_v,
                      float lr, float alpha, float eps,
                      float momentum, float* d_buffer, float weight_decay, size_t n);

    // --- Weight cache management ---
    // Call after optimizer.step() to mark cached weight buffers as stale.
    // Next conv2d call will re-upload weights that have changed.
    void invalidate_weight_cache();

    // --- Shadow cache management ---
    // Call after optimizer.step() to clear device-side shadow buffers.
    // Shadows hold conv2d/batchnorm outputs on the GPU so the next layer
    // can skip H2D.  After parameter updates the forward outputs are stale.
    void invalidate_shadow_cache();

    // --- Memory transfers ---
    void memcpy_h2d(float* d_dst, const float* h_src, size_t n_floats);
    void memcpy_d2h(float* h_dst, const float* d_src, size_t n_floats);
    void memcpy_d2d(float* d_dst, const float* d_src, size_t n_floats);
    void memset_zero(float* d_ptr, size_t n_floats);

private:
    CUDABackend() = default;
    bool initialized_ = false;
    void* cublas_handle_ = nullptr;  // cublasHandle_t
    void* cudnn_handle_ = nullptr;   // cudnnHandle_t
    void* stream_ = nullptr;         // cudaStream_t
    void* workspace_ = nullptr;      // cuDNN workspace buffer
    size_t workspace_size_ = 0;
};

}  // namespace whitematter

#endif
