// CUDA tensor operation wrappers with autograd support.
// This file is pure C++ (not .cu) and compiles with g++/clang.
// All code is guarded by WHITEMATTER_CUDA so the non-CUDA build is unaffected.

#if defined(WHITEMATTER_CUDA)

#include "cuda_tensor_ops.h"
#include "cuda_backend.h"
#include "cuda_memory.h"
#include "../tensor.h"
#include "../autograd.h"

namespace cuda_ops {

// ---------------------------------------------------------------------------
// Element-wise binary ops
// ---------------------------------------------------------------------------

TensorPtr add(const Tensor* self, const TensorPtr& other) {
    // Only handles same-shape fast path; broadcasting falls back to CPU
    bool track = (self->requires_grad || other->requires_grad) && GradMode::is_enabled();
    auto result = Tensor::create_on_device(self->shape, track, whitematter::DeviceType::CUDA);

    whitematter::CUDABackend::instance().elementwise_add(
        self->data(), other->data(), result->data(), self->size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(self)->shared_from_this();
        auto other_ptr = other;
        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result]() {
            auto& be = whitematter::CUDABackend::instance();
            if (self_ptr->requires_grad) {
                be.elementwise_add(self_ptr->grad(), result->grad(),
                                   self_ptr->grad(), self_ptr->size());
            }
            if (other_ptr->requires_grad) {
                be.elementwise_add(other_ptr->grad(), result->grad(),
                                   other_ptr->grad(), other_ptr->size());
            }
        };
    }
    return result;
}

TensorPtr sub(const Tensor* self, const TensorPtr& other) {
    bool track = (self->requires_grad || other->requires_grad) && GradMode::is_enabled();
    auto result = Tensor::create_on_device(self->shape, track, whitematter::DeviceType::CUDA);

    whitematter::CUDABackend::instance().elementwise_sub(
        self->data(), other->data(), result->data(), self->size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(self)->shared_from_this();
        auto other_ptr = other;
        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result]() {
            auto& be = whitematter::CUDABackend::instance();
            if (self_ptr->requires_grad) {
                be.elementwise_add(self_ptr->grad(), result->grad(),
                                   self_ptr->grad(), self_ptr->size());
            }
            if (other_ptr->requires_grad) {
                be.elementwise_sub(other_ptr->grad(), result->grad(),
                                   other_ptr->grad(), other_ptr->size());
            }
        };
    }
    return result;
}

TensorPtr mul(const Tensor* self, const TensorPtr& other) {
    bool track = (self->requires_grad || other->requires_grad) && GradMode::is_enabled();
    auto result = Tensor::create_on_device(self->shape, track, whitematter::DeviceType::CUDA);

    whitematter::CUDABackend::instance().elementwise_mul(
        self->data(), other->data(), result->data(), self->size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(self)->shared_from_this();
        auto other_ptr = other;
        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result]() {
            auto& be = whitematter::CUDABackend::instance();
            size_t n = self_ptr->size();
            auto& pool = whitematter::CUDAMemoryPool::instance();
            auto tmp = pool.acquire_shared(n);
            if (self_ptr->requires_grad) {
                // tmp = grad_out * other_data
                be.elementwise_mul(result->grad(), other_ptr->data(), tmp.get(), n);
                be.elementwise_add(self_ptr->grad(), tmp.get(), self_ptr->grad(), n);
            }
            if (other_ptr->requires_grad) {
                // tmp = grad_out * self_data
                be.elementwise_mul(result->grad(), self_ptr->data(), tmp.get(), n);
                be.elementwise_add(other_ptr->grad(), tmp.get(), other_ptr->grad(), n);
            }
        };
    }
    return result;
}

TensorPtr div(const Tensor* self, const TensorPtr& other) {
    bool track = (self->requires_grad || other->requires_grad) && GradMode::is_enabled();
    auto result = Tensor::create_on_device(self->shape, track, whitematter::DeviceType::CUDA);

    whitematter::CUDABackend::instance().elementwise_div(
        self->data(), other->data(), result->data(), self->size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(self)->shared_from_this();
        auto other_ptr = other;
        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result]() {
            auto& be = whitematter::CUDABackend::instance();
            size_t n = self_ptr->size();
            auto& pool = whitematter::CUDAMemoryPool::instance();
            auto tmp = pool.acquire_shared(n);
            if (self_ptr->requires_grad) {
                // d(a/b)/da = 1/b => grad_in += grad_out / b
                be.elementwise_div(result->grad(), other_ptr->data(), tmp.get(), n);
                be.elementwise_add(self_ptr->grad(), tmp.get(), self_ptr->grad(), n);
            }
            if (other_ptr->requires_grad) {
                // d(a/b)/db = -a/b^2 => grad_in -= grad_out * a / b^2
                // tmp1 = grad_out * self_data
                be.elementwise_mul(result->grad(), self_ptr->data(), tmp.get(), n);
                // tmp2 = b * b
                auto tmp2 = pool.acquire_shared(n);
                be.elementwise_mul(other_ptr->data(), other_ptr->data(), tmp2.get(), n);
                // tmp = tmp / tmp2
                auto tmp3 = pool.acquire_shared(n);
                be.elementwise_div(tmp.get(), tmp2.get(), tmp3.get(), n);
                // other_grad -= tmp3
                be.elementwise_sub(other_ptr->grad(), tmp3.get(), other_ptr->grad(), n);
            }
        };
    }
    return result;
}

TensorPtr scalar_mul(const Tensor* self, float scalar) {
    bool track = self->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create_on_device(self->shape, track, whitematter::DeviceType::CUDA);

    whitematter::CUDABackend::instance().scalar_mul(
        self->data(), scalar, result->data(), self->size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(self)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, scalar]() {
            auto& be = whitematter::CUDABackend::instance();
            size_t n = self_ptr->size();
            auto& pool = whitematter::CUDAMemoryPool::instance();
            auto tmp = pool.acquire_shared(n);
            be.scalar_mul(result->grad(), scalar, tmp.get(), n);
            be.elementwise_add(self_ptr->grad(), tmp.get(), self_ptr->grad(), n);
        };
    }
    return result;
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

TensorPtr relu(const Tensor* self) {
    bool track = self->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create_on_device(self->shape, track, whitematter::DeviceType::CUDA);

    whitematter::CUDABackend::instance().relu_forward(
        self->data(), result->data(), self->size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(self)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            whitematter::CUDABackend::instance().relu_backward(
                self_ptr->grad(), result->grad(), self_ptr->data(), self_ptr->size());
        };
    }
    return result;
}

TensorPtr sigmoid(const Tensor* self) {
    bool track = self->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create_on_device(self->shape, track, whitematter::DeviceType::CUDA);

    whitematter::CUDABackend::instance().sigmoid_forward(
        self->data(), result->data(), self->size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(self)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            whitematter::CUDABackend::instance().sigmoid_backward(
                self_ptr->grad(), result->grad(), result->data(), self_ptr->size());
        };
    }
    return result;
}

TensorPtr tanh_op(const Tensor* self) {
    bool track = self->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create_on_device(self->shape, track, whitematter::DeviceType::CUDA);

    whitematter::CUDABackend::instance().tanh_forward(
        self->data(), result->data(), self->size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(self)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            whitematter::CUDABackend::instance().tanh_backward(
                self_ptr->grad(), result->grad(), result->data(), self_ptr->size());
        };
    }
    return result;
}

TensorPtr silu(const Tensor* self) {
    bool track = self->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create_on_device(self->shape, track, whitematter::DeviceType::CUDA);

    whitematter::CUDABackend::instance().silu_forward(
        self->data(), result->data(), self->size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(self)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            whitematter::CUDABackend::instance().silu_backward(
                self_ptr->grad(), result->grad(), self_ptr->data(), self_ptr->size());
        };
    }
    return result;
}

TensorPtr gelu(const Tensor* self) {
    bool track = self->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create_on_device(self->shape, track, whitematter::DeviceType::CUDA);

    whitematter::CUDABackend::instance().gelu_forward(
        self->data(), result->data(), self->size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(self)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            whitematter::CUDABackend::instance().gelu_backward(
                self_ptr->grad(), result->grad(), self_ptr->data(), self_ptr->size());
        };
    }
    return result;
}

TensorPtr mish(const Tensor* self) {
    // No dedicated CUDA mish kernel in CUDABackend, so compose from silu.
    // mish(x) = x * tanh(softplus(x)), but we can approximate via the forward path.
    // For now, fall back: transfer to CPU, compute, transfer back.
    // Actually, since there's no mish kernel, we do the same as silu but with
    // the mish formula. We'll use a 2-step approach:
    //   1. Compute softplus on GPU (using silu_forward as placeholder -- not correct)
    // Better approach: just return nullptr to signal caller to fall back to CPU.
    // The dispatch in tensor.cpp will handle this.
    (void)self;
    return nullptr;  // Signal: no CUDA kernel, fall back to CPU
}

// ---------------------------------------------------------------------------
// Matmul
// ---------------------------------------------------------------------------

TensorPtr matmul(const Tensor* self, const TensorPtr& other) {
    size_t m = self->shape[0], k = self->shape[1], n = other->shape[1];
    bool track = (self->requires_grad || other->requires_grad) && GradMode::is_enabled();
    auto result = Tensor::create_on_device({m, n}, track, whitematter::DeviceType::CUDA);

    whitematter::CUDABackend::instance().matmul(
        self->data(), other->data(), result->data(),
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));

    if (track) {
        auto self_ptr = const_cast<Tensor*>(self)->shared_from_this();
        auto other_ptr = other;
        result->parents = {self_ptr, other_ptr};
        result->grad_fn = [self_ptr, other_ptr, result, m, k, n]() {
            auto& be = whitematter::CUDABackend::instance();
            auto& pool = whitematter::CUDAMemoryPool::instance();
            if (self_ptr->requires_grad) {
                // grad_A += grad_C @ B^T
                // grad_C is [m,n], B is [k,n] -> B^T is [n,k]
                auto bt = pool.acquire_shared(k * n);
                // Transpose B: naive via kernel would be ideal, but we can use
                // matmul with transposed dimensions. CUDABackend::matmul does C = A @ B.
                // grad_A = grad_C @ B^T. We can compute by noting:
                // (grad_C @ B^T)^T = B @ grad_C^T, then transpose result.
                // Simpler: use bmm or just do it row-major.
                // Actually, cublas handles row-major by interpreting as column-major transpose.
                // For now, we'll use the existing matmul which does C = A*B:
                // We need A_grad[m,k] = C_grad[m,n] * B[k,n]^T
                // Trick: cublas matmul(C_grad, B^T) but our API doesn't have transpose flag.
                // Fallback: transfer to CPU for backward.
                // This is acceptable for the dispatch scaffolding -- backward kernels
                // can be optimized later with cublas transpose flags.
                auto tmp_cgrad = pool.acquire_shared(m * n);
                be.memcpy_d2d(tmp_cgrad.get(), result->grad(), m * n);
                // For now, fall back to CPU-based backward
                // Copy data to CPU, compute, copy back
                std::vector<float> cg(m * n), bd(k * n), ag(m * k);
                be.memcpy_d2h(cg.data(), result->grad(), m * n);
                be.memcpy_d2h(bd.data(), other_ptr->data(), k * n);
                be.memcpy_d2h(ag.data(), self_ptr->grad(), m * k);
                for (size_t i = 0; i < m; i++) {
                    for (size_t l = 0; l < k; l++) {
                        float sum = 0.0f;
                        for (size_t j = 0; j < n; j++) {
                            sum += cg[i * n + j] * bd[l * n + j];
                        }
                        ag[i * k + l] += sum;
                    }
                }
                be.memcpy_h2d(self_ptr->grad(), ag.data(), m * k);
            }
            if (other_ptr->requires_grad) {
                std::vector<float> cg(m * n), ad(m * k), bg(k * n);
                be.memcpy_d2h(cg.data(), result->grad(), m * n);
                be.memcpy_d2h(ad.data(), self_ptr->data(), m * k);
                be.memcpy_d2h(bg.data(), other_ptr->grad(), k * n);
                for (size_t l = 0; l < k; l++) {
                    for (size_t j = 0; j < n; j++) {
                        float sum = 0.0f;
                        for (size_t i = 0; i < m; i++) {
                            sum += ad[i * k + l] * cg[i * n + j];
                        }
                        bg[l * n + j] += sum;
                    }
                }
                be.memcpy_h2d(other_ptr->grad(), bg.data(), k * n);
            }
        };
    }
    return result;
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

TensorPtr sum(const Tensor* self) {
    bool track = self->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create_on_device({1}, track, whitematter::DeviceType::CUDA);

    whitematter::CUDABackend::instance().sum(
        self->data(), result->data(), self->size());

    if (track) {
        auto self_ptr = const_cast<Tensor*>(self)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result]() {
            // grad_in += grad_out[0] (broadcast scalar to all elements)
            auto& be = whitematter::CUDABackend::instance();
            auto& pool = whitematter::CUDAMemoryPool::instance();
            size_t n = self_ptr->size();
            auto tmp = pool.acquire_shared(n);
            be.fill(tmp.get(), 1.0f, n);
            be.scalar_mul(tmp.get(), 0.0f, tmp.get(), n); // zero it
            // Actually: we need grad_out[0] broadcast. Copy scalar, fill, multiply.
            // Simpler: read grad_out[0] to host, fill tmp with that value, add.
            float g;
            be.memcpy_d2h(&g, result->grad(), 1);
            be.fill(tmp.get(), g, n);
            be.elementwise_add(self_ptr->grad(), tmp.get(), self_ptr->grad(), n);
        };
    }
    return result;
}

TensorPtr mean(const Tensor* self) {
    // mean = sum / N
    auto s = sum(self);
    if (!s) return nullptr;
    float inv_n = 1.0f / static_cast<float>(self->size());
    // Scale the result on GPU
    auto result = Tensor::create_on_device({1}, false, whitematter::DeviceType::CUDA);
    whitematter::CUDABackend::instance().scalar_mul(s->data(), inv_n, result->data(), 1);

    bool track = self->requires_grad && GradMode::is_enabled();
    if (track) {
        result->requires_grad = true;
        // Allocate grad for result
        auto& pool = whitematter::CUDAMemoryPool::instance();
        // We need to set up grad -- but create_on_device with track=false didn't allocate it.
        // Recreate properly:
    }
    // Simpler: just use sum then scalar_mul through the tensor API
    // The caller (mean in tensor.cpp) already does sum()->mul(1/N).
    // So we only need to provide sum() dispatch and let mean() compose.
    // Return nullptr to let the caller handle composition.
    (void)result;
    return nullptr;
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

TensorPtr softmax(const Tensor* self, int dim) {
    assert(self->shape.size() == 2);
    if (dim < 0) dim = static_cast<int>(self->shape.size()) + dim;

    bool track = self->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create_on_device(self->shape, track, whitematter::DeviceType::CUDA);

    size_t rows = self->shape[0], cols = self->shape[1];
    whitematter::CUDABackend::instance().softmax_forward(
        self->data(), result->data(), rows, cols);

    if (track) {
        auto self_ptr = const_cast<Tensor*>(self)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, rows, cols]() {
            // Softmax backward: grad_in[i,j] = sum_k(grad_out[i,k] * (delta(j,k) - sm[i,k]) * sm[i,j])
            // Fall back to CPU for backward (complex Jacobian)
            auto& be = whitematter::CUDABackend::instance();
            size_t n = rows * cols;
            std::vector<float> sm(n), go(n), gi(n);
            be.memcpy_d2h(sm.data(), result->data(), n);
            be.memcpy_d2h(go.data(), result->grad(), n);
            be.memcpy_d2h(gi.data(), self_ptr->grad(), n);
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    float sj = sm[i * cols + j];
                    for (size_t k = 0; k < cols; k++) {
                        float sk = sm[i * cols + k];
                        float grad = (j == k) ? sj * (1.0f - sj) : -sj * sk;
                        gi[i * cols + j] += go[i * cols + k] * grad;
                    }
                }
            }
            be.memcpy_h2d(self_ptr->grad(), gi.data(), n);
        };
    }
    return result;
}

TensorPtr log_softmax(const Tensor* self, int dim) {
    assert(self->shape.size() == 2);
    if (dim < 0) dim = static_cast<int>(self->shape.size()) + dim;

    bool track = self->requires_grad && GradMode::is_enabled();
    auto result = Tensor::create_on_device(self->shape, track, whitematter::DeviceType::CUDA);

    size_t rows = self->shape[0], cols = self->shape[1];
    whitematter::CUDABackend::instance().log_softmax_forward(
        self->data(), result->data(), rows, cols);

    if (track) {
        auto self_ptr = const_cast<Tensor*>(self)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, rows, cols]() {
            auto& be = whitematter::CUDABackend::instance();
            size_t n = rows * cols;
            std::vector<float> lsm(n), go(n), gi(n);
            be.memcpy_d2h(lsm.data(), result->data(), n);
            be.memcpy_d2h(go.data(), result->grad(), n);
            be.memcpy_d2h(gi.data(), self_ptr->grad(), n);
            for (size_t i = 0; i < rows; i++) {
                float grad_sum = 0.0f;
                for (size_t j = 0; j < cols; j++) {
                    grad_sum += go[i * cols + j];
                }
                for (size_t j = 0; j < cols; j++) {
                    float softmax_j = std::exp(lsm[i * cols + j]);
                    gi[i * cols + j] += go[i * cols + j] - softmax_j * grad_sum;
                }
            }
            be.memcpy_h2d(self_ptr->grad(), gi.data(), n);
        };
    }
    return result;
}

// ---------------------------------------------------------------------------
// Conv2d
// ---------------------------------------------------------------------------

TensorPtr conv2d(const Tensor* self, const TensorPtr& weight, const TensorPtr& bias,
                  size_t stride, size_t padding, size_t groups, size_t dilation) {
    // CUDABackend conv2d_forward does not support dilation; skip if dilation > 1
    if (dilation > 1) return nullptr;

    size_t batch = self->shape[0];
    size_t in_channels = self->shape[1];
    size_t in_h = self->shape[2];
    size_t in_w = self->shape[3];
    size_t out_channels = weight->shape[0];
    size_t kernel_h = weight->shape[2];
    size_t kernel_w = weight->shape[3];
    size_t out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    size_t out_w = (in_w + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    bool track = (self->requires_grad || weight->requires_grad ||
                  (bias && bias->requires_grad)) && GradMode::is_enabled();
    auto result = Tensor::create_on_device({batch, out_channels, out_h, out_w}, track,
                                            whitematter::DeviceType::CUDA);

    whitematter::CUDABackend::instance().conv2d_forward(
        self->data(), weight->data(), bias ? bias->data() : nullptr,
        result->data(), batch, in_channels, out_channels,
        in_h, in_w, kernel_h, kernel_w, stride, padding, groups);

    if (track) {
        auto self_ptr = const_cast<Tensor*>(self)->shared_from_this();
        auto weight_ptr = weight;
        auto bias_ptr = bias;
        result->parents = {self_ptr, weight_ptr};
        if (bias_ptr) result->parents.push_back(bias_ptr);

        result->grad_fn = [self_ptr, weight_ptr, bias_ptr, result,
                           batch, in_channels, in_h, in_w,
                           out_channels, out_h, out_w,
                           kernel_h, kernel_w, stride, padding, groups]() {
            whitematter::CUDABackend::instance().conv2d_backward(
                self_ptr->data(), weight_ptr->data(),
                result->grad(),
                self_ptr->requires_grad ? self_ptr->grad() : nullptr,
                weight_ptr->requires_grad ? weight_ptr->grad() : nullptr,
                (bias_ptr && bias_ptr->requires_grad) ? bias_ptr->grad() : nullptr,
                batch, in_channels, out_channels,
                in_h, in_w, kernel_h, kernel_w, stride, padding, groups);
        };
    }
    return result;
}

} // namespace cuda_ops

#endif // WHITEMATTER_CUDA
