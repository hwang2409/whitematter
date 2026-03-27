#pragma once
#include "../tensor.h"

#if defined(WHITEMATTER_CUDA)

namespace cuda_ops {

// Element-wise binary ops
TensorPtr add(const Tensor* self, const TensorPtr& other);
TensorPtr sub(const Tensor* self, const TensorPtr& other);
TensorPtr mul(const Tensor* self, const TensorPtr& other);
TensorPtr div(const Tensor* self, const TensorPtr& other);
TensorPtr scalar_mul(const Tensor* self, float scalar);

// Activations
TensorPtr relu(const Tensor* self);
TensorPtr sigmoid(const Tensor* self);
TensorPtr tanh_op(const Tensor* self);
TensorPtr silu(const Tensor* self);
TensorPtr gelu(const Tensor* self);
TensorPtr mish(const Tensor* self);

// Matmul
TensorPtr matmul(const Tensor* self, const TensorPtr& other);

// Reductions
TensorPtr sum(const Tensor* self);
TensorPtr mean(const Tensor* self);

// Softmax
TensorPtr softmax(const Tensor* self, int dim);
TensorPtr log_softmax(const Tensor* self, int dim);

// Conv2d
TensorPtr conv2d(const Tensor* self, const TensorPtr& weight, const TensorPtr& bias,
                  size_t stride, size_t padding, size_t groups, size_t dilation);

} // namespace cuda_ops

#endif // WHITEMATTER_CUDA
