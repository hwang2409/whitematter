#include "../tensor.h"
#include "im2col.h"
#include "matmul_cpu.h"
#include "simd_ops.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#if defined(WHITEMATTER_METAL) && defined(__APPLE__)
#include "../metal/metal_backend.h"
#endif
#if defined(WHITEMATTER_CUDA)
#include "../cuda/cuda_backend.h"
#endif

TensorPtr Tensor::conv2d(const TensorPtr& weight, const TensorPtr& bias,
                          size_t stride, size_t padding) const {
    assert(shape.size() == 4);
    assert(weight->shape.size() == 4);
    assert(shape[1] == weight->shape[1]);

    size_t batch = shape[0];
    size_t in_channels = shape[1];
    size_t in_h = shape[2];
    size_t in_w = shape[3];

    size_t out_channels = weight->shape[0];
    size_t kernel_h = weight->shape[2];
    size_t kernel_w = weight->shape[3];

    size_t out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    size_t out_w = (in_w + 2 * padding - kernel_w) / stride + 1;

    bool track = (requires_grad || weight->requires_grad || (bias && bias->requires_grad))
                 && GradMode::is_enabled();
    auto result = create({batch, out_channels, out_h, out_w}, track);

    size_t col_h = in_channels * kernel_h * kernel_w;
    size_t col_w = out_h * out_w;
    std::vector<float> col_buffer(col_h * col_w);

    for (size_t b = 0; b < batch; b++) {
        const float* input_ptr = data() + b * in_channels * in_h * in_w;
        float* output_ptr = result->data() + b * out_channels * out_h * out_w;

        im2col(input_ptr, col_buffer.data(),
               in_channels, in_h, in_w,
               kernel_h, kernel_w,
               out_h, out_w, stride, padding);

        matmul_blocked(output_ptr, weight->data(), col_buffer.data(),
                       out_channels, col_h, col_w);

        if (bias) {
            for (size_t oc = 0; oc < out_channels; oc++) {
                for (size_t i = 0; i < col_w; i++) {
                    output_ptr[oc * col_w + i] += bias->data()[oc];
                }
            }
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto weight_ptr = weight;
        auto bias_ptr = bias;
        result->parents = {self_ptr, weight_ptr};
        if (bias_ptr) result->parents.push_back(bias_ptr);

        result->grad_fn = [self_ptr, weight_ptr, bias_ptr, result,
                           batch, in_channels, in_h, in_w,
                           out_channels, out_h, out_w,
                           kernel_h, kernel_w, stride, padding]() {

            size_t col_h = in_channels * kernel_h * kernel_w;
            size_t col_w = out_h * out_w;

            if (self_ptr->requires_grad) {
                std::vector<float> weight_T(col_h * out_channels);
                for (size_t oc = 0; oc < out_channels; oc++) {
                    for (size_t k = 0; k < col_h; k++) {
                        weight_T[k * out_channels + oc] = weight_ptr->data()[oc * col_h + k];
                    }
                }

                std::vector<float> col_grad(col_h * col_w);

                for (size_t b = 0; b < batch; b++) {
                    const float* grad_out = result->grad() + b * out_channels * col_w;
                    float* grad_in = self_ptr->grad() + b * in_channels * in_h * in_w;

                    matmul_blocked(col_grad.data(), weight_T.data(), grad_out,
                                   col_h, out_channels, col_w);

                    col2im(col_grad.data(), grad_in,
                           in_channels, in_h, in_w,
                           kernel_h, kernel_w,
                           out_h, out_w, stride, padding);
                }
            }

            if (weight_ptr->requires_grad) {
                std::vector<float> col_buffer(col_h * col_w);
                std::vector<float> col_T(col_w * col_h);

                for (size_t b = 0; b < batch; b++) {
                    const float* input_ptr = self_ptr->data() + b * in_channels * in_h * in_w;
                    const float* grad_out = result->grad() + b * out_channels * col_w;

                    im2col(input_ptr, col_buffer.data(),
                           in_channels, in_h, in_w,
                           kernel_h, kernel_w,
                           out_h, out_w, stride, padding);

                    for (size_t i = 0; i < col_h; i++) {
                        for (size_t j = 0; j < col_w; j++) {
                            col_T[j * col_h + i] = col_buffer[i * col_w + j];
                        }
                    }

                    std::vector<float> grad_w_batch(out_channels * col_h);
                    matmul_blocked(grad_w_batch.data(), grad_out, col_T.data(),
                                   out_channels, col_w, col_h);

                    for (size_t i = 0; i < out_channels * col_h; i++) {
                        weight_ptr->grad()[i] += grad_w_batch[i];
                    }
                }
            }

            if (bias_ptr && bias_ptr->requires_grad) {
                for (size_t oc = 0; oc < out_channels; oc++) {
                    float grad_sum = 0.0f;
                    for (size_t b = 0; b < batch; b++) {
                        for (size_t i = 0; i < col_w; i++) {
                            size_t out_idx = b * out_channels * col_w + oc * col_w + i;
                            grad_sum += result->grad()[out_idx];
                        }
                    }
                    bias_ptr->grad()[oc] += grad_sum;
                }
            }
        };
    }

    return result;
}

TensorPtr Tensor::conv_transpose2d(const TensorPtr& weight, const TensorPtr& bias,
                                    size_t stride, size_t padding,
                                    size_t output_padding) const {
    assert(shape.size() == 4);
    assert(weight->shape.size() == 4);
    assert(shape[1] == weight->shape[0]);
    assert(output_padding < stride);

    size_t batch = shape[0];
    size_t in_channels = shape[1];
    size_t in_h = shape[2];
    size_t in_w = shape[3];

    size_t out_channels = weight->shape[1];
    size_t kernel_h = weight->shape[2];
    size_t kernel_w = weight->shape[3];

    size_t out_h = (in_h - 1) * stride - 2 * padding + kernel_h + output_padding;
    size_t out_w = (in_w - 1) * stride - 2 * padding + kernel_w + output_padding;

    bool track = (requires_grad || weight->requires_grad || (bias && bias->requires_grad))
                 && GradMode::is_enabled();
    auto result = create({batch, out_channels, out_h, out_w}, track);

    std::fill(result->data(), result->data() + result->size(), 0.0f);

    for (size_t b = 0; b < batch; b++) {
        for (size_t ic = 0; ic < in_channels; ic++) {
            for (size_t ih = 0; ih < in_h; ih++) {
                for (size_t iw = 0; iw < in_w; iw++) {
                    size_t in_idx = b * in_channels * in_h * in_w +
                                    ic * in_h * in_w +
                                    ih * in_w + iw;
                    float in_val = data()[in_idx];

                    for (size_t oc = 0; oc < out_channels; oc++) {
                        for (size_t kh = 0; kh < kernel_h; kh++) {
                            for (size_t kw = 0; kw < kernel_w; kw++) {
                                int oh = static_cast<int>(ih * stride + kh) - static_cast<int>(padding);
                                int ow = static_cast<int>(iw * stride + kw) - static_cast<int>(padding);

                                if (oh >= 0 && oh < static_cast<int>(out_h) &&
                                    ow >= 0 && ow < static_cast<int>(out_w)) {
                                    size_t w_idx = ic * out_channels * kernel_h * kernel_w +
                                                   oc * kernel_h * kernel_w +
                                                   kh * kernel_w + kw;
                                    size_t out_idx = b * out_channels * out_h * out_w +
                                                     oc * out_h * out_w +
                                                     static_cast<size_t>(oh) * out_w +
                                                     static_cast<size_t>(ow);
                                    result->data()[out_idx] += in_val * weight->data()[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (bias) {
        for (size_t b = 0; b < batch; b++) {
            for (size_t oc = 0; oc < out_channels; oc++) {
                for (size_t oh = 0; oh < out_h; oh++) {
                    for (size_t ow = 0; ow < out_w; ow++) {
                        size_t idx = b * out_channels * out_h * out_w +
                                     oc * out_h * out_w +
                                     oh * out_w + ow;
                        result->data()[idx] += bias->data()[oc];
                    }
                }
            }
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto weight_ptr = weight;
        auto bias_ptr = bias;
        result->parents = {self_ptr, weight_ptr};
        if (bias_ptr) result->parents.push_back(bias_ptr);

        result->grad_fn = [self_ptr, weight_ptr, bias_ptr, result,
                           batch, in_channels, in_h, in_w,
                           out_channels, out_h, out_w,
                           kernel_h, kernel_w, stride, padding]() {

            if (self_ptr->requires_grad) {
                for (size_t b = 0; b < batch; b++) {
                    for (size_t ic = 0; ic < in_channels; ic++) {
                        for (size_t ih = 0; ih < in_h; ih++) {
                            for (size_t iw = 0; iw < in_w; iw++) {
                                float grad_sum = 0.0f;

                                for (size_t oc = 0; oc < out_channels; oc++) {
                                    for (size_t kh = 0; kh < kernel_h; kh++) {
                                        for (size_t kw = 0; kw < kernel_w; kw++) {
                                            int oh = static_cast<int>(ih * stride + kh) - static_cast<int>(padding);
                                            int ow = static_cast<int>(iw * stride + kw) - static_cast<int>(padding);

                                            if (oh >= 0 && oh < static_cast<int>(out_h) &&
                                                ow >= 0 && ow < static_cast<int>(out_w)) {
                                                size_t w_idx = ic * out_channels * kernel_h * kernel_w +
                                                               oc * kernel_h * kernel_w +
                                                               kh * kernel_w + kw;
                                                size_t out_idx = b * out_channels * out_h * out_w +
                                                                 oc * out_h * out_w +
                                                                 static_cast<size_t>(oh) * out_w +
                                                                 static_cast<size_t>(ow);
                                                grad_sum += result->grad()[out_idx] * weight_ptr->data()[w_idx];
                                            }
                                        }
                                    }
                                }

                                size_t in_idx = b * in_channels * in_h * in_w +
                                                ic * in_h * in_w +
                                                ih * in_w + iw;
                                self_ptr->grad()[in_idx] += grad_sum;
                            }
                        }
                    }
                }
            }

            if (weight_ptr->requires_grad) {
                for (size_t b = 0; b < batch; b++) {
                    for (size_t ic = 0; ic < in_channels; ic++) {
                        for (size_t ih = 0; ih < in_h; ih++) {
                            for (size_t iw = 0; iw < in_w; iw++) {
                                size_t in_idx = b * in_channels * in_h * in_w +
                                                ic * in_h * in_w +
                                                ih * in_w + iw;
                                float in_val = self_ptr->data()[in_idx];

                                for (size_t oc = 0; oc < out_channels; oc++) {
                                    for (size_t kh = 0; kh < kernel_h; kh++) {
                                        for (size_t kw = 0; kw < kernel_w; kw++) {
                                            int oh = static_cast<int>(ih * stride + kh) - static_cast<int>(padding);
                                            int ow = static_cast<int>(iw * stride + kw) - static_cast<int>(padding);

                                            if (oh >= 0 && oh < static_cast<int>(out_h) &&
                                                ow >= 0 && ow < static_cast<int>(out_w)) {
                                                size_t w_idx = ic * out_channels * kernel_h * kernel_w +
                                                               oc * kernel_h * kernel_w +
                                                               kh * kernel_w + kw;
                                                size_t out_idx = b * out_channels * out_h * out_w +
                                                                 oc * out_h * out_w +
                                                                 static_cast<size_t>(oh) * out_w +
                                                                 static_cast<size_t>(ow);
                                                weight_ptr->grad()[w_idx] += in_val * result->grad()[out_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (bias_ptr && bias_ptr->requires_grad) {
                for (size_t oc = 0; oc < out_channels; oc++) {
                    float grad_sum = 0.0f;
                    for (size_t b = 0; b < batch; b++) {
                        for (size_t oh = 0; oh < out_h; oh++) {
                            for (size_t ow = 0; ow < out_w; ow++) {
                                size_t idx = b * out_channels * out_h * out_w +
                                             oc * out_h * out_w +
                                             oh * out_w + ow;
                                grad_sum += result->grad()[idx];
                            }
                        }
                    }
                    bias_ptr->grad()[oc] += grad_sum;
                }
            }
        };
    }

    return result;
}

TensorPtr Tensor::maxpool2d(size_t kernel_size, size_t stride) const {
    assert(shape.size() == 4);

    if (stride == 0) stride = kernel_size;

    size_t batch = shape[0];
    size_t channels = shape[1];
    size_t in_h = shape[2];
    size_t in_w = shape[3];

    size_t out_h = (in_h - kernel_size) / stride + 1;
    size_t out_w = (in_w - kernel_size) / stride + 1;

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create({batch, channels, out_h, out_w}, track);

    std::vector<size_t> max_indices(result->size());

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    float max_val = -std::numeric_limits<float>::max();
                    size_t max_idx = 0;

                    for (size_t kh = 0; kh < kernel_size; kh++) {
                        for (size_t kw = 0; kw < kernel_size; kw++) {
                            size_t ih = oh * stride + kh;
                            size_t iw = ow * stride + kw;
                            size_t input_idx = b * (channels * in_h * in_w) +
                                               c * (in_h * in_w) +
                                               ih * in_w + iw;
                            if (data()[input_idx] > max_val) {
                                max_val = data()[input_idx];
                                max_idx = input_idx;
                            }
                        }
                    }

                    size_t output_idx = b * (channels * out_h * out_w) +
                                        c * (out_h * out_w) +
                                        oh * out_w + ow;
                    result->data()[output_idx] = max_val;
                    max_indices[output_idx] = max_idx;
                }
            }
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, max_indices]() {
            for (size_t i = 0; i < result->size(); i++) {
                self_ptr->grad()[max_indices[i]] += result->grad()[i];
            }
        };
    }

    return result;
}

TensorPtr Tensor::avgpool2d(size_t kernel_size, size_t stride) const {
    assert(shape.size() == 4);

    if (stride == 0) stride = kernel_size;

    size_t batch = shape[0];
    size_t channels = shape[1];
    size_t in_h = shape[2];
    size_t in_w = shape[3];

    size_t out_h = (in_h - kernel_size) / stride + 1;
    size_t out_w = (in_w - kernel_size) / stride + 1;

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create({batch, channels, out_h, out_w}, track);

    float pool_size = static_cast<float>(kernel_size * kernel_size);

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    float sum = 0.0f;

                    for (size_t kh = 0; kh < kernel_size; kh++) {
                        for (size_t kw = 0; kw < kernel_size; kw++) {
                            size_t ih = oh * stride + kh;
                            size_t iw = ow * stride + kw;
                            size_t input_idx = b * (channels * in_h * in_w) +
                                               c * (in_h * in_w) +
                                               ih * in_w + iw;
                            sum += data()[input_idx];
                        }
                    }

                    size_t output_idx = b * (channels * out_h * out_w) +
                                        c * (out_h * out_w) +
                                        oh * out_w + ow;
                    result->data()[output_idx] = sum / pool_size;
                }
            }
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};
        result->grad_fn = [self_ptr, result, batch, channels, in_h, in_w,
                           out_h, out_w, kernel_size, stride, pool_size]() {
            for (size_t b = 0; b < batch; b++) {
                for (size_t c = 0; c < channels; c++) {
                    for (size_t oh = 0; oh < out_h; oh++) {
                        for (size_t ow = 0; ow < out_w; ow++) {
                            size_t out_idx = b * (channels * out_h * out_w) +
                                             c * (out_h * out_w) +
                                             oh * out_w + ow;
                            float grad_val = result->grad()[out_idx] / pool_size;

                            for (size_t kh = 0; kh < kernel_size; kh++) {
                                for (size_t kw = 0; kw < kernel_size; kw++) {
                                    size_t ih = oh * stride + kh;
                                    size_t iw = ow * stride + kw;
                                    size_t input_idx = b * (channels * in_h * in_w) +
                                                       c * (in_h * in_w) +
                                                       ih * in_w + iw;
                                    self_ptr->grad()[input_idx] += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        };
    }

    return result;
}

TensorPtr Tensor::flatten(size_t start_dim) const {
    assert(start_dim < shape.size());

    std::vector<size_t> new_shape;
    size_t flat_size = 1;

    for (size_t i = 0; i < start_dim; i++) {
        new_shape.push_back(shape[i]);
    }
    for (size_t i = start_dim; i < shape.size(); i++) {
        flat_size *= shape[i];
    }
    new_shape.push_back(flat_size);

    return reshape(new_shape);
}
