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
#include "../cuda/cuda_tensor_ops.h"
#endif

// Reusable scratch buffers to avoid heap allocation on every conv forward/backward.
// Thread-local so they're safe with threaded dataloaders.
static thread_local std::vector<float> tl_col_buf;
static thread_local std::vector<float> tl_col_buf2;
static thread_local std::vector<float> tl_scratch;
static thread_local std::vector<float> tl_wino_filter;
static thread_local std::vector<float> tl_wino_input;
static thread_local std::vector<float> tl_wino_M;

static float* get_buf(std::vector<float>& buf, size_t n) {
    if (buf.size() < n) buf.resize(n);
    return buf.data();
}

// ============================================================================
// Winograd F(2x2, 3x3) convolution — forward pass only.
//
// Computes 2x2 output tiles from 4x4 input tiles and 3x3 filters using
// only 16 multiplications per tile instead of 36 (2.25x theoretical speedup).
//
// Valid only for kernel_size=3, stride=1, padding=1.
// ============================================================================

// Inline helper: compute B^T @ d @ B for a 4x4 input tile.
// B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
static inline void winograd_input_transform(const float d[4][4], float V[4][4]) {
    // Compute t = d @ B  (4x4 @ 4x4 -> 4x4)
    // B = [[1,0,0,0],[0,1,-1,1],[−1,1,1,0],[0,0,0,−1]]
    float t[4][4];
    for (int i = 0; i < 4; i++) {
        t[i][0] =  d[i][0] - d[i][2];
        t[i][1] =  d[i][1] + d[i][2];
        t[i][2] = -d[i][1] + d[i][2];
        t[i][3] =  d[i][1] - d[i][3];
    }
    // Compute V = B^T @ t  (4x4)
    for (int j = 0; j < 4; j++) {
        V[0][j] =  t[0][j] - t[2][j];
        V[1][j] =  t[1][j] + t[2][j];
        V[2][j] = -t[1][j] + t[2][j];
        V[3][j] =  t[1][j] - t[3][j];
    }
}

// Inline helper: compute G @ g @ G^T for a 3x3 filter.
// G = [[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]]
static inline void winograd_filter_transform(const float g[3][3], float U[4][4]) {
    // Compute t = g @ G^T  (3x3 @ 3x4 -> 3x4)
    // G^T = [[1,0.5,0.5,0],[0,0.5,-0.5,0],[0,0.5,0.5,1]]
    float t[3][4];
    for (int i = 0; i < 3; i++) {
        t[i][0] = g[i][0];
        t[i][1] = 0.5f * (g[i][0] + g[i][1] + g[i][2]);
        t[i][2] = 0.5f * (g[i][0] - g[i][1] + g[i][2]);
        t[i][3] = g[i][2];
    }
    // Compute U = G @ t  (4x3 @ 3x4 -> 4x4)
    for (int j = 0; j < 4; j++) {
        U[0][j] = t[0][j];
        U[1][j] = 0.5f * (t[0][j] + t[1][j] + t[2][j]);
        U[2][j] = 0.5f * (t[0][j] - t[1][j] + t[2][j]);
        U[3][j] = t[2][j];
    }
}

// Inline helper: compute A^T @ m @ A for a 4x4 element-wise product.
// A^T = [[1,1,1,0],[0,1,-1,-1]]
static inline void winograd_output_transform(const float m[4][4], float Y[2][2]) {
    // Compute t = m @ A  (4x4 @ 4x2 -> 4x2)
    // A = [[1,0],[1,1],[1,-1],[0,-1]]
    float t[4][2];
    for (int i = 0; i < 4; i++) {
        t[i][0] = m[i][0] + m[i][1] + m[i][2];
        t[i][1] = m[i][1] - m[i][2] - m[i][3];
    }
    // Compute Y = A^T @ t  (2x4 @ 4x2 -> 2x2)
    for (int j = 0; j < 2; j++) {
        Y[0][j] = t[0][j] + t[1][j] + t[2][j];
        Y[1][j] = t[1][j] - t[2][j] - t[3][j];
    }
}

static void conv2d_winograd_f2x2_3x3(
    float* output, const float* input, const float* weight, const float* bias_data,
    size_t batch, size_t in_channels, size_t out_channels,
    size_t in_h, size_t in_w, size_t out_h, size_t out_w)
{
    // Number of 2x2 output tiles along each spatial dimension.
    size_t tile_h = (out_h + 1) / 2;  // ceil(out_h / 2)
    size_t tile_w = (out_w + 1) / 2;  // ceil(out_w / 2)
    size_t num_tiles = tile_h * tile_w;

    // ---- Step 1: Pre-transform all filters ----
    // U_all[oc][ic][4][4] stored as flat: [out_channels * in_channels * 16]
    // Reorganized as U_elem[alpha*4+beta][out_channels * in_channels]
    // so that for each (alpha,beta) we have a contiguous [OC x IC] matrix.
    float* U_flat = get_buf(tl_wino_filter, 16 * out_channels * in_channels);

    for (size_t oc = 0; oc < out_channels; oc++) {
        for (size_t ic = 0; ic < in_channels; ic++) {
            const float* w = weight + (oc * in_channels + ic) * 9;  // 3x3 filter
            float g[3][3];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    g[i][j] = w[i * 3 + j];

            float U[4][4];
            winograd_filter_transform(g, U);

            // Store in element-major layout: U_flat[elem * OC * IC + oc * IC + ic]
            for (int a = 0; a < 4; a++)
                for (int b = 0; b < 4; b++)
                    U_flat[(a * 4 + b) * out_channels * in_channels + oc * in_channels + ic] = U[a][b];
        }
    }

    // ---- Per-batch processing ----
    for (size_t n = 0; n < batch; n++) {
        const float* in_batch = input + n * in_channels * in_h * in_w;
        float* out_batch = output + n * out_channels * out_h * out_w;

        // ---- Step 2: Transform all input tiles ----
        // V_flat[elem][in_channels * num_tiles]
        // Layout: V_flat[elem * IC * num_tiles + ic * num_tiles + tile_idx]
        float* V_flat = get_buf(tl_wino_input, 16 * in_channels * num_tiles);

        for (size_t ic = 0; ic < in_channels; ic++) {
            const float* in_ch = in_batch + ic * in_h * in_w;

            for (size_t th = 0; th < tile_h; th++) {
                for (size_t tw = 0; tw < tile_w; tw++) {
                    size_t tile_idx = th * tile_w + tw;

                    // Extract 4x4 input tile with padding handling.
                    // The tile starts at spatial position (th*2 - 1, tw*2 - 1) due to padding=1.
                    int row_start = static_cast<int>(th * 2) - 1;
                    int col_start = static_cast<int>(tw * 2) - 1;

                    float d[4][4];
                    for (int i = 0; i < 4; i++) {
                        int r = row_start + i;
                        for (int j = 0; j < 4; j++) {
                            int c = col_start + j;
                            if (r >= 0 && r < static_cast<int>(in_h) &&
                                c >= 0 && c < static_cast<int>(in_w)) {
                                d[i][j] = in_ch[r * static_cast<int>(in_w) + c];
                            } else {
                                d[i][j] = 0.0f;
                            }
                        }
                    }

                    float V[4][4];
                    winograd_input_transform(d, V);

                    // Store in element-major layout
                    for (int a = 0; a < 4; a++)
                        for (int b = 0; b < 4; b++)
                            V_flat[(a * 4 + b) * in_channels * num_tiles + ic * num_tiles + tile_idx] = V[a][b];
                }
            }
        }

        // ---- Step 3: Batched matrix multiplications ----
        // For each of 16 elements (alpha, beta):
        //   M[elem] = U[elem] @ V[elem]  where
        //   U[elem] is [OC x IC], V[elem] is [IC x num_tiles]
        //   => M[elem] is [OC x num_tiles]
        float* M_flat = get_buf(tl_wino_M, 16 * out_channels * num_tiles);

        for (int elem = 0; elem < 16; elem++) {
            const float* U_elem = U_flat + elem * out_channels * in_channels;
            const float* V_elem = V_flat + elem * in_channels * num_tiles;
            float* M_elem = M_flat + elem * out_channels * num_tiles;

            matmul_blocked(M_elem, U_elem, V_elem,
                           out_channels, in_channels, num_tiles);
        }

        // ---- Step 4: Inverse-transform and scatter to output ----
        // Zero the output first (needed for edge-tile handling)
        std::memset(out_batch, 0, out_channels * out_h * out_w * sizeof(float));

        for (size_t oc = 0; oc < out_channels; oc++) {
            for (size_t th = 0; th < tile_h; th++) {
                for (size_t tw = 0; tw < tile_w; tw++) {
                    size_t tile_idx = th * tile_w + tw;

                    // Gather the 4x4 element-wise product result for this (oc, tile)
                    float m[4][4];
                    for (int a = 0; a < 4; a++)
                        for (int b = 0; b < 4; b++)
                            m[a][b] = M_flat[(a * 4 + b) * out_channels * num_tiles + oc * num_tiles + tile_idx];

                    // Inverse transform to get 2x2 output tile
                    float Y[2][2];
                    winograd_output_transform(m, Y);

                    // Add bias if present
                    float b_val = bias_data ? bias_data[oc] : 0.0f;

                    // Scatter to output, handling edge cases where out_h/out_w is odd
                    size_t oh_base = th * 2;
                    size_t ow_base = tw * 2;
                    float* out_ch = out_batch + oc * out_h * out_w;

                    for (int i = 0; i < 2; i++) {
                        for (int j = 0; j < 2; j++) {
                            size_t oh = oh_base + static_cast<size_t>(i);
                            size_t ow = ow_base + static_cast<size_t>(j);
                            if (oh < out_h && ow < out_w) {
                                out_ch[oh * out_w + ow] = Y[i][j] + b_val;
                            }
                        }
                    }
                }
            }
        }
    }
}

TensorPtr Tensor::conv2d(const TensorPtr& weight, const TensorPtr& bias,
                          size_t stride, size_t padding, size_t groups, size_t dilation) const {
#if defined(WHITEMATTER_CUDA)
    if (device == whitematter::DeviceType::CUDA && weight->device == whitematter::DeviceType::CUDA &&
        whitematter::cuda_backend_available()) {
        auto r = cuda_ops::conv2d(this, weight, bias, stride, padding, groups, dilation);
        if (r) return r;
        // nullptr means unsupported config (e.g. dilation > 1); fall through to CPU
    }
#endif

    assert(shape.size() == 4);
    assert(weight->shape.size() == 4);
    assert(groups >= 1);
    assert(dilation >= 1);

    size_t batch = shape[0];
    size_t in_channels = shape[1];
    size_t in_h = shape[2];
    size_t in_w = shape[3];

    size_t out_channels = weight->shape[0];
    size_t kernel_h = weight->shape[2];
    size_t kernel_w = weight->shape[3];

    assert(in_channels % groups == 0);
    assert(out_channels % groups == 0);
    // weight->shape[1] should be in_channels / groups
    assert(weight->shape[1] == in_channels / groups);

    size_t out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    size_t out_w = (in_w + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    bool track = (requires_grad || weight->requires_grad || (bias && bias->requires_grad))
                 && GradMode::is_enabled();
    auto result = create({batch, out_channels, out_h, out_w}, track);

    size_t in_ch_per_group = in_channels / groups;
    size_t out_ch_per_group = out_channels / groups;

    // Winograd F(2x2, 3x3) fast path: kernel=3, stride=1, padding=1, groups=1, dilation=1 only
    if (groups == 1 && dilation == 1 && kernel_h == 3 && kernel_w == 3 && stride == 1 && padding == 1) {
        conv2d_winograd_f2x2_3x3(result->data(), data(), weight->data(),
                                   bias ? bias->data() : nullptr,
                                   batch, in_channels, out_channels,
                                   in_h, in_w, out_h, out_w);
    } else {
        // im2col + GEMM, per group
        size_t col_h_per_group = in_ch_per_group * kernel_h * kernel_w;
        size_t col_w = out_h * out_w;
        float* col_buffer = get_buf(tl_col_buf, col_h_per_group * col_w);

        for (size_t b = 0; b < batch; b++) {
            for (size_t g = 0; g < groups; g++) {
                const float* input_ptr = data() + b * in_channels * in_h * in_w
                                        + g * in_ch_per_group * in_h * in_w;
                float* output_ptr = result->data() + b * out_channels * out_h * out_w
                                   + g * out_ch_per_group * out_h * out_w;
                const float* weight_ptr_g = weight->data()
                                           + g * out_ch_per_group * in_ch_per_group * kernel_h * kernel_w;

                im2col(input_ptr, col_buffer,
                       in_ch_per_group, in_h, in_w,
                       kernel_h, kernel_w,
                       out_h, out_w, stride, padding, dilation);

                matmul_blocked(output_ptr, weight_ptr_g, col_buffer,
                               out_ch_per_group, col_h_per_group, col_w);

                if (bias) {
                    for (size_t oc = 0; oc < out_ch_per_group; oc++) {
                        size_t bias_idx = g * out_ch_per_group + oc;
                        for (size_t i = 0; i < col_w; i++) {
                            output_ptr[oc * col_w + i] += bias->data()[bias_idx];
                        }
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
                           kernel_h, kernel_w, stride, padding, groups, dilation]() {

            size_t in_ch_per_group = in_channels / groups;
            size_t out_ch_per_group = out_channels / groups;
            size_t col_h_per_group = in_ch_per_group * kernel_h * kernel_w;
            size_t col_w = out_h * out_w;

            if (self_ptr->requires_grad) {
                float* weight_T = get_buf(tl_scratch, col_h_per_group * out_ch_per_group);
                float* col_grad = get_buf(tl_col_buf, col_h_per_group * col_w);

                for (size_t b = 0; b < batch; b++) {
                    for (size_t g = 0; g < groups; g++) {
                        const float* weight_g = weight_ptr->data()
                                               + g * out_ch_per_group * col_h_per_group;
                        // Transpose weight for this group: [out_ch_per_group, col_h_per_group] -> [col_h_per_group, out_ch_per_group]
                        for (size_t oc = 0; oc < out_ch_per_group; oc++) {
                            for (size_t k = 0; k < col_h_per_group; k++) {
                                weight_T[k * out_ch_per_group + oc] = weight_g[oc * col_h_per_group + k];
                            }
                        }

                        const float* grad_out = result->grad()
                                               + b * out_channels * col_w
                                               + g * out_ch_per_group * col_w;
                        float* grad_in = self_ptr->grad()
                                        + b * in_channels * in_h * in_w
                                        + g * in_ch_per_group * in_h * in_w;

                        matmul_blocked(col_grad, weight_T, grad_out,
                                       col_h_per_group, out_ch_per_group, col_w);

                        col2im(col_grad, grad_in,
                               in_ch_per_group, in_h, in_w,
                               kernel_h, kernel_w,
                               out_h, out_w, stride, padding, dilation);
                    }
                }
            }

            if (weight_ptr->requires_grad) {
                float* col_buf = get_buf(tl_col_buf, col_h_per_group * col_w);
                float* col_T = get_buf(tl_col_buf2, col_w * col_h_per_group);

                for (size_t b = 0; b < batch; b++) {
                    for (size_t g = 0; g < groups; g++) {
                        const float* input_ptr = self_ptr->data()
                                                + b * in_channels * in_h * in_w
                                                + g * in_ch_per_group * in_h * in_w;
                        const float* grad_out = result->grad()
                                               + b * out_channels * col_w
                                               + g * out_ch_per_group * col_w;

                        im2col(input_ptr, col_buf,
                               in_ch_per_group, in_h, in_w,
                               kernel_h, kernel_w,
                               out_h, out_w, stride, padding, dilation);

                        for (size_t i = 0; i < col_h_per_group; i++) {
                            for (size_t j = 0; j < col_w; j++) {
                                col_T[j * col_h_per_group + i] = col_buf[i * col_w + j];
                            }
                        }

                        float* grad_w_g = get_buf(tl_scratch, out_ch_per_group * col_h_per_group);
                        matmul_blocked(grad_w_g, grad_out, col_T,
                                       out_ch_per_group, col_w, col_h_per_group);

                        float* weight_grad_g = weight_ptr->grad()
                                              + g * out_ch_per_group * col_h_per_group;
                        for (size_t i = 0; i < out_ch_per_group * col_h_per_group; i++) {
                            weight_grad_g[i] += grad_w_g[i];
                        }
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

// ============================================================================
// Conv1d: 1D convolution using im2col_1d + GEMM
// ============================================================================

// Helper: unfold 1D input into column matrix
// input: [in_channels, length]
// col: [in_channels * kernel_size, out_length]
static void im2col_1d(const float* input, float* col,
                       size_t in_channels, size_t length,
                       size_t kernel_size, size_t out_length,
                       size_t stride, size_t padding) {
    size_t col_rows = in_channels * kernel_size;
    for (size_t c = 0; c < in_channels; c++) {
        for (size_t k = 0; k < kernel_size; k++) {
            size_t row = c * kernel_size + k;
            for (size_t ol = 0; ol < out_length; ol++) {
                int il = static_cast<int>(ol * stride + k) - static_cast<int>(padding);
                if (il >= 0 && il < static_cast<int>(length)) {
                    col[row * out_length + ol] = input[c * length + static_cast<size_t>(il)];
                } else {
                    col[row * out_length + ol] = 0.0f;
                }
            }
        }
    }
    (void)col_rows;
}

// Helper: reverse of im2col_1d, accumulates into input grad
static void col2im_1d(const float* col, float* input,
                       size_t in_channels, size_t length,
                       size_t kernel_size, size_t out_length,
                       size_t stride, size_t padding) {
    for (size_t c = 0; c < in_channels; c++) {
        for (size_t k = 0; k < kernel_size; k++) {
            size_t row = c * kernel_size + k;
            for (size_t ol = 0; ol < out_length; ol++) {
                int il = static_cast<int>(ol * stride + k) - static_cast<int>(padding);
                if (il >= 0 && il < static_cast<int>(length)) {
                    input[c * length + static_cast<size_t>(il)] += col[row * out_length + ol];
                }
            }
        }
    }
}

TensorPtr Tensor::conv1d(const TensorPtr& weight, const TensorPtr& bias,
                          size_t stride, size_t padding) const {
    assert(shape.size() == 3);      // [batch, in_channels, length]
    assert(weight->shape.size() == 3); // [out_channels, in_channels, kernel_size]
    assert(shape[1] == weight->shape[1]);

    size_t batch = shape[0];
    size_t in_channels = shape[1];
    size_t length = shape[2];

    size_t out_channels = weight->shape[0];
    size_t kernel_size = weight->shape[2];

    size_t out_length = (length + 2 * padding - kernel_size) / stride + 1;

    bool track = (requires_grad || weight->requires_grad || (bias && bias->requires_grad))
                 && GradMode::is_enabled();
    auto result = create({batch, out_channels, out_length}, track);

    size_t col_rows = in_channels * kernel_size;
    float* col_buffer = get_buf(tl_col_buf, col_rows * out_length);

    for (size_t b = 0; b < batch; b++) {
        const float* input_ptr = data() + b * in_channels * length;
        float* output_ptr = result->data() + b * out_channels * out_length;

        im2col_1d(input_ptr, col_buffer,
                  in_channels, length,
                  kernel_size, out_length,
                  stride, padding);

        // weight: [out_channels, in_channels * kernel_size]
        // col: [in_channels * kernel_size, out_length]
        // output: [out_channels, out_length]
        matmul_blocked(output_ptr, weight->data(), col_buffer,
                       out_channels, col_rows, out_length);

        if (bias) {
            for (size_t oc = 0; oc < out_channels; oc++) {
                for (size_t i = 0; i < out_length; i++) {
                    output_ptr[oc * out_length + i] += bias->data()[oc];
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
                           batch, in_channels, length,
                           out_channels, out_length,
                           kernel_size, stride, padding]() {

            size_t col_rows = in_channels * kernel_size;

            // Gradient w.r.t. input
            if (self_ptr->requires_grad) {
                float* weight_T = get_buf(tl_scratch, col_rows * out_channels);
                float* col_grad = get_buf(tl_col_buf, col_rows * out_length);

                // Transpose weight: [out_channels, col_rows] -> [col_rows, out_channels]
                for (size_t oc = 0; oc < out_channels; oc++) {
                    for (size_t k = 0; k < col_rows; k++) {
                        weight_T[k * out_channels + oc] = weight_ptr->data()[oc * col_rows + k];
                    }
                }

                for (size_t b = 0; b < batch; b++) {
                    const float* grad_out = result->grad() + b * out_channels * out_length;
                    float* grad_in = self_ptr->grad() + b * in_channels * length;

                    matmul_blocked(col_grad, weight_T, grad_out,
                                   col_rows, out_channels, out_length);

                    col2im_1d(col_grad, grad_in,
                              in_channels, length,
                              kernel_size, out_length,
                              stride, padding);
                }
            }

            // Gradient w.r.t. weight
            if (weight_ptr->requires_grad) {
                float* col_buf = get_buf(tl_col_buf, col_rows * out_length);
                float* col_T = get_buf(tl_col_buf2, out_length * col_rows);

                for (size_t b = 0; b < batch; b++) {
                    const float* input_ptr = self_ptr->data() + b * in_channels * length;
                    const float* grad_out = result->grad() + b * out_channels * out_length;

                    im2col_1d(input_ptr, col_buf,
                              in_channels, length,
                              kernel_size, out_length,
                              stride, padding);

                    // Transpose col: [col_rows, out_length] -> [out_length, col_rows]
                    for (size_t i = 0; i < col_rows; i++) {
                        for (size_t j = 0; j < out_length; j++) {
                            col_T[j * col_rows + i] = col_buf[i * out_length + j];
                        }
                    }

                    float* grad_w = get_buf(tl_scratch, out_channels * col_rows);
                    matmul_blocked(grad_w, grad_out, col_T,
                                   out_channels, out_length, col_rows);

                    for (size_t i = 0; i < out_channels * col_rows; i++) {
                        weight_ptr->grad()[i] += grad_w[i];
                    }
                }
            }

            // Gradient w.r.t. bias
            if (bias_ptr && bias_ptr->requires_grad) {
                for (size_t oc = 0; oc < out_channels; oc++) {
                    float grad_sum = 0.0f;
                    for (size_t b = 0; b < batch; b++) {
                        for (size_t i = 0; i < out_length; i++) {
                            grad_sum += result->grad()[b * out_channels * out_length + oc * out_length + i];
                        }
                    }
                    bias_ptr->grad()[oc] += grad_sum;
                }
            }
        };
    }

    return result;
}

// ============================================================================
// Upsample: nearest and bilinear upsampling
// ============================================================================

TensorPtr Tensor::upsample(size_t scale_factor, const std::string& mode) const {
    assert(shape.size() == 4);  // [batch, channels, H, W]
    assert(mode == "nearest" || mode == "bilinear");

    size_t batch = shape[0];
    size_t channels = shape[1];
    size_t in_h = shape[2];
    size_t in_w = shape[3];
    size_t out_h = in_h * scale_factor;
    size_t out_w = in_w * scale_factor;

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create({batch, channels, out_h, out_w}, track);

    if (mode == "nearest") {
        for (size_t b = 0; b < batch; b++) {
            for (size_t c = 0; c < channels; c++) {
                for (size_t oh = 0; oh < out_h; oh++) {
                    for (size_t ow = 0; ow < out_w; ow++) {
                        size_t ih = oh / scale_factor;
                        size_t iw = ow / scale_factor;
                        size_t out_idx = b * channels * out_h * out_w +
                                         c * out_h * out_w +
                                         oh * out_w + ow;
                        size_t in_idx = b * channels * in_h * in_w +
                                        c * in_h * in_w +
                                        ih * in_w + iw;
                        result->data()[out_idx] = data()[in_idx];
                    }
                }
            }
        }
    } else {
        // Bilinear interpolation
        for (size_t b = 0; b < batch; b++) {
            for (size_t c = 0; c < channels; c++) {
                for (size_t oh = 0; oh < out_h; oh++) {
                    for (size_t ow = 0; ow < out_w; ow++) {
                        // Map output to input coordinates
                        float src_h = (static_cast<float>(oh) + 0.5f) / static_cast<float>(scale_factor) - 0.5f;
                        float src_w = (static_cast<float>(ow) + 0.5f) / static_cast<float>(scale_factor) - 0.5f;

                        int ih0 = static_cast<int>(std::floor(src_h));
                        int iw0 = static_cast<int>(std::floor(src_w));
                        int ih1 = ih0 + 1;
                        int iw1 = iw0 + 1;

                        float h_lerp = src_h - static_cast<float>(ih0);
                        float w_lerp = src_w - static_cast<float>(iw0);

                        // Clamp to valid range
                        ih0 = std::max(0, std::min(ih0, static_cast<int>(in_h) - 1));
                        ih1 = std::max(0, std::min(ih1, static_cast<int>(in_h) - 1));
                        iw0 = std::max(0, std::min(iw0, static_cast<int>(in_w) - 1));
                        iw1 = std::max(0, std::min(iw1, static_cast<int>(in_w) - 1));

                        size_t base = b * channels * in_h * in_w + c * in_h * in_w;
                        float v00 = data()[base + static_cast<size_t>(ih0) * in_w + static_cast<size_t>(iw0)];
                        float v01 = data()[base + static_cast<size_t>(ih0) * in_w + static_cast<size_t>(iw1)];
                        float v10 = data()[base + static_cast<size_t>(ih1) * in_w + static_cast<size_t>(iw0)];
                        float v11 = data()[base + static_cast<size_t>(ih1) * in_w + static_cast<size_t>(iw1)];

                        float val = (1.0f - h_lerp) * (1.0f - w_lerp) * v00 +
                                    (1.0f - h_lerp) * w_lerp * v01 +
                                    h_lerp * (1.0f - w_lerp) * v10 +
                                    h_lerp * w_lerp * v11;

                        size_t out_idx = b * channels * out_h * out_w +
                                         c * out_h * out_w +
                                         oh * out_w + ow;
                        result->data()[out_idx] = val;
                    }
                }
            }
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        auto mode_copy = mode;
        result->parents = {self_ptr};

        result->grad_fn = [self_ptr, result, batch, channels,
                           in_h, in_w, out_h, out_w, scale_factor, mode_copy]() {
            if (!self_ptr->requires_grad) return;

            if (mode_copy == "nearest") {
                // Accumulate gradients to source positions
                for (size_t b = 0; b < batch; b++) {
                    for (size_t c = 0; c < channels; c++) {
                        for (size_t oh = 0; oh < out_h; oh++) {
                            for (size_t ow = 0; ow < out_w; ow++) {
                                size_t ih = oh / scale_factor;
                                size_t iw = ow / scale_factor;
                                size_t out_idx = b * channels * out_h * out_w +
                                                 c * out_h * out_w +
                                                 oh * out_w + ow;
                                size_t in_idx = b * channels * in_h * in_w +
                                                c * in_h * in_w +
                                                ih * in_w + iw;
                                self_ptr->grad()[in_idx] += result->grad()[out_idx];
                            }
                        }
                    }
                }
            } else {
                // Bilinear: distribute gradients weighted by interpolation coefficients
                for (size_t b = 0; b < batch; b++) {
                    for (size_t c = 0; c < channels; c++) {
                        for (size_t oh = 0; oh < out_h; oh++) {
                            for (size_t ow = 0; ow < out_w; ow++) {
                                float src_h = (static_cast<float>(oh) + 0.5f) / static_cast<float>(scale_factor) - 0.5f;
                                float src_w = (static_cast<float>(ow) + 0.5f) / static_cast<float>(scale_factor) - 0.5f;

                                int ih0 = static_cast<int>(std::floor(src_h));
                                int iw0 = static_cast<int>(std::floor(src_w));
                                int ih1 = ih0 + 1;
                                int iw1 = iw0 + 1;

                                float h_lerp = src_h - static_cast<float>(ih0);
                                float w_lerp = src_w - static_cast<float>(iw0);

                                ih0 = std::max(0, std::min(ih0, static_cast<int>(in_h) - 1));
                                ih1 = std::max(0, std::min(ih1, static_cast<int>(in_h) - 1));
                                iw0 = std::max(0, std::min(iw0, static_cast<int>(in_w) - 1));
                                iw1 = std::max(0, std::min(iw1, static_cast<int>(in_w) - 1));

                                size_t out_idx = b * channels * out_h * out_w +
                                                 c * out_h * out_w +
                                                 oh * out_w + ow;
                                float grad_val = result->grad()[out_idx];

                                size_t base = b * channels * in_h * in_w + c * in_h * in_w;
                                self_ptr->grad()[base + static_cast<size_t>(ih0) * in_w + static_cast<size_t>(iw0)] += (1.0f - h_lerp) * (1.0f - w_lerp) * grad_val;
                                self_ptr->grad()[base + static_cast<size_t>(ih0) * in_w + static_cast<size_t>(iw1)] += (1.0f - h_lerp) * w_lerp * grad_val;
                                self_ptr->grad()[base + static_cast<size_t>(ih1) * in_w + static_cast<size_t>(iw0)] += h_lerp * (1.0f - w_lerp) * grad_val;
                                self_ptr->grad()[base + static_cast<size_t>(ih1) * in_w + static_cast<size_t>(iw1)] += h_lerp * w_lerp * grad_val;
                            }
                        }
                    }
                }
            }
        };
    }

    return result;
}

// ============================================================================
// AdaptiveAvgPool2d: pool to fixed output size
// ============================================================================

TensorPtr Tensor::adaptive_avgpool2d(size_t output_h, size_t output_w) const {
    assert(shape.size() == 4);  // [batch, channels, H, W]

    // CPU fallback for CUDA tensors
    if (device != whitematter::DeviceType::CPU) {
        auto cpu_self = to(whitematter::DeviceType::CPU);
        auto cpu_result = cpu_self->adaptive_avgpool2d(output_h, output_w);
        cpu_result->to_inplace(device);
        return cpu_result;
    }

    size_t batch = shape[0];
    size_t channels = shape[1];
    size_t in_h = shape[2];
    size_t in_w = shape[3];

    bool track = requires_grad && GradMode::is_enabled();
    auto result = create({batch, channels, output_h, output_w}, track);

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t oh = 0; oh < output_h; oh++) {
                for (size_t ow = 0; ow < output_w; ow++) {
                    // Compute input range for this output position
                    size_t ih_start = (oh * in_h) / output_h;
                    size_t ih_end = ((oh + 1) * in_h) / output_h;
                    size_t iw_start = (ow * in_w) / output_w;
                    size_t iw_end = ((ow + 1) * in_w) / output_w;

                    float sum = 0.0f;
                    size_t count = (ih_end - ih_start) * (iw_end - iw_start);

                    for (size_t ih = ih_start; ih < ih_end; ih++) {
                        for (size_t iw = iw_start; iw < iw_end; iw++) {
                            size_t in_idx = b * channels * in_h * in_w +
                                            c * in_h * in_w +
                                            ih * in_w + iw;
                            sum += data()[in_idx];
                        }
                    }

                    size_t out_idx = b * channels * output_h * output_w +
                                     c * output_h * output_w +
                                     oh * output_w + ow;
                    result->data()[out_idx] = sum / static_cast<float>(count);
                }
            }
        }
    }

    if (track) {
        auto self_ptr = const_cast<Tensor*>(this)->shared_from_this();
        result->parents = {self_ptr};

        result->grad_fn = [self_ptr, result, batch, channels,
                           in_h, in_w, output_h, output_w]() {
            if (!self_ptr->requires_grad) return;

            for (size_t b = 0; b < batch; b++) {
                for (size_t c = 0; c < channels; c++) {
                    for (size_t oh = 0; oh < output_h; oh++) {
                        for (size_t ow = 0; ow < output_w; ow++) {
                            size_t ih_start = (oh * in_h) / output_h;
                            size_t ih_end = ((oh + 1) * in_h) / output_h;
                            size_t iw_start = (ow * in_w) / output_w;
                            size_t iw_end = ((ow + 1) * in_w) / output_w;

                            size_t count = (ih_end - ih_start) * (iw_end - iw_start);
                            size_t out_idx = b * channels * output_h * output_w +
                                             c * output_h * output_w +
                                             oh * output_w + ow;
                            float grad_val = result->grad()[out_idx] / static_cast<float>(count);

                            for (size_t ih = ih_start; ih < ih_end; ih++) {
                                for (size_t iw = iw_start; iw < iw_end; iw++) {
                                    size_t in_idx = b * channels * in_h * in_w +
                                                    c * in_h * in_w +
                                                    ih * in_w + iw;
                                    self_ptr->grad()[in_idx] += grad_val;
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
