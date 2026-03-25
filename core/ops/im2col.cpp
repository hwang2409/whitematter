#include "im2col.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void im2col(const float* input, float* col,
            size_t in_channels, size_t in_h, size_t in_w,
            size_t kernel_h, size_t kernel_w,
            size_t out_h, size_t out_w,
            size_t stride, size_t padding) {
    size_t col_w = out_h * out_w;
    size_t kernel_size = kernel_h * kernel_w;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t c = 0; c < in_channels; c++) {
        for (size_t k = 0; k < kernel_size; k++) {
            size_t kh = k / kernel_w;
            size_t kw = k % kernel_w;
            size_t row = c * kernel_size + k;
            for (size_t oh = 0; oh < out_h; oh++) {
                int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
                for (size_t ow = 0; ow < out_w; ow++) {
                    int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);

                    size_t col_idx = row * col_w + oh * out_w + ow;
                    if (ih >= 0 && ih < static_cast<int>(in_h) &&
                        iw >= 0 && iw < static_cast<int>(in_w)) {
                        col[col_idx] = input[c * in_h * in_w + ih * in_w + iw];
                    } else {
                        col[col_idx] = 0.0f;
                    }
                }
            }
        }
    }
}

void col2im(const float* col, float* input,
            size_t in_channels, size_t in_h, size_t in_w,
            size_t kernel_h, size_t kernel_w,
            size_t out_h, size_t out_w,
            size_t stride, size_t padding) {
    size_t col_w = out_h * out_w;
    size_t kernel_size = kernel_h * kernel_w;

    #pragma omp parallel for schedule(static)
    for (size_t c = 0; c < in_channels; c++) {
        for (size_t k = 0; k < kernel_size; k++) {
            size_t kh = k / kernel_w;
            size_t kw = k % kernel_w;
            size_t row = c * kernel_size + k;
            for (size_t oh = 0; oh < out_h; oh++) {
                int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
                if (ih < 0 || ih >= static_cast<int>(in_h)) continue;
                for (size_t ow = 0; ow < out_w; ow++) {
                    int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);

                    if (iw >= 0 && iw < static_cast<int>(in_w)) {
                        size_t col_idx = row * col_w + oh * out_w + ow;
                        input[c * in_h * in_w + ih * in_w + iw] += col[col_idx];
                    }
                }
            }
        }
    }
}
