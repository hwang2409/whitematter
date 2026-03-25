#ifndef IM2COL_H
#define IM2COL_H

#include <cstddef>

// im2col: Transform input patches into columns for matrix multiplication.
// Input layout: [in_channels, in_h, in_w] (single image)
// Output layout: [in_channels * kernel_h * kernel_w, out_h * out_w]
void im2col(const float* input, float* col,
            size_t in_channels, size_t in_h, size_t in_w,
            size_t kernel_h, size_t kernel_w,
            size_t out_h, size_t out_w,
            size_t stride, size_t padding);

// col2im: Transform columns back to image (accumulating gradients).
// Input layout: [in_channels * kernel_h * kernel_w, out_h * out_w]
// Output layout: [in_channels, in_h, in_w] (accumulated)
void col2im(const float* col, float* input,
            size_t in_channels, size_t in_h, size_t in_w,
            size_t kernel_h, size_t kernel_w,
            size_t out_h, size_t out_w,
            size_t stride, size_t padding);

#endif
