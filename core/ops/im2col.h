#ifndef IM2COL_H
#define IM2COL_H

#include <cstddef>

void im2col(const float* input, float* col,
            size_t in_channels, size_t in_h, size_t in_w,
            size_t kernel_h, size_t kernel_w,
            size_t out_h, size_t out_w,
            size_t stride, size_t padding);

void col2im(const float* col, float* input,
            size_t in_channels, size_t in_h, size_t in_w,
            size_t kernel_h, size_t kernel_w,
            size_t out_h, size_t out_w,
            size_t stride, size_t padding);

#endif
