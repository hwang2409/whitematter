#ifndef SIMD_OPS_H
#define SIMD_OPS_H

#include <cstddef>

// Architecture-agnostic SIMD function declarations.
// Implementations live in simd_ops_avx.cpp, simd_ops_neon.cpp, or
// simd_ops_fallback.cpp — exactly one provides symbols per build.

void simd_add(float* dst, const float* a, const float* b, size_t n);
void simd_sub(float* dst, const float* a, const float* b, size_t n);
void simd_mul(float* dst, const float* a, const float* b, size_t n);
void simd_scale(float* dst, const float* a, float scalar, size_t n);
void simd_relu(float* dst, const float* a, size_t n);
float simd_dot(const float* a, const float* b, size_t n);

#endif
