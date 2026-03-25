#include "simd_ops.h"

// Scalar fallback — compiled on all platforms but only provides symbols when
// neither x86 SIMD nor ARM NEON is available.
#if !defined(__x86_64__) && !defined(_M_X64) && !defined(__i386__) && !defined(_M_IX86) \
    && !defined(__ARM_NEON) && !defined(__ARM_NEON__)

void simd_add(float* dst, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = a[i] + b[i];
}
void simd_sub(float* dst, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = a[i] - b[i];
}
void simd_mul(float* dst, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = a[i] * b[i];
}
void simd_scale(float* dst, const float* a, float scalar, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = a[i] * scalar;
}
void simd_relu(float* dst, const float* a, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = a[i] > 0 ? a[i] : 0;
}
float simd_dot(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

#endif // fallback
