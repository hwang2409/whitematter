#include "simd_ops.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

void simd_add(float* dst, const float* a, const float* b, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(dst + i, vaddq_f32(va, vb));
    }
    for (; i < n; i++) dst[i] = a[i] + b[i];
}

void simd_sub(float* dst, const float* a, const float* b, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(dst + i, vsubq_f32(va, vb));
    }
    for (; i < n; i++) dst[i] = a[i] - b[i];
}

void simd_mul(float* dst, const float* a, const float* b, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(dst + i, vmulq_f32(va, vb));
    }
    for (; i < n; i++) dst[i] = a[i] * b[i];
}

void simd_scale(float* dst, const float* a, float scalar, size_t n) {
    size_t i = 0;
    float32x4_t vs = vdupq_n_f32(scalar);
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vst1q_f32(dst + i, vmulq_f32(va, vs));
    }
    for (; i < n; i++) dst[i] = a[i] * scalar;
}

void simd_relu(float* dst, const float* a, size_t n) {
    size_t i = 0;
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vst1q_f32(dst + i, vmaxq_f32(va, zero));
    }
    for (; i < n; i++) dst[i] = a[i] > 0 ? a[i] : 0;
}

float simd_dot(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    size_t i = 0;
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vsum = vmlaq_f32(vsum, va, vb);
    }
    float temp[4];
    vst1q_f32(temp, vsum);
    sum = temp[0] + temp[1] + temp[2] + temp[3];
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

#endif // ARM NEON
