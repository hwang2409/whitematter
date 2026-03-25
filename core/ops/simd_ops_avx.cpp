#include "simd_ops.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>

void simd_add(float* dst, const float* a, const float* b, size_t n) {
    size_t i = 0;
    #ifdef __AVX__
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(dst + i, vc);
    }
    #endif
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_add_ps(va, vb);
        _mm_storeu_ps(dst + i, vc);
    }
    for (; i < n; i++) {
        dst[i] = a[i] + b[i];
    }
}

void simd_sub(float* dst, const float* a, const float* b, size_t n) {
    size_t i = 0;
    #ifdef __AVX__
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(dst + i, vc);
    }
    #endif
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_sub_ps(va, vb);
        _mm_storeu_ps(dst + i, vc);
    }
    for (; i < n; i++) {
        dst[i] = a[i] - b[i];
    }
}

void simd_mul(float* dst, const float* a, const float* b, size_t n) {
    size_t i = 0;
    #ifdef __AVX__
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(dst + i, vc);
    }
    #endif
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_mul_ps(va, vb);
        _mm_storeu_ps(dst + i, vc);
    }
    for (; i < n; i++) {
        dst[i] = a[i] * b[i];
    }
}

void simd_scale(float* dst, const float* a, float scalar, size_t n) {
    size_t i = 0;
    #ifdef __AVX__
    __m256 vs = _mm256_set1_ps(scalar);
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vc = _mm256_mul_ps(va, vs);
        _mm256_storeu_ps(dst + i, vc);
    }
    #endif
    __m128 vs4 = _mm_set1_ps(scalar);
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vc = _mm_mul_ps(va, vs4);
        _mm_storeu_ps(dst + i, vc);
    }
    for (; i < n; i++) {
        dst[i] = a[i] * scalar;
    }
}

void simd_relu(float* dst, const float* a, size_t n) {
    size_t i = 0;
    #ifdef __AVX__
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vc = _mm256_max_ps(va, zero);
        _mm256_storeu_ps(dst + i, vc);
    }
    #endif
    __m128 zero4 = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vc = _mm_max_ps(va, zero4);
        _mm_storeu_ps(dst + i, vc);
    }
    for (; i < n; i++) {
        dst[i] = a[i] > 0 ? a[i] : 0;
    }
}

float simd_dot(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    size_t i = 0;
    #ifdef __AVX__
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb));
    }
    float temp[8];
    _mm256_storeu_ps(temp, vsum);
    sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    #endif
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

#endif // x86
