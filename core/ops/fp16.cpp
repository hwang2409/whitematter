#include "fp16.h"
#include <cstring>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NEON_FP16 1
#elif defined(__F16C__) && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
#define USE_F16C 1
#endif

// ---------------------------------------------------------------------------
// Scalar fallback (IEEE 754 bit manipulation)
// ---------------------------------------------------------------------------

uint16_t float_to_half_scalar(float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, 4);

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFF;

    if (exponent <= 0) {
        // Subnormal or zero
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x800000) >> (1 - exponent);
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    } else if (exponent >= 31) {
        // Overflow → infinity (or NaN)
        if (exponent == 31 && mantissa != 0)
            return static_cast<uint16_t>(sign | 0x7C00 | (mantissa >> 13));
        return static_cast<uint16_t>(sign | 0x7C00);
    }

    return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
}

float half_to_float_scalar(uint16_t val) {
    uint32_t sign = (static_cast<uint32_t>(val) & 0x8000) << 16;
    uint32_t exponent = (val >> 10) & 0x1F;
    uint32_t mantissa = val & 0x3FF;

    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            // Subnormal → normalize
            exponent = 1;
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000 | (mantissa << 13);  // Inf or NaN
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

// ---------------------------------------------------------------------------
// Vectorized conversions
// ---------------------------------------------------------------------------

void float_to_half(const float* src, uint16_t* dst, size_t n) {
#if defined(USE_NEON_FP16)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t f32 = vld1q_f32(src + i);
        float16x4_t f16 = vcvt_f16_f32(f32);
        vst1_f16(reinterpret_cast<__fp16*>(dst + i), f16);
    }
    for (; i < n; i++)
        dst[i] = float_to_half_scalar(src[i]);
#elif defined(USE_F16C)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 f32 = _mm256_loadu_ps(src + i);
        __m128i f16 = _mm256_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), f16);
    }
    for (; i < n; i++)
        dst[i] = float_to_half_scalar(src[i]);
#else
    for (size_t i = 0; i < n; i++)
        dst[i] = float_to_half_scalar(src[i]);
#endif
}

void half_to_float(const uint16_t* src, float* dst, size_t n) {
#if defined(USE_NEON_FP16)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float16x4_t f16 = vld1_f16(reinterpret_cast<const __fp16*>(src + i));
        float32x4_t f32 = vcvt_f32_f16(f16);
        vst1q_f32(dst + i, f32);
    }
    for (; i < n; i++)
        dst[i] = half_to_float_scalar(src[i]);
#elif defined(USE_F16C)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i f16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256 f32 = _mm256_cvtph_ps(f16);
        _mm256_storeu_ps(dst + i, f32);
    }
    for (; i < n; i++)
        dst[i] = half_to_float_scalar(src[i]);
#else
    for (size_t i = 0; i < n; i++)
        dst[i] = half_to_float_scalar(src[i]);
#endif
}
