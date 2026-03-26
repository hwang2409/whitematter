#ifndef FP16_H
#define FP16_H

#include <cstdint>
#include <cstddef>

// Convert between IEEE 754 half-precision (fp16) and single-precision (fp32).
// SIMD-accelerated on ARM NEON and x86 F16C.
void float_to_half(const float* src, uint16_t* dst, size_t n);
void half_to_float(const uint16_t* src, float* dst, size_t n);

// Scalar conversions (for single values)
uint16_t float_to_half_scalar(float val);
float half_to_float_scalar(uint16_t val);

#endif
