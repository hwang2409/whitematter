#include "matmul_cpu.h"
#include "simd_ops.h"
#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD / NEON detection (for inner kernel)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
    #define USE_SIMD 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define USE_NEON 1
#endif

static constexpr size_t BLOCK_SIZE = 32;

void matmul_blocked(float* C, const float* A, const float* B,
                    size_t M, size_t K, size_t N) {
    std::memset(C, 0, M * N * sizeof(float));

    // Parallelize over row blocks - each thread handles different output rows
    #pragma omp parallel for schedule(static)
    for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        size_t imax = std::min(i0 + BLOCK_SIZE, M);
        for (size_t k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
            size_t kmax = std::min(k0 + BLOCK_SIZE, K);
            for (size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
                size_t jmax = std::min(j0 + BLOCK_SIZE, N);

                for (size_t i = i0; i < imax; i++) {
                    for (size_t k = k0; k < kmax; k++) {
                        float a_ik = A[i * K + k];
                        #if defined(USE_SIMD) && defined(__AVX__)
                        __m256 va = _mm256_set1_ps(a_ik);
                        size_t j = j0;
                        for (; j + 8 <= jmax; j += 8) {
                            __m256 vb = _mm256_loadu_ps(&B[k * N + j]);
                            __m256 vc = _mm256_loadu_ps(&C[i * N + j]);
                            vc = _mm256_add_ps(vc, _mm256_mul_ps(va, vb));
                            _mm256_storeu_ps(&C[i * N + j], vc);
                        }
                        for (; j < jmax; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                        #elif defined(USE_NEON)
                        float32x4_t va = vdupq_n_f32(a_ik);
                        size_t j = j0;
                        for (; j + 4 <= jmax; j += 4) {
                            float32x4_t vb = vld1q_f32(&B[k * N + j]);
                            float32x4_t vc = vld1q_f32(&C[i * N + j]);
                            vc = vmlaq_f32(vc, va, vb);
                            vst1q_f32(&C[i * N + j], vc);
                        }
                        for (; j < jmax; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                        #else
                        for (size_t j = j0; j < jmax; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                        #endif
                    }
                }
            }
        }
    }
}
