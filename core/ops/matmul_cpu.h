#ifndef MATMUL_CPU_H
#define MATMUL_CPU_H

#include <cstddef>

void matmul_blocked(float* C, const float* A, const float* B,
                    size_t M, size_t K, size_t N);

#endif
