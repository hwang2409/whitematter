#ifndef MATMUL_CPU_H
#define MATMUL_CPU_H

#include <cstddef>

// Blocked matrix multiplication with OpenMP parallelization.
// C = A * B  where A is [M,K], B is [K,N], C is [M,N].
// C is zeroed before accumulation.
void matmul_blocked(float* C, const float* A, const float* B,
                    size_t M, size_t K, size_t N);

#endif
