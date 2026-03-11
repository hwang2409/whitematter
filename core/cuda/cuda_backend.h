#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

#include <cstddef>

namespace whitematter {

// CUDA backend singleton. Only built when CUDA=1.
// matmul: C = A @ B, where A is [M,K], B is [K,N], C is [M,N] (row-major).
class CUDABackend {
public:
    static CUDABackend& instance();

    bool is_available() const;
    void matmul(const float* A, const float* B, float* C, int M, int N, int K);
    void bmm(const float* A, const float* B, float* C, int batch, int M, int K, int N);

private:
    CUDABackend() = default;
    bool available_ = false;
    void init();
};

}  // namespace whitematter

#endif
