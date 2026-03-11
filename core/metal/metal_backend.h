#ifndef METAL_BACKEND_H
#define METAL_BACKEND_H

#include <cstddef>

namespace whitematter {

// Metal backend singleton. Only built when METAL=1 and __APPLE__.
// matmul: C = A @ B, where A is [M,K], B is [K,N], C is [M,N].
class MetalBackend {
public:
    static MetalBackend& instance();

    bool is_available() const;
    void matmul(const float* A, const float* B, float* C, int M, int N, int K);

private:
    MetalBackend() = default;
    bool available_ = false;
    void init();
};

}  // namespace whitematter

#endif
