#ifndef METAL_BACKEND_H
#define METAL_BACKEND_H

#include <cstddef>

namespace whitematter {

// Metal backend singleton. Only built when METAL=1 and __APPLE__.
// Persistent GPU resources are created once and reused across calls.
class MetalBackend {
public:
    static MetalBackend& instance();
    bool is_available() const;

    // Matrix operations
    // C = A @ B, where A is [M,K], B is [K,N], C is [M,N].
    void matmul(const float* A, const float* B, float* C, int M, int N, int K);

    // Element-wise binary operations
    void elementwise_add(const float* a, const float* b, float* c, size_t n);
    void elementwise_sub(const float* a, const float* b, float* c, size_t n);
    void elementwise_mul(const float* a, const float* b, float* c, size_t n);
    void elementwise_div(const float* a, const float* b, float* c, size_t n);

    // Activations
    void relu(const float* x, float* y, size_t n);
    void sigmoid(const float* x, float* y, size_t n);
    void tanh_op(const float* x, float* y, size_t n);

private:
    MetalBackend() = default;
    [[maybe_unused]] bool available_ = false;
    [[maybe_unused]] bool initialized_ = false;
    void init();

    // Persistent Metal resources (opaque pointers to avoid ObjC in header)
    [[maybe_unused]] void* device_ = nullptr;      // id<MTLDevice>
    [[maybe_unused]] void* queue_ = nullptr;       // id<MTLCommandQueue>
    [[maybe_unused]] void* library_ = nullptr;     // id<MTLLibrary>

    // Cached pipeline states (up to 16 kernels)
    struct PipelineEntry {
        const char* name = nullptr;
        void* pipeline = nullptr;  // id<MTLComputePipelineState>
    };
    PipelineEntry pipelines_[16] = {};
    [[maybe_unused]] int pipeline_count_ = 0;

    void* get_pipeline(const char* fn_name);
    void run_elementwise(const char* kernel_name, const float* a, const float* b, float* c, size_t n);
    void run_unary(const char* kernel_name, const float* x, float* y, size_t n);
};

}  // namespace whitematter

#endif
