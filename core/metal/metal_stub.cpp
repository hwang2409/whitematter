// Stub when Metal is not built (Linux/Windows or METAL=0).
#include "metal_backend.h"
#include "../device.h"

namespace whitematter {

MetalBackend& MetalBackend::instance() {
    static MetalBackend inst;
    return inst;
}

bool MetalBackend::is_available() const { return false; }

void MetalBackend::init() {}

void* MetalBackend::get_pipeline(const char*) { return nullptr; }

void MetalBackend::run_elementwise(const char*, const float*, const float*, float*, size_t) {}

void MetalBackend::run_unary(const char*, const float*, float*, size_t) {}

void MetalBackend::matmul(const float*, const float*, float*, int, int, int) {}

void MetalBackend::elementwise_add(const float*, const float*, float*, size_t) {}
void MetalBackend::elementwise_sub(const float*, const float*, float*, size_t) {}
void MetalBackend::elementwise_mul(const float*, const float*, float*, size_t) {}
void MetalBackend::elementwise_div(const float*, const float*, float*, size_t) {}

void MetalBackend::relu(const float*, float*, size_t) {}
void MetalBackend::sigmoid(const float*, float*, size_t) {}
void MetalBackend::tanh_op(const float*, float*, size_t) {}

bool metal_backend_available() { return false; }

}  // namespace whitematter
