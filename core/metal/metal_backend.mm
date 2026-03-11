#ifdef __APPLE__

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_backend.h"
#include "../device.h"
#include <cstring>
#include <stdexcept>

namespace whitematter {

static const char* kMatmulSource = R"(
#include <metal_stdlib>
using namespace metal;
kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint col = gid.y;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
)";

MetalBackend& MetalBackend::instance() {
    static MetalBackend inst;
    return inst;
}

void MetalBackend::init() {
    if (available_) return;
#if __MAC_OS_X_VERSION_MIN_REQUIRED >= 100000
    if (@available(macOS 10.14, *)) {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {
            NSError* err = nil;
            id<MTLLibrary> lib = [device newLibraryWithSource:@(kMatmulSource) options:nil error:&err];
            if (lib && !err) {
                available_ = true;
            }
        }
    }
#endif
}

bool MetalBackend::is_available() const {
    const_cast<MetalBackend*>(this)->init();
    return available_;
}

void MetalBackend::matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    init();
    if (!available_) return;  // fallback to CPU is handled by caller

#if __MAC_OS_X_VERSION_MIN_REQUIRED >= 100000
    if (@available(macOS 10.14, *)) {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return;
        NSError* err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:@(kMatmulSource) options:nil error:&err];
        if (!lib || err) return;
        id<MTLFunction> fn = [lib newFunctionWithName:@"matmul"];
        if (!fn) return;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pipeline || err) return;

        size_t aBytes = (size_t)M * K * sizeof(float);
        size_t bBytes = (size_t)K * N * sizeof(float);
        size_t cBytes = (size_t)M * N * sizeof(float);

        id<MTLBuffer> bufA = [device newBufferWithLength:aBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithLength:bBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithLength:cBytes options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufC) return;

        memcpy(bufA.contents, A, aBytes);
        memcpy(bufB.contents, B, bBytes);

        uint32_t m = (uint32_t)M, n = (uint32_t)N, k = (uint32_t)K;
        id<MTLBuffer> bufM = [device newBufferWithBytes:&m length:sizeof(m) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufN = [device newBufferWithBytes:&n length:sizeof(n) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [device newBufferWithBytes:&k length:sizeof(k) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufC offset:0 atIndex:2];
        [enc setBuffer:bufM offset:0 atIndex:3];
        [enc setBuffer:bufN offset:0 atIndex:4];
        [enc setBuffer:bufK offset:0 atIndex:5];

        MTLSize threadgroups = MTLSizeMake((M + 15) / 16, (N + 15) / 16, 1);
        MTLSize threadsPerGroup = MTLSizeMake(16, 16, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(C, bufC.contents, cBytes);
    }
#endif
}

}  // namespace whitematter

// Provide metal_backend_available for device.cpp when Metal is built.
namespace whitematter {
bool metal_backend_available() {
    return MetalBackend::instance().is_available();
}
}  // namespace whitematter

#endif  // __APPLE__
