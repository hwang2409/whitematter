#ifdef __APPLE__

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_backend.h"
#include "../device.h"
#include <cstring>
#include <mach/vm_page_size.h>

namespace whitematter {

// All shader kernels embedded as a single source string.
static const char* kAllKernelSource = R"(
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

constant uint TILE = 16;

kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    threadgroup float* As [[threadgroup(0)]],
    threadgroup float* Bs [[threadgroup(1)]]
) {
    uint row = gid.x;
    uint col = gid.y;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (uint tile = 0; tile < (K + TILE - 1) / TILE; tile++) {
        uint kBase = tile * TILE;
        for (uint k = 0; k < TILE && (kBase + k) < K; k++) {
            sum += A[row * K + kBase + k] * B[(kBase + k) * N + col];
        }
    }
    C[row * N + col] = sum;
}

kernel void elementwise_add(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    C[gid] = A[gid] + B[gid];
}

kernel void elementwise_mul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    C[gid] = A[gid] * B[gid];
}

kernel void elementwise_sub(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    C[gid] = A[gid] - B[gid];
}

kernel void elementwise_div(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    C[gid] = A[gid] / B[gid];
}

kernel void relu_kernel(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = X[gid];
    Y[gid] = x > 0.0f ? x : 0.0f;
}

kernel void sigmoid_kernel(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    Y[gid] = 1.0f / (1.0f + exp(-X[gid]));
}

kernel void tanh_kernel(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    Y[gid] = tanh(X[gid]);
}
)";

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

MetalBackend& MetalBackend::instance() {
    static MetalBackend inst;
    return inst;
}

// ---------------------------------------------------------------------------
// Lazy one-time init: create device, queue, and compile shader library
// ---------------------------------------------------------------------------

void MetalBackend::init() {
    if (initialized_) return;
    initialized_ = true;

    if (@available(macOS 10.14, *)) {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return;

        NSError* err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:@(kAllKernelSource)
                                                  options:nil
                                                    error:&err];
        if (!lib || err) return;

        // Retain ObjC objects as void* (prevent ARC from releasing them)
        device_  = (void*)CFBridgingRetain(device);
        queue_   = (void*)CFBridgingRetain(queue);
        library_ = (void*)CFBridgingRetain(lib);

        available_ = true;
    }
}

bool MetalBackend::is_available() const {
    const_cast<MetalBackend*>(this)->init();
    return available_;
}

// ---------------------------------------------------------------------------
// Pipeline cache: compile + cache MTLComputePipelineState per kernel name
// ---------------------------------------------------------------------------

void* MetalBackend::get_pipeline(const char* fn_name) {
    // Linear search through cached pipelines
    for (int i = 0; i < pipeline_count_; i++) {
        if (strcmp(pipelines_[i].name, fn_name) == 0) {
            return pipelines_[i].pipeline;
        }
    }

    // Not cached yet -- compile it
    if (pipeline_count_ >= 16) return nullptr;  // cache full

    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)library_;
    id<MTLFunction> fn = [lib newFunctionWithName:@(fn_name)];
    if (!fn) return nullptr;

    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    NSError* err = nil;
    id<MTLComputePipelineState> pipe = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pipe || err) return nullptr;

    void* retained = (void*)CFBridgingRetain(pipe);
    pipelines_[pipeline_count_].name = fn_name;
    pipelines_[pipeline_count_].pipeline = retained;
    pipeline_count_++;
    return retained;
}

// ---------------------------------------------------------------------------
// Helper: page-align size for newBufferWithBytesNoCopy
// ---------------------------------------------------------------------------

static size_t page_align(size_t bytes) {
    size_t ps = vm_page_size;
    return (bytes + ps - 1) & ~(ps - 1);
}

// ---------------------------------------------------------------------------
// run_elementwise: dispatches a binary elementwise kernel (3 buffers)
// ---------------------------------------------------------------------------

void MetalBackend::run_elementwise(const char* kernel_name,
                                   const float* a, const float* b, float* c,
                                   size_t n) {
    if (!available_ || n == 0) return;

    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> pipe =
        (__bridge id<MTLComputePipelineState>)get_pipeline(kernel_name);
    if (!pipe) return;

    size_t bytes = n * sizeof(float);
    size_t aligned = page_align(bytes);

    // Try zero-copy buffers (requires page-aligned address and size)
    id<MTLBuffer> bufA = [dev newBufferWithBytesNoCopy:(void*)a
                                                length:aligned
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
    id<MTLBuffer> bufB = [dev newBufferWithBytesNoCopy:(void*)b
                                                length:aligned
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
    id<MTLBuffer> bufC = [dev newBufferWithBytesNoCopy:(void*)c
                                                length:aligned
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];

    // Fall back to copy if zero-copy fails (alignment issues)
    bool needCopyBack = false;
    if (!bufA) {
        bufA = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        memcpy(bufA.contents, a, bytes);
    }
    if (!bufB) {
        bufB = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        memcpy(bufB.contents, b, bytes);
    }
    if (!bufC) {
        bufC = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        needCopyBack = true;
    }

    if (!bufA || !bufB || !bufC) return;

    id<MTLCommandBuffer> cmdBuf = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufC offset:0 atIndex:2];

    NSUInteger tgSize = MIN(pipe.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
    MTLSize threads = MTLSizeMake(n, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(tgSize, 1, 1);
    [enc dispatchThreads:threads threadsPerThreadgroup:threadgroupSize];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    if (needCopyBack) memcpy(c, bufC.contents, bytes);
}

// ---------------------------------------------------------------------------
// run_unary: dispatches a unary kernel (2 buffers: input + output)
// ---------------------------------------------------------------------------

void MetalBackend::run_unary(const char* kernel_name,
                             const float* x, float* y,
                             size_t n) {
    if (!available_ || n == 0) return;

    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
    id<MTLComputePipelineState> pipe =
        (__bridge id<MTLComputePipelineState>)get_pipeline(kernel_name);
    if (!pipe) return;

    size_t bytes = n * sizeof(float);
    size_t aligned = page_align(bytes);

    // Try zero-copy buffers
    id<MTLBuffer> bufX = [dev newBufferWithBytesNoCopy:(void*)x
                                                length:aligned
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
    id<MTLBuffer> bufY = [dev newBufferWithBytesNoCopy:(void*)y
                                                length:aligned
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];

    // Fall back to copy if zero-copy fails
    bool needCopyBackX = false;
    (void)needCopyBackX;
    bool needCopyBack = false;
    if (!bufX) {
        bufX = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        memcpy(bufX.contents, x, bytes);
    }
    if (!bufY) {
        bufY = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        needCopyBack = true;
    }

    if (!bufX || !bufY) return;

    id<MTLCommandBuffer> cmdBuf = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:bufX offset:0 atIndex:0];
    [enc setBuffer:bufY offset:0 atIndex:1];

    NSUInteger tgSize = MIN(pipe.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
    MTLSize threads = MTLSizeMake(n, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(tgSize, 1, 1);
    [enc dispatchThreads:threads threadsPerThreadgroup:threadgroupSize];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    if (needCopyBack) memcpy(y, bufY.contents, bytes);
}

// ---------------------------------------------------------------------------
// Public API: matmul
// ---------------------------------------------------------------------------

void MetalBackend::matmul(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    init();
    if (!available_) return;

    if (@available(macOS 10.14, *)) {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
        id<MTLComputePipelineState> pipe =
            (__bridge id<MTLComputePipelineState>)get_pipeline("matmul");
        if (!pipe) return;

        size_t aBytes = (size_t)M * K * sizeof(float);
        size_t bBytes = (size_t)K * N * sizeof(float);
        size_t cBytes = (size_t)M * N * sizeof(float);

        size_t aAligned = page_align(aBytes);
        size_t bAligned = page_align(bBytes);
        size_t cAligned = page_align(cBytes);

        // Try zero-copy buffers
        id<MTLBuffer> bufA = [dev newBufferWithBytesNoCopy:(void*)A
                                                    length:aAligned
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
        id<MTLBuffer> bufB = [dev newBufferWithBytesNoCopy:(void*)B
                                                    length:bAligned
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
        id<MTLBuffer> bufC = [dev newBufferWithBytesNoCopy:(void*)C
                                                    length:cAligned
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];

        // Fall back to copy if zero-copy fails
        bool needCopyBack = false;
        if (!bufA) {
            bufA = [dev newBufferWithLength:aBytes options:MTLResourceStorageModeShared];
            if (!bufA) return;
            memcpy(bufA.contents, A, aBytes);
        }
        if (!bufB) {
            bufB = [dev newBufferWithLength:bBytes options:MTLResourceStorageModeShared];
            if (!bufB) return;
            memcpy(bufB.contents, B, bBytes);
        }
        if (!bufC) {
            bufC = [dev newBufferWithLength:cBytes options:MTLResourceStorageModeShared];
            if (!bufC) return;
            needCopyBack = true;
        }

        uint32_t m = (uint32_t)M, n = (uint32_t)N, k = (uint32_t)K;
        id<MTLBuffer> bufM = [dev newBufferWithBytes:&m length:sizeof(m) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufN = [dev newBufferWithBytes:&n length:sizeof(n) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [dev newBufferWithBytes:&k length:sizeof(k) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pipe];
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

        if (needCopyBack) memcpy(C, bufC.contents, cBytes);
    }
}

// ---------------------------------------------------------------------------
// Public API: element-wise operations
// ---------------------------------------------------------------------------

void MetalBackend::elementwise_add(const float* a, const float* b, float* c, size_t n) {
    init();
    run_elementwise("elementwise_add", a, b, c, n);
}

void MetalBackend::elementwise_sub(const float* a, const float* b, float* c, size_t n) {
    init();
    run_elementwise("elementwise_sub", a, b, c, n);
}

void MetalBackend::elementwise_mul(const float* a, const float* b, float* c, size_t n) {
    init();
    run_elementwise("elementwise_mul", a, b, c, n);
}

void MetalBackend::elementwise_div(const float* a, const float* b, float* c, size_t n) {
    init();
    run_elementwise("elementwise_div", a, b, c, n);
}

// ---------------------------------------------------------------------------
// Public API: activations
// ---------------------------------------------------------------------------

void MetalBackend::relu(const float* x, float* y, size_t n) {
    init();
    run_unary("relu_kernel", x, y, n);
}

void MetalBackend::sigmoid(const float* x, float* y, size_t n) {
    init();
    run_unary("sigmoid_kernel", x, y, n);
}

void MetalBackend::tanh_op(const float* x, float* y, size_t n) {
    init();
    run_unary("tanh_kernel", x, y, n);
}

}  // namespace whitematter

// Provide metal_backend_available for device.cpp when Metal is built.
namespace whitematter {
bool metal_backend_available() {
    return MetalBackend::instance().is_available();
}
}  // namespace whitematter

#endif  // __APPLE__
