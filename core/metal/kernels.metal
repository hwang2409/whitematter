#include <metal_stdlib>
using namespace metal;

// Tiled matmul: C = A @ B. A [M,K], B [K,N], C [M,N].
// Threadgroup memory for shared tiles.
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

// Tiled version with threadgroup memory (optional, for larger matrices)
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
