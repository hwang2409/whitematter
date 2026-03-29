#include <cstdio>
#include <chrono>
#include "tensor.h"
#include "layer.h"

int main() {
    printf("whitematter benchmark\n");
    printf("====================\n\n");

    // Matmul benchmarks
    printf("Matrix Multiply (GEMM):\n");
    int sizes[][3] = {{128,128,128}, {256,256,256}, {512,512,512}, {1024,1024,1024}};
    for (auto& s : sizes) {
        auto A = Tensor::randn({(size_t)s[0], (size_t)s[1]}, false);
        auto B = Tensor::randn({(size_t)s[1], (size_t)s[2]}, false);

        // Warmup
        auto C = A->matmul(B);

        auto t0 = std::chrono::high_resolution_clock::now();
        int iters = s[0] <= 256 ? 100 : 10;
        for (int i = 0; i < iters; i++) C = A->matmul(B);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        double gflops = 2.0 * s[0] * s[1] * s[2] / (ms * 1e6);
        printf("  [%4d x %4d] x [%4d x %4d]: %7.2f ms  (%5.1f GFLOPS)\n",
               s[0], s[1], s[1], s[2], ms, gflops);
    }

    // Conv2d benchmarks
    printf("\nConv2d (3x3, stride=1, padding=1):\n");
    int conv_sizes[][4] = {{1,64,32,32}, {1,64,56,56}, {1,128,28,28}, {1,256,14,14}, {1,512,7,7}};
    int out_channels[] = {64, 64, 128, 256, 512};
    for (int i = 0; i < 5; i++) {
        auto& s = conv_sizes[i];
        Conv2d conv(s[1], out_channels[i], 3, 1, 1);
        auto input = Tensor::randn({(size_t)s[0], (size_t)s[1], (size_t)s[2], (size_t)s[3]}, false);

        // Warmup
        auto out = conv.forward(input);

        auto t0 = std::chrono::high_resolution_clock::now();
        int iters = 10;
        for (int j = 0; j < iters; j++) out = conv.forward(input);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        printf("  [%d, %3d, %2d, %2d] → [%d, %3d, %2d, %2d]: %7.2f ms\n",
               s[0], s[1], s[2], s[3], s[0], out_channels[i], s[2], s[3], ms);
    }

    return 0;
}
