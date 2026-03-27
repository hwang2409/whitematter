#pragma once
#include <cstdio>
#include <cstdlib>

#ifdef WHITEMATTER_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", __FILE__, __LINE__, (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#include <cudnn.h>

#define CUDNN_CHECK(call) do { \
    cudnnStatus_t status = (call); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#endif // WHITEMATTER_CUDA
