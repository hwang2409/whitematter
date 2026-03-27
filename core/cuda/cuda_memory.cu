#include "cuda_memory.h"
#include "cuda_check.h"
#include <cuda_runtime.h>

namespace whitematter {

CUDAMemoryPool& CUDAMemoryPool::instance() {
    static CUDAMemoryPool pool;
    return pool;
}

size_t CUDAMemoryPool::size_class(size_t n) {
    if (n <= 1) return 1;
    // Round up to next power of 2
    size_t v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

float* CUDAMemoryPool::acquire(size_t n_floats) {
    if (n_floats == 0) return nullptr;
    // For small allocs, round to power-of-2 for reuse. For large allocs, exact size.
    size_t alloc_size = (n_floats <= 65536) ? size_class(n_floats) : n_floats;

    std::lock_guard<std::mutex> lock(mu_);
    auto it = free_lists_.find(alloc_size);
    if (it != free_lists_.end() && !it->second.empty()) {
        float* ptr = it->second.back();
        it->second.pop_back();
        return ptr;
    }

    // No free buffer; allocate from device
    float* ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, alloc_size * sizeof(float));
    if (err != cudaSuccess) {
        // OOM: try freeing cached buffers and retrying
        for (auto& [sz, list] : free_lists_) {
            for (float* p : list) cudaFree(p);
            list.clear();
        }
        err = cudaMallocManaged(&ptr, alloc_size * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA OOM: failed to allocate %zu floats (%.1f MB)\n",
                    alloc_size, alloc_size * sizeof(float) / (1024.0 * 1024.0));
            return nullptr;
        }
    }
    return ptr;
}

std::shared_ptr<float> CUDAMemoryPool::acquire_shared(size_t n_floats) {
    float* raw = acquire(n_floats);
    if (!raw) return nullptr;
    size_t n = n_floats;  // capture for deleter
    return std::shared_ptr<float>(raw, [n](float* p) {
        CUDAMemoryPool::instance().release(p, n);
    });
}

void CUDAMemoryPool::release(float* ptr, size_t n_floats) {
    if (!ptr || n_floats == 0) return;
    size_t bucket = size_class(n_floats);

    std::lock_guard<std::mutex> lock(mu_);
    free_lists_[bucket].push_back(ptr);
}

CUDAMemoryPool::~CUDAMemoryPool() {
    for (auto& [bucket, ptrs] : free_lists_) {
        for (float* p : ptrs) {
            cudaFree(p);
        }
    }
    free_lists_.clear();
}

}  // namespace whitematter
