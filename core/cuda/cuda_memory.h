#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace whitematter {

// GPU memory pool with size-class buckets (round up to next power of 2).
// Thread-safe: GPU memory is a global resource shared across threads.
class CUDAMemoryPool {
public:
    static CUDAMemoryPool& instance();

    // Acquire a device buffer of at least n_floats floats.
    // Returns a device pointer (NOT host-accessible).
    float* acquire(size_t n_floats);

    // Acquire shared ownership of a device buffer.
    // The custom deleter calls release() when the last shared_ptr is destroyed.
    std::shared_ptr<float> acquire_shared(size_t n_floats);

    // Return a device buffer previously obtained from acquire().
    // n_floats must match the value originally passed to acquire().
    void release(float* ptr, size_t n_floats);

    // Round up to next power of 2 (size class).
    static size_t size_class(size_t n);

    ~CUDAMemoryPool();

private:
    CUDAMemoryPool() = default;
    CUDAMemoryPool(const CUDAMemoryPool&) = delete;
    CUDAMemoryPool& operator=(const CUDAMemoryPool&) = delete;

    std::mutex mu_;
    // Bucket: size_class -> list of free device pointers
    std::unordered_map<size_t, std::vector<float*>> free_lists_;
};

}  // namespace whitematter
