#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <cstddef>

// Thread-safe memory pool for float buffers using size-class buckets
// (next power of 2). Used by Tensor for data and grad storage.
class MemoryPool {
public:
    static MemoryPool& instance();

    // Acquire a buffer of at least n floats. Returns pointer to n floats (actual
    // allocation may be larger due to size class). Caller must not free; use release().
    float* acquire(size_t n);

    // Return a buffer previously obtained from acquire(original_n).
    // original_n must be the same value passed to acquire (used for bucket lookup).
    void release(float* ptr, size_t original_n);

    // Round requested size up to the pool's size class (for tests/tooling).
    static size_t size_class(size_t n);

private:
    MemoryPool() = default;
    ~MemoryPool() = default;
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    struct Impl;
    Impl* impl();
    static const size_t kMinSizeClass = 1;
};

#endif
