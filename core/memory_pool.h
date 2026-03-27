#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <cstddef>
#include <memory>

namespace memory_pool_detail { struct ThreadCache; }

// Thread-safe memory pool for float buffers using size-class buckets
// (next power of 2). Used by Tensor for data and grad storage.
class MemoryPool {
    friend struct memory_pool_detail::ThreadCache;
public:
    static MemoryPool& instance();

    // Acquire a buffer of at least n floats. Returns pointer to n floats (actual
    // allocation may be larger due to size class). Caller must not free; use release().
    float* acquire(size_t n);

    // Acquire shared ownership of a buffer. Memory is returned to the pool when
    // the last shared_ptr is destroyed (custom deleter calls release).
    std::shared_ptr<float> acquire_shared(size_t n);

    // Return a buffer previously obtained from acquire(original_n).
    // original_n must be the same value passed to acquire (used for bucket lookup).
    void release(float* ptr, size_t original_n);

    // Round requested size up to the pool's size class (for tests/tooling).
    static size_t size_class(size_t n);

    // Custom allocator hooks (for CUDA managed memory).
    // When set, these replace std::malloc/std::free for all pool allocations.
    using AllocFn = float*(*)(size_t n_bytes);
    using FreeFn = void(*)(float* ptr);
    static void set_allocator(AllocFn alloc, FreeFn free);

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
