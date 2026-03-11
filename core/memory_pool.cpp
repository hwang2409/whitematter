#include "memory_pool.h"
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstdlib>
#include <algorithm>
#include <cmath>

namespace {

size_t next_power_of_two(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
#if (SIZE_MAX > 0xFFFFFFFF)
    n |= n >> 32;
#endif
    return n + 1;
}

}  // namespace

size_t MemoryPool::size_class(size_t n) {
    if (n == 0) return MemoryPool::kMinSizeClass;
    return std::max(kMinSizeClass, next_power_of_two(n));
}

struct MemoryPool::Impl {
    std::mutex mutex;
    std::unordered_map<size_t, std::vector<float*>> buckets;

    float* acquire(size_t n) {
        size_t cls = MemoryPool::size_class(n);
        std::lock_guard<std::mutex> lock(mutex);
        auto& free_list = buckets[cls];
        if (!free_list.empty()) {
            float* ptr = free_list.back();
            free_list.pop_back();
            return ptr;
        }
        float* ptr = static_cast<float*>(std::malloc(cls * sizeof(float)));
        if (!ptr) return nullptr;
        return ptr;
    }

    void release(float* ptr, size_t original_n) {
        if (!ptr) return;
        size_t cls = MemoryPool::size_class(original_n);
        std::lock_guard<std::mutex> lock(mutex);
        buckets[cls].push_back(ptr);
    }
};

MemoryPool::Impl* MemoryPool::impl() {
    static Impl s_impl;
    return &s_impl;
}

MemoryPool& MemoryPool::instance() {
    static MemoryPool s_pool;
    return s_pool;
}

float* MemoryPool::acquire(size_t n) {
    return impl()->acquire(n);
}

void MemoryPool::release(float* ptr, size_t original_n) {
    impl()->release(ptr, original_n);
}
