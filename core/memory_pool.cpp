#include "memory_pool.h"
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstdlib>
#include <algorithm>
#include <cmath>

// Custom allocator hooks (default to malloc/free)
static MemoryPool::AllocFn g_alloc_fn = nullptr;
static MemoryPool::FreeFn g_free_fn = nullptr;

static float* pool_alloc(size_t n_bytes) {
    if (g_alloc_fn) return g_alloc_fn(n_bytes);
    return static_cast<float*>(std::malloc(n_bytes));
}

[[maybe_unused]]
static void pool_free(float* ptr) {
    if (g_free_fn) g_free_fn(ptr);
    else std::free(ptr);
}

void MemoryPool::set_allocator(AllocFn alloc, FreeFn free) {
    g_alloc_fn = alloc;
    g_free_fn = free;
}

namespace {

const size_t kMaxPerClassPerThread = 64;
const size_t kRefillBatch = 32;

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
        float* ptr = pool_alloc(cls * sizeof(float));
        if (!ptr) return nullptr;
        return ptr;
    }

    void release(float* ptr, size_t original_n) {
        if (!ptr) return;
        size_t cls = MemoryPool::size_class(original_n);
        std::lock_guard<std::mutex> lock(mutex);
        buckets[cls].push_back(ptr);
    }

    // Batch refill: under lock, move up to `count` buffers from global bucket
    // (or allocate new) into `out`. Caller consumes from `out` without holding lock.
    void acquire_batch(size_t cls, size_t count, std::vector<float*>& out) {
        std::lock_guard<std::mutex> lock(mutex);
        auto& free_list = buckets[cls];
        while (out.size() < count) {
            if (!free_list.empty()) {
                out.push_back(free_list.back());
                free_list.pop_back();
            } else {
                float* ptr = pool_alloc(cls * sizeof(float));
                if (!ptr) break;
                out.push_back(ptr);
            }
        }
    }

    // Batch release: under lock, push all pointers from `from` into global bucket, then clear `from`.
    void release_batch(std::vector<float*>& from, size_t cls) {
        if (from.empty()) return;
        std::lock_guard<std::mutex> lock(mutex);
        auto& free_list = buckets[cls];
        for (float* ptr : from)
            free_list.push_back(ptr);
        from.clear();
    }

    // Release a single buffer (used by thread-local cache destructor).
    void release_one(float* ptr, size_t cls) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mutex);
        buckets[cls].push_back(ptr);
    }
};

namespace memory_pool_detail {

struct ThreadCache {
    std::unordered_map<size_t, std::vector<float*>> buckets;
    MemoryPool::Impl* global_ = nullptr;

    ~ThreadCache() {
        if (!global_) return;
        for (auto& kv : buckets) {
            size_t cls = kv.first;
            for (float* ptr : kv.second)
                global_->release_one(ptr, cls);
            kv.second.clear();
        }
    }

    float* acquire(size_t n, MemoryPool::Impl* global) {
        global_ = global;
        size_t cls = MemoryPool::size_class(n);
        auto& local = buckets[cls];
        if (!local.empty()) {
            float* ptr = local.back();
            local.pop_back();
            return ptr;
        }
        std::vector<float*> batch;
        global->acquire_batch(cls, kRefillBatch, batch);
        if (batch.empty()) return nullptr;
        for (size_t i = 1; i < batch.size(); ++i)
            local.push_back(batch[i]);
        return batch[0];
    }

    void release(float* ptr, size_t original_n, MemoryPool::Impl* global) {
        if (!ptr) return;
        global_ = global;
        size_t cls = MemoryPool::size_class(original_n);
        auto& local = buckets[cls];
        if (local.size() < kMaxPerClassPerThread) {
            local.push_back(ptr);
            return;
        }
        size_t drain_count = kMaxPerClassPerThread / 2;
        std::vector<float*> to_global;
        for (size_t i = 0; i < drain_count && !local.empty(); ++i) {
            to_global.push_back(local.back());
            local.pop_back();
        }
        global->release_batch(to_global, cls);
        local.push_back(ptr);
    }
};

thread_local ThreadCache t_thread_cache;
}  // namespace memory_pool_detail

MemoryPool::Impl* MemoryPool::impl() {
    static Impl s_impl;
    return &s_impl;
}

MemoryPool& MemoryPool::instance() {
    static MemoryPool s_pool;
    return s_pool;
}

float* MemoryPool::acquire(size_t n) {
    return memory_pool_detail::t_thread_cache.acquire(n, impl());
}

void MemoryPool::release(float* ptr, size_t original_n) {
    memory_pool_detail::t_thread_cache.release(ptr, original_n, impl());
}

void MemoryPool::trim() {
    auto* p = impl();
    std::lock_guard<std::mutex> lock(p->mutex);
    for (auto& [cls, free_list] : p->buckets) {
        for (float* ptr : free_list) {
            if (g_free_fn) g_free_fn(ptr);
            else std::free(ptr);
        }
        free_list.clear();
    }
}

std::shared_ptr<float> MemoryPool::acquire_shared(size_t n) {
    float* ptr = acquire(n);
    if (!ptr) return nullptr;
    return std::shared_ptr<float>(ptr, [n](float* p) {
        MemoryPool::instance().release(p, n);
    });
}
