# dongha.md — What I Added to This Project

This file records plans I implemented in the whitematter project. Add new entries when you implement a plan.

---

## 1. Zero-Copy Views for Reshape / Squeeze / Unsqueeze

**Summary:** Tensor storage was switched from raw pool pointers to `std::shared_ptr<float>` with a pool deleter. Reshape, squeeze, and unsqueeze were changed to O(1) views that share the data buffer instead of copying.

**From this chat:** Implementation followed the plan in order: (1) MemoryPool `acquire_shared`, (2) Tensor switch to shared_ptr storage, (3) view constructor and reshape as view, (4) squeeze and unsqueeze as views, (5) full test suite. One fix during implementation: in the from-vector constructor the parameter `data` shadowed the member `data()`; `std::memcpy` was updated to use `data_storage_.get()` instead of `data()`. `operator[]` and `at()` now use `data()`. Full test run: 184 tests passed; an earlier run had a transient bus error that did not reproduce.

### Changes

- **Memory pool (`core/memory_pool.h`, `core/memory_pool.cpp`)**
  - Added `#include <memory>` in the header.
  - Added `std::shared_ptr<float> acquire_shared(size_t n)` that returns a shared_ptr with a custom deleter calling `release(ptr, n)` so memory is returned to the pool when the last owner is destroyed. Deleter captures `n` for the correct size-class bucket.
  - Kept `acquire` and `release` for compatibility.

- **Tensor storage (`core/tensor.h`, `core/tensor.cpp`)**
  - Replaced `float* data_ptr_` / `float* grad_ptr_` with `std::shared_ptr<float> data_storage_` / `grad_storage_` (kept `data_size_` / `grad_size_`).
  - `data()` / `grad()` now return `data_storage_.get()` / `grad_storage_.get()` (or nullptr if empty); `grad_empty()` is `!grad_storage_`.
  - Default constructor: only sets `data_size_` / `grad_size_` to 0.
  - Destructor: no manual `release`; shared_ptr deleters return memory to the pool.
  - Shape-only and from-vector constructors: use `acquire_shared` for data and grad; on grad alloc failure, data shared_ptr goes out of scope and deleter runs.
  - From-vector constructor: `std::memcpy(data_storage_.get(), data.data(), ...)` to avoid parameter name clash with `data()`.
  - `operator[]` and `at()`: use `data()` instead of `data_ptr_`.
  - `backward()` lazy grad: use `acquire_shared(1)` and set `grad_storage_`, `grad_size_ = 1`.

- **View constructor**
  - Added private constructor:  
    `Tensor(std::shared_ptr<float> data_storage, size_t data_size, const std::vector<size_t>& shape, bool requires_grad)`.  
    View shares `data_storage_`; allocates its own `grad_storage_` (via `acquire_shared(data_size_)`) when `requires_grad` so backward can accumulate into the base without double-counting.

- **Reshape / squeeze / unsqueeze as views**
  - **reshape:** Computes `total = product(new_shape)`, asserts `total == size()`. Creates view with `std::make_shared<Tensor>(data_storage_, data_size_, new_shape, track)`. No `std::vector<float>`, no copy. If track and existing grad, copies grad into result. Sets `result->parents = {self_ptr}` and `result->grad_fn` (simd_add for grad accumulation).
  - **squeeze:** Builds `new_shape` (remove size-1 dim(s), or single dim if given). If `new_shape == shape`, returns `shared_from_this()`. Otherwise creates view same way as reshape; sets parents and grad_fn.
  - **unsqueeze:** Builds `new_shape` (insert 1 at dim). Creates view; sets parents and grad_fn when tracking.
  - **flatten:** Unchanged; it calls `reshape`, so it is a view automatically.
  - **permute:** Left as copy-based (no view semantics).

### Verification

- Full test suite: 184 tests passed (tensor 39, autograd 23, layers 74, loss 22, optimizer 26).

---

## 2. Thread-Local Pool Caches

**Summary:** The memory pool now uses a thread-local cache per worker; threads only lock the global mutex when refilling from or draining to the global pool in batches (and when the thread exits), which reduces lock contention under OpenMP.

**From this chat:** Implementation followed the Thread-Local Pool Caches plan: added `ThreadCache` (thread_local) with per–size-class buckets and a per-thread cap, batch refill/drain on `Impl`, routing of `acquire`/`release` through TLS, thread-exit drain in the cache destructor, then full test suite and this dongha update. A friend declaration was added so `memory_pool_detail::ThreadCache` can use the private `Impl` type.

### Changes

- **core/memory_pool.cpp**
  - Constants in anonymous namespace: `kMaxPerClassPerThread` (64), `kRefillBatch` (32).
  - `Impl`: added `acquire_batch(cls, count, out)` (under lock: fill `out` from global bucket or malloc) and `release_batch(from, cls)` (under lock: push all from `from` into global bucket, clear `from`); added `release_one(ptr, cls)` for thread-exit drain.
  - `memory_pool_detail::ThreadCache`: per-thread `unordered_map<size_t, vector<float*>>` buckets and `Impl* global_` (set on first use). `acquire(n, global)`: serve from local bucket if non-empty; else call `global->acquire_batch`, keep rest in local, return one. `release(ptr, original_n, global)`: push to local if under cap; else drain half of that size-class to global via `release_batch`, then push current ptr to local. Destructor: if `global_` non-null, release all cached buffers to global via `release_one`.
  - `thread_local ThreadCache t_thread_cache` in `memory_pool_detail`. `MemoryPool::acquire`/`release` call into `memory_pool_detail::t_thread_cache`; `acquire_shared` unchanged and thus uses the TLS path automatically.

- **core/memory_pool.h**
  - Forward declaration `namespace memory_pool_detail { struct ThreadCache; }` and `friend struct memory_pool_detail::ThreadCache;` so `ThreadCache` can use `MemoryPool::Impl*`.

### Verification

- Full test suite: 184 tests passed (tensor 39, autograd 23, layers 74, loss 22, optimizer 26).

---

*When you implement another plan, add a new numbered section above this line.*
