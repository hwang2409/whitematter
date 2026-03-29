# Architecture

WhiteMatter is a from-scratch deep learning framework in C++17 with a PyTorch-like API. It implements tensors with automatic differentiation, a module system for composing layers, and hardware backends for CPU (with BLAS), CUDA (cuDNN/cuBLAS), and Metal. The project also includes a Python/Next.js training platform (`platform/` + `frontend/`) and an ONNX export pipeline for browser inference. Key design decisions: pool-backed memory with `shared_ptr<float>` for zero-copy views, closure-based autograd rather than a tape, and transparent GPU offload where CPU tensors can use cuDNN without explicit device placement.

## Directory Layout

```
core/                      Tensor, autograd, loss, optimizer, dataloader, device abstraction
core/layers/               Module implementations: linear, conv, attention, normalization, etc.
core/ops/                  CPU kernels: im2col, matmul (BLAS dispatch), SIMD ops, Winograd, fp16
core/cuda/                 CUDA backend: cuDNN conv/batchnorm, cuBLAS matmul, memory pool, kernels
core/metal/                Metal backend: MSL kernels, MPS integration (macOS GPU)
core/serialization/        Binary checkpoint format, ONNX export/import
datasets/                  Dataset loaders (MNIST, CIFAR-10) with async prefetching
examples/                  Training scripts: CNNs, ResNet, MobileNetV2, GPT, GAN, autoencoder
tests/                     Unit tests: tensor ops, autograd, layers, loss, optimizer, grad checking
platform/                  Python backend (FastAPI): training jobs, dataset processing, inference
frontend/                  Next.js web UI for the training platform
demo/                      Browser-based CIFAR-10 demo using ONNX Runtime Web
```

## Tensor and Memory

Tensors store data in a `shared_ptr<float>` obtained from `MemoryPool`, a singleton that maintains thread-local free lists bucketed by size class (next power of 2). When a `shared_ptr` is destroyed, its custom deleter returns the buffer to the pool instead of freeing it.

```cpp
// core/tensor.h (private members)
std::shared_ptr<float> data_storage_;    // pool-backed float buffer
std::shared_ptr<uint16_t> half_storage_; // fp16 storage (when dtype == Float16)
size_t data_size_;
std::shared_ptr<float> grad_storage_;    // allocated lazily, only when requires_grad
size_t grad_size_;
```

Views (from `reshape`, `squeeze`, `unsqueeze`) share the same `data_storage_` and allocate their own gradient buffer. This means reshaping is zero-copy.

The pool supports custom allocator hooks via `MemoryPool::set_allocator()` -- the CUDA rewrite plan uses this to swap in `cudaMallocManaged`.

Shape is stored as `std::vector<size_t>`. There is no stride array; tensors are always contiguous.

## Autograd

Each tensor operation (e.g., `matmul`, `relu`, `conv2d`) creates a result tensor and, if gradient tracking is active, captures a `grad_fn` closure and a `parents` vector on the result:

```cpp
// From core/tensor.h
std::function<void()> grad_fn;     // backward closure
std::vector<TensorPtr> parents;    // inputs this tensor depends on
```

`backward()` works in three steps:

1. Seeds `grad[0] = 1.0f` on the scalar loss tensor.
2. Calls `build_topo()` -- a recursive DFS that produces a topological ordering of the computation graph.
3. Iterates in reverse topological order, calling each node's `grad_fn()` which accumulates into parent gradients.
4. Clears `grad_fn` and `parents` on all visited nodes (frees the graph after each backward pass).

`GradMode` is a global toggle. `NoGradGuard` disables it in a scoped block (RAII), used during evaluation and inference.

## Layers

All layers inherit from `Module` (`core/layer.h`):

```cpp
class Module {
public:
    virtual TensorPtr forward(const TensorPtr& input) = 0;
    virtual std::vector<TensorPtr> parameters() { return {}; }
    virtual void to(whitematter::DeviceType device);  // moves params to device
    TensorPtr operator()(const TensorPtr& input);     // calls forward()
};
```

`Sequential` composes modules into a pipeline and provides `train()`/`eval()` which toggle `Dropout` and `BatchNorm2d` behavior. `parameters()` recursively collects all learnable tensors.

Available layers: `Linear`, `Conv2d`, `Conv1d`, `ConvTranspose2d`, `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`, `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `RMSNorm`, `Dropout`, `Embedding`, `LSTM`, `GRU`, `MultiHeadAttention`, `GroupedQueryAttention`, `KVCache`, `SinusoidalPositionalEncoding`, `Upsample`, `Flatten`, and activation modules (`ReLU`, `SiLU`, `GELU`, `Mish`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`).

## Operations Dispatch

An operation like `conv2d` follows this dispatch chain (`core/ops/conv_ops.cpp`):

1. **`Tensor::conv2d()` called** with weight, bias, stride, padding, groups, dilation.

2. **CUDA device tensors** -- if both input and weight are on CUDA and the backend is available, dispatches to `cuda_ops::conv2d()` which uses `CUDAMemoryPool` for device buffers and calls `CUDABackend::conv2d_forward` (cuDNN). Falls through to CPU on unsupported configs (e.g., dilation > 1).

3. **Transparent cuDNN offload** -- if tensors are on CPU but CUDA is available and `dilation == 1`, the function copies host data to GPU via `CUDABackend::conv2d_forward` (which handles H2D/D2H internally), runs cuDNN, and copies results back. No explicit `.to(CUDA)` needed.

4. **Winograd fast path** -- for 3x3 kernels with stride=1, padding=1, groups=1, dilation=1, uses the Winograd F(2x2, 3x3) transform (16 multiplies per 2x2 output tile vs 36 for direct convolution).

5. **CPU fallback** -- `im2col` rearranges the input into a column matrix, then `matmul_blocked` computes the GEMM. `matmul_blocked` itself dispatches:
   - Apple Accelerate `cblas_sgemm` on macOS (always linked)
   - OpenBLAS `cblas_sgemm` on Linux (when `OPENBLAS=1`)
   - Hand-written SIMD kernel (AVX or NEON) with OpenMP threading as last resort
   - Transparent cuBLAS offload for large matrices (M,N >= 512) when no CPU BLAS is available

## CUDA Backend

The CUDA backend (`core/cuda/`) is a singleton with cuBLAS and cuDNN handles, compiled only when `CUDA=1`.

**CUDAMemoryPool** (`cuda_memory.h`): GPU-side equivalent of `MemoryPool`. Uses power-of-2 size classes with a mutex-protected free list. `acquire_shared()` returns `shared_ptr<float>` with a custom deleter that recycles device memory.

**CUDABackend** (`cuda_backend.h`): Exposes operations that work on device pointers:
- `matmul` / `bmm` via cuBLAS
- `conv2d_forward` / `conv2d_backward` via cuDNN (with auto-tuned algorithm selection and workspace allocation)
- `batchnorm_forward` / `batchnorm_backward` via cuDNN
- Element-wise ops, activations, reductions, pooling, loss, and optimizer steps as custom CUDA kernels
- `*_host` variants that accept CPU pointers and handle H2D/D2H copies internally (the "transparent offload" pattern)

**Weight caching**: `invalidate_weight_cache()` marks cached GPU weight buffers as stale after an optimizer step. The next forward pass re-uploads changed weights.

When `CUDA=0`, `core/cuda/cuda_stub.cpp` provides no-op implementations so the rest of the code compiles without CUDA headers.

## Build System

The project uses a GNU Makefile. Key flags:

| Flag | Effect |
|------|--------|
| `OPENBLAS=1` | Link OpenBLAS for CPU matmul (auto-detected on Linux) |
| `CUDA=1` | Enable CUDA backend; compiles `.cu` files with `nvcc` |
| `METAL=1` | Enable Metal backend on macOS (links Metal frameworks) |
| `DEBUG=1` | Define `WHITEMATTER_DEBUG` for verbose logging |

CUDA files are compiled with `nvcc --gpu-architecture=sm_75`. Non-CUDA builds link a stub file instead. On macOS, Apple Accelerate is always linked for BLAS. The build produces a static library `build/libwhitematter.a` that example binaries link against.

```bash
make                          # build core library + default examples
make CUDA=1 resnet18-cuda     # build with CUDA support
make test                     # build and run all unit tests
make test-autograd             # run just autograd tests
make bench                    # build and run benchmarks
```

To add a new training example: create `examples/foo.cpp`, add a build target in the Makefile following the existing pattern (compile `.o`, link against `libwhitematter.a`).

## Testing

Tests use a custom framework (`tests/test_framework.h`) with no external dependencies.

**Core types**: `TestSuite` holds named test functions. `TestRunner` runs suites and reports pass/fail counts with timing.

**Assertion macros**: `TEST_ASSERT(cond)`, `TEST_ASSERT_EQ(a, b)`, `TEST_ASSERT_NEAR(a, b, eps)`, `TEST_ASSERT_SHAPE(tensor, expected_shape)`.

**Test suites** (`tests/`): `test_tensor.cpp`, `test_autograd.cpp`, `test_layers.cpp`, `test_loss.cpp`, `test_optimizer.cpp`, `test_grad_check.cpp`. Each file defines a `create_*_tests()` function that returns a `TestSuite*`.

**Adding a test**: Add a test function to the relevant suite's `create_*_tests()`:

```cpp
suite->add_test("my_new_op", [&]() {
    auto x = Tensor::randn({2, 3}, true);
    auto y = x->my_new_op();
    TEST_ASSERT_SHAPE(y, {2, 3});
    TEST_ASSERT_NEAR(y->data()[0], expected, 1e-5f);
});
```

The runner supports filtering: `./run_tests --tensor`, `--autograd`, `--layers`, `--loss`, `--optimizer`, `--gradcheck`.

## Data Pipeline

Two binary tensor formats are used:

**Python preprocessing format** (magic `0x54454E53` = "TENS"): Used by `examples/preprocess_*.py` scripts and the platform's dataset service. Layout: `magic (4B) | ndim (4B) | shape (ndim x 8B uint64) | float32 data`. Python scripts convert raw images/text into these `.bin` files, which C++ training code reads directly.

**C++ checkpoint format** (magic `0x574D5400` = "WMT\0"): Used by `core/serialization/serialize.cpp` for model/optimizer checkpoints. Layout: `magic (4B) | ndim (4B) | shape (ndim x 4B uint32) | float32 data`. Also supports compound checkpoint files (model + optimizer + epoch/loss/accuracy metadata).

**Dataset loaders** (`datasets/`): `CIFAR10Dataset` and `MNISTDataset` read raw binary files into tensors with normalization. `CIFAR10DataLoader` and `AsyncCIFAR10DataLoader` handle batching, shuffling, and optional data augmentation (random crop + horizontal flip). `ThreadedDataLoader` (`core/dataloader.h`) is a generic loader with worker threads and a `BatchQueue` for prefetching.

**Adding a new dataset**: Create `datasets/foo.h` and `datasets/foo.cpp` with a struct holding image/label tensors and a loader class. Write a Python preprocessing script in `examples/` to convert raw data to the `.bin` format. Add the `.o` target to the Makefile's `DATASET_OBJS`.

**ONNX export**: `core/onnx_export.h` serializes a `Sequential` model to ONNX format for inference with ONNX Runtime (including browser-based inference via `demo/`).
