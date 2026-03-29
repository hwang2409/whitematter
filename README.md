# whitematter

A neural network framework built from scratch in C++.

Complete deep learning from the ground up: tensors with autograd, 30+ layer types, GPU acceleration via CUDA/cuDNN and Metal, training and inference. Not a wrapper around PyTorch -- every operation, every backward pass, every SIMD kernel is written by hand.

**[Live demo: CIFAR-10 classifier running in your browser](https://hwang2409.github.io/blog/cifar10-classifier)**

## Features

| Category | What's included |
|----------|----------------|
| **Core** | Tensor with autograd, NumPy-style broadcasting, memory pool, ONNX export/import |
| **Layers** | Conv2d (grouped, dilated), Conv1d, ConvTranspose2d, Linear, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Upsample, Flatten |
| **Normalization** | BatchNorm2d, LayerNorm, GroupNorm, RMSNorm |
| **Attention** | MultiHeadAttention, GroupedQueryAttention, KV cache, sinusoidal positional encoding |
| **Recurrent** | LSTM, GRU |
| **Activations** | ReLU, GELU, SiLU, Mish, Sigmoid, Tanh, Softmax, LogSoftmax |
| **Loss** | CrossEntropy, MSE, L1, SmoothL1, NLL, BCE, BCEWithLogits, KLDiv, Focal, BinaryFocal |
| **Optimizers** | SGD (momentum), Adam, AdamW, RMSprop |
| **Schedulers** | StepLR, ExponentialLR, CosineAnnealing, CosineWarmRestarts, ReduceLROnPlateau |
| **Training** | Gradient clipping, gradient accumulation, mixed precision (fp16 + GradScaler), early stopping, checkpointing |
| **Performance** | Apple Accelerate BLAS, OpenBLAS, NEON/AVX SIMD, Winograd convolution, flash attention, OpenMP |
| **GPU** | CUDA (cuDNN conv/batchnorm, cuBLAS matmul, custom kernels), Metal (macOS) |
| **Export** | ONNX export, browser inference via ONNX Runtime Web |

## Building

```bash
# CPU only (uses Apple Accelerate on macOS)
make

# With OpenBLAS (Linux)
make OPENBLAS=1

# With CUDA (NVIDIA GPU)
make OPENBLAS=1 CUDA=1

# With Metal (macOS GPU)
make METAL=1
```

## Example: Train ResNet-18 on CIFAR-10

```bash
# Download data
mkdir -p data && cd data
curl -LO https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar xzf cifar-10-binary.tar.gz && mv cifar-10-batches-bin/*.bin .
cd ..

# Train (CPU)
make resnet18-cifar10
./build/resnet18_cifar10 data 64

# Train (CUDA)
make OPENBLAS=1 CUDA=1 resnet18-cuda
./build/resnet18_cifar10_cuda data 64
```

## Example: Train on ImageNette (224x224)

```bash
python examples/preprocess_imagenette.py
make OPENBLAS=1 CUDA=1 resnet18-imagenette
./build/resnet18_imagenette data/imagenette
```

## Example: GPT on Shakespeare

```bash
python examples/preprocess_shakespeare.py
make OPENBLAS=1 gpt_shakespeare
./build/gpt_shakespeare
```

## Quick tour

**Tensors and autograd:**

```cpp
auto a = Tensor::randn({3, 4}, true);   // requires_grad=true
auto b = Tensor::xavier(4, 2, true);
auto c = a->matmul(b)->relu()->sum();
c->backward();                           // gradients flow through the whole graph
```

**Building a model:**

```cpp
Sequential model({
    new Conv2d(3, 64, 3, 1, 1),
    new BatchNorm2d(64),
    new ReLU(),
    new MaxPool2d(2, 2),
    new Flatten(),
    new Linear(64 * 16 * 16, 10)
});

CrossEntropyLoss criterion;
AdamW optimizer(model.parameters(), 0.001f);

for (auto [x, y] : dataloader) {
    optimizer.zero_grad();
    auto loss = criterion(model.forward(x), y);
    loss->backward();
    optimizer.step();
}
```

## Architecture

```
core/                Tensor, autograd engine, memory pool
core/layers/         All layer implementations (conv, attention, recurrent, norm, ...)
core/ops/            SIMD kernels (AVX/NEON), im2col, Winograd conv, matmul, fp16
core/cuda/           CUDA backend: cuDNN, cuBLAS, custom kernels
core/metal/          Metal GPU backend (macOS)
core/serialization/  Checkpoint save/load, ONNX export/import
datasets/            CIFAR-10, MNIST loaders
examples/            Training programs (ResNet-18, MobileNetV2, GPT, GAN, autoencoder, ...)
tests/               Unit tests including numerical gradient checks
demo/                Browser-based ONNX inference demo
platform/            Web training platform (FastAPI backend)
bindings/            Python bindings (pybind11)
```

## Tests

```bash
make build/run_tests
./build/run_tests
```

243 tests covering tensors, autograd, layers, loss functions, optimizers, and numerical gradient verification.

## By the numbers

- ~40,000 lines of C/C++
- 30+ layer types
- 243 unit tests
- 3 GPU/SIMD backends (CUDA, Metal, AVX/NEON)
- Reference models: ResNet-18, MobileNetV2, GPT, DCGAN, LSTM text generation

## License

MIT
