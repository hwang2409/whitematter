# whitematter: a NN framework

A lightweight PyTorch-like neural network framework written in C++ with automatic differentiation (autograd), SIMD optimizations, and an MNIST example.

## Quick Start

```bash
# Build
make

# Download MNIST data
mkdir -p data && cd data
curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz && cd ..

# Run training
./ml
```

## Framework Structure

```
├── tensor.h/cpp      # Core tensor with autograd
├── layer.h/cpp       # Neural network layers
├── loss.h/cpp        # Loss functions
├── optimizer.h/cpp   # Parameter optimizers
├── serialize.h/cpp   # Model save/load functionality
├── mnist.h/cpp             # MNIST data loader
├── cifar10.h/cpp           # CIFAR-10 data loader
├── ml.cpp                  # MLP training example
├── cnn_mnist.cpp           # CNN training example (MNIST)
├── cnn_cifar10.cpp         # CNN training example (CIFAR-10)
├── transformer_example.cpp # Transformer language model example
└── Makefile                # Build configuration
```

## Usage Guide

### Tensors

The `Tensor` class is the core data structure with automatic gradient computation:

```cpp
#include "tensor.h"

// Create tensors
auto a = Tensor::randn({3, 4}, true);      // 3x4 random tensor, requires_grad=true
auto b = Tensor::zeros({4, 2}, true);      // 4x2 zeros
auto c = Tensor::ones({3, 2}, false);      // 3x2 ones, no gradients
auto w = Tensor::xavier(784, 256, true);   // Xavier initialization

// Operations (automatically tracked for backprop)
auto d = a->matmul(b);          // Matrix multiplication
auto e = d->add(c);             // Addition (supports broadcasting)
auto f = e->relu();             // ReLU activation
auto g = f->sum();              // Sum to scalar

// Backpropagation
g->backward();                  // Compute gradients
a->grad;                        // Access gradients
a->zero_grad();                 // Reset gradients
```

**Available tensor operations:**
- Arithmetic: `add`, `sub`, `mul`, `div`, `neg` (with broadcasting)
- Matrix: `matmul`, `bmm`, `transpose`, `reshape`, `slice`, `concat`, `stack`
- Shape: `squeeze`, `unsqueeze`, `flatten`, `permute`
- Activations: `relu`, `sigmoid`, `tanh_`, `softmax`, `log_softmax`
- Reductions: `sum`, `mean`, `max`, `min`, `argmax`, `argmin`
- Element-wise: `log_`, `exp_`, `pow`, `sqrt`, `abs`, `clamp`
- Augmentation: `flip_horizontal`, `random_flip_horizontal`, `pad2d`, `crop`, `random_crop`

**Broadcasting:**
Arithmetic operations (`add`, `sub`, `mul`, `div`) support NumPy-style broadcasting:
```cpp
auto a = Tensor::randn({2, 3}, true);
auto b = Tensor::randn({3}, true);      // Broadcasts to [2, 3]
auto c = a->add(b);                      // Shape: [2, 3]

auto x = Tensor::randn({3, 1}, true);
auto y = Tensor::randn({1, 4}, true);
auto z = x->mul(y);                      // Outer product, Shape: [3, 4]

auto bias = Tensor::randn({1}, true);    // Scalar broadcast
auto out = a->add(bias);                 // Shape: [2, 3]
```

**Math operations:**
```cpp
auto a = Tensor::create({4, 9, 16}, {3}, true);
auto b = a->sqrt();           // [2, 3, 4]
auto c = a->pow(2.0f);        // [16, 81, 256]
auto d = a->pow(0.5f);        // Same as sqrt

// L2 norm: sqrt(sum(x^2))
auto x = Tensor::randn({10}, true);
auto norm = x->pow(2.0f)->sum()->sqrt();

// Element-wise power with broadcasting
auto bases = Tensor::randn({2, 3}, true);
auto exps = Tensor::create({1, 2, 3}, {3}, true);
auto result = bases->pow(exps);  // Each column raised to different power

// Absolute value and clamping
auto y = Tensor::create({-3, -1, 2, 5}, {4}, true);
auto abs_y = y->abs();              // [3, 1, 2, 5]
auto clamped = y->clamp(-2, 3);     // [-2, -1, 2, 3]
```

**Max/Min operations:**
```cpp
auto a = Tensor::create({1, 5, 3, 2, 8, 4}, {2, 3}, true);
// [[1, 5, 3], [2, 8, 4]]

// Reduction along dimension
auto max0 = a->max(0);           // max along dim 0: [2, 8, 4]
auto max1 = a->max(1);           // max along dim 1: [5, 8]
auto min1 = a->min(1);           // min along dim 1: [1, 2]
auto max_keep = a->max(1, true); // keepdim: [[5], [8]] shape [2, 1]

// Element-wise max/min with broadcasting
auto threshold = Tensor::create({3}, {1}, true);
auto clamped_low = a->max(threshold);   // ReLU-like: max(a, 3)
auto clamped_high = a->min(threshold);  // cap at 3: min(a, 3)

// Gradient flows only to the "winning" element
auto loss = a->max(1)->sum();
loss->backward();  // grad is 1 at max positions, 0 elsewhere

// Get indices of max/min values (no gradients, returns integer indices)
auto argmax_idx = a->argmax(1);    // [1, 1] - column indices of max values
auto argmin_idx = a->argmin(1);    // [0, 0] - column indices of min values
auto argmax_keep = a->argmax(1, true);  // [[1], [1]] - keepdim preserves shape
```

**Batch operations:**
```cpp
// Batch matrix multiplication: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
auto a = Tensor::randn({8, 16, 32}, true);   // 8 batches of 16x32 matrices
auto b = Tensor::randn({8, 32, 64}, true);   // 8 batches of 32x64 matrices
auto c = a->bmm(b);                          // Shape: [8, 16, 64]

// Attention scores computation: Q @ K^T
auto Q = Tensor::randn({4, 8, 16}, true);    // [batch, seq_len, head_dim]
auto K = Tensor::randn({4, 8, 16}, true);
auto scores = Q->bmm(K->permute({0, 2, 1})); // Shape: [4, 8, 8]
```

**Combining tensors:**
```cpp
// Concatenate along existing dimension
auto a = Tensor::randn({2, 3}, true);
auto b = Tensor::randn({2, 3}, true);
auto c = Tensor::concat({a, b}, 0);  // Shape: [4, 3]
auto d = Tensor::concat({a, b}, 1);  // Shape: [2, 6]

// Stack along new dimension
auto e = Tensor::stack({a, b}, 0);   // Shape: [2, 2, 3]
auto f = Tensor::stack({a, b}, -1);  // Shape: [2, 3, 2]
```

**Reshaping tensors:**
```cpp
auto a = Tensor::randn({2, 3}, true);
auto b = a->unsqueeze(0);    // Shape: [1, 2, 3]
auto c = a->unsqueeze(-1);   // Shape: [2, 3, 1]
auto d = b->squeeze(0);      // Shape: [2, 3]
auto e = Tensor::randn({1, 2, 1, 3, 1}, true);
auto f = e->squeeze();       // Remove all size-1 dims -> [2, 3]

// Permute dimensions (reorder axes)
auto g = Tensor::randn({2, 3, 4}, true);
auto h = g->permute({2, 0, 1});   // Shape: [4, 2, 3]
auto i = g->permute({0, 2, 1});   // Shape: [2, 4, 3]
// NCHW -> NHWC conversion
auto nchw = Tensor::randn({8, 3, 32, 32}, true);
auto nhwc = nchw->permute({0, 2, 3, 1});  // Shape: [8, 32, 32, 3]
```

### Layers

Build networks using the `Module` interface:

```cpp
#include "layer.h"

// Individual layers
auto linear = new Linear(784, 256);   // Fully connected
auto relu = new ReLU();               // Activation
auto sigmoid = new Sigmoid();
auto softmax = new Softmax(-1);       // Along last dimension
auto dropout = new Dropout(0.5);      // 50% dropout

// Sequential model
Sequential model({
    new Linear(784, 256),
    new ReLU(),
    new Dropout(0.2),
    new Linear(256, 128),
    new ReLU(),
    new Linear(128, 10)
});

// Forward pass
auto output = model.forward(input);

// Access parameters
auto params = model.parameters();  // Returns vector<TensorPtr>

// Training/eval mode (affects Dropout)
model.train();
model.eval();
```

**Available layers:**
- `Linear(in_features, out_features)` - Fully connected layer
- `ReLU()` - ReLU activation
- `Sigmoid()` - Sigmoid activation
- `Tanh()` - Tanh activation
- `Softmax(dim)` - Softmax activation
- `LogSoftmax(dim)` - Log-softmax activation
- `Dropout(p)` - Dropout regularization
- `Conv2d(in_channels, out_channels, kernel_size, stride, padding)` - 2D convolution
- `MaxPool2d(kernel_size, stride)` - 2D max pooling
- `AvgPool2d(kernel_size, stride)` - 2D average pooling
- `BatchNorm2d(num_features, eps, momentum)` - 2D batch normalization
- `LayerNorm(normalized_shape, eps)` - Layer normalization (for transformers)
- `Flatten()` - Flatten spatial dimensions
- `Embedding(num_embeddings, embedding_dim)` - Embedding lookup table for NLP
- `LSTM(input_size, hidden_size, batch_first)` - LSTM recurrent layer
- `GRU(input_size, hidden_size, batch_first)` - GRU recurrent layer
- `MultiHeadAttention(embed_dim, num_heads)` - Multi-head attention for transformers
- `Sequential({...})` - Container for chaining layers

### Loss Functions

```cpp
#include "loss.h"

CrossEntropyLoss criterion;           // For multi-class classification
MSELoss mse_criterion;                // For regression (L2 loss)
L1Loss l1_criterion;                  // For regression (mean absolute error)
SmoothL1Loss smooth_l1;               // Huber loss (robust to outliers)
SmoothL1Loss smooth_l1_custom(0.5f);  // Huber loss with custom beta threshold
NLLLoss nll_criterion;                // Negative log likelihood
BCELoss bce_criterion;                // Binary cross entropy (expects probabilities)
BCEWithLogitsLoss bce_logits;         // BCE with built-in sigmoid (numerically stable)
KLDivLoss kl_div;                     // KL divergence (for knowledge distillation)
FocalLoss focal;                      // Focal loss for imbalanced multi-class
FocalLoss focal_custom(2.0f, 0.25f);  // gamma=2, alpha=0.25
BinaryFocalLoss binary_focal;         // Binary focal loss for imbalanced binary

// Compute loss
auto loss = criterion(predictions, targets);
loss->backward();
```

### Optimizers

```cpp
#include "optimizer.h"

// Stochastic Gradient Descent with momentum
SGD optimizer(model.parameters(), /*lr=*/0.01, /*momentum=*/0.9);

// Adam optimizer
Adam optimizer(model.parameters(), /*lr=*/0.001, /*beta1=*/0.9, /*beta2=*/0.999);

// AdamW optimizer (Adam with decoupled weight decay)
AdamW optimizer(model.parameters(), /*lr=*/0.001, /*beta1=*/0.9, /*beta2=*/0.999,
                /*eps=*/1e-8, /*weight_decay=*/0.01);

// RMSprop optimizer
RMSprop optimizer(model.parameters(), /*lr=*/0.01, /*alpha=*/0.99, /*eps=*/1e-8,
                  /*momentum=*/0.0, /*weight_decay=*/0.0);

// Training step
optimizer.zero_grad();      // Clear gradients
auto loss = criterion(model.forward(x), y);
loss->backward();           // Compute gradients
optimizer.step();           // Update parameters
```

### Learning Rate Schedulers

```cpp
#include "optimizer.h"

SGD optimizer(model.parameters(), 0.1f, 0.9f);

// Step decay: multiply LR by gamma every step_size epochs
StepLR scheduler(&optimizer, 30, 0.1f);  // decay by 0.1 every 30 epochs

// Exponential decay: multiply LR by gamma every epoch
ExponentialLR scheduler(&optimizer, 0.95f);

// Cosine annealing: smoothly decrease LR to eta_min over T_max epochs
CosineAnnealingLR scheduler(&optimizer, 100, 0.0001f);

// Cosine with warm restarts: reset LR periodically
CosineAnnealingWarmRestarts scheduler(&optimizer, 10, 2, 0.0001f);  // T_0=10, T_mult=2

// Reduce on plateau: reduce LR when metric stops improving
ReduceLROnPlateau scheduler(&optimizer, 0.1f, 10, 0.0001f, true);  // factor, patience, min_lr, mode_min

// Call at end of each epoch
for (int epoch = 0; epoch < num_epochs; epoch++) {
    train_one_epoch();
    scheduler.step();                    // For most schedulers
    // scheduler.step(validation_loss);  // For ReduceLROnPlateau
}
```

### Gradient Clipping

```cpp
#include "optimizer.h"

auto params = model.parameters();

// Clip by norm: scale gradients if total norm > max_norm
loss->backward();
clip_grad_norm_(params, 1.0f);  // max_norm = 1.0
optimizer.step();

// Clip by value: clamp each gradient to [-clip_value, clip_value]
loss->backward();
clip_grad_value_(params, 0.5f);  // clip to [-0.5, 0.5]
optimizer.step();

// Get gradient norm (useful for monitoring)
float grad_norm = get_grad_norm(params);
```

### Disabling Gradient Tracking

Use `NoGradGuard` for inference to improve performance:

```cpp
{
    NoGradGuard no_grad;  // Disables gradient tracking in this scope
    auto output = model.forward(input);  // No computation graph built
    // ~3x faster inference
}
// Gradients automatically re-enabled when guard goes out of scope
```

### Data Loading

**MNIST:**
```cpp
#include "mnist.h"

// Load MNIST dataset
auto train_data = load_mnist_train("data/");
auto test_data = load_mnist_test("data/");

// Create data loader with batching and shuffling
DataLoader train_loader(train_data, /*batch_size=*/64, /*shuffle=*/true);
DataLoader test_loader(test_data, /*batch_size=*/64, /*shuffle=*/false);

// Iterate over batches
while (train_loader.has_next()) {
    auto [images, labels] = train_loader.next_batch();
    // images: [batch_size, 784], labels: [batch_size]
}

train_loader.reset();  // Reset for next epoch
```

**CIFAR-10:**
```cpp
#include "cifar10.h"

// Download CIFAR-10 binary files to data/ directory first
// Files needed: data_batch_1.bin ... data_batch_5.bin, test_batch.bin

// Load CIFAR-10 dataset (automatically normalized with ImageNet stats)
auto train_data = load_cifar10_train("data/");  // 50,000 images
auto test_data = load_cifar10_test("data/");    // 10,000 images

// Create data loader with augmentation (random crop + horizontal flip)
CIFAR10DataLoader train_loader(train_data, 64, /*shuffle=*/true, /*augment=*/true);
CIFAR10DataLoader test_loader(test_data, 64, /*shuffle=*/false, /*augment=*/false);

while (train_loader.has_next()) {
    auto [images, labels] = train_loader.next_batch();
    // images: [batch_size, 3, 32, 32], labels: [batch_size]
}

// Get class name
const char* name = cifar10_class_name(3);  // "cat"
```

### Data Augmentation

```cpp
// For images with shape [N, C, H, W] or [C, H, W]

// Horizontal flip
auto flipped = img->flip_horizontal();
auto maybe_flipped = img->random_flip_horizontal(0.5f);  // 50% chance

// Padding and cropping
auto padded = img->pad2d(4);  // Zero-pad by 4 pixels on each side
auto cropped = padded->crop(2, 2, 32, 32);  // Crop from (top=2, left=2)
auto random_cropped = padded->random_crop(32, 32);  // Random position

// Standard CIFAR-10 augmentation pipeline
auto augmented = img->pad2d(4)->random_crop(32, 32)->random_flip_horizontal(0.5f);
```

## Complete Training Example

```cpp
#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "mnist.h"

int main() {
    // Load data
    auto train_data = load_mnist_train("data/");
    DataLoader train_loader(train_data, 64, true);

    // Define model
    Sequential model({
        new Linear(784, 256),
        new ReLU(),
        new Linear(256, 10)
    });

    // Loss and optimizer
    CrossEntropyLoss criterion;
    SGD optimizer(model.parameters(), 0.01, 0.9);

    // Training loop
    for (int epoch = 0; epoch < 10; epoch++) {
        train_loader.reset();
        float total_loss = 0;
        int batches = 0;

        while (train_loader.has_next()) {
            auto [images, labels] = train_loader.next_batch();

            optimizer.zero_grad();
            auto output = model.forward(images);
            auto loss = criterion(output, labels);
            loss->backward();
            optimizer.step();

            total_loss += loss->item();
            batches++;
        }

        printf("Epoch %d, Loss: %.4f\n", epoch + 1, total_loss / batches);
    }

    return 0;
}
```

## CNN Example (MNIST)

A convolutional neural network example is included in `cnn_mnist.cpp`:

```bash
make cnn_mnist
./cnn_mnist
```

Architecture:
```
Conv2d(1, 16, 3) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
Conv2d(16, 32, 3) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
Flatten -> Linear(1568, 128) -> ReLU -> Linear(128, 10)
```

Results: **~98.5% test accuracy** after 1 epoch (~207k parameters)

## CNN Example (CIFAR-10)

A VGG-style CNN for CIFAR-10 color image classification in `cnn_cifar10.cpp`:

```bash
# Download CIFAR-10 data
mkdir -p data && cd data
curl -LO https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz
mv cifar-10-batches-bin/*.bin .
cd ..

# Build and run
make cnn_cifar10
./cnn_cifar10
```

Architecture:
```
Block 1: Conv(3,32,3) -> BN -> ReLU -> Conv(32,32,3) -> BN -> ReLU -> MaxPool(2)
Block 2: Conv(32,64,3) -> BN -> ReLU -> Conv(64,64,3) -> BN -> ReLU -> MaxPool(2)
Block 3: Conv(64,128,3) -> BN -> ReLU -> Conv(128,128,3) -> BN -> ReLU -> MaxPool(2)
Classifier: Flatten -> Linear(2048,256) -> ReLU -> Dropout(0.5) -> Linear(256,10)
```

Features:
- **~815k parameters** (~3.1 MB)
- Adam optimizer with cosine annealing learning rate schedule
- Data augmentation: random crop (pad=4) + horizontal flip
- im2col + GEMM + OpenMP optimized convolution (~1.5s/batch on Apple M1)
- Expected: **~75-85% test accuracy** after 20 epochs (~19 min/epoch)

## Transformer Example

A simple transformer language model example is included in `transformer_example.cpp`:

```bash
make transformer_example
./transformer_example
```

Architecture:
```
Embedding -> Positional Encoding -> 2x Transformer Blocks -> Linear
Each block: MultiHeadAttention -> LayerNorm -> FFN -> LayerNorm
```

The model learns to predict the next character in a simple repeating pattern ("abcdef"):
- **17k parameters** with embed_dim=32, num_heads=4, 2 layers
- Trains to near-zero loss in ~100 epochs
- Generates perfect sequences from any starting prompt

Sample output:
```
Prompt 'abc' -> abcdefabcdefabcdefabcdef...
Prompt 'f'   -> fabcdefabcdefabcdefabcdef...
```

## Performance

The framework includes several optimizations:

- **SIMD Vectorization**: ARM NEON (Apple Silicon) and x86 SSE/AVX support
- **Blocked Matrix Multiplication**: Cache-friendly 32x32 block tiling
- **im2col + GEMM Convolution**: Converts conv2d to optimized matrix multiplication
- **OpenMP Parallelization**: Multi-threaded convolution and GEMM operations
- **NoGradGuard**: Skip computation graph building during inference
- **O3 Optimization**: Aggressive compiler optimizations enabled

### macOS OpenMP Setup
```bash
brew install libomp
```

### Benchmarks (Apple M1, 10 threads)

| Model | Batch Time | Epoch Time |
|-------|------------|------------|
| Simple CNN (2 conv) | ~76 ms | ~1 min |
| VGG-style (6 conv) | ~1.5 s | ~19 min |

Typical MNIST training: ~18 seconds/epoch on Apple M1.

## Build Options

```bash
make            # Build optimized release
make debug      # Build with debug symbols
make clean      # Remove build artifacts
make run        # Build and run
```

## Requirements

- C++17 compatible compiler (g++, clang++)
- No external dependencies

---

## Platform Architecture

whitematter includes a self-service ML training platform where users can upload custom datasets, design models with natural language, and deploy trained models via API.

### Workflow

1. **Upload** - Users upload labeled datasets (ZIP of folders, one folder per class)
2. **Design** - Describe the model in natural language; LLM suggests architecture
3. **Train** - Generated C++ code compiles and trains the model
4. **Deploy** - Trained models are exposed via REST API

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React)                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Datasets │ │  Design  │ │ Presets  │ │  Models  │ │ Predict  │          │
│  │   Tab    │ │   Tab    │ │   Tab    │ │   Tab    │ │   Tab    │          │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
└───────┼────────────┼────────────┼────────────┼────────────┼─────────────────┘
        │            │            │            │            │
        ▼            ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FastAPI SERVER (server.py)                           │
│                                                                              │
│  /datasets/upload    /design/suggest    /train         /models    /predict  │
│  /datasets/{id}      /design/validate   /train/custom  /models/{id}         │
│                      /design/refine     /train/{job}              /api/{id} │
└───────┬────────────────────┬────────────────────┬───────────────────┬───────┘
        │                    │                    │                   │
        ▼                    ▼                    ▼                   ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐    ┌─────────────┐
│ DatasetManager│    │  LLMService   │    │ CodeGenerator │    │ whitematter │
│               │    │  (Claude API) │    │  (Templates)  │    │   (C++)     │
│ - extract ZIP │    │               │    │               │    │             │
│ - detect type │    │ - suggest     │    │ - generate    │    │ - Tensor    │
│ - preprocess  │    │ - refine      │    │   train.cpp   │    │ - Layers    │
└───────┬───────┘    └───────────────┘    │ - Makefile    │    │ - Optimizer │
        │                                  └───────┬───────┘    │ - Loss      │
        ▼                                          │            └──────┬──────┘
┌───────────────┐                                  ▼                   │
│ Preprocessors │                          ┌───────────────┐           │
│               │                          │   Compiler    │           │
│ - image       │                          │               │           │
│ - text        │                          │ - make train  │           │
│ - tabular     │                          │ - subprocess  │           │
└───────┬───────┘                          └───────┬───────┘           │
        │                                          │                   │
        ▼                                          ▼                   │
┌─────────────────────────────────────────────────────────────────────┴───────┐
│                              FILE SYSTEM                                     │
│                                                                              │
│  uploads/{dataset_id}/          generated/{job_id}/       models/           │
│    raw/                           train.cpp                 {model}.bin     │
│    processed/                     Makefile                  {model}.json    │
│      train_images.bin             build.log                                 │
│      train_labels.bin             train (executable)                        │
│      test_images.bin                                                        │
│      test_labels.bin                                                        │
│    metadata.json                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow (Custom Training)

```
1. Upload    2. Design       3. Generate     4. Compile    5. Train      6. Predict
   ZIP          Prompt          C++ Code        Binary        Model         API
    │             │                │               │            │             │
    ▼             ▼                ▼               ▼            ▼             ▼
┌───────┐   ┌─────────┐      ┌──────────┐    ┌────────┐   ┌────────┐   ┌────────┐
│ ZIP   │──▶│ Claude  │─────▶│ Template │───▶│  g++   │──▶│ ./train│──▶│ model  │
│ file  │   │   API   │      │ Engine   │    │  make  │   │        │   │ .bin   │
└───────┘   └─────────┘      └──────────┘    └────────┘   └────────┘   └────────┘
    │             │                │
    ▼             ▼                ▼
 Extract     Architecture      train.cpp
 & Process   JSON              Makefile
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| C++ Engine | `tensor.cpp`, `layer.cpp`, `optimizer.cpp`, `loss.cpp` | Core autograd & NN ops |
| Python Bindings | `whitematter_py.cpp` | pybind11 inference wrapper |
| Server | `server.py` | FastAPI REST endpoints |
| Dataset Manager | `dataset_manager.py` | Upload, extract, preprocess |
| Image Processor | `preprocessing/image_processor.py` | Resize, normalize, binarize |
| Code Generator | `codegen/generator.py` | Architecture JSON to C++ |
| LLM Service | `llm/service.py` | Claude API for design |
| Frontend | `frontend/src/` | React UI |

### API Endpoints

**Dataset Management:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/datasets/upload` | POST | Upload ZIP dataset |
| `/datasets` | GET | List all datasets |
| `/datasets/{id}` | GET | Get dataset metadata |
| `/datasets/{id}` | DELETE | Delete dataset |

**Architecture Design:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/design/suggest` | POST | Get LLM architecture suggestion |
| `/design/validate` | POST | Validate architecture JSON |
| `/design/refine` | POST | Refine architecture with feedback |

**Training:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/train` | POST | Start preset training |
| `/train/custom` | POST | Start custom training |
| `/train/{job_id}` | GET | Get training status |
| `/train/{job_id}` | DELETE | Cancel training |

**Inference:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict with preset model |
| `/api/{model_id}/predict` | POST | Predict with custom model |
| `/api/{model_id}/info` | GET | Get model metadata |

### Running the Platform

```bash
# Start the backend server
python server.py

# In another terminal, start the frontend
cd frontend && npm run dev

# Open http://localhost:5173 in browser
```

### Architecture JSON Schema

Models are defined using a JSON schema:

```json
{
  "name": "my_classifier",
  "description": "CNN for image classification",
  "data_type": "image",
  "input_shape": [3, 32, 32],
  "num_classes": 10,
  "layers": [
    {"type": "conv2d", "params": {"in_channels": 3, "out_channels": 32, "kernel_size": 3, "padding": 1}},
    {"type": "batchnorm2d", "params": {"num_features": 32}},
    {"type": "relu", "params": {}},
    {"type": "maxpool2d", "params": {"kernel_size": 2}},
    {"type": "flatten", "params": {}},
    {"type": "linear", "params": {"in_features": 8192, "out_features": 10}}
  ],
  "training": {
    "optimizer": {"type": "adam", "params": {"learning_rate": 0.001}},
    "scheduler": {"type": "cosine", "params": {"T_max": 50}},
    "epochs": 50,
    "batch_size": 64
  }
}
```

**Available layer types:** `conv2d`, `linear`, `relu`, `sigmoid`, `tanh`, `softmax`, `dropout`, `flatten`, `maxpool2d`, `avgpool2d`, `batchnorm2d`, `layernorm`, `embedding`, `lstm`, `gru`

---

## TODO

Future improvements to make this framework more extensive:

### Layers
- [x] Conv2d - 2D convolutional layer
- [x] MaxPool2d - Max pooling layer
- [x] AvgPool2d - Average pooling layer
- [x] Flatten - Flatten spatial dimensions
- [x] BatchNorm2d - Batch normalization
- [x] LayerNorm - Layer normalization
- [x] Embedding - Embedding layer for NLP
- [x] LSTM - Long Short-Term Memory recurrent layer
- [x] GRU - Gated Recurrent Unit layer
- [x] MultiHeadAttention - Transformer multi-head attention

### Tensor Operations
- [x] Broadcasting for add/sub/mul/div operations
- [x] Concatenate / Stack tensors
- [x] Squeeze / Unsqueeze dimensions
- [x] Permute / View operations
- [x] Convolution operations (conv2d)
- [x] Pooling operations (maxpool2d)
- [x] Batch matrix multiplication (bmm)
- [x] Max / Min reduction and element-wise operations
- [x] Argmax / Argmin operations

### Optimizers
- [x] AdamW - Adam with decoupled weight decay
- [x] RMSprop - RMSprop with optional momentum
- [x] Learning rate schedulers (StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau)
- [x] Gradient clipping (clip_grad_norm_, clip_grad_value_)

### Loss Functions
- [x] BCELoss / BCEWithLogitsLoss - Binary cross entropy
- [x] L1Loss - Mean absolute error
- [x] SmoothL1Loss - Huber loss
- [x] KLDivLoss - KL divergence
- [x] FocalLoss / BinaryFocalLoss - For imbalanced classification

### Data & Training
- [x] CIFAR-10 data loader with normalization
- [x] Data augmentation (random flip, crop, padding)
- [ ] Multi-threaded data loading
- [ ] Mixed precision training (fp16)
- [ ] Gradient accumulation
- [ ] Early stopping

### Infrastructure
- [ ] GPU support (CUDA/Metal)
- [ ] Model summary / parameter count
- [ ] TensorBoard-style logging
- [ ] ONNX export
- [ ] Pretrained model zoo
- [ ] Unit tests

### Examples
- [x] CIFAR-10 image classification (cnn_cifar10.cpp)
- [x] Simple CNN example (cnn_mnist.cpp)
- [x] Transformer language model (transformer_example.cpp)
- [ ] RNN text generation
- [ ] Autoencoder
- [ ] GAN
