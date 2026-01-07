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
├── mnist.h/cpp       # MNIST data loader
├── ml.cpp            # Example training script
└── Makefile          # Build configuration
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
- Arithmetic: `add`, `sub`, `mul`, `div`, `neg`
- Matrix: `matmul`, `transpose`, `reshape`, `slice`
- Activations: `relu`, `sigmoid`, `tanh_`, `softmax`, `log_softmax`
- Reductions: `sum`, `mean`
- Element-wise: `log_`, `exp_`

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
- `Sequential({...})` - Container for chaining layers

### Loss Functions

```cpp
#include "loss.h"

CrossEntropyLoss criterion;           // For classification
MSELoss mse_criterion;                // For regression
NLLLoss nll_criterion;                // Negative log likelihood

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

// Training step
optimizer.zero_grad();      // Clear gradients
auto loss = criterion(model.forward(x), y);
loss->backward();           // Compute gradients
optimizer.step();           // Update parameters
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

## Performance

The framework includes several optimizations:

- **SIMD Vectorization**: ARM NEON (Apple Silicon) and x86 SSE/AVX support
- **Blocked Matrix Multiplication**: Cache-friendly 32x32 block tiling
- **NoGradGuard**: Skip computation graph building during inference
- **O3 Optimization**: Aggressive compiler optimizations enabled

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
