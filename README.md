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
./build/ml
```

## Framework Structure

```
├── core/                       # C++ core framework
│   ├── tensor.h/cpp            # Core tensor with autograd
│   ├── layer.h/cpp             # Neural network layers
│   ├── loss.h/cpp              # Loss functions
│   ├── optimizer.h/cpp         # Parameter optimizers
│   ├── serialize.h/cpp         # Model save/load
│   ├── dataloader.h/cpp        # Multi-threaded data loading
│   ├── model_zoo.h/cpp         # Pretrained model registry
│   ├── onnx_export.h/cpp       # ONNX format export
│   ├── amp.h                   # Mixed precision training (fp16)
│   └── logging.h               # Training logger and metrics
├── datasets/                   # Dataset loaders
│   ├── mnist.h/cpp             # MNIST data loader
│   └── cifar10.h/cpp           # CIFAR-10 data loader
├── examples/                   # Training examples
│   ├── ml.cpp                  # MLP training
│   ├── cnn_mnist.cpp           # CNN (MNIST)
│   ├── cnn_cifar10.cpp         # CNN (CIFAR-10)
│   └── transformer_example.cpp # Transformer LM
├── bindings/                   # Language bindings
│   └── whitematter_py.cpp      # Python bindings (pybind11)
├── platform/                   # ML platform server
│   ├── server.py               # FastAPI server
│   ├── dataset_manager.py      # Dataset upload/processing
│   ├── codegen/                # C++ code generation
│   ├── llm/                    # Claude API integration
│   └── preprocessing/          # Data preprocessors
├── frontend/                   # React web UI
├── build/                      # Build artifacts (*.o, binaries)
├── models/                     # Trained model files
├── data/                       # Training datasets
└── Makefile                    # Build configuration
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
- `ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)` - 2D transposed convolution (upsampling)
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

**Transposed Convolution (Upsampling):**
```cpp
// ConvTranspose2d upsamples spatial dimensions (used in autoencoders, GANs, segmentation)
// Output size: (H - 1) * stride - 2 * padding + kernel_size + output_padding

// Double spatial dimensions with stride=2
auto upsample = new ConvTranspose2d(64, 32, 4, 2, 1);  // [N, 64, 8, 8] -> [N, 32, 16, 16]

// Decoder for autoencoder
Sequential decoder({
    new ConvTranspose2d(256, 128, 4, 2, 1),  // 4x4 -> 8x8
    new ReLU(),
    new ConvTranspose2d(128, 64, 4, 2, 1),   // 8x8 -> 16x16
    new ReLU(),
    new ConvTranspose2d(64, 3, 4, 2, 1),     // 16x16 -> 32x32
    new Sigmoid()                             // Output in [0, 1]
});
```

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

### Mixed Precision Training

Mixed precision training uses fp16 (half-precision) for faster computation while maintaining fp32 accuracy through loss scaling:

```cpp
#include "amp.h"

Sequential model({
    new Linear(784, 256),
    new ReLU(),
    new Linear(256, 10)
});

SGD optimizer(model.parameters(), 0.001f);
GradScaler scaler;  // Default: init_scale=65536, growth_interval=2000
CrossEntropyLoss criterion;

for (int epoch = 0; epoch < epochs; epoch++) {
    for (auto [x, y] : dataloader) {
        optimizer.zero_grad();

        // Forward pass (in real fp16, this would use half-precision)
        auto output = model.forward(x);
        auto loss = criterion(output, y);

        // Scale loss for numerical stability
        auto scaled_loss = scaler.scale(loss);
        scaled_loss->backward();

        // Unscale gradients and check for overflow
        scaler.unscale(&optimizer);

        // Gradient clipping (optional, works with scaled gradients)
        clip_grad_norm_(model.parameters(), 1.0f);

        // Step optimizer only if gradients are finite
        scaler.step(&optimizer, true);  // true = already unscaled

        // Update scale factor based on overflow history
        scaler.update();
    }
}
```

**GradScaler API:**
```cpp
// Constructor with custom settings
GradScaler scaler(
    256.0f,    // init_scale: starting scale factor
    2.0f,      // growth_factor: multiply scale by this after growth_interval good steps
    0.5f,      // backoff_factor: multiply scale by this on overflow
    2000,      // growth_interval: steps between scale increases
    true       // enabled: set to false to disable scaling
);

// Core methods
float scale = scaler.get_scale();           // Get current scale factor
auto scaled = scaler.scale(loss);           // Scale a tensor
bool finite = scaler.unscale(&optimizer);   // Unscale gradients, returns true if all finite
scaler.step(&optimizer, already_unscaled);  // Step if gradients are finite
scaler.update();                            // Adjust scale based on overflow history
```

**HalfTensor for memory optimization:**
```cpp
// Store weights in fp16 to save memory (50% reduction)
auto weights = Tensor::randn({1000, 1000}, false);
HalfTensor half_weights(weights);

printf("Original: %zu bytes\n", weights->data.size() * 4);    // 4MB
printf("Half: %zu bytes\n", half_weights.data.size() * 2);    // 2MB

// Convert back to fp32 for computation
auto restored = half_weights.to_float();
```

**FP16 conversion utilities:**
```cpp
// Convert single values
uint16_t h = float_to_half(3.14159f);
float f = half_to_float(h);

// Convert vectors
std::vector<uint16_t> half_data = to_half(float_data);
std::vector<float> float_data = from_half(half_data);
```

### Gradient Accumulation

Gradient accumulation enables training with effectively larger batch sizes when memory is limited. Instead of updating weights after every batch, accumulate gradients over multiple mini-batches:

```cpp
#include "optimizer.h"

Sequential model({
    new Linear(784, 256),
    new ReLU(),
    new Linear(256, 10)
});

SGD optimizer(model.parameters(), 0.01f);
CrossEntropyLoss criterion;
GradientAccumulator accumulator(4);  // Effective batch = mini_batch * 4

for (auto [x, y] : dataloader) {
    auto loss = criterion(model.forward(x), y);

    // Scales loss by 1/4 and calls backward
    accumulator.backward(loss);

    // Step only when we've accumulated 4 batches
    if (accumulator.should_step()) {
        optimizer.step();
        optimizer.zero_grad();
        accumulator.reset();
    }
}
```

**GradientAccumulator API:**
```cpp
// Create accumulator (effective_batch = mini_batch * accumulation_steps)
GradientAccumulator accumulator(4);

// Option 1: Combined scale + backward
accumulator.backward(loss);

// Option 2: Manual control
auto scaled_loss = accumulator.scale(loss);  // loss / accumulation_steps
scaled_loss->backward();
accumulator.increment();

// Check state
accumulator.should_step();              // True when ready for optimizer step
accumulator.is_last_step();             // True on final accumulation step
accumulator.current_step();             // Current step (0 to accumulation_steps-1)
accumulator.get_accumulation_steps();   // Total steps
accumulator.get_scale_factor();         // 1.0 / accumulation_steps

// Reset after optimizer step
accumulator.reset();
```

**Combined with mixed precision:**
```cpp
GradientAccumulator accumulator(4);
GradScaler scaler(256.0f);

for (auto [x, y] : dataloader) {
    auto loss = criterion(model.forward(x), y);

    // Scale for accumulation, then for mixed precision
    auto scaled = scaler.scale(accumulator.scale(loss));
    scaled->backward();
    accumulator.increment();

    if (accumulator.should_step()) {
        scaler.unscale(&optimizer);
        scaler.step(&optimizer, true);
        scaler.update();
        optimizer.zero_grad();
        accumulator.reset();
    }
}
```

### Early Stopping

Early stopping prevents overfitting by halting training when validation metrics stop improving:

```cpp
#include "optimizer.h"

Sequential model({...});
SGD optimizer(model.parameters(), 0.01f);
CrossEntropyLoss criterion;

EarlyStopping early_stopping(10);  // patience = 10 epochs

for (int epoch = 0; epoch < max_epochs; epoch++) {
    // Training
    train_one_epoch(model, train_loader);

    // Validation
    float val_loss = evaluate(model, val_loader);

    // Check if we should stop
    if (early_stopping.step(val_loss)) {
        printf("Early stopping at epoch %d\n", epoch);
        printf("Best val loss: %.4f at epoch %d\n",
               early_stopping.best_metric(), early_stopping.best_epoch());
        break;
    }
}
```

**EarlyStopping API:**
```cpp
// Create early stopping monitor
// patience: epochs to wait after last improvement
// min_delta: minimum change to qualify as improvement
// mode_min: true = lower is better (loss), false = higher is better (accuracy)
EarlyStopping early_stopping(10, 0.001f, true);   // For loss
EarlyStopping early_stopping(10, 0.0f, false);    // For accuracy

// Step and check
bool should_stop = early_stopping.step(metric);

// Query state
early_stopping.best_metric();                // Best value seen
early_stopping.best_epoch();                 // Epoch of best value
early_stopping.epochs_without_improvement(); // Current patience counter
early_stopping.should_stop();                // Whether training should stop

// Reset for new training run
early_stopping.reset();
```

**ModelCheckpoint - Save best model automatically:**
```cpp
#include "optimizer.h"
#include "serialize.h"

EarlyStopping early_stopping(10);
ModelCheckpoint checkpoint("best_model.bin", true);  // mode_min=true for loss

for (int epoch = 0; epoch < max_epochs; epoch++) {
    train_one_epoch();
    float val_loss = evaluate();

    // Save model if validation improved
    if (checkpoint.step(val_loss, &model)) {
        printf("Saved best model (val_loss=%.4f)\n", val_loss);
    }

    // Check early stopping
    if (early_stopping.step(val_loss)) {
        printf("Early stopping at epoch %d\n", epoch);
        break;
    }
}

// Restore best model for inference/testing
checkpoint.restore(&model);
```

### Model Summary

Inspect model architecture, parameter counts, and memory usage:

```cpp
#include "layer.h"

Sequential model({
    new Conv2d(1, 16, 3, 1, 1),
    new BatchNorm2d(16),
    new ReLU(),
    new MaxPool2d(2, 2),
    new Flatten(),
    new Linear(3136, 128),
    new ReLU(),
    new Linear(128, 10)
});

// PyTorch-style layer-by-layer summary with output shapes
model.summary({1, 1, 28, 28});  // Pass input shape for shape tracking
```

**Output:**
```
==============================================================================
Layer (type)                    Output Shape              Param #
==============================================================================
Conv2d(1, 16, kernel_size=3)    [1, 16, 28, 28]           160
BatchNorm2d(16)                  [1, 16, 28, 28]           32
ReLU                             [1, 16, 28, 28]           0
MaxPool2d(kernel_size=2)         [1, 16, 14, 14]           0
Flatten                          [1, 3136]                 0
Linear(3136, 128)                [1, 128]                  401,536
ReLU                             [1, 128]                  0
Linear(128, 10)                  [1, 10]                   1,290
==============================================================================
Total params: 403,018
Trainable params: 403,018
Non-trainable params: 0
==============================================================================
```

**Utility functions:**
```cpp
// Get detailed model info
ModelSummary info = get_model_summary(&model);
printf("Total params: %zu\n", info.total_params);
printf("Trainable: %zu\n", info.trainable_params);
printf("Param memory (fp32): %zu bytes\n", info.param_memory_bytes);
printf("Param memory (fp16): %zu bytes\n", info.param_memory_fp16_bytes);
printf("Gradient memory: %zu bytes\n", info.grad_memory_bytes);
printf("Total training memory: %zu bytes\n", info.total_memory_bytes);

// Convenience functions
size_t total = count_parameters(&model);
size_t trainable = count_trainable_parameters(&model);

// Human-readable formatting
std::string params = format_number(1234567);     // "1,234,567"
std::string memory = format_memory(1024*1024);   // "1.00 MB"

// Simple summary printout
print_model_info(&model, "My CNN");
```

### Training Logger

Log metrics during training with TensorBoard-style tracking and export:

```cpp
#include "logging.h"

TrainingLogger logger("logs", "my_experiment");
logger.set_total_epochs(10);
logger.set_total_steps(100);  // Steps per epoch (for progress bar)

for (int epoch = 0; epoch < 10; epoch++) {
    logger.new_epoch();

    for (int batch = 0; batch < 100; batch++) {
        // Train...
        float loss = train_batch();
        float acc = compute_accuracy();

        // Log batch metrics (for epoch averaging)
        logger.log_batch("loss", loss);
        logger.log_batch("accuracy", acc);

        // Show progress bar
        logger.print_progress();
    }

    // Log epoch-level metrics
    logger.log("train_loss", logger.epoch_mean("loss"));
    logger.log("train_acc", logger.epoch_mean("accuracy"));
    logger.log("lr", optimizer.lr);
    logger.step();

    logger.print_epoch_summary();
}

// Save logs and print summary
logger.save_csv();   // logs/my_experiment_metrics.csv
logger.save_json();  // logs/my_experiment_metrics.json
logger.print_summary();
```

**Console output:**
```
Epoch 3/10 [============>       ] 60% loss: 0.4523 accuracy: 0.8721 (1m 23s)
Epoch 3/10 - loss: 0.4512 accuracy: 0.8734 - 2m 18s

==============================================================================
Training Summary: my_experiment
==============================================================================
Total epochs:  10
Total steps:   1000
Elapsed time:  23m 45s
------------------------------------------------------------------------------
train_loss:     min:   0.1234  max:   1.2345  mean:   0.4567  std:   0.2345
train_acc:      min:   0.7500  max:   0.9500  mean:   0.8750  std:   0.0500
==============================================================================
```

**MetricTracker for statistics:**
```cpp
// Track running statistics for any metric
MetricTracker tracker;
for (float value : batch_losses) {
    tracker.update(value);
}
printf("Mean: %.4f, Std: %.4f, Min: %.4f, Max: %.4f\n",
       tracker.mean(), tracker.std(), tracker.min(), tracker.max());
```

**ProgressBar for loops:**
```cpp
ProgressBar bar(1000, 40, "Training: ");
for (int i = 0; i < 1000; i++) {
    // Do work...
    bar.update();
}
bar.finish();
// Output: Training: [=================>              ] 50% 500/1000 [1.2s < 1.2s]
```

**CSV export format:**
```csv
step,epoch,timestamp,train_loss,train_acc,lr
0,1,12.34,0.9876,0.7500,0.001
1,2,25.67,0.5432,0.8500,0.001
...
```

**JSON export format:**
```json
{
  "experiment": "my_experiment",
  "total_steps": 10,
  "total_epochs": 10,
  "elapsed_seconds": 1234.56,
  "summary": {
    "train_loss": {"min": 0.12, "max": 1.23, "mean": 0.45, "std": 0.23, "last": 0.15}
  },
  "history": [
    {"step": 0, "epoch": 1, "timestamp": 12.34, "train_loss": 0.9876}
  ]
}
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

### Model Zoo

The model zoo provides pre-defined architectures and pretrained weights:

```cpp
#include "model_zoo.h"

// List available models
auto models = list_models();  // {"mnist_mlp", "mnist_cnn", "cifar10_simple", ...}

// Get model info
auto info = ModelZoo::instance().get_info("mnist_cnn");
printf("Params: %zu, Expected accuracy: %.1f%%\n", info.num_params, info.accuracy);

// Create model architecture (random weights)
Sequential* model = create_model("mnist_cnn");

// Load pretrained model (architecture + weights)
Sequential* pretrained = load_pretrained("mnist_cnn");

// Save a trained model to the zoo
ModelZoo::instance().save_to_zoo("mnist_cnn", model);

// Change weights directory (default: "pretrained/")
ModelZoo::instance().set_weights_dir("my_models/");
```

**Available models:**

| Model | Dataset | Input Shape | Params | Expected Accuracy |
|-------|---------|-------------|--------|-------------------|
| `mnist_mlp` | MNIST | [1, 784] | 203K | ~97.5% |
| `mnist_cnn` | MNIST | [1, 1, 28, 28] | 207K | ~98.5% |
| `cifar10_simple` | CIFAR-10 | [1, 3, 32, 32] | 310K | ~75% |
| `cifar10_vgg` | CIFAR-10 | [1, 3, 32, 32] | 3.2M | ~85% |
| `tiny_mlp` | synthetic | [1, 4] | 42 | 100% |

**Training and saving to the zoo:**
```cpp
// Train a model
Sequential* model = create_model("mnist_cnn");
// ... training loop ...

// Save to pretrained/mnist_cnn.bin
ModelZoo::instance().save_to_zoo("mnist_cnn", model);

// Later, load the pretrained model
Sequential* loaded = load_pretrained("mnist_cnn");
```

### ONNX Export

Export models to ONNX format for use with ONNX Runtime, TensorRT, or other frameworks:

```cpp
#include "onnx_export.h"

// Simple export with input shape
Sequential* model = load_pretrained("mnist_cnn");
export_onnx(model, "model.onnx", {1, 1, 28, 28});

// Export with options
ONNXExportOptions options;
options.model_name = "my_model";
options.input_shape = {1, 1, 28, 28};
options.verbose = true;
export_onnx(model, "model.onnx", options);

// Get export info (for debugging)
std::string info = get_onnx_export_info(model, {1, 1, 28, 28});
```

**Supported layers for ONNX export:**
- `Linear` → Gemm
- `Conv2d` → Conv
- `ReLU`, `Sigmoid`, `Tanh` → Relu, Sigmoid, Tanh
- `Softmax` → Softmax
- `Flatten` → Flatten
- `MaxPool2d`, `AvgPool2d` → MaxPool, AveragePool
- `BatchNorm2d` → BatchNormalization
- `Dropout` → Identity (inference mode)

**Using the exported model:**
```python
import onnxruntime as ort
import numpy as np

# Load and run inference
sess = ort.InferenceSession("mnist_cnn.onnx")
x = np.random.randn(1, 1, 28, 28).astype(np.float32)
y = sess.run(None, {"input": x})[0]
print(f"Predicted class: {np.argmax(y)}")
```

### Data Loading

**Basic DataLoader (MNIST):**
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

**ThreadedDataLoader (Multi-threaded with Prefetching):**
```cpp
#include "dataloader.h"

// Convert dataset to generic Dataset struct
Dataset train_dataset{train_data.images, train_data.labels, train_data.num_samples};

// Create threaded data loader
// Args: dataset, batch_size, shuffle, num_workers, prefetch_factor
ThreadedDataLoader train_loader(train_dataset, 64, true, 2, 2);  // 2 workers, prefetch 2 batches each
ThreadedDataLoader test_loader(test_dataset, 64, false, 0);       // 0 workers = synchronous

// Training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    train_loader.reset();  // Shuffles data, starts worker threads
    while (train_loader.has_next()) {
        auto [images, labels] = train_loader.next_batch_pair();
        // Workers prefetch next batches while you train
    }
}
```

The `ThreadedDataLoader` uses background threads to prefetch batches while the main thread processes the current batch, improving training throughput on CPU-bound workloads.

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

## Autoencoder Example

A convolutional autoencoder using ConvTranspose2d for image reconstruction on MNIST:

```bash
make autoencoder
./build/autoencoder
```

Architecture:
```
Encoder: Conv2d(1,16,3,s=2) -> Conv2d(16,32,3,s=2) -> Conv2d(32,64,3,s=2) -> Linear(1024,32)
         28x28 -> 14x14 -> 7x7 -> 4x4 -> 32-dim latent vector

Decoder: Linear(32,1024) -> ConvTranspose2d(64,32) -> ConvTranspose2d(32,16) -> ConvTranspose2d(16,1)
         32-dim latent -> 4x4 -> 7x7 -> 14x14 -> 28x28
```

Features:
- **~85k parameters** with 32-dimensional latent space
- Uses ConvTranspose2d for learned upsampling (decoder)
- MSE loss for pixel-wise reconstruction
- Demonstrates latent space interpolation between digits
- ASCII art visualization of reconstructions

Sample output:
```
Epoch 10/10  Train Loss: 0.012345  Test Loss: 0.012567

Reconstruction Examples:
Original:                    Reconstructed:
    .::---==+++**##%@@           .::---==+++**##%@@
    :::---==+++**##%%@           .::---==+++**##%%@
    ...                          ...

Latent Space Interpolation (digit 3 -> digit 7):
[image 1]  [image 2]  [image 3]  [image 4]  [image 5]
```

## GAN Example

A Deep Convolutional GAN (DCGAN) for generating handwritten digits:

```bash
make gan
./build/gan
```

Architecture:
```
Generator (noise -> image):
  Linear(100, 4096) -> Reshape(256,4,4) -> ConvTranspose2d -> 7x7 -> ConvTranspose2d -> 14x14 -> ConvTranspose2d -> 28x28

Discriminator (image -> real/fake):
  Conv2d(1,64) -> 14x14 -> Conv2d(64,128) -> 7x7 -> Conv2d(128,256) -> 4x4 -> Linear -> sigmoid
```

Features:
- **~1.5M parameters** (Generator: ~1M, Discriminator: ~500K)
- 100-dimensional latent space
- Adam optimizer with beta1=0.5 (standard for GANs)
- Label smoothing (real labels = 0.9) for training stability
- Dropout in discriminator to prevent overfitting
- Demonstrates latent space interpolation and diversity checking

Training dynamics:
- D(x): Discriminator output on real images (should stay ~0.5-0.8)
- D(G(z)): Discriminator output on fake images (should rise from ~0 to ~0.5)
- Balanced training when both losses are similar

Sample output:
```
Epoch 20/20  D_loss: 0.8234  G_loss: 1.2345  D(x): 0.72  D(G(z)): 0.45

Generated samples (epoch 20):
    .::---==+++**##%@@  .::---==+++**##%@@  .::---==+++**##%@@
    :::---==+++**##%%@  :::---==+++**##%%@  :::---==+++**##%%@

Latent Space Interpolation:
[z1] -> [interp1] -> [interp2] -> [interp3] -> [z2]
```

## RNN Text Generation Example

A character-level RNN language model for generating Shakespeare-style text:

```bash
make rnn
./build/rnn_text_gen
```

Architecture:
```
Embedding(vocab_size, 128) -> LSTM(128, 256) -> LSTM(256, 256) -> Dropout(0.3) -> Linear(256, vocab_size)
```

Features:
- **~500K parameters** with 2-layer LSTM and 256 hidden units
- Character-level language modeling (predicts next character)
- Embedded Shakespeare corpus (~2KB) for training
- Temperature-based sampling for text generation:
  - Low temperature (0.5): More conservative, repetitive text
  - Medium temperature (0.8): Balanced creativity and coherence
  - High temperature (1.2): More random, creative text
- Gradient clipping (max norm = 5.0) for stable RNN training
- Custom sequence cross-entropy loss with proper gradient computation

Training output:
```
Epoch 50/50  Loss: 1.2345  Perplexity: 3.44

Generated text (temperature=0.8):
----------------------------------------
ROMEO:
What light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon...
----------------------------------------
```

Sample generation at different temperatures:
```
Temperature 0.5 (conservative):
  "the the the the and the..."

Temperature 0.8 (balanced):
  "What dreams may come when we have shuffled off..."

Temperature 1.2 (creative):
  "Twas brillig sloathy toves did gyre..."
```

## Performance

The framework includes several optimizations:

- **SIMD Vectorization**: ARM NEON (Apple Silicon) and x86 SSE/AVX support
- **Blocked Matrix Multiplication**: Cache-friendly 32x32 block tiling
- **im2col + GEMM Convolution**: Converts conv2d to optimized matrix multiplication
- **OpenMP Parallelization**: Multi-threaded convolution and GEMM operations
- **Threaded Data Loading**: Background workers prefetch batches during training
- **Mixed Precision (fp16)**: GradScaler for loss scaling, HalfTensor for memory optimization
- **Gradient Accumulation**: Train with larger effective batch sizes on limited memory
- **Early Stopping**: Prevent overfitting with automatic training termination
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
make test       # Run unit tests
```

## Unit Tests

Comprehensive unit tests are provided in the `tests/` directory:

```bash
# Run all tests
make test

# Run specific test suites
make test-tensor      # Tensor operations
make test-autograd    # Automatic differentiation
make test-layers      # Neural network layers
make test-loss        # Loss functions
make test-optimizer   # Optimizers and schedulers
```

**Test coverage:**
- **Tensor Operations (39 tests)**: Creation, arithmetic, matrix ops, reductions, shape manipulation
- **Autograd (23 tests)**: Gradient computation for all differentiable operations
- **Layers (42 tests)**: Forward pass, parameters, gradients for all layer types
- **Loss Functions (22 tests)**: Correctness and gradients for all losses
- **Optimizers (26 tests)**: SGD, Adam, AdamW, RMSprop, schedulers, early stopping

Sample output:
```
################################################################################
#                         WHITEMATTER UNIT TESTS                               #
################################################################################

================================================================================
Test Suite: Tensor Operations
================================================================================
  [PASS] zeros (0.01ms)
  [PASS] ones (0.00ms)
  [PASS] matmul_2d (0.52ms)
  ...
--------------------------------------------------------------------------------
Results: 39 passed, 0 failed, 39 total (0.95ms)
================================================================================

################################################################################
TOTAL: 152 passed, 0 failed (0.01s)
################################################################################
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
| C++ Engine | `core/tensor.cpp`, `core/layer.cpp`, etc. | Core autograd & NN ops |
| Data Loader | `core/dataloader.cpp` | Multi-threaded batch prefetching |
| Model Zoo | `core/model_zoo.cpp` | Pretrained model registry |
| ONNX Export | `core/onnx_export.cpp` | Export to ONNX format |
| Mixed Precision | `core/amp.h` | GradScaler, HalfTensor, fp16 utils |
| Training Logger | `core/logging.h` | TrainingLogger, MetricTracker, CSV/JSON |
| Python Bindings | `bindings/whitematter_py.cpp` | pybind11 inference wrapper |
| Server | `platform/server.py` | FastAPI REST endpoints |
| Dataset Manager | `platform/dataset_manager.py` | Upload, extract, preprocess |
| Image Processor | `platform/preprocessing/image_processor.py` | Resize, normalize, binarize |
| Code Generator | `platform/codegen/generator.py` | Architecture JSON to C++ |
| LLM Service | `platform/llm/service.py` | Claude API for design |
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
# Build the C++ framework first
make

# Start the backend server
cd platform && python server.py

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
- [x] ConvTranspose2d - 2D transposed convolution (upsampling for decoders/GANs)
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
- [x] Multi-threaded data loading with prefetching
- [x] Mixed precision training (fp16, GradScaler, HalfTensor)
- [x] Gradient accumulation (GradientAccumulator)
- [x] Early stopping (EarlyStopping, ModelCheckpoint)

### Infrastructure
- [ ] GPU support (CUDA/Metal)
- [x] Model summary / parameter count (summary(), ModelSummary, format utilities)
- [x] TensorBoard-style logging (TrainingLogger, MetricTracker, CSV/JSON export)
- [x] ONNX export
- [x] Pretrained model zoo
- [x] Unit tests (152 tests across tensor, autograd, layers, loss, optimizer)

### Examples
- [x] CIFAR-10 image classification (cnn_cifar10.cpp)
- [x] Simple CNN example (cnn_mnist.cpp)
- [x] Transformer language model (transformer_example.cpp)
- [x] RNN text generation (character-level Shakespeare)
- [x] Autoencoder (convolutional, using ConvTranspose2d)
- [x] GAN (DCGAN for MNIST digit generation)
