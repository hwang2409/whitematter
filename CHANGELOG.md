# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed


## [0.5.0] - 2026-03-12

### Added
- **Web platform**: Full-stack ML training platform with FastAPI backend and Next.js frontend
- **AI architecture designer**: Describe models in natural language; Claude suggests architectures
- **Live training dashboard**: Real-time training charts, stat cards, recent activity, quick actions
- **One-click deploy**: Deploy trained models to AWS EC2 as inference APIs
- **Model cards**: Browse, inspect, and manage trained models with metadata
- **Predict playground**: Interactive inference UI with animations
- **Settings page**: Light/dark theme toggle, masked secrets, connection indicator
- **Authentication system**: JWT-based auth with user registration and login
- **S3-compatible storage**: Upload datasets via R2/B2/S3 or import from URL/Hugging Face
- **Docker deployment**: Multi-stage Dockerfile, docker-compose for dev (SQLite) and prod (PostgreSQL)
- **Python bindings**: Full training API exposed via pybind11 (`import whitematter as wm`)
- **Early stopping**: `EarlyStopping` and `ModelCheckpoint` for automatic training termination
- **GAN example**: DCGAN for generating handwritten digits with latent space interpolation
- **RNN example**: Character-level LSTM language model with temperature-based text generation
- **Autoencoder example**: Convolutional autoencoder with ConvTranspose2d for image reconstruction

### Changed
- Unified frontend design system under MUI with accent theme and vertical sidebar
- Server modularized into routes, config, schemas, and dependencies

### Fixed
- Early stopping `mode_max` behavior under `-ffast-math` compiler flag

## [0.4.0] - 2026-02-15

### Added
- **ONNX import/export**: Bidirectional ONNX support for interoperability with other frameworks
- **FP16 export**: Export model initializers in Float16 for edge-friendly models
- **Device auto-selection**: `Device::auto_()` picks Metal > CUDA > CPU automatically
- **CUDA backend**: cuBLAS-accelerated matmul and batched matmul on Linux/cloud GPUs

## [0.3.0] - 2026-01-20

### Added
- **Metal backend**: GPU-accelerated matmul on macOS via Metal compute shaders
- **Mixed precision training**: `GradScaler`, `HalfTensor`, and fp16 conversion utilities
- **Gradient accumulation**: Train with larger effective batch sizes on limited memory
- **Training logger**: TensorBoard-style metric tracking with CSV/JSON export
- **Model summary**: PyTorch-style layer-by-layer summary with output shapes and param counts
- **Gradient clipping**: `clip_grad_norm_` and `clip_grad_value_` utilities
- **Learning rate schedulers**: StepLR, ExponentialLR, CosineAnnealing, CosineWarmRestarts, ReduceLROnPlateau

## [0.2.0] - 2025-12-10

### Added
- **CNN layers**: Conv2d, ConvTranspose2d, MaxPool2d, AvgPool2d, BatchNorm2d
- **Transformer layers**: MultiHeadAttention, LayerNorm, Embedding
- **Recurrent layers**: LSTM, GRU with batch_first support
- **Additional losses**: FocalLoss, BinaryFocalLoss, BCEWithLogitsLoss, KLDivLoss, SmoothL1Loss
- **Additional optimizers**: AdamW (decoupled weight decay), RMSprop
- **Data augmentation**: Random crop, horizontal flip, padding for image tensors
- **CIFAR-10 support**: Dataset loader with ImageNet normalization
- **Model zoo**: Pre-defined architectures with pretrained weight management
- **Threaded data loading**: Background workers prefetch batches during training

## [0.1.0] - 2025-11-01

### Added
- Core tensor library with automatic differentiation (autograd)
- SIMD-optimized operations (ARM NEON, x86 AVX/FMA)
- OpenMP-parallelized matrix multiplication and convolutions
- Basic layers: Linear, ReLU, Sigmoid, Tanh, Softmax, Dropout, Sequential
- Loss functions: CrossEntropyLoss, MSELoss, L1Loss, NLLLoss, BCELoss
- Optimizers: SGD (with momentum), Adam
- MNIST dataset loader and MLP training example
- Model serialization (save/load)
- im2col + GEMM convolution implementation
- CMake and Makefile build systems

[0.5.0]: https://github.com/hwang2409/whitematter/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/hwang2409/whitematter/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/hwang2409/whitematter/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/hwang2409/whitematter/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hwang2409/whitematter/releases/tag/v0.1.0
