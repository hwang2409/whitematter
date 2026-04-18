# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-04-18

Complete rewrite from C++ to pure Python.

### Added
- Pure-Python tensor with full autograd (backward through entire computation graph)
- NumPy-style broadcasting
- 30+ layer types: Conv2d, ConvTranspose2d, Linear, BatchNorm2d, LayerNorm, GroupNorm, RMSNorm, MultiHeadAttention, GroupedQueryAttention, LSTM, GRU, Embedding, and more
- 10 loss functions: CrossEntropy, MSE, BCE, Focal, KLDiv, SmoothL1, etc.
- 4 optimizers: SGD (momentum), Adam, AdamW, RMSprop
- 5 LR schedulers: StepLR, ExponentialLR, CosineAnnealing, CosineWarmRestarts, ReduceLROnPlateau
- DataLoader with batching and shuffling
- Model serialization via NumPy `.npz`
- Gradient clipping and accumulation
- KV cache for autoregressive inference
- 98 unit tests with numerical gradient verification
- Example training scripts: ResNet-18, GPT, GAN, autoencoder, RNN text generation, transformer

### Removed
- C++ core, CUDA/Metal backends, pybind11 bindings
- Web platform (FastAPI + Next.js)
- Docker deployment configs
- ONNX export/import

[1.0.0]: https://github.com/hwang2409/whitematter/releases/tag/v1.0.0
