# Architecture

WhiteMatter is a pure-Python deep learning framework. Every operation — forward pass, backward pass, im2col convolution — is implemented from scratch using only NumPy.

## Core: Tensor + Autograd

`tensor.py` implements `Tensor`, the fundamental data type. Each tensor wraps a NumPy array and optionally tracks a computation graph for automatic differentiation.

- **Forward**: operations like `matmul`, `relu`, `conv2d` produce new tensors and record a `_backward` closure
- **Backward**: `tensor.backward()` walks the graph in reverse topological order, calling each `_backward` to accumulate gradients in `.grad`
- **Context manager**: `autograd.no_grad()` disables gradient tracking for inference

## nn: Layers and Losses

`nn/module.py` defines `Module`, the base class. It uses `__setattr__` to auto-register child modules and parameters, providing `parameters()`, `train()`, `eval()`, and `zero_grad()`.

Layer implementations:

| File | Layers |
|------|--------|
| `linear.py` | Linear |
| `conv.py` | Conv2d, ConvTranspose2d (im2col + GEMM) |
| `normalization.py` | BatchNorm2d, LayerNorm, GroupNorm, RMSNorm |
| `attention.py` | MultiHeadAttention, GroupedQueryAttention |
| `recurrent.py` | LSTM, GRU |
| `activation.py` | ReLU, GELU, SiLU, Mish, Sigmoid, Tanh, Softmax, LogSoftmax |
| `pooling.py` | MaxPool2d, AvgPool2d, AdaptiveAvgPool2d |
| `embedding.py` | Embedding |
| `positional.py` | SinusoidalPositionalEncoding |
| `dropout.py` | Dropout |
| `container.py` | Sequential |
| `loss.py` | CrossEntropyLoss, MSELoss, BCELoss, FocalLoss, etc. |

### Convolutions

Conv2d and ConvTranspose2d use im2col to reshape input patches into a 2D matrix, then delegate to NumPy's matrix multiply (GEMM). The backward pass computes gradients through the same im2col/col2im pathway.

## optim: Optimizers and Schedulers

`optim/` implements SGD, Adam, AdamW, and RMSprop. Each optimizer stores per-parameter state (momentum buffers, Adam moments) and updates parameters in-place via `step()`.

Learning rate schedulers (StepLR, CosineAnnealing, etc.) wrap an optimizer and adjust `lr` each epoch.

## data: DataLoader

`data/` provides `DataLoader` for batching and shuffling datasets. Datasets are simple objects with `__len__` and `__getitem__`.

## serialization

`serialization.py` saves and loads model state dicts as NumPy `.npz` files.
