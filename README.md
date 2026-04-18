# whitematter

A pure-Python deep learning framework built on NumPy. Tensors with autograd, 30+ layer types, optimizers, and training utilities — no compiled extensions required.

## Install

```bash
pip install -e .
```

## Quick start

```python
import whitematter as wm
from whitematter import nn, optim

# Tensors with automatic differentiation
a = wm.Tensor.randn(3, 4, requires_grad=True)
b = wm.Tensor.randn(4, 2, requires_grad=True)
c = a.matmul(b).relu().sum()
c.backward()  # gradients flow through the whole graph

# Build a model
model = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid(),
)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for x, y in dataloader:
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
```

## Features

| Category | What's included |
|----------|----------------|
| **Core** | Tensor with autograd, broadcasting, serialization |
| **Layers** | Conv2d (grouped, dilated), ConvTranspose2d, Linear, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Flatten |
| **Normalization** | BatchNorm2d, LayerNorm, GroupNorm, RMSNorm |
| **Attention** | MultiHeadAttention, GroupedQueryAttention, KV cache, sinusoidal positional encoding |
| **Recurrent** | LSTM, GRU |
| **Activations** | ReLU, GELU, SiLU, Mish, Sigmoid, Tanh, Softmax, LogSoftmax |
| **Loss** | CrossEntropy, MSE, L1, SmoothL1, NLL, BCE, BCEWithLogits, KLDiv, Focal, BinaryFocal |
| **Optimizers** | SGD (momentum), Adam, AdamW, RMSprop |
| **Schedulers** | StepLR, ExponentialLR, CosineAnnealing, CosineWarmRestarts, ReduceLROnPlateau |
| **Training** | Gradient clipping, gradient accumulation, early stopping, checkpointing |
| **Data** | DataLoader with batching and shuffling |

## Project structure

```
whitematter/          Python library
  tensor.py           Tensor with autograd
  autograd.py         Gradient context management
  nn/                 Layers, losses, containers
  optim/              Optimizers and LR schedulers
  data/               DataLoader
  serialization.py    Save/load checkpoints
examples/             Training scripts (ResNet, GPT, GAN, etc.)
tests/                Unit tests with numerical gradient checks
data/                 Training data (MNIST, CIFAR-10, Shakespeare, etc.)
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

98 tests covering tensors, autograd, layers, loss functions, optimizers, and numerical gradient verification.

## License

MIT
