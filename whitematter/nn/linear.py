"""Linear (fully connected) layer."""

import numpy as np
from ..tensor import Tensor
from .module import Module


class Linear(Module):
    """Linear transformation: y = x @ W^T + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bound = 1.0 / np.sqrt(in_features)
        self.weight = Tensor(
            np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32),
            requires_grad=True,
        )
        if bias:
            self.bias = Tensor(
                np.random.uniform(-bound, bound, (out_features,)).astype(np.float32),
                requires_grad=True,
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weight.transpose())
        if self.bias is not None:
            out = out.add(self.bias)
        return out

    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features}, bias={self.bias is not None})"
