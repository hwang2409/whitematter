"""Container modules: Sequential, Flatten, Upsample."""

import numpy as np
from ..tensor import Tensor, _acc
from .module import Module
from typing import List


class Sequential(Module):
    """A sequential container of modules."""

    def __init__(self, *layers):
        super().__init__()
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = layers[0]
        for i, layer in enumerate(layers):
            self.register_module(str(i), layer)
        self._layer_list: List[Module] = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._layer_list:
            x = layer(x)
        return x

    def __getitem__(self, idx):
        return self._layer_list[idx]

    def __len__(self):
        return len(self._layer_list)

    def __repr__(self):
        lines = ["Sequential("]
        for i, layer in enumerate(self._layer_list):
            lines.append(f"  ({i}): {layer}")
        lines.append(")")
        return "\n".join(lines)


class Flatten(Module):
    """Flatten dimensions from start_dim to end_dim."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)

    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"


class Upsample(Module):
    """Upsample by scale_factor using nearest neighbor interpolation."""

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        s = self.scale_factor
        data = x.data
        N, C, H, W = data.shape
        out = data.repeat(s, axis=2).repeat(s, axis=3)
        result = x._make_result(out, [x])
        if result.requires_grad:
            def _backward(grad):
                if x.requires_grad:
                    g = grad.reshape(N, C, H, s, W, s).sum(axis=(3, 5))
                    x.grad = _acc(x.grad, g)
            result._grad_fn = _backward
        return result

    def __repr__(self):
        return f"Upsample(scale_factor={self.scale_factor})"
