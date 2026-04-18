"""Dropout layer."""

from __future__ import annotations

import numpy as np
from ..tensor import Tensor, _acc
from .module import Module


class Dropout(Module):
    """Inverted dropout: scales during training, identity during eval."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self._training or self.p == 0.0:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32) / (1.0 - self.p)
        result = x._make_result(x.data * mask, [x])
        if result.requires_grad:
            def _backward(grad):
                if x.requires_grad:
                    x.grad = _acc(x.grad, grad * mask)
            result._grad_fn = _backward
        return result

    def __repr__(self):
        return f"Dropout(p={self.p})"
