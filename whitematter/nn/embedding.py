"""Embedding layer."""

from __future__ import annotations

import numpy as np
from ..tensor import Tensor, _acc
from .module import Module


class Embedding(Module):
    """Lookup table embedding."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02,
            requires_grad=True,
        )

    def forward(self, indices: Tensor) -> Tensor:
        idx = indices.data.astype(np.int64)
        out_data = self.weight.data[idx]
        result = self.weight._make_result(out_data, [self.weight])
        if result.requires_grad:
            weight = self.weight
            def _backward(grad):
                if weight.requires_grad:
                    g = np.zeros_like(weight.data)
                    np.add.at(g, idx, grad)
                    weight.grad = _acc(weight.grad, g)
            result._grad_fn = _backward
        return result

    def __repr__(self):
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"
