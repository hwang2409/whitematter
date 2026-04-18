"""Positional encoding modules."""

from __future__ import annotations

import numpy as np
from ..tensor import Tensor
from .module import Module


class SinusoidalPositionalEncoding(Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        pe = np.zeros((max_len, embed_dim), dtype=np.float32)
        pos = np.arange(max_len, dtype=np.float32).reshape(-1, 1)
        div = np.exp(np.arange(0, embed_dim, 2, dtype=np.float32) * -(np.log(10000.0) / embed_dim))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        self.pe = pe  # not a parameter (fixed)

    def forward(self, x: Tensor) -> Tensor:
        T = x.shape[1] if x.ndim == 3 else x.shape[0]
        pe_tensor = Tensor(self.pe[:T])
        return x + pe_tensor

    def __repr__(self):
        return f"SinusoidalPositionalEncoding(max_len={self.pe.shape[0]}, embed_dim={self.pe.shape[1]})"
