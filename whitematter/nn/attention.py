"""Attention modules: MultiHeadAttention, GroupedQueryAttention, KVCache."""

from __future__ import annotations

import numpy as np
from ..tensor import Tensor
from .module import Module
from .linear import Linear


class KVCache:
    """Key-value cache for efficient autoregressive generation."""

    def __init__(self, batch_size: int, max_seq_len: int, embed_dim: int):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.k_cache = np.zeros((batch_size, max_seq_len, embed_dim), dtype=np.float32)
        self.v_cache = np.zeros((batch_size, max_seq_len, embed_dim), dtype=np.float32)
        self.current_len = 0

    def append(self, k: Tensor, v: Tensor):
        seq_len = k.shape[1] if k.ndim == 3 else 1
        end = self.current_len + seq_len
        if k.ndim == 3:
            self.k_cache[:, self.current_len:end, :] = k.data
            self.v_cache[:, self.current_len:end, :] = v.data
        else:
            self.k_cache[:, self.current_len, :] = k.data
            self.v_cache[:, self.current_len, :] = v.data
        self.current_len = end

    def keys(self) -> Tensor:
        return Tensor(self.k_cache[:, :self.current_len, :])

    def values(self) -> Tensor:
        return Tensor(self.v_cache[:, :self.current_len, :])

    def reset(self):
        self.current_len = 0


class MultiHeadAttention(Module):
    """Multi-head scaled dot-product attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.W_q = Linear(embed_dim, embed_dim, bias=False)
        self.W_k = Linear(embed_dim, embed_dim, bias=False)
        self.W_v = Linear(embed_dim, embed_dim, bias=False)
        self.W_o = Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: Tensor, mask: Tensor = None, kv_cache: KVCache = None) -> Tensor:
        N, T, E = x.shape
        H, D = self.num_heads, self.head_dim

        Q = self.W_q(x)  # (N, T, E)
        K = self.W_k(x)
        V = self.W_v(x)

        if kv_cache is not None:
            kv_cache.append(K, V)
            K = kv_cache.keys()
            V = kv_cache.values()

        S = K.shape[1]  # key sequence length (may differ from T with cache)

        # Reshape: (N, T, H, D) -> (N, H, T, D)
        Q = Q.reshape(N, T, H, D).permute(0, 2, 1, 3)
        K = K.reshape(N, S, H, D).permute(0, 2, 1, 3)
        V = V.reshape(N, S, H, D).permute(0, 2, 1, 3)

        # Attention scores: (N, H, T, D) @ (N, H, D, S) -> (N, H, T, S)
        scores = Q.matmul(K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores + mask

        attn = scores.softmax(dim=-1)

        # Context: (N, H, T, S) @ (N, H, S, D) -> (N, H, T, D)
        ctx = attn.matmul(V)

        # Reshape back: (N, H, T, D) -> (N, T, E)
        ctx = ctx.permute(0, 2, 1, 3).reshape(N, T, E)

        return self.W_o(ctx)

    def __repr__(self):
        return f"MultiHeadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"


class GroupedQueryAttention(Module):
    """Grouped-query attention: fewer KV heads than query heads."""

    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.groups = num_heads // num_kv_heads

        self.W_q = Linear(embed_dim, embed_dim, bias=False)
        self.W_k = Linear(embed_dim, self.kv_dim, bias=False)
        self.W_v = Linear(embed_dim, self.kv_dim, bias=False)
        self.W_o = Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        N, T, E = x.shape
        H, D = self.num_heads, self.head_dim
        KVH = self.num_kv_heads

        Q = self.W_q(x).reshape(N, T, H, D).permute(0, 2, 1, 3)       # (N, H, T, D)
        K = self.W_k(x).reshape(N, T, KVH, D).permute(0, 2, 1, 3)     # (N, KVH, T, D)
        V = self.W_v(x).reshape(N, T, KVH, D).permute(0, 2, 1, 3)     # (N, KVH, T, D)

        # Repeat KV heads to match query heads
        # (N, KVH, T, D) -> (N, KVH, 1, T, D) -> (N, KVH, groups, T, D) -> (N, H, T, D)
        K = K.unsqueeze(2).expand(N, KVH, self.groups, T, D).reshape(N, H, T, D)
        V = V.unsqueeze(2).expand(N, KVH, self.groups, T, D).reshape(N, H, T, D)

        scores = Q.matmul(K.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask
        attn = scores.softmax(dim=-1)
        ctx = attn.matmul(V)
        ctx = ctx.permute(0, 2, 1, 3).reshape(N, T, E)
        return self.W_o(ctx)

    def __repr__(self):
        return f"GroupedQueryAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads})"


def causal_mask(seq_len: int) -> Tensor:
    """Create a causal (upper-triangular) attention mask."""
    mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
    return Tensor(mask.reshape(1, 1, seq_len, seq_len))
