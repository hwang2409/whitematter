"""Gradient utilities: clipping, accumulation, early stopping, checkpointing."""

from __future__ import annotations

import numpy as np
from typing import List, Optional
from ..tensor import Tensor


def clip_grad_norm_(params: List[Tensor], max_norm: float) -> float:
    """Clip gradient norm in-place. Returns the total norm."""
    total_norm_sq = 0.0
    grads = []
    for p in params:
        if p.grad is not None:
            total_norm_sq += float((p.grad ** 2).sum())
            grads.append(p)
    total_norm = np.sqrt(total_norm_sq)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in grads:
            p.grad *= scale
    return total_norm


def clip_grad_value_(params: List[Tensor], clip_value: float):
    """Clip gradient values in-place."""
    for p in params:
        if p.grad is not None:
            np.clip(p.grad, -clip_value, clip_value, out=p.grad)


class GradientAccumulator:
    """Accumulate gradients over multiple backward passes, then average."""

    def __init__(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self._count = 0

    def backward(self, loss: Tensor):
        loss.backward()
        self._count += 1

    def should_step(self) -> bool:
        return self._count >= self.accumulation_steps

    def reset(self):
        self._count = 0


class EarlyStopping:
    """Stop training when a metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self._best: Optional[float] = None
        self._counter = 0

    def __call__(self, metric: float) -> bool:
        if self._best is None or metric < self._best - self.min_delta:
            self._best = metric
            self._counter = 0
            return False
        self._counter += 1
        return self._counter >= self.patience


class ModelCheckpoint:
    """Save model when metric improves."""

    def __init__(self, path: str):
        self.path = path
        self._best: Optional[float] = None

    def __call__(self, metric: float, model) -> bool:
        from .. import serialization

        if self._best is None or metric < self._best:
            self._best = metric
            serialization.save(self.path, model)
            return True
        return False
