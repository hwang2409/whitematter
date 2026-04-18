"""Learning rate schedulers."""

from __future__ import annotations

import math
from .optimizer import Optimizer


class _Scheduler:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr

    def step(self, *args, **kwargs):
        raise NotImplementedError


class StepLR(_Scheduler):
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self._epoch = 0

    def step(self):
        self._epoch += 1
        self.optimizer.lr = self.base_lr * (self.gamma ** (self._epoch // self.step_size))


class ExponentialLR(_Scheduler):
    def __init__(self, optimizer: Optimizer, gamma: float):
        super().__init__(optimizer)
        self.gamma = gamma
        self._epoch = 0

    def step(self):
        self._epoch += 1
        self.optimizer.lr = self.base_lr * (self.gamma ** self._epoch)


class CosineAnnealingLR(_Scheduler):
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self._epoch = 0

    def step(self):
        self._epoch += 1
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self._epoch / self.T_max)
        ) / 2


class CosineAnnealingWarmRestarts(_Scheduler):
    def __init__(
        self, optimizer: Optimizer, T_0: int, T_mult: int = 1, eta_min: float = 0.0
    ):
        super().__init__(optimizer)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self._step = 0
        self._T_i = T_0
        self._T_cur = 0

    def step(self):
        self._step += 1
        self._T_cur += 1
        if self._T_cur >= self._T_i:
            self._T_cur = 0
            self._T_i *= self.T_mult
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self._T_cur / self._T_i)
        ) / 2


class LinearWarmupCosineDecay(_Scheduler):
    """Step-based scheduler: linear warmup then cosine decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self._step = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            self.optimizer.lr = self.base_lr * (self._step / self.warmup_steps)
        else:
            progress = (self._step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            self.optimizer.lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )


class ReduceLROnPlateau(_Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-7,
    ):
        super().__init__(optimizer)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self._best = float("inf")
        self._bad = 0

    def step(self, metric: float):
        if metric < self._best:
            self._best = metric
            self._bad = 0
        else:
            self._bad += 1
            if self._bad > self.patience:
                self.optimizer.lr = max(self.optimizer.lr * self.factor, self.min_lr)
                self._bad = 0
