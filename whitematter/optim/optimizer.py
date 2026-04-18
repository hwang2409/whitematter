"""Optimizer implementations: SGD, Adam, AdamW, RMSprop."""

from __future__ import annotations

import numpy as np
from typing import List
from ..tensor import Tensor


class Optimizer:
    """Base optimizer."""

    def __init__(self, params: List[Tensor], lr: float):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum."""

    def __init__(self, params: List[Tensor], lr: float = 0.01, momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self._velocity = [np.zeros_like(p.data) for p in params] if momentum else None

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.momentum:
                self._velocity[i] = self.momentum * self._velocity[i] + g
                p.data -= self.lr * self._velocity[i]
            else:
                p.data -= self.lr * g


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self._m = [np.zeros_like(p.data) for p in params]
        self._v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * g ** 2
            m_hat = self._m[i] / (1 - b1 ** self.t)
            v_hat = self._v[i] / (1 - b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay)."""

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self._m = [np.zeros_like(p.data) for p in params]
        self._v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * g ** 2
            m_hat = self._m[i] / (1 - b1 ** self.t)
            v_hat = self._v[i] / (1 - b2 ** self.t)
            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)


class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._v = [np.zeros_like(p.data) for p in params]
        self._buf = [np.zeros_like(p.data) for p in params] if momentum else None

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.copy()
            if self.weight_decay:
                g += self.weight_decay * p.data
            self._v[i] = self.alpha * self._v[i] + (1 - self.alpha) * g ** 2
            if self.momentum:
                self._buf[i] = self.momentum * self._buf[i] + g / (np.sqrt(self._v[i]) + self.eps)
                p.data -= self.lr * self._buf[i]
            else:
                p.data -= self.lr * g / (np.sqrt(self._v[i]) + self.eps)
