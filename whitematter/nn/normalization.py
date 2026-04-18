"""Normalization layers: BatchNorm2d, LayerNorm, GroupNorm, RMSNorm."""

from __future__ import annotations

import numpy as np
from ..tensor import Tensor, _acc
from .module import Module


class BatchNorm2d(Module):
    """Batch normalization over 4D input (N, C, H, W)."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.bias = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            # (N, C) case
            return self._forward_2d(x)
        N, C, H, W = x.shape
        if self._training:
            mean = x.data.mean(axis=(0, 2, 3))
            var = x.data.var(axis=(0, 2, 3))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_norm = (x.data - mean.reshape(1, C, 1, 1)) * inv_std.reshape(1, C, 1, 1)
        out_data = self.weight.data.reshape(1, C, 1, 1) * x_norm + self.bias.data.reshape(1, C, 1, 1)

        result = x._make_result(out_data, [x, self.weight, self.bias])
        if result.requires_grad:
            gamma = self.weight.data
            def _backward(grad):
                M = N * H * W  # number of elements per channel
                if self.bias.requires_grad:
                    self.bias.grad = _acc(self.bias.grad, grad.sum(axis=(0, 2, 3)))
                if self.weight.requires_grad:
                    self.weight.grad = _acc(self.weight.grad, (grad * x_norm).sum(axis=(0, 2, 3)))
                if x.requires_grad:
                    dx_norm = grad * gamma.reshape(1, C, 1, 1)
                    dvar = (dx_norm * (x.data - mean.reshape(1, C, 1, 1)) * -0.5 * (var + self.eps).reshape(1, C, 1, 1) ** (-1.5)).sum(axis=(0, 2, 3))
                    dmean = (-dx_norm * inv_std.reshape(1, C, 1, 1)).sum(axis=(0, 2, 3))
                    x.grad = _acc(x.grad, (
                        dx_norm * inv_std.reshape(1, C, 1, 1)
                        + dvar.reshape(1, C, 1, 1) * 2 * (x.data - mean.reshape(1, C, 1, 1)) / M
                        + dmean.reshape(1, C, 1, 1) / M
                    ).astype(np.float32))
            result._grad_fn = _backward
        return result

    def _forward_2d(self, x: Tensor) -> Tensor:
        N, C = x.shape
        if self._training:
            mean = x.data.mean(axis=0)
            var = x.data.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_norm = (x.data - mean) * inv_std
        out_data = self.weight.data * x_norm + self.bias.data
        result = x._make_result(out_data, [x, self.weight, self.bias])
        if result.requires_grad:
            gamma = self.weight.data
            def _backward(grad):
                if self.bias.requires_grad:
                    self.bias.grad = _acc(self.bias.grad, grad.sum(axis=0))
                if self.weight.requires_grad:
                    self.weight.grad = _acc(self.weight.grad, (grad * x_norm).sum(axis=0))
                if x.requires_grad:
                    dx_norm = grad * gamma
                    dvar = (dx_norm * (x.data - mean) * -0.5 * (var + self.eps) ** (-1.5)).sum(axis=0)
                    dmean = (-dx_norm * inv_std).sum(axis=0)
                    x.grad = _acc(x.grad, (dx_norm * inv_std + dvar * 2 * (x.data - mean) / N + dmean / N).astype(np.float32))
            result._grad_fn = _backward
        return result

    def __repr__(self):
        return f"BatchNorm2d({self.num_features})"


class LayerNorm(Module):
    """Layer normalization."""

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        shape = self.normalized_shape
        self.weight = Tensor(np.ones(shape, dtype=np.float32), requires_grad=True)
        self.bias = Tensor(np.zeros(shape, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        ndim = len(self.normalized_shape)
        axes = tuple(range(x.ndim - ndim, x.ndim))
        mean = x.data.mean(axis=axes, keepdims=True)
        var = x.data.var(axis=axes, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_norm = (x.data - mean) * inv_std
        out_data = self.weight.data * x_norm + self.bias.data

        result = x._make_result(out_data, [x, self.weight, self.bias])
        if result.requires_grad:
            gamma = self.weight.data
            norm_size = 1
            for s in self.normalized_shape:
                norm_size *= s

            def _backward(grad):
                if self.bias.requires_grad:
                    # Sum over all dims except the normalized ones
                    batch_axes = tuple(range(x.ndim - ndim))
                    self.bias.grad = _acc(self.bias.grad, grad.sum(axis=batch_axes if batch_axes else None))
                if self.weight.requires_grad:
                    batch_axes = tuple(range(x.ndim - ndim))
                    self.weight.grad = _acc(self.weight.grad, (grad * x_norm).sum(axis=batch_axes if batch_axes else None))
                if x.requires_grad:
                    dx_norm = grad * gamma
                    dmean = (-dx_norm * inv_std).sum(axis=axes, keepdims=True)
                    dvar = (dx_norm * (x.data - mean) * -0.5 * (var + self.eps) ** (-1.5)).sum(axis=axes, keepdims=True)
                    x.grad = _acc(x.grad, (
                        dx_norm * inv_std + dvar * 2 * (x.data - mean) / norm_size + dmean / norm_size
                    ).astype(np.float32))
            result._grad_fn = _backward
        return result

    def __repr__(self):
        return f"LayerNorm({self.normalized_shape})"


class GroupNorm(Module):
    """Group normalization."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        assert num_channels % num_groups == 0
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = Tensor(np.ones(num_channels, dtype=np.float32), requires_grad=True)
        self.bias = Tensor(np.zeros(num_channels, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        C = self.num_channels
        G = self.num_groups
        # Reshape to (N, G, C//G, *spatial)
        orig_shape = x.shape
        spatial = x.shape[2:] if x.ndim > 2 else ()
        group_shape = (N, G, C // G) + spatial
        x_grouped = x.data.reshape(group_shape)

        axes = tuple(range(2, len(group_shape)))
        mean = x_grouped.mean(axis=axes, keepdims=True)
        var = x_grouped.var(axis=axes, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_norm_grouped = (x_grouped - mean) * inv_std
        x_norm = x_norm_grouped.reshape(orig_shape)

        if x.ndim == 4:
            out_data = self.weight.data.reshape(1, C, 1, 1) * x_norm + self.bias.data.reshape(1, C, 1, 1)
        elif x.ndim == 2:
            out_data = self.weight.data * x_norm + self.bias.data
        else:
            shape = [1, C] + [1] * (x.ndim - 2)
            out_data = self.weight.data.reshape(shape) * x_norm + self.bias.data.reshape(shape)

        result = x._make_result(out_data, [x, self.weight, self.bias])
        if result.requires_grad:
            gamma = self.weight.data
            group_size = x.data.size // (N * G)

            def _backward(grad):
                if self.bias.requires_grad:
                    batch_axes = tuple(i for i in range(x.ndim) if i != 1)
                    self.bias.grad = _acc(self.bias.grad, grad.sum(axis=batch_axes))
                if self.weight.requires_grad:
                    batch_axes = tuple(i for i in range(x.ndim) if i != 1)
                    self.weight.grad = _acc(self.weight.grad, (grad * x_norm).sum(axis=batch_axes))
                if x.requires_grad:
                    if x.ndim == 4:
                        dx_norm = grad * gamma.reshape(1, C, 1, 1)
                    elif x.ndim == 2:
                        dx_norm = grad * gamma
                    else:
                        shape = [1, C] + [1] * (x.ndim - 2)
                        dx_norm = grad * gamma.reshape(shape)
                    dx_grouped = dx_norm.reshape(group_shape)
                    dmean = (-dx_grouped * inv_std).sum(axis=axes, keepdims=True)
                    dvar = (dx_grouped * (x_grouped - mean) * -0.5 * (var + self.eps) ** (-1.5)).sum(axis=axes, keepdims=True)
                    dx = dx_grouped * inv_std + dvar * 2 * (x_grouped - mean) / group_size + dmean / group_size
                    x.grad = _acc(x.grad, dx.reshape(orig_shape).astype(np.float32))
            result._grad_fn = _backward
        return result

    def __repr__(self):
        return f"GroupNorm({self.num_groups}, {self.num_channels})"


class RMSNorm(Module):
    """Root Mean Square normalization."""

    def __init__(self, normalized_shape, eps: float = 1e-8):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Tensor(np.ones(self.normalized_shape, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        ndim = len(self.normalized_shape)
        axes = tuple(range(x.ndim - ndim, x.ndim))
        rms = np.sqrt((x.data ** 2).mean(axis=axes, keepdims=True) + self.eps)
        x_norm = x.data / rms
        out_data = self.weight.data * x_norm

        result = x._make_result(out_data, [x, self.weight])
        if result.requires_grad:
            gamma = self.weight.data
            norm_size = 1
            for s in self.normalized_shape:
                norm_size *= s

            def _backward(grad):
                if self.weight.requires_grad:
                    batch_axes = tuple(range(x.ndim - ndim))
                    self.weight.grad = _acc(self.weight.grad, (grad * x_norm).sum(
                        axis=batch_axes if batch_axes else None
                    ))
                if x.requires_grad:
                    dx_norm = grad * gamma
                    # d/dx (x / rms) = 1/rms - x^2 / (rms^3 * n)
                    x_grad = dx_norm / rms - x.data * (dx_norm * x.data).sum(axis=axes, keepdims=True) / (rms ** 3 * norm_size)
                    x.grad = _acc(x.grad, x_grad.astype(np.float32))
            result._grad_fn = _backward
        return result

    def __repr__(self):
        return f"RMSNorm({self.normalized_shape})"
