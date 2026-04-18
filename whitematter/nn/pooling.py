"""Pooling layers: MaxPool2d, AvgPool2d, AdaptiveAvgPool2d."""

from __future__ import annotations

import numpy as np
from ..tensor import Tensor, _acc
from .module import Module


class MaxPool2d(Module):
    """2D max pooling."""

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        out_h = (H + 2 * p - k) // s + 1
        out_w = (W + 2 * p - k) // s + 1

        if p > 0:
            x_pad = np.pad(x.data, ((0, 0), (0, 0), (p, p), (p, p)), constant_values=-np.inf)
        else:
            x_pad = x.data

        out = np.zeros((N, C, out_h, out_w), dtype=np.float32)
        max_indices = np.zeros((N, C, out_h, out_w), dtype=np.int64)

        for oh in range(out_h):
            for ow in range(out_w):
                region = x_pad[:, :, oh * s : oh * s + k, ow * s : ow * s + k]
                region_flat = region.reshape(N, C, -1)
                idx = region_flat.argmax(axis=2)
                out[:, :, oh, ow] = np.take_along_axis(region_flat, idx[:, :, None], axis=2).squeeze(2)
                kh_idx, kw_idx = np.divmod(idx, k)
                max_indices[:, :, oh, ow] = (oh * s + kh_idx) * (W + 2 * p) + (ow * s + kw_idx)

        result = x._make_result(out, [x])
        if result.requires_grad:
            x_shape = x.shape

            def _backward(grad):
                if x.requires_grad:
                    if p > 0:
                        dx = np.zeros((N, C, H + 2 * p, W + 2 * p), dtype=np.float32)
                    else:
                        dx = np.zeros((N, C, H, W), dtype=np.float32)
                    W_padded = W + 2 * p
                    for oh in range(out_h):
                        for ow in range(out_w):
                            idx = max_indices[:, :, oh, ow]
                            ih, iw = np.divmod(idx, W_padded)
                            np.add.at(
                                dx,
                                (np.arange(N)[:, None], np.arange(C)[None, :], ih, iw),
                                grad[:, :, oh, ow],
                            )
                    if p > 0:
                        dx = dx[:, :, p:-p, p:-p]
                    x.grad = _acc(x.grad, dx)
            result._grad_fn = _backward
        return result

    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride})"


class AvgPool2d(Module):
    """2D average pooling."""

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        out_h = (H + 2 * p - k) // s + 1
        out_w = (W + 2 * p - k) // s + 1

        if p > 0:
            x_pad = np.pad(x.data, ((0, 0), (0, 0), (p, p), (p, p)))
        else:
            x_pad = x.data

        out = np.zeros((N, C, out_h, out_w), dtype=np.float32)
        pool_size = k * k
        for oh in range(out_h):
            for ow in range(out_w):
                region = x_pad[:, :, oh * s : oh * s + k, ow * s : ow * s + k]
                out[:, :, oh, ow] = region.mean(axis=(2, 3))

        result = x._make_result(out, [x])
        if result.requires_grad:
            def _backward(grad):
                if x.requires_grad:
                    if p > 0:
                        dx = np.zeros((N, C, H + 2 * p, W + 2 * p), dtype=np.float32)
                    else:
                        dx = np.zeros((N, C, H, W), dtype=np.float32)
                    for oh in range(out_h):
                        for ow in range(out_w):
                            dx[:, :, oh * s : oh * s + k, ow * s : ow * s + k] += (
                                grad[:, :, oh : oh + 1, ow : ow + 1] / pool_size
                            )
                    if p > 0:
                        dx = dx[:, :, p:-p, p:-p]
                    x.grad = _acc(x.grad, dx)
            result._grad_fn = _backward
        return result

    def __repr__(self):
        return f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride})"


class AdaptiveAvgPool2d(Module):
    """Adaptive average pooling to target output size."""

    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        oh, ow = self.output_size

        out = np.zeros((N, C, oh, ow), dtype=np.float32)
        for i in range(oh):
            h_start = (i * H) // oh
            h_end = ((i + 1) * H) // oh
            for j in range(ow):
                w_start = (j * W) // ow
                w_end = ((j + 1) * W) // ow
                out[:, :, i, j] = x.data[:, :, h_start:h_end, w_start:w_end].mean(axis=(2, 3))

        result = x._make_result(out, [x])
        if result.requires_grad:
            def _backward(grad):
                if x.requires_grad:
                    dx = np.zeros_like(x.data)
                    for i in range(oh):
                        h_start = (i * H) // oh
                        h_end = ((i + 1) * H) // oh
                        for j in range(ow):
                            w_start = (j * W) // ow
                            w_end = ((j + 1) * W) // ow
                            count = (h_end - h_start) * (w_end - w_start)
                            dx[:, :, h_start:h_end, w_start:w_end] += (
                                grad[:, :, i : i + 1, j : j + 1] / count
                            )
                    x.grad = _acc(x.grad, dx)
            result._grad_fn = _backward
        return result

    def __repr__(self):
        return f"AdaptiveAvgPool2d(output_size={self.output_size})"
