"""Convolution layers: Conv1d, Conv2d, ConvTranspose2d."""

from __future__ import annotations

import numpy as np
from ..tensor import Tensor, _acc
from .module import Module


def _im2col(x: np.ndarray, kh: int, kw: int, stride: int, padding: int, dilation: int = 1) -> np.ndarray:
    """Transform input patches to columns for GEMM-based convolution.

    x: (N, C, H, W)
    Returns: (N, C*kh*kw, out_h*out_w)
    """
    N, C, H, W = x.shape
    dk_h = (kh - 1) * dilation + 1
    dk_w = (kw - 1) * dilation + 1
    out_h = (H + 2 * padding - dk_h) // stride + 1
    out_w = (W + 2 * padding - dk_w) // stride + 1

    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    cols = np.zeros((N, C, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        i_d = i * dilation
        i_max = i_d + stride * out_h
        for j in range(kw):
            j_d = j * dilation
            j_max = j_d + stride * out_w
            cols[:, :, i, j, :, :] = x[:, :, i_d:i_max:stride, j_d:j_max:stride]

    return cols.reshape(N, C * kh * kw, out_h * out_w)


def _col2im(cols: np.ndarray, x_shape, kh: int, kw: int, stride: int, padding: int, dilation: int = 1) -> np.ndarray:
    """Reverse of im2col: accumulate column gradients back to image."""
    N, C, H, W = x_shape
    dk_h = (kh - 1) * dilation + 1
    dk_w = (kw - 1) * dilation + 1
    out_h = (H + 2 * padding - dk_h) // stride + 1
    out_w = (W + 2 * padding - dk_w) // stride + 1

    cols_reshaped = cols.reshape(N, C, kh, kw, out_h, out_w)
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    for i in range(kh):
        i_d = i * dilation
        i_max = i_d + stride * out_h
        for j in range(kw):
            j_d = j * dilation
            j_max = j_d + stride * out_w
            x_padded[:, :, i_d:i_max:stride, j_d:j_max:stride] += cols_reshaped[:, :, i, j, :, :]

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded


class Conv2d(Module):
    """2D convolution using im2col + GEMM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        assert in_channels % groups == 0
        assert out_channels % groups == 0

        fan_in = in_channels * kernel_size * kernel_size // groups
        bound = 1.0 / np.sqrt(fan_in)
        self.weight = Tensor(
            np.random.uniform(
                -bound, bound,
                (out_channels, in_channels // groups, kernel_size, kernel_size),
            ).astype(np.float32),
            requires_grad=True,
        )
        if bias:
            self.bias = Tensor(
                np.random.uniform(-bound, bound, (out_channels,)).astype(np.float32),
                requires_grad=True,
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        kh = kw = self.kernel_size
        s, p, d, g = self.stride, self.padding, self.dilation, self.groups

        dk_h = (kh - 1) * d + 1
        dk_w = (kw - 1) * d + 1
        out_h = (H + 2 * p - dk_h) // s + 1
        out_w = (W + 2 * p - dk_w) // s + 1
        out_c = self.out_channels

        cols = _im2col(x.data, kh, kw, s, p, d)  # (N, C*kh*kw, out_h*out_w)
        w_reshaped = self.weight.data.reshape(out_c, -1)  # (out_c, C/g*kh*kw)

        if g == 1:
            # out = w @ cols  →  (N, out_c, out_h*out_w)
            out = np.einsum("ij,njk->nik", w_reshaped, cols)
        else:
            c_per_g = C // g
            oc_per_g = out_c // g
            col_h = c_per_g * kh * kw
            out = np.zeros((N, out_c, out_h * out_w), dtype=np.float32)
            for gi in range(g):
                cols_g = cols[:, gi * col_h : (gi + 1) * col_h, :]
                w_g = w_reshaped[gi * oc_per_g : (gi + 1) * oc_per_g, :]
                out[:, gi * oc_per_g : (gi + 1) * oc_per_g, :] = np.einsum(
                    "ij,njk->nik", w_g, cols_g
                )

        if self.bias is not None:
            out += self.bias.data.reshape(1, -1, 1)

        out_data = out.reshape(N, out_c, out_h, out_w)
        result = x._make_result(out_data, [x, self.weight] + ([self.bias] if self.bias is not None else []))

        if result.requires_grad:
            x_shape = x.shape
            weight = self.weight
            bias = self.bias

            def _backward(grad):
                grad_flat = grad.reshape(N, out_c, -1)  # (N, out_c, out_h*out_w)

                if bias is not None and bias.requires_grad:
                    bias.grad = _acc(bias.grad, grad_flat.sum(axis=(0, 2)))

                if g == 1:
                    if weight.requires_grad:
                        # grad_w = grad_flat @ cols^T, summed over batch
                        weight.grad = _acc(weight.grad, np.einsum(
                            "nik,njk->ij", grad_flat, cols
                        ).reshape(weight.shape))

                    if x.requires_grad:
                        # col_grad = w^T @ grad_flat: (out_c, col_h)^T @ (N, out_c, spatial) -> (N, col_h, spatial)
                        col_grad = np.einsum("oi,nok->nik", w_reshaped, grad_flat)
                        x.grad = _acc(x.grad, _col2im(col_grad, x_shape, kh, kw, s, p, d))
                else:
                    c_per_g = C // g
                    oc_per_g = out_c // g
                    col_h = c_per_g * kh * kw
                    if weight.requires_grad:
                        wg = np.zeros_like(weight.data).reshape(out_c, -1)
                        for gi in range(g):
                            cols_g = cols[:, gi * col_h : (gi + 1) * col_h, :]
                            grad_g = grad_flat[:, gi * oc_per_g : (gi + 1) * oc_per_g, :]
                            wg[gi * oc_per_g : (gi + 1) * oc_per_g, :] = np.einsum(
                                "nik,njk->ij", grad_g, cols_g
                            )
                        weight.grad = _acc(weight.grad, wg.reshape(weight.shape))

                    if x.requires_grad:
                        col_grad = np.zeros_like(cols)
                        for gi in range(g):
                            w_g = w_reshaped[gi * oc_per_g : (gi + 1) * oc_per_g, :]
                            grad_g = grad_flat[:, gi * oc_per_g : (gi + 1) * oc_per_g, :]
                            col_grad[:, gi * col_h : (gi + 1) * col_h, :] = np.einsum(
                                "oi,nok->nik", w_g, grad_g
                            )
                        x.grad = _acc(x.grad, _col2im(col_grad, x_shape, kh, kw, s, p, d))

            result._grad_fn = _backward

        return result

    def __repr__(self):
        return (
            f"Conv2d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding})"
        )


class Conv1d(Module):
    """1D convolution — implemented as Conv2d with height=1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        # Re-init with proper 1D kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_size
        bound = 1.0 / np.sqrt(fan_in)
        self.weight = Tensor(
            np.random.uniform(-bound, bound, (out_channels, in_channels, kernel_size)).astype(
                np.float32
            ),
            requires_grad=True,
        )
        if bias:
            self.bias = Tensor(
                np.random.uniform(-bound, bound, (out_channels,)).astype(np.float32),
                requires_grad=True,
            )
        else:
            self.bias = None
        # Remove the internal Conv2d from modules
        self._modules.pop("conv", None)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, L) -> (N, C, 1, L), conv2d, squeeze
        x4d = x.unsqueeze(2)
        # Use Conv2d logic but with kernel (1, kernel_size)
        N, C, _, L = x4d.shape
        kh, kw = 1, self.kernel_size
        s, p = self.stride, self.padding

        out_w = (L + 2 * p - kw) // s + 1
        cols = _im2col(x4d.data, kh, kw, s, p, 1)
        w = self.weight.data.reshape(self.out_channels, -1)
        out = np.einsum("ij,njk->nik", w, cols)
        if self.bias is not None:
            out += self.bias.data.reshape(1, -1, 1)

        out_data = out.reshape(N, self.out_channels, out_w)
        result = x._make_result(out_data, [x, self.weight] + ([self.bias] if self.bias is not None else []))

        if result.requires_grad:
            x_shape = x4d.shape
            weight = self.weight
            bias = self.bias

            def _backward(grad):
                grad_flat = grad.reshape(N, self.out_channels, -1)
                if bias is not None and bias.requires_grad:
                    bias.grad = _acc(bias.grad, grad_flat.sum(axis=(0, 2)))
                if weight.requires_grad:
                    weight.grad = _acc(weight.grad, np.einsum(
                        "nik,njk->ij", grad_flat, cols
                    ).reshape(weight.shape))
                if x.requires_grad:
                    col_grad = np.einsum("oi,nok->nik", w, grad_flat)
                    x_grad_4d = _col2im(col_grad, x_shape, kh, kw, s, p, 1)
                    x.grad = _acc(x.grad, x_grad_4d.squeeze(axis=2))
            result._grad_fn = _backward

        return result

    def __repr__(self):
        return f"Conv1d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size})"


class ConvTranspose2d(Module):
    """2D transposed convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        fan_in = in_channels * kernel_size * kernel_size
        bound = 1.0 / np.sqrt(fan_in)
        # Note: weight shape is (in_channels, out_channels, kh, kw) for transposed conv
        self.weight = Tensor(
            np.random.uniform(
                -bound, bound, (in_channels, out_channels, kernel_size, kernel_size)
            ).astype(np.float32),
            requires_grad=True,
        )
        if bias:
            self.bias = Tensor(
                np.random.uniform(-bound, bound, (out_channels,)).astype(np.float32),
                requires_grad=True,
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        N, C_in, H_in, W_in = x.shape
        k = self.kernel_size
        s, p, op = self.stride, self.padding, self.output_padding

        H_out = (H_in - 1) * s - 2 * p + k + op
        W_out = (W_in - 1) * s - 2 * p + k + op

        # Insert zeros between input elements (stride > 1)
        if s > 1:
            H_dilated = H_in + (H_in - 1) * (s - 1)
            W_dilated = W_in + (W_in - 1) * (s - 1)
            dilated = np.zeros((N, C_in, H_dilated, W_dilated), dtype=np.float32)
            dilated[:, :, ::s, ::s] = x.data
        else:
            dilated = x.data

        # Pad and correlate with flipped kernel
        # Effective padding for transposed conv
        pad_h = k - 1 - p
        pad_w = k - 1 - p
        if pad_h > 0 or pad_w > 0 or op > 0:
            dilated = np.pad(
                dilated,
                ((0, 0), (0, 0), (pad_h, pad_h + op), (pad_w, pad_w + op)),
            )

        # Flip weight and reshape: (C_in, C_out, k, k) -> (C_out, C_in, k, k) flipped
        w_flipped = self.weight.data[:, :, ::-1, ::-1]  # (C_in, C_out, k, k)
        w_conv = w_flipped.transpose(1, 0, 2, 3)  # (C_out, C_in, k, k)

        cols = _im2col(dilated, k, k, 1, 0, 1)
        w_r = w_conv.reshape(self.out_channels, -1)
        out = np.einsum("ij,njk->nik", w_r, cols)

        if self.bias is not None:
            out += self.bias.data.reshape(1, -1, 1)

        out_data = out.reshape(N, self.out_channels, H_out, W_out)
        result = x._make_result(out_data, [x, self.weight] + ([self.bias] if self.bias is not None else []))

        if result.requires_grad:
            weight = self.weight
            bias = self.bias

            def _backward(grad):
                if bias is not None and bias.requires_grad:
                    bias.grad = _acc(bias.grad, grad.sum(axis=(0, 2, 3)))

                if weight.requires_grad:
                    wg = np.zeros_like(weight.data)
                    for n in range(N):
                        for ci in range(C_in):
                            for co in range(self.out_channels):
                                for kh in range(k):
                                    for kw in range(k):
                                        val = 0.0
                                        for hi in range(H_in):
                                            for wi in range(W_in):
                                                oh = hi * s + kh - p
                                                ow = wi * s + kw - p
                                                if 0 <= oh < H_out and 0 <= ow < W_out:
                                                    val += x.data[n, ci, hi, wi] * grad[n, co, oh, ow]
                                        wg[ci, co, kh, kw] += val
                    weight.grad = _acc(weight.grad, wg)

                if x.requires_grad:
                    w_r2 = weight.data.transpose(1, 0, 2, 3).reshape(C_in, -1)
                    grad_cols = _im2col(grad, k, k, s, p, 1)
                    x_grad = np.einsum("ij,njk->nik", w_r2, grad_cols)
                    x.grad = _acc(x.grad, x_grad.reshape(x.shape))

            result._grad_fn = _backward

        return result

    def __repr__(self):
        return (
            f"ConvTranspose2d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
        )
