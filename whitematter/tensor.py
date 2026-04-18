"""Tensor with NumPy backend and autograd support."""

from __future__ import annotations

import numpy as np
from typing import Optional, Sequence, Tuple, Union, List

from . import autograd

Shape = Tuple[int, ...]
DimArg = Optional[Union[int, Tuple[int, ...]]]


def _acc(current, new):
    """Accumulate gradient: handles None case."""
    if current is None:
        return new
    return current + new


def _unbroadcast(grad: np.ndarray, target_shape: Shape) -> np.ndarray:
    """Reduce broadcasted gradient back to target shape."""
    if grad.shape == target_shape:
        return grad
    padded = (1,) * (grad.ndim - len(target_shape)) + target_shape
    reduce_dims = tuple(
        i for i, (g, t) in enumerate(zip(grad.shape, padded)) if t == 1 and g != 1
    )
    leading = tuple(range(grad.ndim - len(target_shape)))
    reduce_dims = leading + tuple(d for d in reduce_dims if d not in leading)
    if reduce_dims:
        grad = grad.sum(axis=reduce_dims, keepdims=True)
    return grad.reshape(target_shape)


class Tensor:
    """Tensor with automatic differentiation."""

    __slots__ = ("data", "grad", "requires_grad", "_grad_fn", "_parents", "_name")

    def __init__(
        self,
        data: Union[np.ndarray, list, float, int],
        requires_grad: bool = False,
        dtype=np.float32,
    ):
        if isinstance(data, np.ndarray):
            self.data = data.astype(dtype) if data.dtype != dtype else data
        else:
            self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._grad_fn = None
        self._parents: List[Tensor] = []
        self._name: Optional[str] = None

    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def item(self) -> float:
        return float(self.data.flat[0])

    def numpy(self) -> np.ndarray:
        return self.data

    def detach(self) -> Tensor:
        t = Tensor(self.data, requires_grad=False)
        return t

    def clone(self) -> Tensor:
        t = Tensor(self.data.copy(), requires_grad=self.requires_grad)
        return t

    @staticmethod
    def zeros(*shape, requires_grad=False) -> Tensor:
        return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)

    @staticmethod
    def ones(*shape, requires_grad=False) -> Tensor:
        return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, requires_grad=False) -> Tensor:
        return Tensor(
            np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad
        )

    @staticmethod
    def rand(*shape, requires_grad=False) -> Tensor:
        return Tensor(
            np.random.rand(*shape).astype(np.float32), requires_grad=requires_grad
        )

    @staticmethod
    def full(shape, fill_value, requires_grad=False) -> Tensor:
        return Tensor(
            np.full(shape, fill_value, dtype=np.float32), requires_grad=requires_grad
        )

    @staticmethod
    def from_numpy(arr: np.ndarray, requires_grad=False) -> Tensor:
        return Tensor(arr, requires_grad=requires_grad)

    @staticmethod
    def xavier(fan_in: int, fan_out: int, requires_grad=True) -> Tensor:
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return Tensor(
            np.random.randn(fan_out, fan_in).astype(np.float32) * std,
            requires_grad=requires_grad,
        )

    @staticmethod
    def kaiming_normal(fan_in: int, fan_out: int, requires_grad=True) -> Tensor:
        std = np.sqrt(2.0 / fan_in)
        return Tensor(
            np.random.randn(fan_out, fan_in).astype(np.float32) * std,
            requires_grad=requires_grad,
        )

    @staticmethod
    def kaiming_uniform(fan_in: int, fan_out: int, requires_grad=True) -> Tensor:
        bound = np.sqrt(6.0 / fan_in)
        return Tensor(
            np.random.uniform(-bound, bound, (fan_out, fan_in)).astype(np.float32),
            requires_grad=requires_grad,
        )

    def _build_topo(self) -> List[Tensor]:
        """Topological sort of the computation graph."""
        topo = []
        visited = set()

        def _visit(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            for p in t._parents:
                _visit(p)
            topo.append(t)

        _visit(self)
        return topo

    def backward(self, grad: Optional[np.ndarray] = None):
        """Run backpropagation."""
        if grad is None:
            assert self.data.size == 1, "backward() requires scalar output or explicit grad"
            grad = np.ones_like(self.data)

        self.grad = grad
        for t in reversed(self._build_topo()):
            if t._grad_fn is not None and t.grad is not None:
                t._grad_fn(t.grad)

    def zero_grad(self):
        self.grad = None

    def _make_result(self, data: np.ndarray, parents: list) -> Tensor:
        """Create result tensor, tracking grad if needed."""
        needs_grad = autograd.is_grad_enabled() and any(
            p.requires_grad for p in parents
        )
        result = Tensor(data, requires_grad=needs_grad)
        if needs_grad:
            result._parents = parents
        return result

    def add(self, other: Tensor) -> Tensor:
        result = self._make_result(self.data + other.data, [self, other])
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, _unbroadcast(grad, self.shape))
                if other.requires_grad:
                    other.grad = _acc(other.grad, _unbroadcast(grad, other.shape))
            result._grad_fn = _backward
        return result

    def sub(self, other: Tensor) -> Tensor:
        result = self._make_result(self.data - other.data, [self, other])
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, _unbroadcast(grad, self.shape))
                if other.requires_grad:
                    other.grad = _acc(other.grad, _unbroadcast(-grad, other.shape))
            result._grad_fn = _backward
        return result

    def mul(self, other: Union[Tensor, float, int]) -> Tensor:
        if isinstance(other, (float, int)):
            return self._scalar_mul(float(other))
        result = self._make_result(self.data * other.data, [self, other])
        if result.requires_grad:
            s_data, o_data = self.data, other.data
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, _unbroadcast(grad * o_data, self.shape))
                if other.requires_grad:
                    other.grad = _acc(other.grad, _unbroadcast(grad * s_data, other.shape))
            result._grad_fn = _backward
        return result

    def div(self, other: Union[Tensor, float, int]) -> Tensor:
        if isinstance(other, (float, int)):
            return self._scalar_mul(1.0 / float(other))
        result = self._make_result(self.data / other.data, [self, other])
        if result.requires_grad:
            s_data, o_data = self.data, other.data
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, _unbroadcast(grad / o_data, self.shape))
                if other.requires_grad:
                    other.grad = _acc(other.grad, _unbroadcast(
                        -grad * s_data / (o_data ** 2), other.shape
                    ))
            result._grad_fn = _backward
        return result

    def _scalar_mul(self, scalar: float) -> Tensor:
        result = self._make_result(self.data * scalar, [self])
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, grad * scalar)
            result._grad_fn = _backward
        return result

    def neg(self) -> Tensor:
        result = self._make_result(-self.data, [self])
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, (-grad))
            result._grad_fn = _backward
        return result

    def matmul(self, other: Tensor) -> Tensor:
        result = self._make_result(self.data @ other.data, [self, other])
        if result.requires_grad:
            s_data, o_data = self.data, other.data
            def _backward(grad):
                if self.requires_grad:
                    g = grad @ o_data.swapaxes(-2, -1)
                    self.grad = _acc(self.grad, _unbroadcast(g, self.shape))
                if other.requires_grad:
                    g = s_data.swapaxes(-2, -1) @ grad
                    other.grad = _acc(other.grad, _unbroadcast(g, other.shape))
            result._grad_fn = _backward
        return result

    def bmm(self, other: Tensor) -> Tensor:
        """Batch matrix multiply: (B, M, K) @ (B, K, N) -> (B, M, N)."""
        return self.matmul(other)

    def sum(self, dim: DimArg = None, keepdim: bool = False) -> Tensor:
        result_data = self.data.sum(axis=dim, keepdims=keepdim)
        result = self._make_result(np.atleast_1d(result_data), [self])
        if result.requires_grad:
            shape = self.shape
            def _backward(grad):
                if self.requires_grad:
                    g = grad
                    if not keepdim and dim is not None:
                        if isinstance(dim, int):
                            g = np.expand_dims(g, axis=dim)
                        else:
                            for d in sorted(dim):
                                g = np.expand_dims(g, axis=d)
                    self.grad = _acc(self.grad, np.broadcast_to(g, shape).copy())
            result._grad_fn = _backward
        return result

    def mean(self, dim: DimArg = None, keepdim: bool = False) -> Tensor:
        result_data = self.data.mean(axis=dim, keepdims=keepdim)
        result = self._make_result(np.atleast_1d(result_data), [self])
        if result.requires_grad:
            shape = self.shape
            if dim is None:
                n = self.data.size
            elif isinstance(dim, int):
                n = shape[dim]
            else:
                n = 1
                for d in dim:
                    n *= shape[d]

            def _backward(grad):
                if self.requires_grad:
                    g = grad
                    if not keepdim and dim is not None:
                        if isinstance(dim, int):
                            g = np.expand_dims(g, axis=dim)
                        else:
                            for d in sorted(dim):
                                g = np.expand_dims(g, axis=d)
                    self.grad = _acc(self.grad, np.broadcast_to(g / n, shape).copy())
            result._grad_fn = _backward
        return result

    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        if dim is None:
            result_data = np.atleast_1d(self.data.max())
            result = self._make_result(result_data, [self])
            if result.requires_grad:
                s_data = self.data
                def _backward(grad):
                    if self.requires_grad:
                        mask = (s_data == s_data.max()).astype(np.float32)
                        mask /= mask.sum()
                        self.grad = _acc(self.grad, grad.flat[0] * mask)
                result._grad_fn = _backward
            return result
        result_data = self.data.max(axis=dim, keepdims=keepdim)
        result = self._make_result(result_data, [self])
        if result.requires_grad:
            s_data = self.data
            def _backward(grad):
                if self.requires_grad:
                    g = grad
                    if not keepdim:
                        g = np.expand_dims(g, axis=dim)
                    max_vals = s_data.max(axis=dim, keepdims=True)
                    mask = (s_data == max_vals).astype(np.float32)
                    mask /= mask.sum(axis=dim, keepdims=True)
                    self.grad = _acc(self.grad, g * mask)
            result._grad_fn = _backward
        return result

    def min(self, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        if dim is None:
            result_data = np.atleast_1d(self.data.min())
            result = self._make_result(result_data, [self])
            if result.requires_grad:
                s_data = self.data
                def _backward(grad):
                    if self.requires_grad:
                        mask = (s_data == s_data.min()).astype(np.float32)
                        mask /= mask.sum()
                        self.grad = _acc(self.grad, grad.flat[0] * mask)
                result._grad_fn = _backward
            return result
        result_data = self.data.min(axis=dim, keepdims=keepdim)
        result = self._make_result(result_data, [self])
        if result.requires_grad:
            s_data = self.data
            def _backward(grad):
                if self.requires_grad:
                    g = grad
                    if not keepdim:
                        g = np.expand_dims(g, axis=dim)
                    min_vals = s_data.min(axis=dim, keepdims=True)
                    mask = (s_data == min_vals).astype(np.float32)
                    mask /= mask.sum(axis=dim, keepdims=True)
                    self.grad = _acc(self.grad, g * mask)
            result._grad_fn = _backward
        return result

    def relu(self) -> Tensor:
        result = self._make_result(np.maximum(self.data, 0), [self])
        if result.requires_grad:
            s_data = self.data
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, grad * (s_data > 0).astype(np.float32))
            result._grad_fn = _backward
        return result

    def sigmoid(self) -> Tensor:
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -88, 88)))
        result = self._make_result(s, [self])
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, grad * s * (1 - s))
            result._grad_fn = _backward
        return result

    def tanh(self) -> Tensor:
        t = np.tanh(self.data)
        result = self._make_result(t, [self])
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, grad * (1 - t ** 2))
            result._grad_fn = _backward
        return result

    def silu(self) -> Tensor:
        sig = 1.0 / (1.0 + np.exp(-np.clip(self.data, -88, 88)))
        result = self._make_result(self.data * sig, [self])
        if result.requires_grad:
            s_data = self.data
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, grad * (sig + s_data * sig * (1 - sig)))
            result._grad_fn = _backward
        return result

    def gelu(self) -> Tensor:
        from scipy.special import erf  # noqa: F811

        x = self.data
        sqrt2 = np.sqrt(2.0)
        cdf = 0.5 * (1.0 + erf(x / sqrt2))
        result = self._make_result(x * cdf, [self])
        if result.requires_grad:
            pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, grad * (cdf + x * pdf))
            result._grad_fn = _backward
        return result

    def mish(self) -> Tensor:
        x = self.data
        sp = np.log1p(np.exp(np.clip(x, -20, 20)))
        t = np.tanh(sp)
        result = self._make_result(x * t, [self])
        if result.requires_grad:
            sig = 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, grad * (t + x * (1 - t ** 2) * sig))
            result._grad_fn = _backward
        return result

    def exp(self) -> Tensor:
        e = np.exp(self.data)
        result = self._make_result(e, [self])
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, grad * e)
            result._grad_fn = _backward
        return result

    def log(self) -> Tensor:
        result = self._make_result(np.log(self.data + 1e-8), [self])
        if result.requires_grad:
            s_data = self.data
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, grad / (s_data + 1e-8))
            result._grad_fn = _backward
        return result

    def pow(self, exponent: Union[float, int, Tensor]) -> Tensor:
        if isinstance(exponent, Tensor):
            result = self._make_result(
                np.power(self.data, exponent.data), [self, exponent]
            )
            if result.requires_grad:
                s_data, e_data, r_data = self.data, exponent.data, result.data
                def _backward(grad):
                    if self.requires_grad:
                        g = grad * e_data * np.power(s_data, e_data - 1)
                        self.grad = _acc(self.grad, _unbroadcast(g, self.shape))
                    if exponent.requires_grad:
                        g = grad * r_data * np.log(np.maximum(s_data, 1e-8))
                        exponent.grad = _acc(exponent.grad, _unbroadcast(g, exponent.shape))
                result._grad_fn = _backward
            return result
        result = self._make_result(np.power(self.data, exponent), [self])
        if result.requires_grad:
            s_data = self.data
            exp_val = exponent
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, grad * exp_val * np.power(s_data, exp_val - 1))
            result._grad_fn = _backward
        return result

    def sqrt(self) -> Tensor:
        r = np.sqrt(self.data)
        result = self._make_result(r, [self])
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, grad / (2 * r + 1e-12))
            result._grad_fn = _backward
        return result

    def abs(self) -> Tensor:
        result = self._make_result(np.abs(self.data), [self])
        if result.requires_grad:
            s_data = self.data
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, grad * np.sign(s_data))
            result._grad_fn = _backward
        return result

    def clamp(self, min_val: float = float("-inf"), max_val: float = float("inf")) -> Tensor:
        result = self._make_result(np.clip(self.data, min_val, max_val), [self])
        if result.requires_grad:
            s_data = self.data
            def _backward(grad):
                if self.requires_grad:
                    mask = ((s_data >= min_val) & (s_data <= max_val)).astype(np.float32)
                    self.grad = _acc(self.grad, grad * mask)
            result._grad_fn = _backward
        return result

    def softmax(self, dim: int = -1) -> Tensor:
        x = self.data
        x_max = x.max(axis=dim, keepdims=True)
        e = np.exp(x - x_max)
        s = e / e.sum(axis=dim, keepdims=True)
        result = self._make_result(s, [self])
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    # Jacobian: s_i * (delta_ij - s_j)
                    dot = (grad * s).sum(axis=dim, keepdims=True)
                    self.grad = _acc(self.grad, s * (grad - dot))
            result._grad_fn = _backward
        return result

    def log_softmax(self, dim: int = -1) -> Tensor:
        x = self.data
        x_max = x.max(axis=dim, keepdims=True)
        log_sum_exp = x_max + np.log(np.exp(x - x_max).sum(axis=dim, keepdims=True))
        ls = x - log_sum_exp
        result = self._make_result(ls, [self])
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    s = np.exp(ls)
                    self.grad = _acc(self.grad, grad - s * grad.sum(axis=dim, keepdims=True))
            result._grad_fn = _backward
        return result

    def reshape(self, *shape) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        result = self._make_result(self.data.reshape(shape), [self])
        if result.requires_grad:
            orig_shape = self.shape
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, grad.reshape(orig_shape))
            result._grad_fn = _backward
        return result

    def view(self, *shape) -> Tensor:
        return self.reshape(*shape)

    def transpose(self, dim0: int = -2, dim1: int = -1) -> Tensor:
        result = self._make_result(np.swapaxes(self.data, dim0, dim1), [self])
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, np.swapaxes(grad, dim0, dim1))
            result._grad_fn = _backward
        return result

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def permute(self, *dims) -> Tensor:
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        result = self._make_result(np.transpose(self.data, dims), [self])
        if result.requires_grad:
            inv = [0] * len(dims)
            for i, d in enumerate(dims):
                inv[d] = i
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, np.transpose(grad, inv))
            result._grad_fn = _backward
        return result

    def squeeze(self, dim: Optional[int] = None) -> Tensor:
        result = self._make_result(self.data.squeeze(axis=dim), [self])
        if result.requires_grad:
            orig_shape = self.shape
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, grad.reshape(orig_shape))
            result._grad_fn = _backward
        return result

    def unsqueeze(self, dim: int) -> Tensor:
        result = self._make_result(np.expand_dims(self.data, axis=dim), [self])
        if result.requires_grad:
            orig_shape = self.shape
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, grad.reshape(orig_shape))
            result._grad_fn = _backward
        return result

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        s = list(self.shape)
        if end_dim < 0:
            end_dim = len(s) + end_dim
        new_shape = s[:start_dim] + [-1] + s[end_dim + 1 :]
        return self.reshape(*new_shape)

    def expand(self, *shape) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        result = self._make_result(np.broadcast_to(self.data, shape).copy(), [self])
        if result.requires_grad:
            orig_shape = self.shape
            def _backward(grad):
                if self.requires_grad:
                    self.grad = _acc(self.grad, _unbroadcast(grad, orig_shape))
            result._grad_fn = _backward
        return result

    def __getitem__(self, idx) -> Tensor:
        result = self._make_result(self.data[idx], [self])
        if result.requires_grad:
            orig_shape = self.shape
            def _backward(grad):
                if self.requires_grad:
                    full_grad = np.zeros(orig_shape, dtype=np.float32)
                    np.add.at(full_grad, idx, grad)
                    self.grad = _acc(self.grad, full_grad)
            result._grad_fn = _backward
        return result

    @staticmethod
    def concat(tensors: List[Tensor], dim: int = 0) -> Tensor:
        result = tensors[0]._make_result(
            np.concatenate([t.data for t in tensors], axis=dim), tensors
        )
        if result.requires_grad:
            shapes = [t.shape[dim] for t in tensors]
            def _backward(grad):
                splits = np.split(grad, np.cumsum(shapes[:-1]), axis=dim)
                for t, g in zip(tensors, splits):
                    if t.requires_grad:
                        t.grad = _acc(t.grad, g)
            result._grad_fn = _backward
        return result

    @staticmethod
    def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
        result = tensors[0]._make_result(
            np.stack([t.data for t in tensors], axis=dim), tensors
        )
        if result.requires_grad:
            def _backward(grad):
                grads = np.split(grad, len(tensors), axis=dim)
                for t, g in zip(tensors, grads):
                    if t.requires_grad:
                        t.grad = _acc(t.grad, g.squeeze(axis=dim))
            result._grad_fn = _backward
        return result

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        return self.add(other)

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        return other.add(self)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        return self.sub(other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        return other.sub(self)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        return other.div(self)

    def __neg__(self):
        return self.neg()

    def __matmul__(self, other):
        return self.matmul(other)

    def __pow__(self, exponent):
        return self.pow(exponent)

    def __repr__(self):
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{grad_str})"

    def __len__(self):
        return self.shape[0]
