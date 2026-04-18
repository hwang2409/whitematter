"""Tests for Tensor operations and autograd."""

import numpy as np
import pytest
from whitematter.tensor import Tensor


def _grad_check(fn, inputs, eps=1e-4, atol=1e-2, rtol=5e-2):
    """Numerical gradient check via central finite differences."""
    out = fn(*inputs)
    out.backward(np.ones_like(out.data))
    analytic_grads = [inp.grad.copy() if inp.grad is not None else None for inp in inputs]

    for i, inp in enumerate(inputs):
        if not inp.requires_grad or analytic_grads[i] is None:
            continue
        numerical = np.zeros_like(inp.data)
        orig_data = inp.data.copy()
        for idx in np.ndindex(inp.shape):
            data_p = orig_data.copy()
            data_p[idx] += eps
            data_m = orig_data.copy()
            data_m[idx] -= eps
            inp_p = [Tensor(data_p, requires_grad=True) if j == i else Tensor(x.data.copy(), requires_grad=x.requires_grad) for j, x in enumerate(inputs)]
            inp_m = [Tensor(data_m, requires_grad=True) if j == i else Tensor(x.data.copy(), requires_grad=x.requires_grad) for j, x in enumerate(inputs)]
            numerical[idx] = (fn(*inp_p).data.sum() - fn(*inp_m).data.sum()) / (2 * eps)

        np.testing.assert_allclose(
            analytic_grads[i], numerical, atol=atol, rtol=rtol,
            err_msg=f"Gradient check failed for input {i}"
        )


class TestFactory:
    def test_zeros(self):
        t = Tensor.zeros(2, 3)
        assert t.shape == (2, 3)
        np.testing.assert_array_equal(t.data, 0)

    def test_ones(self):
        t = Tensor.ones(3, 4)
        np.testing.assert_array_equal(t.data, 1)

    def test_randn_shape(self):
        t = Tensor.randn(5, 10)
        assert t.shape == (5, 10)

    def test_xavier(self):
        t = Tensor.xavier(100, 200)
        assert t.shape == (200, 100)
        assert abs(t.data.std() - np.sqrt(2 / 300)) < 0.1

    def test_kaiming(self):
        t = Tensor.kaiming_normal(100, 200)
        assert t.shape == (200, 100)


class TestArithmetic:
    def test_add(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a.add(b)
        c.backward(np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(c.data, [5, 7, 9])
        np.testing.assert_array_equal(a.grad, [1, 1, 1])
        np.testing.assert_array_equal(b.grad, [1, 1, 1])

    def test_add_broadcast(self):
        a = Tensor(np.ones((3, 4)), requires_grad=True)
        b = Tensor(np.ones((4,)), requires_grad=True)
        c = a + b
        c.backward(np.ones((3, 4), dtype=np.float32))
        assert a.grad.shape == (3, 4)
        assert b.grad.shape == (4,)
        np.testing.assert_array_almost_equal(b.grad, [3, 3, 3, 3])

    def test_sub(self):
        a = Tensor([5, 6, 7], requires_grad=True)
        b = Tensor([1, 2, 3], requires_grad=True)
        c = a - b
        c.backward(np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(a.grad, [1, 1, 1])
        np.testing.assert_array_equal(b.grad, [-1, -1, -1])

    def test_mul(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        c = a * b
        c.backward(np.ones(2, dtype=np.float32))
        np.testing.assert_array_equal(a.grad, [4, 5])
        np.testing.assert_array_equal(b.grad, [2, 3])

    def test_div(self):
        a = Tensor([6.0, 8.0], requires_grad=True)
        b = Tensor([2.0, 4.0], requires_grad=True)
        c = a / b
        np.testing.assert_array_almost_equal(c.data, [3, 2])

    def test_neg(self):
        a = Tensor([1, -2, 3], requires_grad=True)
        c = -a
        c.backward(np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(c.data, [-1, 2, -3])
        np.testing.assert_array_equal(a.grad, [-1, -1, -1])

    def test_scalar_mul(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        c = a * 3
        c.backward(np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(c.data, [3, 6, 9])
        np.testing.assert_array_equal(a.grad, [3, 3, 3])

    def test_matmul(self):
        a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        b = Tensor(np.random.randn(4, 5).astype(np.float32), requires_grad=True)
        _grad_check(lambda x, y: x.matmul(y), [a, b])

    def test_matmul_batched(self):
        a = Tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
        b = Tensor(np.random.randn(2, 4, 5).astype(np.float32), requires_grad=True)
        _grad_check(lambda x, y: x.matmul(y), [a, b])


class TestReductions:
    def test_sum_all(self):
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        s = a.sum()
        s.backward()
        np.testing.assert_array_equal(a.grad, np.ones((2, 2)))

    def test_sum_dim(self):
        a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        _grad_check(lambda x: x.sum(dim=1), [a])

    def test_mean(self):
        a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        _grad_check(lambda x: x.mean(), [a])

    def test_mean_dim(self):
        a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        _grad_check(lambda x: x.mean(dim=0), [a])

    def test_max(self):
        a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        m = a.max(dim=1)
        assert m.shape == (3,)

    def test_min(self):
        a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        m = a.min(dim=0)
        assert m.shape == (4,)


class TestActivations:
    def test_relu(self):
        a = Tensor([-1, 0, 1, 2], requires_grad=True)
        c = a.relu()
        np.testing.assert_array_equal(c.data, [0, 0, 1, 2])
        c.backward(np.ones(4, dtype=np.float32))
        np.testing.assert_array_equal(a.grad, [0, 0, 1, 1])

    def test_sigmoid(self):
        a = Tensor(np.random.randn(5).astype(np.float32), requires_grad=True)
        _grad_check(lambda x: x.sigmoid(), [a])

    def test_tanh(self):
        a = Tensor(np.random.randn(5).astype(np.float32), requires_grad=True)
        _grad_check(lambda x: x.tanh(), [a])

    def test_silu(self):
        a = Tensor(np.random.randn(5).astype(np.float32), requires_grad=True)
        _grad_check(lambda x: x.silu(), [a])

    def test_exp(self):
        a = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        _grad_check(lambda x: x.exp(), [a])

    def test_log(self):
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        _grad_check(lambda x: x.log(), [a])

    def test_pow(self):
        a = Tensor([2.0, 3.0, 4.0], requires_grad=True)
        _grad_check(lambda x: x.pow(2.0), [a])

    def test_sqrt(self):
        a = Tensor([1.0, 4.0, 9.0], requires_grad=True)
        _grad_check(lambda x: x.sqrt(), [a])

    def test_abs(self):
        a = Tensor([-2.0, -1.0, 1.0, 2.0], requires_grad=True)
        _grad_check(lambda x: x.abs(), [a])

    def test_clamp(self):
        a = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        c = a.clamp(-1, 1)
        np.testing.assert_array_equal(c.data, [-1, -1, 0, 1, 1])


class TestSoftmax:
    def test_softmax(self):
        a = Tensor(np.random.randn(2, 5).astype(np.float32), requires_grad=True)
        s = a.softmax(dim=-1)
        np.testing.assert_allclose(s.data.sum(axis=-1), [1, 1], atol=1e-6)
        _grad_check(lambda x: x.softmax(dim=-1), [a])

    def test_log_softmax(self):
        a = Tensor(np.random.randn(2, 5).astype(np.float32), requires_grad=True)
        _grad_check(lambda x: x.log_softmax(dim=-1), [a])


class TestShapeOps:
    def test_reshape(self):
        a = Tensor(np.arange(12).reshape(3, 4).astype(np.float32), requires_grad=True)
        b = a.reshape(6, 2)
        assert b.shape == (6, 2)
        b.backward(np.ones((6, 2), dtype=np.float32))
        assert a.grad.shape == (3, 4)

    def test_transpose(self):
        a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        b = a.transpose()
        assert b.shape == (4, 3)
        _grad_check(lambda x: x.transpose(), [a])

    def test_permute(self):
        a = Tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
        b = a.permute(2, 0, 1)
        assert b.shape == (4, 2, 3)
        _grad_check(lambda x: x.permute(2, 0, 1), [a])

    def test_squeeze_unsqueeze(self):
        a = Tensor(np.random.randn(3, 1, 4).astype(np.float32), requires_grad=True)
        b = a.squeeze(1)
        assert b.shape == (3, 4)
        c = b.unsqueeze(0)
        assert c.shape == (1, 3, 4)

    def test_concat(self):
        a = Tensor(np.ones((2, 3)).astype(np.float32), requires_grad=True)
        b = Tensor(np.ones((2, 3)).astype(np.float32) * 2, requires_grad=True)
        c = Tensor.concat([a, b], dim=0)
        assert c.shape == (4, 3)
        c.backward(np.ones((4, 3), dtype=np.float32))
        np.testing.assert_array_equal(a.grad, np.ones((2, 3)))

    def test_stack(self):
        a = Tensor(np.ones(3).astype(np.float32), requires_grad=True)
        b = Tensor(np.ones(3).astype(np.float32) * 2, requires_grad=True)
        c = Tensor.stack([a, b], dim=0)
        assert c.shape == (2, 3)

    def test_getitem(self):
        a = Tensor(np.arange(12).reshape(3, 4).astype(np.float32), requires_grad=True)
        b = a[1]
        assert b.shape == (4,)
        b.backward(np.ones(4, dtype=np.float32))
        expected = np.zeros((3, 4), dtype=np.float32)
        expected[1] = 1
        np.testing.assert_array_equal(a.grad, expected)

    def test_flatten(self):
        a = Tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
        b = a.flatten(1)
        assert b.shape == (2, 12)


class TestGradAccumulation:
    def test_reuse(self):
        """Gradient accumulates when a tensor is used multiple times."""
        a = Tensor([2.0], requires_grad=True)
        b = a + a  # da/db = 2
        b.backward(np.array([1.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(a.grad, [2.0])

    def test_chain(self):
        """Multi-step chain rule."""
        x = Tensor([3.0], requires_grad=True)
        y = x * x  # y = x^2
        z = y * x  # z = x^3
        z.backward(np.array([1.0], dtype=np.float32))
        # dz/dx = 3x^2 = 27
        np.testing.assert_array_almost_equal(x.grad, [27.0])
