"""Tests for autograd utilities."""

import numpy as np
from whitematter import Tensor, no_grad, is_grad_enabled


def test_no_grad_context():
    assert is_grad_enabled()
    with no_grad():
        assert not is_grad_enabled()
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a + b
        assert not c.requires_grad
    assert is_grad_enabled()


def test_no_grad_decorator():
    @no_grad()
    def f():
        a = Tensor([1, 2, 3], requires_grad=True)
        b = a * 2
        return b

    r = f()
    assert not r.requires_grad


def test_detach():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = a.detach()
    assert not b.requires_grad


def test_backward_scalar():
    x = Tensor([2.0], requires_grad=True)
    y = x * x
    y.backward()
    np.testing.assert_array_almost_equal(x.grad, [4.0])


def test_backward_complex_graph():
    """Test graph with diamond shape: z depends on y1 and y2 which both depend on x."""
    x = Tensor([3.0], requires_grad=True)
    y1 = x * 2  # 6
    y2 = x * 3  # 9
    z = y1 + y2  # 15
    z.backward()
    np.testing.assert_array_almost_equal(x.grad, [5.0])  # dy1/dx + dy2/dx = 2 + 3
