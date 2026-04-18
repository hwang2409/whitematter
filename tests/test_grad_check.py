"""Numerical gradient checking for all operations."""

import numpy as np
import pytest
from whitematter import Tensor
from whitematter.nn import Linear, Conv2d, LayerNorm, RMSNorm, BatchNorm2d, Embedding


def _finite_diff_check(fn, inputs, eps=1e-4, atol=1e-2, rtol=5e-2):
    """Check analytical gradients against finite differences.

    Re-creates tensors for each perturbation to avoid mutating captured closures.
    """
    # Get analytical grads
    out = fn(*inputs)
    out.backward(np.ones_like(out.data))
    analytic = {i: inp.grad.copy() for i, inp in enumerate(inputs) if inp.requires_grad and inp.grad is not None}

    for i, inp in enumerate(inputs):
        if not inp.requires_grad or i not in analytic:
            continue
        numerical = np.zeros_like(inp.data)
        orig_data = inp.data.copy()
        for idx in np.ndindex(inp.shape):
            # Perturb +eps
            data_plus = orig_data.copy()
            data_plus[idx] += eps
            inputs_plus = []
            for j, inp2 in enumerate(inputs):
                if j == i:
                    inputs_plus.append(Tensor(data_plus, requires_grad=True))
                else:
                    inputs_plus.append(Tensor(inp2.data.copy(), requires_grad=inp2.requires_grad))
            fp = fn(*inputs_plus).data.sum()

            # Perturb -eps
            data_minus = orig_data.copy()
            data_minus[idx] -= eps
            inputs_minus = []
            for j, inp2 in enumerate(inputs):
                if j == i:
                    inputs_minus.append(Tensor(data_minus, requires_grad=True))
                else:
                    inputs_minus.append(Tensor(inp2.data.copy(), requires_grad=inp2.requires_grad))
            fm = fn(*inputs_minus).data.sum()

            numerical[idx] = (fp - fm) / (2 * eps)

        np.testing.assert_allclose(
            analytic[i], numerical, atol=atol, rtol=rtol,
            err_msg=f"Grad check failed for tensor shape {inp.shape}"
        )


# --- Binary ops ---

def test_add_grad():
    a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    _finite_diff_check(lambda a, b: a + b, [a, b])

def test_mul_grad():
    a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    _finite_diff_check(lambda a, b: a * b, [a, b])

def test_div_grad():
    a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.rand(3, 4).astype(np.float32) + 0.5, requires_grad=True)
    _finite_diff_check(lambda a, b: a / b, [a, b])

def test_matmul_grad():
    a = Tensor(np.random.randn(3, 5).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(5, 4).astype(np.float32), requires_grad=True)
    _finite_diff_check(lambda a, b: a @ b, [a, b])

# --- Activations ---

@pytest.mark.parametrize("fn_name", ["relu", "sigmoid", "tanh", "silu", "exp", "abs"])
def test_activation_grad(fn_name):
    x = Tensor(np.random.randn(8).astype(np.float32) * 0.5, requires_grad=True)
    if fn_name == "relu":
        x.data[np.abs(x.data) < 0.01] = 0.5
    _finite_diff_check(lambda x: getattr(x, fn_name)(), [x])

def test_softmax_grad():
    x = Tensor(np.random.randn(2, 5).astype(np.float32), requires_grad=True)
    _finite_diff_check(lambda x: x.softmax(dim=-1), [x])

def test_log_softmax_grad():
    x = Tensor(np.random.randn(2, 5).astype(np.float32), requires_grad=True)
    _finite_diff_check(lambda x: x.log_softmax(dim=-1), [x])

# --- Layers ---

def test_linear_grad():
    np.random.seed(42)
    l = Linear(4, 3)
    x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    _finite_diff_check(lambda x: l(x), [x])

def test_conv2d_grad():
    np.random.seed(42)
    c = Conv2d(1, 2, 3, padding=1)
    x = Tensor(np.random.randn(1, 1, 4, 4).astype(np.float32), requires_grad=True)
    _finite_diff_check(lambda x: c(x), [x])

def test_layernorm_grad():
    np.random.seed(42)
    ln = LayerNorm(8)
    x = Tensor(np.random.randn(2, 8).astype(np.float32), requires_grad=True)
    _finite_diff_check(lambda x: ln(x), [x])

def test_rmsnorm_grad():
    np.random.seed(42)
    rn = RMSNorm(8)
    x = Tensor(np.random.randn(2, 8).astype(np.float32), requires_grad=True)
    _finite_diff_check(lambda x: rn(x), [x])

# --- Broadcast ---

def test_broadcast_add_grad():
    a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(4,).astype(np.float32), requires_grad=True)
    _finite_diff_check(lambda a, b: a + b, [a, b])

def test_broadcast_mul_grad():
    a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(1, 4).astype(np.float32), requires_grad=True)
    _finite_diff_check(lambda a, b: a * b, [a, b])
