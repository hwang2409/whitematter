"""Tests for loss functions."""

import numpy as np
import pytest
from whitematter import Tensor
from whitematter.nn import (
    MSELoss, L1Loss, SmoothL1Loss,
    CrossEntropyLoss, NLLLoss,
    BCELoss, BCEWithLogitsLoss,
)


def _grad_check(fn, inputs, eps=1e-4, atol=1e-2, rtol=5e-2):
    out = fn(*inputs)
    out.backward(np.ones_like(out.data))
    analytic = [inp.grad.copy() for inp in inputs if inp.requires_grad and inp.grad is not None]

    idx = 0
    for k, inp in enumerate(inputs):
        if not inp.requires_grad:
            continue
        numerical = np.zeros_like(inp.data)
        orig_data = inp.data.copy()
        for i in np.ndindex(inp.shape):
            data_p = orig_data.copy()
            data_p[i] += eps
            data_m = orig_data.copy()
            data_m[i] -= eps
            inp_p = [Tensor(data_p, requires_grad=True) if j == k else Tensor(x.data.copy(), requires_grad=x.requires_grad) for j, x in enumerate(inputs)]
            inp_m = [Tensor(data_m, requires_grad=True) if j == k else Tensor(x.data.copy(), requires_grad=x.requires_grad) for j, x in enumerate(inputs)]
            numerical[i] = (fn(*inp_p).data.sum() - fn(*inp_m).data.sum()) / (2 * eps)

        np.testing.assert_allclose(analytic[idx], numerical, atol=atol, rtol=rtol)
        idx += 1


class TestMSE:
    def test_forward(self):
        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([1.0, 2.0, 3.0])
        loss = MSELoss()(pred, target)
        np.testing.assert_almost_equal(loss.item(), 0.0)

    def test_grad(self):
        pred = Tensor(np.random.randn(5).astype(np.float32), requires_grad=True)
        target = Tensor(np.random.randn(5).astype(np.float32))
        _grad_check(lambda p, t: MSELoss()(p, t), [pred, target])


class TestCrossEntropy:
    def test_forward(self):
        logits = Tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        targets = Tensor(np.array([0, 1]))
        loss = CrossEntropyLoss()(logits, targets)
        assert loss.item() < 0.1

    def test_grad(self):
        logits = Tensor(np.random.randn(4, 5).astype(np.float32), requires_grad=True)
        targets = Tensor(np.array([0, 1, 2, 3]))
        _grad_check(lambda l, t: CrossEntropyLoss()(l, t), [logits, targets])


class TestBCE:
    def test_forward(self):
        pred = Tensor([0.9, 0.1])
        target = Tensor([1.0, 0.0])
        loss = BCELoss()(pred, target)
        assert loss.item() < 0.2

    def test_logits_grad(self):
        logits = Tensor(np.random.randn(4).astype(np.float32), requires_grad=True)
        target = Tensor(np.array([1, 0, 1, 0], dtype=np.float32))
        _grad_check(lambda l, t: BCEWithLogitsLoss()(l, t), [logits, target])


class TestL1:
    def test_forward(self):
        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([1.5, 2.5, 3.5])
        loss = L1Loss()(pred, target)
        np.testing.assert_almost_equal(loss.item(), 0.5)


class TestSmoothL1:
    def test_forward(self):
        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([1.5, 2.5, 3.5])
        loss = SmoothL1Loss()(pred, target)
        assert loss.item() > 0
