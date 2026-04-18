"""Tests for optimizers."""

import numpy as np
from whitematter import Tensor
from whitematter.nn import Linear, Sequential, ReLU, MSELoss
from whitematter.optim import SGD, Adam, AdamW, RMSprop


def _train_xor(optimizer_cls, lr=0.1, **kwargs):
    """Train XOR to verify optimizer works."""
    np.random.seed(42)
    model = Sequential(Linear(2, 16), ReLU(), Linear(16, 1))
    opt = optimizer_cls(model.parameters(), lr=lr, **kwargs)
    loss_fn = MSELoss()

    X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
    Y = Tensor(np.array([[0], [1], [1], [0]], dtype=np.float32))

    for _ in range(500):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()

    return loss.item()


def test_sgd():
    loss = _train_xor(SGD, lr=0.5, momentum=0.9)
    assert loss < 0.05, f"SGD failed to converge: loss={loss}"


def test_adam():
    loss = _train_xor(Adam, lr=0.01)
    assert loss < 0.05, f"Adam failed to converge: loss={loss}"


def test_adamw():
    loss = _train_xor(AdamW, lr=0.01, weight_decay=0.01)
    assert loss < 0.05, f"AdamW failed to converge: loss={loss}"


def test_rmsprop():
    loss = _train_xor(RMSprop, lr=0.01)
    assert loss < 0.05, f"RMSprop failed to converge: loss={loss}"


def test_zero_grad():
    model = Sequential(Linear(2, 4), Linear(4, 1))
    opt = SGD(model.parameters(), lr=0.01)
    x = Tensor(np.ones((1, 2), dtype=np.float32), requires_grad=True)
    y = model(x).sum()
    y.backward()
    for p in model.parameters():
        assert p.grad is not None
    opt.zero_grad()
    for p in model.parameters():
        assert p.grad is None
