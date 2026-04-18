"""Tests for neural network layers."""

import numpy as np
import pytest
from whitematter import Tensor
from whitematter.nn import (
    Linear, ReLU, Sequential, Flatten,
    Conv2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d,
    BatchNorm2d, LayerNorm, RMSNorm,
    Dropout, Embedding,
    MultiHeadAttention,
)
from whitematter.nn.attention import causal_mask


class TestLinear:
    def test_forward(self):
        l = Linear(4, 3)
        x = Tensor(np.random.randn(2, 4).astype(np.float32))
        y = l(x)
        assert y.shape == (2, 3)

    def test_backward(self):
        l = Linear(4, 3)
        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        y = l(x)
        loss = y.sum()
        loss.backward()
        assert l.weight.grad is not None
        assert l.bias.grad is not None
        assert x.grad is not None


class TestConv2d:
    def test_forward(self):
        c = Conv2d(3, 16, 3, padding=1)
        x = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))
        y = c(x)
        assert y.shape == (2, 16, 8, 8)

    def test_backward(self):
        c = Conv2d(1, 4, 3, padding=1)
        x = Tensor(np.random.randn(1, 1, 6, 6).astype(np.float32), requires_grad=True)
        y = c(x)
        loss = y.sum()
        loss.backward()
        assert c.weight.grad is not None
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_stride(self):
        c = Conv2d(1, 1, 3, stride=2, padding=1)
        x = Tensor(np.random.randn(1, 1, 8, 8).astype(np.float32))
        y = c(x)
        assert y.shape == (1, 1, 4, 4)


class TestPooling:
    def test_maxpool(self):
        p = MaxPool2d(2)
        x = Tensor(np.random.randn(1, 1, 4, 4).astype(np.float32), requires_grad=True)
        y = p(x)
        assert y.shape == (1, 1, 2, 2)
        y.sum().backward()
        assert x.grad is not None

    def test_avgpool(self):
        p = AvgPool2d(2)
        x = Tensor(np.random.randn(1, 1, 4, 4).astype(np.float32), requires_grad=True)
        y = p(x)
        assert y.shape == (1, 1, 2, 2)

    def test_adaptive_avgpool(self):
        p = AdaptiveAvgPool2d(1)
        x = Tensor(np.random.randn(2, 3, 7, 7).astype(np.float32), requires_grad=True)
        y = p(x)
        assert y.shape == (2, 3, 1, 1)


class TestNormalization:
    def test_batchnorm2d(self):
        bn = BatchNorm2d(8)
        x = Tensor(np.random.randn(4, 8, 3, 3).astype(np.float32), requires_grad=True)
        y = bn(x)
        assert y.shape == x.shape
        y.sum().backward()
        assert bn.weight.grad is not None

    def test_layernorm(self):
        ln = LayerNorm(16)
        x = Tensor(np.random.randn(2, 10, 16).astype(np.float32), requires_grad=True)
        y = ln(x)
        assert y.shape == x.shape
        y.sum().backward()
        assert x.grad is not None

    def test_rmsnorm(self):
        rn = RMSNorm(16)
        x = Tensor(np.random.randn(2, 10, 16).astype(np.float32), requires_grad=True)
        y = rn(x)
        assert y.shape == x.shape


class TestDropout:
    def test_train_mode(self):
        d = Dropout(0.5)
        x = Tensor(np.ones((100, 100)).astype(np.float32))
        y = d(x)
        # Some values should be zero
        assert (y.data == 0).any()
        # Mean should be approximately 1 due to inverted dropout
        np.testing.assert_allclose(y.data.mean(), 1.0, atol=0.1)

    def test_eval_mode(self):
        d = Dropout(0.5)
        d.eval()
        x = Tensor(np.ones((10, 10)).astype(np.float32))
        y = d(x)
        np.testing.assert_array_equal(y.data, x.data)


class TestEmbedding:
    def test_forward(self):
        e = Embedding(100, 16)
        idx = Tensor(np.array([0, 5, 10, 50]))
        y = e(idx)
        assert y.shape == (4, 16)

    def test_backward(self):
        e = Embedding(10, 4)
        idx = Tensor(np.array([0, 1, 0]))
        y = e(idx)
        y.sum().backward()
        assert e.weight.grad is not None
        # Index 0 appears twice, so its gradient should be double
        np.testing.assert_array_almost_equal(
            e.weight.grad[0], np.ones(4) * 2
        )


class TestAttention:
    def test_mha_forward(self):
        mha = MultiHeadAttention(32, 4)
        x = Tensor(np.random.randn(2, 10, 32).astype(np.float32), requires_grad=True)
        y = mha(x)
        assert y.shape == (2, 10, 32)

    def test_mha_causal(self):
        mha = MultiHeadAttention(16, 2)
        x = Tensor(np.random.randn(1, 5, 16).astype(np.float32), requires_grad=True)
        mask = causal_mask(5)
        y = mha(x, mask=mask)
        assert y.shape == (1, 5, 16)
        y.sum().backward()
        assert x.grad is not None


class TestSequential:
    def test_forward(self):
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2),
        )
        x = Tensor(np.random.randn(3, 4).astype(np.float32))
        y = model(x)
        assert y.shape == (3, 2)

    def test_parameters(self):
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2),
        )
        params = model.parameters()
        # Linear(4,8): weight + bias, Linear(8,2): weight + bias = 4 params
        assert len(params) == 4

    def test_train_eval(self):
        model = Sequential(Linear(4, 4), Dropout(0.5))
        model.eval()
        assert not model.training
        model.train()
        assert model.training
