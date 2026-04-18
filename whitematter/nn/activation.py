"""Activation function modules."""

from ..tensor import Tensor
from .module import Module


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.silu()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class Mish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.mish()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Softmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(self.dim)


class LogSoftmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.log_softmax(self.dim)
