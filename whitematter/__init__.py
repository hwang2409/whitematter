"""WhiteMatter — a pure-Python deep learning framework built on NumPy."""

from .tensor import Tensor
from .autograd import no_grad, is_grad_enabled
from . import nn
from . import optim
from . import data
from . import serialization

__version__ = "1.0.0"
__all__ = ["Tensor", "no_grad", "is_grad_enabled", "nn", "optim", "data", "serialization"]
