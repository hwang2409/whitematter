"""Model serialization via numpy."""

from __future__ import annotations

import numpy as np
from .tensor import Tensor
from .nn.module import Module


def save(path: str, model: Module):
    state = {}
    for name, param in model.named_parameters():
        state[name] = param.data
    np.savez(path, **state)


def load(path: str, model: Module):
    if not path.endswith(".npz"):
        path = path + ".npz"
    data = np.load(path)
    own = dict(model.named_parameters())
    for name in data.files:
        if name in own:
            own[name].data[:] = data[name].astype(np.float32)
