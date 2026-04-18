"""Base dataset class."""

from __future__ import annotations

from typing import Tuple
import numpy as np


class Dataset:
    """Abstract base dataset. Subclass and implement __len__ and __getitem__."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
