"""DataLoader with shuffle and optional threaded prefetch."""

from __future__ import annotations

import numpy as np
from typing import Iterator, Tuple, Optional
from .dataset import Dataset
from ..tensor import Tensor

try:
    from concurrent.futures import ThreadPoolExecutor
except ImportError:
    ThreadPoolExecutor = None


class DataLoader:
    """Iterate over a dataset in batches."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)

        def _make_batch(batch_indices):
            items = [self.dataset[i] for i in batch_indices]
            xs = np.stack([it[0] for it in items])
            ys = np.stack([it[1] for it in items])
            return Tensor(xs), Tensor(ys)

        starts = list(range(0, n, self.batch_size))
        batches_indices = []
        for s in starts:
            end = s + self.batch_size
            if end > n and self.drop_last:
                continue
            batches_indices.append(indices[s : min(end, n)])

        if self.num_workers > 0 and ThreadPoolExecutor is not None:
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                yield from pool.map(_make_batch, batches_indices)
        else:
            for bi in batches_indices:
                yield _make_batch(bi)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
