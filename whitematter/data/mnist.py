"""MNIST dataset loader."""

from __future__ import annotations

import gzip
import os
import struct
import numpy as np
from .dataset import Dataset

_URLS = {
    "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}


def _read_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    return data.astype(np.float32) / 255.0


def _read_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def _download(url: str, dest: str):
    import urllib.request
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, dest)


class MNIST(Dataset):
    """MNIST handwritten digits dataset."""

    def __init__(self, root: str = "./data/mnist", train: bool = True, flatten: bool = True):
        super().__init__()
        self.flatten = flatten

        prefix = "train" if train else "test"
        img_file = os.path.join(root, f"{prefix}_images.gz")
        lbl_file = os.path.join(root, f"{prefix}_labels.gz")

        # Try to download if not present
        if not os.path.exists(img_file):
            key = f"{'train' if train else 'test'}_images"
            _download(_URLS[key], img_file)
        if not os.path.exists(lbl_file):
            key = f"{'train' if train else 'test'}_labels"
            _download(_URLS[key], lbl_file)

        self.images = _read_idx_images(img_file)
        self.labels = _read_idx_labels(lbl_file)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index):
        img = self.images[index]
        if self.flatten:
            img = img.reshape(-1)  # 784
        else:
            img = img.reshape(1, 28, 28)  # (1, H, W) for conv
        return img, self.labels[index]
