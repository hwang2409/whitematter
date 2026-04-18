"""CIFAR-10 dataset loader."""

from __future__ import annotations

import os
import pickle
import numpy as np
from .dataset import Dataset


def _load_batch(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    images = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(d[b"labels"], dtype=np.int64)
    return images, labels


class CIFAR10(Dataset):
    """CIFAR-10 dataset (requires pre-downloaded data)."""

    CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    def __init__(self, root: str = "./data/cifar10", train: bool = True):
        super().__init__()
        if train:
            images_list, labels_list = [], []
            for i in range(1, 6):
                path = os.path.join(root, f"data_batch_{i}")
                imgs, lbls = _load_batch(path)
                images_list.append(imgs)
                labels_list.append(lbls)
            self.images = np.concatenate(images_list)
            self.labels = np.concatenate(labels_list)
        else:
            path = os.path.join(root, "test_batch")
            self.images, self.labels = _load_batch(path)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
