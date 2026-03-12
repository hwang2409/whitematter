#!/usr/bin/env python3
"""
One-time setup: Download MNIST, process into .bin format, save to presets/mnist/.
In production, upload the resulting files to R2 under presets/mnist/.
"""
import struct
import gzip
import json
import shutil
import urllib.request
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PRESETS_DIR = PROJECT_ROOT / "presets" / "mnist"
RAW_DIR = PRESETS_DIR / "raw"

MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_mnist():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MNIST_URLS.items():
        dest = RAW_DIR / name.replace(".gz", "")
        if dest.exists():
            print(f"  Already exists: {dest.name}")
            continue
        print(f"  Downloading {name}...")
        data = urllib.request.urlopen(url).read()
        with open(dest, "wb") as f:
            f.write(gzip.decompress(data))


def save_tensor(path, data):
    TENSOR_MAGIC = 0x54454E53
    with open(path, "wb") as f:
        f.write(struct.pack("I", TENSOR_MAGIC))
        f.write(struct.pack("I", len(data.shape)))
        for dim in data.shape:
            f.write(struct.pack("Q", dim))
        data = np.ascontiguousarray(data, dtype=np.float32)
        f.write(data.tobytes())


def read_idx_images(filepath):
    with open(filepath, "rb") as f:
        struct.unpack(">I", f.read(4))  # magic
        num = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, 1, rows, cols).astype(np.float32) / 255.0, rows, cols


def read_idx_labels(filepath):
    with open(filepath, "rb") as f:
        struct.unpack(">I", f.read(4))  # magic
        struct.unpack(">I", f.read(4))  # num
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32)


def main():
    print("Step 1: Downloading MNIST...")
    download_mnist()

    print("Step 2: Processing...")
    train_images, rows, cols = read_idx_images(RAW_DIR / "train-images-idx3-ubyte")
    train_labels = read_idx_labels(RAW_DIR / "train-labels-idx1-ubyte")
    test_images, _, _ = read_idx_images(RAW_DIR / "t10k-images-idx3-ubyte")
    test_labels = read_idx_labels(RAW_DIR / "t10k-labels-idx1-ubyte")

    mean = [float(train_images.mean())]
    std_val = [float(max(train_images.std(), 1e-7))]
    train_images = (train_images - mean[0]) / std_val[0]
    test_images = (test_images - mean[0]) / std_val[0]

    print("Step 3: Saving .bin files...")
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    save_tensor(PRESETS_DIR / "train_images.bin", train_images)
    save_tensor(PRESETS_DIR / "train_labels.bin", train_labels)
    save_tensor(PRESETS_DIR / "test_images.bin", test_images)
    save_tensor(PRESETS_DIR / "test_labels.bin", test_labels)

    config = {
        "target_size": [rows, cols],
        "channels": 1,
        "mean": mean,
        "std": std_val,
        "num_classes": 10,
        "class_names": [str(i) for i in range(10)],
        "train_samples": len(train_images),
        "test_samples": len(test_images),
        "input_shape": [1, rows, cols],
    }
    with open(PRESETS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Cleanup raw files
    shutil.rmtree(RAW_DIR)

    total_mb = sum(f.stat().st_size for f in PRESETS_DIR.glob("*.bin")) / 1024 / 1024
    print(f"Done! Saved to {PRESETS_DIR} ({total_mb:.1f} MB)")
    print("For production: upload presets/mnist/ contents to R2 bucket under presets/mnist/")


if __name__ == "__main__":
    main()
