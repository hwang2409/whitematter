#!/usr/bin/env python3
"""Download and preprocess microsoft/cats_vs_dogs for whitematter C++ training.

Run from repo root:
    pip install datasets Pillow numpy
    python examples/preprocess_cats_dogs.py

Produces data/cats_dogs/{train,test}_{images,labels}.bin in whitematter tensor
format, ready for the C++ dataloader.
"""

import io
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SIZE = (224, 224)
CHANNELS = 3
TOTAL_IMAGES = 2_000
TRAIN_SPLIT = 0.8
SEED = 42

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "cats_dogs"

TENSOR_MAGIC = 0x54454E53  # "TENS" in ASCII


# ---------------------------------------------------------------------------
# Tensor I/O — matches platform/preprocessing/image_processor.py
# ---------------------------------------------------------------------------
def save_tensor(filepath: Path, data: np.ndarray) -> None:
    """Save a numpy array in the whitematter binary tensor format."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    with open(filepath, "wb") as f:
        f.write(struct.pack("<I", TENSOR_MAGIC))
        f.write(struct.pack("<I", data.ndim))
        for dim in data.shape:
            f.write(struct.pack("<Q", dim))
        f.write(data.tobytes())


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def load_and_preprocess(image: Image.Image) -> np.ndarray:
    """Resize to 224x224 RGB, normalise to [0,1], return CHW float32 array."""
    img = image.convert("RGB")
    img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # HWC, [0,1]
    arr = arr.transpose(2, 0, 1)  # CHW
    return arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "The 'datasets' package is required.  Install it with:\n"
            "    pip install datasets"
        )

    np.random.seed(SEED)

    # ------------------------------------------------------------------
    # 1. Download dataset from HuggingFace
    # ------------------------------------------------------------------
    print("Downloading microsoft/cats_vs_dogs from HuggingFace ...")
    t0 = time.time()
    ds = load_dataset("microsoft/cats_vs_dogs", split="train")
    print(f"  Downloaded {len(ds)} samples in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 2. Shuffle and truncate to TOTAL_IMAGES
    # ------------------------------------------------------------------
    ds = ds.shuffle(seed=SEED)
    if len(ds) > TOTAL_IMAGES:
        ds = ds.select(range(TOTAL_IMAGES))
    print(f"  Using {len(ds)} samples (truncated to {TOTAL_IMAGES})")

    # ------------------------------------------------------------------
    # 3. Process images — resize, normalise, convert to CHW
    # ------------------------------------------------------------------
    images = []
    labels = []
    skipped = 0
    t0 = time.time()

    for idx, sample in enumerate(ds):
        if (idx + 1) % 500 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(
                f"  Processing {idx + 1}/{len(ds)}  "
                f"({len(images)} valid, {skipped} skipped)  "
                f"[{elapsed:.1f}s]"
            )

        try:
            pil_image = sample["image"]
            # Some entries may be stored as bytes rather than a PIL Image
            if not isinstance(pil_image, Image.Image):
                pil_image = Image.open(io.BytesIO(pil_image))
            arr = load_and_preprocess(pil_image)
            images.append(arr)
            labels.append(int(sample["labels"]))
        except Exception as exc:
            skipped += 1
            if skipped <= 10:
                print(f"    [warn] Skipping sample {idx}: {exc}")

    print(
        f"  Done processing. {len(images)} valid images, "
        f"{skipped} corrupt/skipped in {time.time() - t0:.1f}s"
    )

    if not images:
        sys.exit("No valid images found — cannot continue.")

    # ------------------------------------------------------------------
    # 4. Stack into numpy arrays
    # ------------------------------------------------------------------
    all_images = np.stack(images, axis=0)  # [N, 3, 224, 224]
    all_labels = np.array(labels, dtype=np.float32)  # [N]

    # ------------------------------------------------------------------
    # 5. Train / test split
    # ------------------------------------------------------------------
    n = len(all_images)
    indices = np.random.permutation(n)
    split_idx = int(n * TRAIN_SPLIT)

    train_images = all_images[indices[:split_idx]]
    train_labels = all_labels[indices[:split_idx]]
    test_images = all_images[indices[split_idx:]]
    test_labels = all_labels[indices[split_idx:]]

    print(f"  Train: {train_images.shape}  Test: {test_images.shape}")

    # ------------------------------------------------------------------
    # 6. Save in whitematter tensor format
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, tensor in [
        ("train_images.bin", train_images),
        ("train_labels.bin", train_labels),
        ("test_images.bin", test_images),
        ("test_labels.bin", test_labels),
    ]:
        path = OUTPUT_DIR / name
        save_tensor(path, tensor)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  Saved {path}  shape={tensor.shape}  ({size_mb:.1f} MB)")

    # ------------------------------------------------------------------
    # 7. Write a small metadata file (useful for the C++ side)
    # ------------------------------------------------------------------
    config = {
        "target_size": list(TARGET_SIZE),
        "channels": CHANNELS,
        "num_classes": 2,
        "class_names": ["cat", "dog"],
        "train_samples": int(train_images.shape[0]),
        "test_samples": int(test_images.shape[0]),
        "input_shape": [CHANNELS, TARGET_SIZE[0], TARGET_SIZE[1]],
        "normalization": "divide_by_255",
    }
    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved {config_path}")

    print("\nAll done. You can now train with the C++ library using:")
    print(f"  data dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
