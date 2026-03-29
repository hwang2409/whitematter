#!/usr/bin/env python3
"""Download and preprocess ImageNette2-320 for whitematter C++ training.

Run from repo root:
    pip install Pillow numpy
    python examples/preprocess_imagenette.py

Produces data/imagenette/{train,test}_{images,labels}.bin in whitematter tensor
format, ready for the C++ dataloader.
"""

import json
import os
import struct
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SIZE = (224, 224)
CHANNELS = 3

DOWNLOAD_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "imagenette"
DOWNLOAD_DIR = REPO_ROOT / "data" / "imagenette" / "_raw"

TENSOR_MAGIC = 0x54454E53  # "TENS" in ASCII

# Synset ID -> (label index, human-readable name)
SYNSET_MAP = {
    "n01440764": (0, "tench"),
    "n02102040": (1, "English springer"),
    "n02979186": (2, "cassette player"),
    "n03000684": (3, "chain saw"),
    "n03028079": (4, "church"),
    "n03394916": (5, "French horn"),
    "n03417042": (6, "garbage truck"),
    "n03425413": (7, "gas pump"),
    "n03445777": (8, "golf ball"),
    "n03888257": (9, "parachute"),
}

# ImageNet normalization (stored in config, not applied here — raw [0,1] pixels)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
# Download
# ---------------------------------------------------------------------------
def download_imagenette(download_dir: Path) -> Path:
    """Download and extract ImageNette2-320, return path to extracted dir."""
    download_dir.mkdir(parents=True, exist_ok=True)
    tgz_path = download_dir / "imagenette2-320.tgz"
    extracted_dir = download_dir / "imagenette2-320"

    if extracted_dir.exists():
        print("ImageNette2-320 already extracted, skipping download.")
        return extracted_dir

    if not tgz_path.exists():
        print("Downloading ImageNette2-320...")
        t0 = time.time()

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(
                    f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct}%)"
                )
                sys.stdout.flush()

        urllib.request.urlretrieve(DOWNLOAD_URL, tgz_path, reporthook=_progress)
        print(f"\n  Downloaded in {time.time() - t0:.1f}s")
    else:
        print("Archive already downloaded, skipping download.")

    print("Extracting...")
    t0 = time.time()
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=download_dir)
    print(f"  Extracted in {time.time() - t0:.1f}s")

    return extracted_dir


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def load_and_preprocess(image_path: Path) -> np.ndarray:
    """Resize shorter side to 256, center crop 224x224, normalise to [0,1], CHW."""
    img = Image.open(image_path).convert("RGB")

    # Resize so shorter side is 256, maintain aspect ratio
    w, h = img.size
    if w < h:
        new_w = 256
        new_h = int(h * 256 / w)
    else:
        new_h = 256
        new_w = int(w * 256 / h)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Center crop to 224x224
    w, h = img.size
    left = (w - TARGET_SIZE[0]) // 2
    top = (h - TARGET_SIZE[1]) // 2
    img = img.crop((left, top, left + TARGET_SIZE[0], top + TARGET_SIZE[1]))

    arr = np.array(img, dtype=np.float32) / 255.0  # HWC, [0,1]
    arr = arr.transpose(2, 0, 1)  # CHW
    return arr


# ---------------------------------------------------------------------------
# Process a split (train or val)
# ---------------------------------------------------------------------------
def process_split(split_dir: Path, split_name: str, images_path: Path, labels_path: Path) -> int:
    """Process all images and stream directly to binary files (low memory).
    Returns the number of valid images."""
    labels = []
    skipped = 0
    t0 = time.time()

    # Collect all image paths with their labels
    image_entries = []
    for synset_dir in sorted(split_dir.iterdir()):
        if not synset_dir.is_dir():
            continue
        synset_id = synset_dir.name
        if synset_id not in SYNSET_MAP:
            continue
        label_idx, _ = SYNSET_MAP[synset_id]
        for img_path in sorted(synset_dir.iterdir()):
            if img_path.suffix.lower() in (".jpeg", ".jpg", ".png"):
                image_entries.append((img_path, label_idx))

    total = len(image_entries)
    print(f"Processing {split_name} set: {total} images")

    # First pass: process images and collect valid ones + labels
    valid_count = 0
    # Write images to a temp file, then prepend header
    tmp_path = images_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as img_f:
        for idx, (img_path, label_idx) in enumerate(image_entries):
            if (idx + 1) % 1000 == 0 or idx == 0:
                elapsed = time.time() - t0
                print(
                    f"  {idx + 1}/{total}  "
                    f"({valid_count} valid, {skipped} skipped)  "
                    f"[{elapsed:.1f}s]"
                )
            try:
                arr = load_and_preprocess(img_path)
                img_f.write(arr.tobytes())
                labels.append(label_idx)
                valid_count += 1
            except Exception as exc:
                skipped += 1
                if skipped <= 10:
                    print(f"    [warn] Skipping {img_path.name}: {exc}")

    elapsed = time.time() - t0
    print(
        f"  Done. {valid_count} valid images, "
        f"{skipped} corrupt/skipped in {elapsed:.1f}s"
    )

    if valid_count == 0:
        sys.exit(f"No valid images found in {split_dir}")

    # Write final images file with header
    with open(images_path, "wb") as f:
        f.write(struct.pack("<I", TENSOR_MAGIC))
        f.write(struct.pack("<I", 4))  # ndim=4
        for dim in [valid_count, 3, TARGET_SIZE[0], TARGET_SIZE[1]]:
            f.write(struct.pack("<Q", dim))
        # Append raw data from temp file
        with open(tmp_path, "rb") as tmp:
            while True:
                chunk = tmp.read(64 * 1024 * 1024)  # 64MB chunks
                if not chunk:
                    break
                f.write(chunk)
    tmp_path.unlink()

    # Write labels
    all_labels = np.array(labels, dtype=np.float32)
    save_tensor(labels_path, all_labels)

    return valid_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ------------------------------------------------------------------
    # 1. Download and extract
    # ------------------------------------------------------------------
    extracted_dir = download_imagenette(DOWNLOAD_DIR)

    train_dir = extracted_dir / "train"
    val_dir = extracted_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        sys.exit(
            f"Expected train/ and val/ directories in {extracted_dir}, "
            "but they are missing."
        )

    # ------------------------------------------------------------------
    # 2. Process and save (streaming — low memory)
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_count = process_split(train_dir, "train",
                                OUTPUT_DIR / "train_images.bin",
                                OUTPUT_DIR / "train_labels.bin")
    test_count = process_split(val_dir, "val",
                               OUTPUT_DIR / "test_images.bin",
                               OUTPUT_DIR / "test_labels.bin")

    for name in ["train_images.bin", "train_labels.bin", "test_images.bin", "test_labels.bin"]:
        path = OUTPUT_DIR / name
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path}  ({size_mb:.1f} MB)")

    # ------------------------------------------------------------------
    # 4. Write config metadata
    # ------------------------------------------------------------------
    class_names = [
        name for _, (_, name) in sorted(
            SYNSET_MAP.items(), key=lambda item: item[1][0]
        )
    ]

    config = {
        "target_size": list(TARGET_SIZE),
        "channels": CHANNELS,
        "num_classes": 10,
        "class_names": class_names,
        "synset_ids": sorted(SYNSET_MAP.keys(), key=lambda s: SYNSET_MAP[s][0]),
        "train_samples": train_count,
        "test_samples": test_count,
        "input_shape": [CHANNELS, TARGET_SIZE[0], TARGET_SIZE[1]],
        "normalization": "divide_by_255",
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
    }
    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved {config_path}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print(f"\nSaved to {OUTPUT_DIR}/")
    print(f"  train: {train_count} images [{train_count}, 3, {TARGET_SIZE[0]}, {TARGET_SIZE[1]}]")
    print(f"  test:  {test_count} images [{test_count}, 3, {TARGET_SIZE[0]}, {TARGET_SIZE[1]}]")
    print("\nAll done. You can now train with the C++ library using:")
    print(f"  data dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
