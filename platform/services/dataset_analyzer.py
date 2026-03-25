import logging
import struct
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

from db import DatasetFormat, DataType

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

logger = logging.getLogger(__name__)


def detect_format(raw_dir: Path) -> Tuple[DatasetFormat, Dict]:
    """Detect dataset format from extracted files."""
    all_items = list(raw_dir.iterdir())
    files = [f for f in all_items if f.is_file()]
    dirs = [d for d in all_items if d.is_dir()]

    # Check for MNIST IDX format
    idx_images = [f for f in files if
                  ('images' in f.name.lower() or 'image' in f.name.lower()) and
                  ('ubyte' in f.name.lower() or 'idx' in f.name.lower())]
    idx_labels = [f for f in files if
                  ('labels' in f.name.lower() or 'label' in f.name.lower()) and
                  ('ubyte' in f.name.lower() or 'idx' in f.name.lower())]

    if idx_images and idx_labels:
        train_images = [f for f in idx_images if 'train' in f.name.lower()]
        train_labels = [f for f in idx_labels if 'train' in f.name.lower()]
        return DatasetFormat.MNIST_IDX, {
            "images_file": train_images[0] if train_images else idx_images[0],
            "labels_file": train_labels[0] if train_labels else idx_labels[0],
            "all_images": idx_images,
            "all_labels": idx_labels
        }

    # Check for folder-per-class
    if dirs:
        has_samples = any(any(d.iterdir()) for d in dirs)
        if has_samples:
            return DatasetFormat.FOLDER_PER_CLASS, {"class_dirs": dirs}

    # Flat images
    image_files = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
    if image_files:
        return DatasetFormat.FLAT_IMAGES, {"image_files": image_files}

    return DatasetFormat.UNKNOWN, {"files": files, "dirs": dirs}


def analyze_dataset(format_type: DatasetFormat, format_info: Dict, raw_dir: Path) -> Dict:
    """Analyze dataset and return metadata."""
    if format_type == DatasetFormat.MNIST_IDX:
        return _analyze_mnist_idx(format_info)
    elif format_type == DatasetFormat.FOLDER_PER_CLASS:
        return _analyze_folder_per_class(format_info)
    elif format_type == DatasetFormat.FLAT_IMAGES:
        return _analyze_flat_images(format_info)
    else:
        return {
            'data_type': DataType.UNKNOWN.value,
            'num_classes': 0,
            'class_names': [],
            'total_samples': 0,
            'input_shape': []
        }


def _analyze_mnist_idx(info: Dict) -> Dict:
    """Analyze MNIST IDX format."""
    images_file = info["images_file"]
    labels_file = info["labels_file"]

    with open(images_file, 'rb') as f:
        struct.unpack('>I', f.read(4))  # magic
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]

    with open(labels_file, 'rb') as f:
        struct.unpack('>I', f.read(4))  # magic
        struct.unpack('>I', f.read(4))  # num_labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    unique_labels = sorted(set(labels))
    class_names = [str(l) for l in unique_labels]
    samples_per_class = {str(l): int(np.sum(labels == l)) for l in unique_labels}

    return {
        'data_type': DataType.IMAGE.value,
        'num_classes': len(unique_labels),
        'class_names': class_names,
        'samples_per_class': samples_per_class,
        'total_samples': num_images,
        'input_shape': [1, rows, cols]
    }


def _analyze_folder_per_class(info: Dict) -> Dict:
    """Analyze folder-per-class format."""
    class_dirs = info["class_dirs"]
    class_names = sorted([d.name for d in class_dirs])
    samples_per_class = {}
    sample_files = []

    for class_dir in class_dirs:
        files = [f for f in class_dir.iterdir() if f.is_file()]
        samples_per_class[class_dir.name] = len(files)
        sample_files.extend(files[:5])

    # Determine input shape from sample
    input_shape = [3, 32, 32]  # Default
    if sample_files:
        try:
            img = Image.open(sample_files[0])
            w, h = img.size
            channels = 3 if img.mode == 'RGB' else 1
            target = min(max(w, h, 32), 224)
            if target <= 32:
                target = 32
            elif target <= 64:
                target = 64
            elif target <= 128:
                target = 128
            else:
                target = 224
            input_shape = [channels, target, target]
        except:
            pass

    return {
        'data_type': DataType.IMAGE.value,
        'num_classes': len(class_names),
        'class_names': class_names,
        'samples_per_class': samples_per_class,
        'total_samples': sum(samples_per_class.values()),
        'input_shape': input_shape
    }


def _analyze_flat_images(info: Dict) -> Dict:
    """Analyze flat images folder."""
    image_files = info["image_files"]

    input_shape = [3, 32, 32]
    if image_files:
        try:
            img = Image.open(image_files[0])
            w, h = img.size
            channels = 3 if img.mode == 'RGB' else 1
            target = min(max(w, h, 32), 224)
            input_shape = [channels, target, target]
        except:
            pass

    return {
        'data_type': DataType.IMAGE.value,
        'num_classes': 0,
        'class_names': [],
        'total_samples': len(image_files),
        'input_shape': input_shape
    }
