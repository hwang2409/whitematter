import json
import logging
import struct
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (32, 32),
        channels: int = 3,
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ):
        self.target_size = target_size
        self.channels = channels
        self.normalize = normalize
        # Default to ImageNet-like stats if not provided
        self.mean = mean or ([0.485, 0.456, 0.406] if channels == 3 else [0.5])
        self.std = std or ([0.229, 0.224, 0.225] if channels == 3 else [0.5])

    def process_dataset(
        self,
        raw_dir: Path,
        output_dir: Path,
        class_names: List[str],
        train_split: float = 0.8
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)

        all_images = []
        all_labels = []

        for class_idx, class_name in enumerate(class_names):
            class_dir = raw_dir / class_name
            if not class_dir.exists():
                continue

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}:
                    try:
                        img_array = self._load_and_preprocess(img_path)
                        all_images.append(img_array)
                        all_labels.append(class_idx)
                    except Exception as e:
                        logger.warning("Failed to load %s: %s", img_path, e)

        if not all_images:
            raise ValueError("No valid images found in dataset")

        images = np.stack(all_images, axis=0)
        labels = np.array(all_labels, dtype=np.float32)


        indices = np.random.permutation(len(images))
        split_idx = int(len(images) * train_split)

        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_images = images[train_indices]
        train_labels = labels[train_indices]
        test_images = images[test_indices]
        test_labels = labels[test_indices]

        if self.normalize:
            self.mean = train_images.mean(axis=(0, 2, 3)).tolist()
            self.std = train_images.std(axis=(0, 2, 3)).tolist()
            self.std = [max(s, 1e-7) for s in self.std]

            mean = np.array(self.mean).reshape(1, -1, 1, 1)
            std = np.array(self.std).reshape(1, -1, 1, 1)
            train_images = (train_images - mean) / std
            test_images = (test_images - mean) / std

        self._save_tensor(output_dir / "train_images.bin", train_images)
        self._save_tensor(output_dir / "train_labels.bin", train_labels)
        self._save_tensor(output_dir / "test_images.bin", test_images)
        self._save_tensor(output_dir / "test_labels.bin", test_labels)

        config = {
            "target_size": list(self.target_size),
            "channels": self.channels,
            "mean": self.mean,
            "std": self.std,
            "num_classes": len(class_names),
            "class_names": class_names,
            "train_samples": len(train_images),
            "test_samples": len(test_images),
            "input_shape": [self.channels, self.target_size[0], self.target_size[1]]
        }
        with open(output_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        return config

    def _load_and_preprocess(self, img_path: Path) -> np.ndarray:
        with Image.open(img_path) as img:
            if self.channels == 3:
                img = img.convert('RGB')
            else:
                img = img.convert('L')

            img = img.resize(self.target_size, Image.Resampling.BILINEAR)

            arr = np.array(img, dtype=np.float32) / 255.0

            if self.channels == 3:
                arr = arr.transpose(2, 0, 1)
            else:
                arr = arr[np.newaxis, ...]  # [H, W] -> [1, H, W]

            return arr

    def _save_tensor(self, path: Path, data: np.ndarray):
        TENSOR_MAGIC = 0x54454E53

        with open(path, 'wb') as f:
            f.write(struct.pack('I', TENSOR_MAGIC))
            f.write(struct.pack('I', len(data.shape)))
            for dim in data.shape:
                f.write(struct.pack('Q', dim))
            data = np.ascontiguousarray(data, dtype=np.float32)
            f.write(data.tobytes())

    @staticmethod
    def load_tensor(path: Path) -> np.ndarray:
        TENSOR_MAGIC = 0x54454E53

        with open(path, 'rb') as f:
            magic = struct.unpack('I', f.read(4))[0]
            if magic != TENSOR_MAGIC:
                raise ValueError(f"Invalid tensor file: {path}")

            ndim = struct.unpack('I', f.read(4))[0]
            shape = []
            for _ in range(ndim):
                shape.append(struct.unpack('Q', f.read(8))[0])

            data = np.frombuffer(f.read(), dtype=np.float32)
            return data.reshape(shape)
