"""
Dataset Manager - handles dataset upload, extraction, validation, and processing.
Supports multiple dataset formats: folder-per-class, MNIST IDX, CSV, and more.
"""

import json
import shutil
import struct
import uuid
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


class DataType(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    TABULAR = "tabular"
    UNKNOWN = "unknown"


class DatasetFormat(str, Enum):
    FOLDER_PER_CLASS = "folder_per_class"  # Standard: class_name/images
    MNIST_IDX = "mnist_idx"  # MNIST binary IDX format
    CSV_LABELS = "csv_labels"  # CSV with image paths and labels
    FLAT_IMAGES = "flat_images"  # Flat folder with images (no labels)
    UNKNOWN = "unknown"


@dataclass
class DatasetMetadata:
    id: str
    name: str
    data_type: DataType
    created_at: str
    num_classes: int
    class_names: List[str]
    samples_per_class: Dict[str, int]
    total_samples: int
    input_shape: List[int]
    format: str = "unknown"
    status: str = "processing"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d['data_type'] = self.data_type.value
        return d


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
TEXT_EXTENSIONS = {'.txt'}
TABULAR_EXTENSIONS = {'.csv', '.tsv'}


class DatasetManager:
    def __init__(self, uploads_dir: Path = Path("uploads")):
        self.uploads_dir = uploads_dir
        self.uploads_dir.mkdir(exist_ok=True)

    def create_dataset(self, name: Optional[str] = None) -> str:
        """Create a new dataset and return its ID."""
        dataset_id = str(uuid.uuid4())[:8]
        dataset_dir = self.uploads_dir / dataset_id
        dataset_dir.mkdir(exist_ok=True)
        (dataset_dir / "raw").mkdir(exist_ok=True)
        (dataset_dir / "processed").mkdir(exist_ok=True)

        if name is None:
            name = f"dataset_{dataset_id}"

        metadata = DatasetMetadata(
            id=dataset_id,
            name=name,
            data_type=DataType.UNKNOWN,
            created_at=datetime.now().isoformat(),
            num_classes=0,
            class_names=[],
            samples_per_class={},
            total_samples=0,
            input_shape=[],
            format="unknown",
            status="created"
        )
        self._save_metadata(dataset_id, metadata)
        return dataset_id

    def extract_zip(self, dataset_id: str, zip_path: Path) -> DatasetMetadata:
        """Extract ZIP file and analyze contents."""
        dataset_dir = self.uploads_dir / dataset_id
        raw_dir = dataset_dir / "raw"

        metadata = self._load_metadata(dataset_id)
        metadata.status = "extracting"
        self._save_metadata(dataset_id, metadata)

        try:
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(raw_dir)

            # Handle nested folder (common when zipping a folder)
            contents = list(raw_dir.iterdir())
            if len(contents) == 1 and contents[0].is_dir():
                # Move contents up one level
                nested_dir = contents[0]
                for item in nested_dir.iterdir():
                    shutil.move(str(item), str(raw_dir / item.name))
                nested_dir.rmdir()

            # Analyze structure
            metadata = self._analyze_dataset(dataset_id, metadata)
            metadata.status = "ready"
            self._save_metadata(dataset_id, metadata)

        except Exception as e:
            metadata.status = "error"
            metadata.error = str(e)
            self._save_metadata(dataset_id, metadata)
            raise

        return metadata

    def _is_idx_file(self, filepath: Path) -> Tuple[bool, str]:
        """Check if a file is an IDX format file by reading magic number.
        Returns (is_idx, type) where type is 'images' or 'labels' or 'unknown'."""
        try:
            with open(filepath, 'rb') as f:
                magic = struct.unpack('>I', f.read(4))[0]
                # IDX magic numbers: 0x0801 = labels (ubyte), 0x0803 = images (ubyte, 3D)
                if magic == 0x00000801:
                    return True, 'labels'
                elif magic == 0x00000803:
                    return True, 'images'
                # Also check for other IDX types
                elif (magic >> 8) == 0x0008:
                    return True, 'unknown'
        except:
            pass
        return False, 'unknown'

    def _detect_format(self, raw_dir: Path) -> Tuple[DatasetFormat, Dict[str, Any]]:
        """Detect the dataset format and return format info."""
        all_items = list(raw_dir.iterdir())
        files = [f for f in all_items if f.is_file()]
        dirs = [d for d in all_items if d.is_dir()]

        print(f"[DEBUG] Detecting format in {raw_dir}")
        print(f"[DEBUG] Found {len(files)} files, {len(dirs)} dirs")
        for f in files:
            print(f"[DEBUG] File: {f.name} (is_file={f.is_file()})")

        # Check for MNIST IDX format by filename pattern
        idx_images = [f for f in files if
                      ('images' in f.name.lower() or 'image' in f.name.lower()) and
                      ('ubyte' in f.name.lower() or 'idx' in f.name.lower())]
        idx_labels = [f for f in files if
                      ('labels' in f.name.lower() or 'label' in f.name.lower()) and
                      ('ubyte' in f.name.lower() or 'idx' in f.name.lower())]

        print(f"[DEBUG] IDX by name - images: {[f.name for f in idx_images]}, labels: {[f.name for f in idx_labels]}")

        # If filename detection didn't work, try reading magic numbers
        if not (idx_images and idx_labels):
            print("[DEBUG] Trying magic number detection...")
            for f in files:
                is_idx, idx_type = self._is_idx_file(f)
                if is_idx:
                    print(f"[DEBUG] {f.name} is IDX type: {idx_type}")
                    if idx_type == 'images':
                        idx_images.append(f)
                    elif idx_type == 'labels':
                        idx_labels.append(f)

        if idx_images and idx_labels:
            print(f"[DEBUG] Detected MNIST IDX format!")
            # Prefer train files over test files
            train_images = [f for f in idx_images if 'train' in f.name.lower()]
            train_labels = [f for f in idx_labels if 'train' in f.name.lower()]
            return DatasetFormat.MNIST_IDX, {
                "images_file": train_images[0] if train_images else idx_images[0],
                "labels_file": train_labels[0] if train_labels else idx_labels[0],
                "all_images": idx_images,
                "all_labels": idx_labels
            }

        # Check for folder-per-class structure
        if dirs:
            # Verify folders contain files
            has_samples = any(any(d.iterdir()) for d in dirs)
            if has_samples:
                print(f"[DEBUG] Detected folder-per-class format")
                return DatasetFormat.FOLDER_PER_CLASS, {"class_dirs": dirs}

        # Check for CSV with labels
        csv_files = [f for f in files if f.suffix.lower() in {'.csv', '.tsv'}]
        if csv_files:
            return DatasetFormat.CSV_LABELS, {"csv_file": csv_files[0]}

        # Check for flat images
        image_files = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
        if image_files:
            return DatasetFormat.FLAT_IMAGES, {"image_files": image_files}

        print(f"[DEBUG] Unknown format")
        return DatasetFormat.UNKNOWN, {"files": files, "dirs": dirs}

    def _analyze_dataset(self, dataset_id: str, metadata: DatasetMetadata) -> DatasetMetadata:
        """Analyze extracted dataset structure."""
        raw_dir = self.uploads_dir / dataset_id / "raw"

        # Detect format
        format_type, format_info = self._detect_format(raw_dir)
        metadata.format = format_type.value

        if format_type == DatasetFormat.MNIST_IDX:
            return self._analyze_mnist_idx(metadata, format_info)
        elif format_type == DatasetFormat.FOLDER_PER_CLASS:
            return self._analyze_folder_per_class(metadata, format_info)
        elif format_type == DatasetFormat.CSV_LABELS:
            return self._analyze_csv_labels(metadata, format_info)
        elif format_type == DatasetFormat.FLAT_IMAGES:
            return self._analyze_flat_images(metadata, format_info)
        else:
            # Try to make sense of whatever is there
            return self._analyze_unknown(metadata, raw_dir)

    def _analyze_mnist_idx(self, metadata: DatasetMetadata, info: Dict) -> DatasetMetadata:
        """Analyze MNIST IDX format dataset."""
        images_file = info["images_file"]
        labels_file = info["labels_file"]

        # Read IDX header for images
        with open(images_file, 'rb') as f:
            magic = struct.unpack('>I', f.read(4))[0]
            num_images = struct.unpack('>I', f.read(4))[0]
            rows = struct.unpack('>I', f.read(4))[0]
            cols = struct.unpack('>I', f.read(4))[0]

        # Read labels to get class info
        with open(labels_file, 'rb') as f:
            magic = struct.unpack('>I', f.read(4))[0]
            num_labels = struct.unpack('>I', f.read(4))[0]
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        unique_labels = sorted(set(labels))
        class_names = [str(l) for l in unique_labels]
        samples_per_class = {str(l): int(np.sum(labels == l)) for l in unique_labels}

        metadata.data_type = DataType.IMAGE
        metadata.num_classes = len(unique_labels)
        metadata.class_names = class_names
        metadata.samples_per_class = samples_per_class
        metadata.total_samples = num_images
        metadata.input_shape = [1, rows, cols]  # Grayscale

        return metadata

    def _analyze_folder_per_class(self, metadata: DatasetMetadata, info: Dict) -> DatasetMetadata:
        """Analyze folder-per-class structure."""
        class_dirs = info["class_dirs"]

        class_names = sorted([d.name for d in class_dirs])
        samples_per_class = {}
        all_extensions = set()
        sample_files = []

        for class_dir in class_dirs:
            files = [f for f in class_dir.iterdir() if f.is_file()]
            samples_per_class[class_dir.name] = len(files)
            sample_files.extend(files[:5])
            all_extensions.update(f.suffix.lower() for f in files)

        data_type = self._detect_data_type(all_extensions, sample_files)
        input_shape = self._determine_input_shape(data_type, sample_files)

        metadata.data_type = data_type
        metadata.num_classes = len(class_names)
        metadata.class_names = class_names
        metadata.samples_per_class = samples_per_class
        metadata.total_samples = sum(samples_per_class.values())
        metadata.input_shape = input_shape

        return metadata

    def _analyze_csv_labels(self, metadata: DatasetMetadata, info: Dict) -> DatasetMetadata:
        """Analyze CSV with labels format."""
        import csv
        csv_file = info["csv_file"]

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Try to find label column
        label_col = None
        for col in ['label', 'class', 'category', 'target', 'y']:
            if col in reader.fieldnames:
                label_col = col
                break

        if label_col:
            labels = [row[label_col] for row in rows]
            unique_labels = sorted(set(labels))
            class_names = [str(l) for l in unique_labels]
            samples_per_class = {l: labels.count(l) for l in unique_labels}

            metadata.num_classes = len(unique_labels)
            metadata.class_names = class_names
            metadata.samples_per_class = samples_per_class

        metadata.data_type = DataType.TABULAR
        metadata.total_samples = len(rows)
        metadata.input_shape = [len(reader.fieldnames)]

        return metadata

    def _analyze_flat_images(self, metadata: DatasetMetadata, info: Dict) -> DatasetMetadata:
        """Analyze flat folder of images (no labels)."""
        image_files = info["image_files"]

        metadata.data_type = DataType.IMAGE
        metadata.num_classes = 0
        metadata.class_names = []
        metadata.samples_per_class = {}
        metadata.total_samples = len(image_files)
        metadata.input_shape = self._determine_input_shape(DataType.IMAGE, image_files[:5])

        return metadata

    def _analyze_unknown(self, metadata: DatasetMetadata, raw_dir: Path) -> DatasetMetadata:
        """Try to analyze unknown format."""
        all_files = list(raw_dir.rglob("*"))
        files_only = [f for f in all_files if f.is_file()]

        metadata.total_samples = len(files_only)
        metadata.class_names = []
        metadata.samples_per_class = {}

        return metadata

    def _detect_data_type(self, extensions: set, sample_files: List[Path]) -> DataType:
        """Detect the type of data in the dataset."""
        if extensions & IMAGE_EXTENSIONS:
            return DataType.IMAGE
        elif extensions & TEXT_EXTENSIONS:
            return DataType.TEXT
        elif extensions & TABULAR_EXTENSIONS:
            if sample_files:
                try:
                    import csv
                    with open(sample_files[0], 'r') as f:
                        reader = csv.reader(f)
                        row = next(reader, None)
                        if row and len(row) > 1:
                            return DataType.TABULAR
                        return DataType.TEXT
                except:
                    pass
            return DataType.TABULAR
        return DataType.UNKNOWN

    def _determine_input_shape(self, data_type: DataType, sample_files: List[Path]) -> List[int]:
        """Determine input shape based on data type and samples."""
        if data_type == DataType.IMAGE and sample_files:
            try:
                from PIL import Image
                img_files = [f for f in sample_files if f.suffix.lower() in IMAGE_EXTENSIONS]
                if img_files:
                    with Image.open(img_files[0]) as img:
                        w, h = img.size
                        channels = 3 if img.mode == 'RGB' else 1
                        target_size = min(max(w, h, 32), 224)
                        if target_size <= 32:
                            target_size = 32
                        elif target_size <= 64:
                            target_size = 64
                        elif target_size <= 128:
                            target_size = 128
                        else:
                            target_size = 224
                        return [channels, target_size, target_size]
            except:
                pass
            return [3, 32, 32]

        elif data_type == DataType.TEXT:
            return [256]

        elif data_type == DataType.TABULAR and sample_files:
            try:
                import csv
                csv_files = [f for f in sample_files if f.suffix.lower() in TABULAR_EXTENSIONS]
                if csv_files:
                    with open(csv_files[0], 'r') as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        if header:
                            return [len(header)]
            except:
                pass
            return [10]

        return []

    def get_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get dataset metadata."""
        return self._load_metadata(dataset_id)

    def list_datasets(self) -> List[DatasetMetadata]:
        """List all datasets."""
        datasets = []
        for path in self.uploads_dir.iterdir():
            if path.is_dir():
                meta = self._load_metadata(path.name)
                if meta:
                    datasets.append(meta)
        return sorted(datasets, key=lambda x: x.created_at, reverse=True)

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset."""
        dataset_dir = self.uploads_dir / dataset_id
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            return True
        return False

    def get_preview(self, dataset_id: str, num_samples: int = 18) -> Dict[str, Any]:
        """Get a preview of the dataset with base64-encoded images."""
        import base64
        from io import BytesIO
        from PIL import Image

        metadata = self._load_metadata(dataset_id)
        if not metadata:
            return {}

        raw_dir = self.uploads_dir / dataset_id / "raw"
        preview = {
            "metadata": metadata.to_dict(),
            "samples": []
        }

        # Handle different formats
        if metadata.format == DatasetFormat.MNIST_IDX.value:
            preview["samples"] = self._preview_mnist_idx(raw_dir, metadata, num_samples)
        elif metadata.format == DatasetFormat.FOLDER_PER_CLASS.value:
            preview["samples"] = self._preview_folder_per_class(raw_dir, metadata, num_samples)
        elif metadata.format == DatasetFormat.FLAT_IMAGES.value:
            preview["samples"] = self._preview_flat_images(raw_dir, num_samples)
        elif metadata.data_type == DataType.IMAGE:
            # Try generic image preview
            preview["samples"] = self._preview_any_images(raw_dir, num_samples)

        return preview

    def _preview_mnist_idx(self, raw_dir: Path, metadata: DatasetMetadata, num_samples: int) -> List[Dict]:
        """Generate preview for MNIST IDX format."""
        import base64
        from io import BytesIO
        from PIL import Image

        samples = []

        # Find the training images file (prefer train over test)
        # IMPORTANT: Only consider actual files, not directories
        files = [f for f in raw_dir.iterdir() if f.is_file()]
        images_file = None
        labels_file = None

        for f in files:
            name_lower = f.name.lower()
            # Must contain 'ubyte' or 'idx' to be valid MNIST file
            if not ('ubyte' in name_lower or 'idx' in name_lower):
                continue
            if 'train' in name_lower and 'images' in name_lower:
                images_file = f
            elif 'images' in name_lower and images_file is None:
                images_file = f
            if 'train' in name_lower and 'labels' in name_lower:
                labels_file = f
            elif 'labels' in name_lower and labels_file is None:
                labels_file = f

        print(f"[DEBUG] Preview MNIST - images: {images_file}, labels: {labels_file}")

        if not images_file or not labels_file:
            print(f"[DEBUG] Missing files for MNIST preview")
            return samples

        try:
            # Read images
            with open(images_file, 'rb') as f:
                magic = struct.unpack('>I', f.read(4))[0]
                num_images = struct.unpack('>I', f.read(4))[0]
                rows = struct.unpack('>I', f.read(4))[0]
                cols = struct.unpack('>I', f.read(4))[0]
                # Read only what we need
                num_to_read = min(num_samples * 10, num_images)  # Read extra to get variety
                image_data = np.frombuffer(f.read(num_to_read * rows * cols), dtype=np.uint8)
                image_data = image_data.reshape(num_to_read, rows, cols)

            # Read labels
            with open(labels_file, 'rb') as f:
                magic = struct.unpack('>I', f.read(4))[0]
                num_labels = struct.unpack('>I', f.read(4))[0]
                labels = np.frombuffer(f.read(num_to_read), dtype=np.uint8)

            # Get samples from different classes
            unique_labels = sorted(set(labels[:num_to_read]))
            samples_per_label = max(1, num_samples // len(unique_labels))

            for label in unique_labels:
                indices = np.where(labels[:num_to_read] == label)[0][:samples_per_label]
                for idx in indices:
                    img_array = image_data[idx]
                    img = Image.fromarray(img_array, mode='L')
                    img = img.resize((64, 64), Image.Resampling.NEAREST)  # Scale up for visibility

                    buffer = BytesIO()
                    img.save(buffer, format='PNG')
                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                    samples.append({
                        "image": img_base64,
                        "label": str(label)
                    })

                    if len(samples) >= num_samples:
                        break
                if len(samples) >= num_samples:
                    break

        except Exception as e:
            import traceback
            print(f"[DEBUG] Failed to preview MNIST IDX: {e}")
            traceback.print_exc()

        return samples

    def _preview_folder_per_class(self, raw_dir: Path, metadata: DatasetMetadata, num_samples: int) -> List[Dict]:
        """Generate preview for folder-per-class format."""
        import base64
        from io import BytesIO
        from PIL import Image

        samples = []
        samples_per_class = max(1, num_samples // max(len(metadata.class_names), 1))

        for class_name in metadata.class_names[:10]:
            class_dir = raw_dir / class_name
            if class_dir.exists():
                files = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS][:samples_per_class]
                for f in files:
                    try:
                        with Image.open(f) as img:
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            img.thumbnail((128, 128))
                            buffer = BytesIO()
                            img.save(buffer, format='PNG')
                            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            samples.append({
                                "image": img_base64,
                                "label": class_name
                            })
                    except Exception as e:
                        print(f"Failed to load preview image {f}: {e}")
                        continue

            if len(samples) >= num_samples:
                break

        return samples

    def _preview_flat_images(self, raw_dir: Path, num_samples: int) -> List[Dict]:
        """Generate preview for flat images folder."""
        import base64
        from io import BytesIO
        from PIL import Image

        samples = []
        image_files = [f for f in raw_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS][:num_samples]

        for f in image_files:
            try:
                with Image.open(f) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.thumbnail((128, 128))
                    buffer = BytesIO()
                    img.save(buffer, format='PNG')
                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    samples.append({
                        "image": img_base64,
                        "label": f.stem
                    })
            except Exception as e:
                print(f"Failed to load preview image {f}: {e}")
                continue

        return samples

    def _preview_any_images(self, raw_dir: Path, num_samples: int) -> List[Dict]:
        """Try to find and preview any images in the directory."""
        import base64
        from io import BytesIO
        from PIL import Image

        samples = []

        # Recursively find images
        for ext in IMAGE_EXTENSIONS:
            for f in raw_dir.rglob(f"*{ext}"):
                if len(samples) >= num_samples:
                    break
                try:
                    with Image.open(f) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.thumbnail((128, 128))
                        buffer = BytesIO()
                        img.save(buffer, format='PNG')
                        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        # Use parent folder name as label if available
                        label = f.parent.name if f.parent != raw_dir else f.stem
                        samples.append({
                            "image": img_base64,
                            "label": label
                        })
                except Exception as e:
                    continue
            if len(samples) >= num_samples:
                break

        return samples

    def _save_metadata(self, dataset_id: str, metadata: DatasetMetadata):
        """Save dataset metadata to JSON."""
        path = self.uploads_dir / dataset_id / "metadata.json"
        with open(path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def _load_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Load dataset metadata from JSON."""
        path = self.uploads_dir / dataset_id / "metadata.json"
        if not path.exists():
            return None
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                data['data_type'] = DataType(data['data_type'])
                # Handle old metadata without format field
                if 'format' not in data:
                    data['format'] = 'unknown'
                return DatasetMetadata(**data)
        except Exception as e:
            print(f"Failed to load metadata: {e}")
            return None
