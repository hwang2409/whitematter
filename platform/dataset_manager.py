"""
Dataset Manager - handles dataset upload, extraction, validation, and processing.
"""

import json
import shutil
import uuid
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np


class DataType(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    TABULAR = "tabular"
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

    def _analyze_dataset(self, dataset_id: str, metadata: DatasetMetadata) -> DatasetMetadata:
        """Analyze extracted dataset structure."""
        raw_dir = self.uploads_dir / dataset_id / "raw"

        # Scan for class folders
        class_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]

        if not class_dirs:
            raise ValueError("No class folders found. Expected structure: folder_per_class/samples")

        class_names = sorted([d.name for d in class_dirs])
        samples_per_class = {}
        all_extensions = set()
        sample_files = []

        for class_dir in class_dirs:
            files = [f for f in class_dir.iterdir() if f.is_file()]
            samples_per_class[class_dir.name] = len(files)
            sample_files.extend(files[:5])  # Keep some samples for analysis
            all_extensions.update(f.suffix.lower() for f in files)

        # Detect data type
        data_type = self._detect_data_type(all_extensions, sample_files)

        # Determine input shape based on data type
        input_shape = self._determine_input_shape(data_type, sample_files)

        metadata.data_type = data_type
        metadata.num_classes = len(class_names)
        metadata.class_names = class_names
        metadata.samples_per_class = samples_per_class
        metadata.total_samples = sum(samples_per_class.values())
        metadata.input_shape = input_shape

        return metadata

    def _detect_data_type(self, extensions: set, sample_files: List[Path]) -> DataType:
        """Detect the type of data in the dataset."""
        if extensions & IMAGE_EXTENSIONS:
            return DataType.IMAGE
        elif extensions & TEXT_EXTENSIONS:
            return DataType.TEXT
        elif extensions & TABULAR_EXTENSIONS:
            # Check if CSV has multiple columns (tabular) or single (text)
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
                # Check first image
                img_files = [f for f in sample_files if f.suffix.lower() in IMAGE_EXTENSIONS]
                if img_files:
                    with Image.open(img_files[0]) as img:
                        w, h = img.size
                        channels = 3 if img.mode == 'RGB' else 1
                        # Default to 32x32 if images are larger (will resize)
                        target_size = min(max(w, h, 32), 224)
                        # Round to common sizes
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
            return [3, 32, 32]  # Default

        elif data_type == DataType.TEXT:
            return [256]  # Default max sequence length

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
            return [10]  # Default

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

    def get_preview(self, dataset_id: str, samples_per_class: int = 3) -> Dict[str, Any]:
        """Get a preview of the dataset."""
        metadata = self._load_metadata(dataset_id)
        if not metadata:
            return {}

        raw_dir = self.uploads_dir / dataset_id / "raw"
        preview = {
            "metadata": metadata.to_dict(),
            "samples": {}
        }

        for class_name in metadata.class_names[:10]:  # Limit to 10 classes
            class_dir = raw_dir / class_name
            if class_dir.exists():
                files = list(class_dir.iterdir())[:samples_per_class]
                preview["samples"][class_name] = [f.name for f in files]

        return preview

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
                return DatasetMetadata(**data)
        except:
            return None
