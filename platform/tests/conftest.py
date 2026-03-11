"""
Shared pytest fixtures for platform tests.
"""

import os
import shutil
import struct
import tempfile
import zipfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def uploads_dir(temp_dir: Path) -> Path:
    """Create a temporary uploads directory."""
    uploads = temp_dir / "uploads"
    uploads.mkdir()
    return uploads


@pytest.fixture
def sample_mnist_images(temp_dir: Path) -> Path:
    """Create a sample MNIST-style IDX images file."""
    filepath = temp_dir / "train-images-idx3-ubyte"

    # Create a minimal MNIST-style file
    num_images = 10
    rows = 28
    cols = 28

    with open(filepath, 'wb') as f:
        # Magic number for images (2051 = 0x0803)
        f.write(struct.pack('>I', 0x00000803))
        # Number of images
        f.write(struct.pack('>I', num_images))
        # Rows
        f.write(struct.pack('>I', rows))
        # Cols
        f.write(struct.pack('>I', cols))
        # Image data (random pixels)
        np.random.seed(42)
        images = np.random.randint(0, 256, (num_images, rows, cols), dtype=np.uint8)
        f.write(images.tobytes())

    return filepath


@pytest.fixture
def sample_mnist_labels(temp_dir: Path) -> Path:
    """Create a sample MNIST-style IDX labels file."""
    filepath = temp_dir / "train-labels-idx1-ubyte"

    num_labels = 10
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)

    with open(filepath, 'wb') as f:
        # Magic number for labels (2049 = 0x0801)
        f.write(struct.pack('>I', 0x00000801))
        # Number of labels
        f.write(struct.pack('>I', num_labels))
        # Labels
        f.write(labels.tobytes())

    return filepath


@pytest.fixture
def folder_per_class_structure(temp_dir: Path) -> Path:
    """Create a folder-per-class dataset structure."""
    dataset_dir = temp_dir / "folder_dataset"
    dataset_dir.mkdir()

    classes = ["cat", "dog", "bird"]
    for class_name in classes:
        class_dir = dataset_dir / class_name
        class_dir.mkdir()
        # Create dummy image files (just files with image extensions)
        for i in range(5):
            (class_dir / f"sample_{i}.png").write_bytes(
                # PNG magic bytes + minimal valid header
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
                b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f'
                b'\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
            )

    return dataset_dir


@pytest.fixture
def csv_dataset(temp_dir: Path) -> Path:
    """Create a sample CSV dataset."""
    csv_file = temp_dir / "dataset.csv"
    csv_file.write_text(
        "id,feature1,feature2,label\n"
        "1,0.5,0.3,cat\n"
        "2,0.7,0.2,dog\n"
        "3,0.1,0.9,cat\n"
        "4,0.8,0.4,bird\n"
        "5,0.2,0.6,dog\n"
    )
    return csv_file


@pytest.fixture
def valid_zip_file(folder_per_class_structure: Path, temp_dir: Path) -> Path:
    """Create a valid ZIP file from the folder-per-class structure."""
    zip_path = temp_dir / "dataset.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for file_path in folder_per_class_structure.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(folder_per_class_structure)
                zf.write(file_path, arcname)
    return zip_path


@pytest.fixture
def zip_with_path_traversal(temp_dir: Path) -> Path:
    """Create a ZIP file with path traversal attempt."""
    zip_path = temp_dir / "malicious.zip"

    # Create zip with manipulated paths
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # Normal file
        zf.writestr("normal.txt", "normal content")
        # Path traversal attempt
        zf.writestr("../../../etc/passwd", "malicious content")

    return zip_path


@pytest.fixture
def corrupted_zip_file(temp_dir: Path) -> Path:
    """Create a corrupted ZIP file."""
    zip_path = temp_dir / "corrupted.zip"
    zip_path.write_bytes(b"PK\x03\x04corrupted data that is not a valid zip")
    return zip_path


@pytest.fixture
def zip_bomb_style_file(temp_dir: Path) -> Path:
    """Create a ZIP file with high compression ratio (simulated zip bomb characteristics)."""
    zip_path = temp_dir / "suspicious.zip"

    # Create a file that compresses very well (repeating data)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # Large file with highly compressible data
        large_data = b'A' * (20 * 1024 * 1024)  # 20 MB of 'A's
        zf.writestr("large_file.txt", large_data)

    return zip_path


@pytest.fixture
def empty_zip_file(temp_dir: Path) -> Path:
    """Create an empty but valid ZIP file."""
    zip_path = temp_dir / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        pass  # Create empty zip
    return zip_path


@pytest.fixture
def mnist_zip(sample_mnist_images: Path, sample_mnist_labels: Path, temp_dir: Path) -> Path:
    """Create a ZIP file containing MNIST IDX format files."""
    zip_path = temp_dir / "mnist.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(sample_mnist_images, sample_mnist_images.name)
        zf.write(sample_mnist_labels, sample_mnist_labels.name)
    return zip_path
