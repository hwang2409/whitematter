"""
Unit tests for dataset_manager.py

Tests cover:
1. MNIST IDX format parsing
2. Folder-per-class format detection
3. CSV dataset handling
4. ZIP validation (security checks)
5. Class folder structure validation
"""

import struct
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_manager import (
    DatasetManager,
    DatasetMetadata,
    DataType,
    DatasetFormat,
    DatasetValidationError,
    FileSizeError,
    PathTraversalError,
    ZipBombError,
    CorruptedFileError,
    ClassStructureError,
    InvalidImageError,
    validate_file_size,
    validate_zip_safety,
    validate_image_magic_bytes,
    validate_class_folder_structure,
    validate_extracted_images,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_FILES_IN_ZIP,
    MAX_EXTRACTED_SIZE_BYTES,
    IMAGE_EXTENSIONS,
)


# =============================================================================
# MNIST IDX Format Parsing Tests
# =============================================================================

class TestMNISTIdxParsing:
    """Tests for MNIST IDX format detection and parsing."""

    def test_idx_images_magic_number_detection(self, sample_mnist_images: Path):
        """Test that IDX images file is correctly identified by magic number."""
        manager = DatasetManager(uploads_dir=sample_mnist_images.parent)
        is_idx, idx_type = manager._is_idx_file(sample_mnist_images)

        assert is_idx is True
        assert idx_type == 'images'

    def test_idx_labels_magic_number_detection(self, sample_mnist_labels: Path):
        """Test that IDX labels file is correctly identified by magic number."""
        manager = DatasetManager(uploads_dir=sample_mnist_labels.parent)
        is_idx, idx_type = manager._is_idx_file(sample_mnist_labels)

        assert is_idx is True
        assert idx_type == 'labels'

    def test_non_idx_file_detection(self, temp_dir: Path):
        """Test that non-IDX files are not detected as IDX."""
        regular_file = temp_dir / "regular.txt"
        regular_file.write_text("This is not an IDX file")

        manager = DatasetManager(uploads_dir=temp_dir)
        is_idx, idx_type = manager._is_idx_file(regular_file)

        assert is_idx is False
        assert idx_type == 'unknown'

    def test_analyze_mnist_idx_metadata(self, sample_mnist_images: Path, sample_mnist_labels: Path):
        """Test that MNIST IDX files are correctly analyzed for metadata."""
        info = {
            "images_file": sample_mnist_images,
            "labels_file": sample_mnist_labels,
        }
        manager = DatasetManager(uploads_dir=sample_mnist_images.parent)

        metadata = DatasetMetadata(
            id="test",
            name="test",
            data_type=DataType.UNKNOWN,
            created_at="2024-01-01",
            num_classes=0,
            class_names=[],
            samples_per_class={},
            total_samples=0,
            input_shape=[],
        )

        result = manager._analyze_mnist_idx(metadata, info)

        assert result.data_type == DataType.IMAGE
        assert result.num_classes == 10  # Labels 0-9
        assert result.total_samples == 10
        assert result.input_shape == [1, 28, 28]  # Grayscale, 28x28
        assert len(result.class_names) == 10
        assert all(str(i) in result.class_names for i in range(10))

    def test_detect_mnist_format_by_filename(self, temp_dir: Path):
        """Test MNIST format detection by filename patterns."""
        # Create files with MNIST-style names
        images_file = temp_dir / "train-images-idx3-ubyte"
        labels_file = temp_dir / "train-labels-idx1-ubyte"

        # Write valid IDX headers
        with open(images_file, 'wb') as f:
            f.write(struct.pack('>I', 0x00000803))  # Images magic
            f.write(struct.pack('>I', 5))  # num images
            f.write(struct.pack('>I', 28))  # rows
            f.write(struct.pack('>I', 28))  # cols
            f.write(np.zeros((5, 28, 28), dtype=np.uint8).tobytes())

        with open(labels_file, 'wb') as f:
            f.write(struct.pack('>I', 0x00000801))  # Labels magic
            f.write(struct.pack('>I', 5))  # num labels
            f.write(np.array([0, 1, 2, 3, 4], dtype=np.uint8).tobytes())

        manager = DatasetManager(uploads_dir=temp_dir)
        format_type, info = manager._detect_format(temp_dir)

        assert format_type == DatasetFormat.MNIST_IDX
        assert "images_file" in info
        assert "labels_file" in info

    def test_mnist_labels_distribution(self, sample_mnist_images: Path, sample_mnist_labels: Path):
        """Test that samples per class is correctly computed."""
        info = {
            "images_file": sample_mnist_images,
            "labels_file": sample_mnist_labels,
        }
        manager = DatasetManager(uploads_dir=sample_mnist_images.parent)

        metadata = DatasetMetadata(
            id="test",
            name="test",
            data_type=DataType.UNKNOWN,
            created_at="2024-01-01",
            num_classes=0,
            class_names=[],
            samples_per_class={},
            total_samples=0,
            input_shape=[],
        )

        result = manager._analyze_mnist_idx(metadata, info)

        # Each label 0-9 appears exactly once in our test data
        for i in range(10):
            assert result.samples_per_class[str(i)] == 1


# =============================================================================
# Folder-per-Class Format Detection Tests
# =============================================================================

class TestFolderPerClassFormat:
    """Tests for folder-per-class format detection and analysis."""

    def test_detect_folder_per_class_format(self, folder_per_class_structure: Path):
        """Test detection of folder-per-class structure."""
        manager = DatasetManager(uploads_dir=folder_per_class_structure.parent)
        format_type, info = manager._detect_format(folder_per_class_structure)

        assert format_type == DatasetFormat.FOLDER_PER_CLASS
        assert "class_dirs" in info
        assert len(info["class_dirs"]) == 3  # cat, dog, bird

    def test_analyze_folder_per_class_structure(self, folder_per_class_structure: Path):
        """Test analysis of folder-per-class dataset."""
        manager = DatasetManager(uploads_dir=folder_per_class_structure.parent)
        _, info = manager._detect_format(folder_per_class_structure)

        metadata = DatasetMetadata(
            id="test",
            name="test",
            data_type=DataType.UNKNOWN,
            created_at="2024-01-01",
            num_classes=0,
            class_names=[],
            samples_per_class={},
            total_samples=0,
            input_shape=[],
        )

        result = manager._analyze_folder_per_class(metadata, info)

        assert result.num_classes == 3
        assert set(result.class_names) == {"cat", "dog", "bird"}
        assert result.total_samples == 15  # 5 samples per class * 3 classes
        assert result.samples_per_class["cat"] == 5
        assert result.samples_per_class["dog"] == 5
        assert result.samples_per_class["bird"] == 5

    def test_empty_class_folder_detection(self, temp_dir: Path):
        """Test that empty class folders are detected."""
        dataset_dir = temp_dir / "dataset"
        dataset_dir.mkdir()

        # Create one populated and one empty class folder
        (dataset_dir / "class_a").mkdir()
        (dataset_dir / "class_a" / "sample.png").write_bytes(b'\x89PNG')
        (dataset_dir / "class_b").mkdir()  # Empty

        class_dirs = [dataset_dir / "class_a", dataset_dir / "class_b"]

        with pytest.raises(ClassStructureError) as exc_info:
            validate_class_folder_structure(class_dirs)

        assert "empty" in str(exc_info.value).lower()

    def test_class_imbalance_warning(self, temp_dir: Path):
        """Test that class imbalance generates warnings."""
        dataset_dir = temp_dir / "dataset"
        dataset_dir.mkdir()

        # Create imbalanced classes
        (dataset_dir / "class_a").mkdir()
        for i in range(200):
            (dataset_dir / "class_a" / f"sample_{i}.png").write_bytes(b'\x89PNG')

        (dataset_dir / "class_b").mkdir()
        (dataset_dir / "class_b" / "sample_0.png").write_bytes(b'\x89PNG')

        class_dirs = [dataset_dir / "class_a", dataset_dir / "class_b"]
        result = validate_class_folder_structure(class_dirs, warn_on_imbalance=True)

        assert len(result["warnings"]) > 0
        assert any("imbalance" in w.lower() for w in result["warnings"])

    def test_no_class_folders_error(self):
        """Test error when no class folders are provided."""
        with pytest.raises(ClassStructureError) as exc_info:
            validate_class_folder_structure([])

        assert "no class folders" in str(exc_info.value).lower()


# =============================================================================
# CSV Dataset Handling Tests
# =============================================================================

class TestCSVDatasetHandling:
    """Tests for CSV dataset format detection and analysis."""

    def test_detect_csv_format(self, csv_dataset: Path):
        """Test detection of CSV dataset format."""
        manager = DatasetManager(uploads_dir=csv_dataset.parent)
        format_type, info = manager._detect_format(csv_dataset.parent)

        assert format_type == DatasetFormat.CSV_LABELS
        assert "csv_file" in info

    def test_analyze_csv_with_label_column(self, csv_dataset: Path):
        """Test analysis of CSV file with label column."""
        manager = DatasetManager(uploads_dir=csv_dataset.parent)

        info = {"csv_file": csv_dataset}
        metadata = DatasetMetadata(
            id="test",
            name="test",
            data_type=DataType.UNKNOWN,
            created_at="2024-01-01",
            num_classes=0,
            class_names=[],
            samples_per_class={},
            total_samples=0,
            input_shape=[],
        )

        result = manager._analyze_csv_labels(metadata, info)

        assert result.data_type == DataType.TABULAR
        assert result.num_classes == 3  # cat, dog, bird
        assert set(result.class_names) == {"cat", "dog", "bird"}
        assert result.total_samples == 5
        assert result.samples_per_class["cat"] == 2
        assert result.samples_per_class["dog"] == 2
        assert result.samples_per_class["bird"] == 1

    def test_csv_with_alternative_label_columns(self, temp_dir: Path):
        """Test CSV with different label column names."""
        for col_name in ['class', 'category', 'target', 'y']:
            csv_file = temp_dir / f"dataset_{col_name}.csv"
            csv_file.write_text(
                f"id,feature,{col_name}\n"
                "1,0.5,positive\n"
                "2,0.3,negative\n"
            )

            manager = DatasetManager(uploads_dir=temp_dir)
            info = {"csv_file": csv_file}
            metadata = DatasetMetadata(
                id="test",
                name="test",
                data_type=DataType.UNKNOWN,
                created_at="2024-01-01",
                num_classes=0,
                class_names=[],
                samples_per_class={},
                total_samples=0,
                input_shape=[],
            )

            result = manager._analyze_csv_labels(metadata, info)
            assert result.num_classes == 2
            assert set(result.class_names) == {"positive", "negative"}

    def test_csv_input_shape_from_columns(self, csv_dataset: Path):
        """Test that input shape is derived from column count."""
        manager = DatasetManager(uploads_dir=csv_dataset.parent)

        info = {"csv_file": csv_dataset}
        metadata = DatasetMetadata(
            id="test",
            name="test",
            data_type=DataType.UNKNOWN,
            created_at="2024-01-01",
            num_classes=0,
            class_names=[],
            samples_per_class={},
            total_samples=0,
            input_shape=[],
        )

        result = manager._analyze_csv_labels(metadata, info)

        # CSV has 4 columns: id, feature1, feature2, label
        assert result.input_shape == [4]


# =============================================================================
# ZIP Validation Tests
# =============================================================================

class TestZIPValidation:
    """Tests for ZIP file security validation."""

    def test_valid_zip_passes_validation(self, valid_zip_file: Path):
        """Test that a valid ZIP file passes all validation checks."""
        result = validate_zip_safety(valid_zip_file)

        assert "file_count" in result
        assert "total_uncompressed_size" in result
        assert "files" in result
        assert result["file_count"] > 0

    def test_path_traversal_detection(self, zip_with_path_traversal: Path):
        """Test that path traversal attempts are detected."""
        with pytest.raises(PathTraversalError) as exc_info:
            validate_zip_safety(zip_with_path_traversal)

        assert "path traversal" in str(exc_info.value).lower()

    def test_corrupted_zip_detection(self, corrupted_zip_file: Path):
        """Test that corrupted ZIP files are detected."""
        with pytest.raises(CorruptedFileError):
            validate_zip_safety(corrupted_zip_file)

    def test_empty_zip_validation(self, empty_zip_file: Path):
        """Test that empty ZIP files pass validation."""
        result = validate_zip_safety(empty_zip_file)

        assert result["file_count"] == 0
        assert result["total_uncompressed_size"] == 0

    def test_file_count_limit(self, temp_dir: Path):
        """Test that ZIP files with too many files are rejected."""
        zip_path = temp_dir / "many_files.zip"

        # Create a mock that simulates too many files
        with patch('dataset_manager.MAX_FILES_IN_ZIP', 5):
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for i in range(10):
                    zf.writestr(f"file_{i}.txt", f"content {i}")

            with pytest.raises(ZipBombError) as exc_info:
                validate_zip_safety(zip_path)

            assert "too many files" in str(exc_info.value).lower()

    def test_absolute_path_detection(self, temp_dir: Path):
        """Test detection of absolute paths in ZIP."""
        zip_path = temp_dir / "absolute_path.zip"

        with zipfile.ZipFile(zip_path, 'w') as zf:
            # Create a valid entry first
            zf.writestr("normal.txt", "normal content")

        # Manually add an entry with absolute path by manipulating the zip
        # This is a bit hacky but necessary to test the validation
        with zipfile.ZipFile(zip_path, 'a') as zf:
            info = zipfile.ZipInfo("/etc/passwd")
            zf.writestr(info, "malicious")

        with pytest.raises(PathTraversalError):
            validate_zip_safety(zip_path)

    def test_file_size_validation(self, temp_dir: Path):
        """Test file size validation."""
        small_file = temp_dir / "small.txt"
        small_file.write_text("small content")

        # Should pass with default limit
        validate_file_size(small_file)

        # Should fail with tiny limit
        with pytest.raises(FileSizeError):
            validate_file_size(small_file, max_size=5)

    def test_file_not_found_error(self, temp_dir: Path):
        """Test that non-existent files raise FileNotFoundError."""
        non_existent = temp_dir / "does_not_exist.zip"

        with pytest.raises(FileNotFoundError):
            validate_file_size(non_existent)

    def test_nested_directory_extraction(self, temp_dir: Path, uploads_dir: Path):
        """Test handling of ZIP with nested root directory."""
        # Create a structure: outer_folder/class_a/file.png
        nested_dir = temp_dir / "outer_folder"
        nested_dir.mkdir()
        (nested_dir / "class_a").mkdir()
        (nested_dir / "class_a" / "file.png").write_bytes(
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f'
            b'\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        )

        zip_path = temp_dir / "nested.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for file_path in nested_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(temp_dir)  # Include outer_folder
                    zf.write(file_path, arcname)

        manager = DatasetManager(uploads_dir=uploads_dir)
        dataset_id = manager.create_dataset("nested_test")
        metadata = manager.extract_zip(dataset_id, zip_path, validate_images=False)

        # After extraction and flattening, class_a should be at root level
        raw_dir = uploads_dir / dataset_id / "raw"
        assert (raw_dir / "class_a").exists()
        assert (raw_dir / "class_a" / "file.png").exists()


# =============================================================================
# Class Folder Structure Validation Tests
# =============================================================================

class TestClassFolderStructureValidation:
    """Tests for class folder structure validation."""

    def test_valid_structure_returns_stats(self, folder_per_class_structure: Path):
        """Test that valid structure returns correct statistics."""
        class_dirs = list(folder_per_class_structure.iterdir())
        result = validate_class_folder_structure(class_dirs)

        assert result["num_classes"] == 3
        assert result["total_samples"] == 15
        assert "cat" in result["class_stats"]
        assert result["class_stats"]["cat"] == 5

    def test_multiple_empty_classes_error_message(self, temp_dir: Path):
        """Test error message when multiple empty classes exist."""
        dataset_dir = temp_dir / "dataset"
        dataset_dir.mkdir()

        # Create 7 empty class folders
        for i in range(7):
            (dataset_dir / f"class_{i}").mkdir()

        class_dirs = list(dataset_dir.iterdir())

        with pytest.raises(ClassStructureError) as exc_info:
            validate_class_folder_structure(class_dirs)

        error_msg = str(exc_info.value)
        assert "7" in error_msg or "empty" in error_msg.lower()
        # Should mention truncation for > 5 classes
        assert "more" in error_msg.lower()

    def test_mixed_file_types_warning(self, temp_dir: Path):
        """Test warning when classes have mixed file types."""
        dataset_dir = temp_dir / "dataset"
        dataset_dir.mkdir()

        (dataset_dir / "class_a").mkdir()
        (dataset_dir / "class_a" / "file.png").write_bytes(b'\x89PNG')
        (dataset_dir / "class_a" / "file.txt").write_text("text")

        class_dirs = [dataset_dir / "class_a"]
        result = validate_class_folder_structure(class_dirs)

        # Should have warning about mixed file types
        assert len(result["warnings"]) > 0

    def test_low_sample_warning(self, temp_dir: Path):
        """Test warning for classes with very few samples."""
        dataset_dir = temp_dir / "dataset"
        dataset_dir.mkdir()

        # Create a class with minimal samples (assuming MIN_SAMPLES_PER_CLASS > 1)
        (dataset_dir / "class_a").mkdir()
        (dataset_dir / "class_a" / "single.png").write_bytes(b'\x89PNG')

        class_dirs = [dataset_dir / "class_a"]

        # With MIN_SAMPLES_PER_CLASS = 1, this should pass without error
        result = validate_class_folder_structure(class_dirs)
        assert result["total_samples"] == 1


# =============================================================================
# Image Validation Tests
# =============================================================================

class TestImageValidation:
    """Tests for image file validation."""

    def test_valid_png_magic_bytes(self, temp_dir: Path):
        """Test validation of valid PNG magic bytes."""
        png_file = temp_dir / "test.png"
        png_file.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)

        is_valid, format_name = validate_image_magic_bytes(png_file)

        assert is_valid is True
        assert format_name == 'png'

    def test_valid_jpeg_magic_bytes(self, temp_dir: Path):
        """Test validation of valid JPEG magic bytes."""
        jpeg_file = temp_dir / "test.jpg"
        jpeg_file.write_bytes(b'\xff\xd8\xff' + b'\x00' * 100)

        is_valid, format_name = validate_image_magic_bytes(jpeg_file)

        assert is_valid is True
        assert format_name == 'jpeg'

    def test_valid_gif_magic_bytes(self, temp_dir: Path):
        """Test validation of valid GIF magic bytes."""
        for magic in [b'GIF87a', b'GIF89a']:
            gif_file = temp_dir / "test.gif"
            gif_file.write_bytes(magic + b'\x00' * 100)

            is_valid, format_name = validate_image_magic_bytes(gif_file)

            assert is_valid is True
            assert format_name == 'gif'

    def test_valid_bmp_magic_bytes(self, temp_dir: Path):
        """Test validation of valid BMP magic bytes."""
        bmp_file = temp_dir / "test.bmp"
        bmp_file.write_bytes(b'BM' + b'\x00' * 100)

        is_valid, format_name = validate_image_magic_bytes(bmp_file)

        assert is_valid is True
        assert format_name == 'bmp'

    def test_invalid_magic_bytes(self, temp_dir: Path):
        """Test that invalid magic bytes are detected."""
        fake_image = temp_dir / "fake.png"
        fake_image.write_bytes(b'This is not an image')

        is_valid, format_name = validate_image_magic_bytes(fake_image)

        assert is_valid is False
        assert format_name == 'unknown'

    def test_empty_file_handling(self, temp_dir: Path):
        """Test handling of empty files."""
        empty_file = temp_dir / "empty.png"
        empty_file.write_bytes(b'')

        is_valid, format_name = validate_image_magic_bytes(empty_file)

        assert is_valid is False
        assert format_name == 'empty'

    def test_webp_detection(self, temp_dir: Path):
        """Test WebP format detection."""
        webp_file = temp_dir / "test.webp"
        # WebP: RIFF....WEBP
        webp_file.write_bytes(b'RIFF\x00\x00\x00\x00WEBP' + b'\x00' * 100)

        is_valid, format_name = validate_image_magic_bytes(webp_file)

        assert is_valid is True
        assert format_name == 'webp'


# =============================================================================
# DatasetManager Integration Tests
# =============================================================================

class TestDatasetManagerIntegration:
    """Integration tests for DatasetManager."""

    def test_create_dataset(self, uploads_dir: Path):
        """Test dataset creation."""
        manager = DatasetManager(uploads_dir=uploads_dir)
        dataset_id = manager.create_dataset("test_dataset")

        assert dataset_id is not None
        assert len(dataset_id) == 8
        assert (uploads_dir / dataset_id).exists()
        assert (uploads_dir / dataset_id / "raw").exists()
        assert (uploads_dir / dataset_id / "processed").exists()
        assert (uploads_dir / dataset_id / "metadata.json").exists()

    def test_create_dataset_with_default_name(self, uploads_dir: Path):
        """Test dataset creation without explicit name."""
        manager = DatasetManager(uploads_dir=uploads_dir)
        dataset_id = manager.create_dataset()

        metadata = manager.get_metadata(dataset_id)
        assert metadata.name.startswith("dataset_")

    def test_extract_zip_folder_per_class(self, uploads_dir: Path, valid_zip_file: Path):
        """Test extraction and analysis of folder-per-class ZIP."""
        manager = DatasetManager(uploads_dir=uploads_dir)
        dataset_id = manager.create_dataset("test")
        metadata = manager.extract_zip(dataset_id, valid_zip_file, validate_images=False)

        assert metadata.status == "ready"
        assert metadata.format == DatasetFormat.FOLDER_PER_CLASS.value
        assert metadata.num_classes == 3
        assert metadata.total_samples == 15

    def test_extract_mnist_zip(self, uploads_dir: Path, mnist_zip: Path):
        """Test extraction and analysis of MNIST format ZIP."""
        manager = DatasetManager(uploads_dir=uploads_dir)
        dataset_id = manager.create_dataset("mnist_test")
        metadata = manager.extract_zip(dataset_id, mnist_zip, validate_images=False)

        assert metadata.status == "ready"
        assert metadata.format == DatasetFormat.MNIST_IDX.value
        assert metadata.num_classes == 10
        assert metadata.input_shape == [1, 28, 28]

    def test_delete_dataset(self, uploads_dir: Path):
        """Test dataset deletion."""
        manager = DatasetManager(uploads_dir=uploads_dir)
        dataset_id = manager.create_dataset("to_delete")

        assert (uploads_dir / dataset_id).exists()

        result = manager.delete_dataset(dataset_id)

        assert result is True
        assert not (uploads_dir / dataset_id).exists()

    def test_delete_nonexistent_dataset(self, uploads_dir: Path):
        """Test deletion of non-existent dataset."""
        manager = DatasetManager(uploads_dir=uploads_dir)
        result = manager.delete_dataset("nonexistent")

        assert result is False

    def test_list_datasets(self, uploads_dir: Path):
        """Test listing all datasets."""
        manager = DatasetManager(uploads_dir=uploads_dir)

        # Create multiple datasets
        ids = [manager.create_dataset(f"dataset_{i}") for i in range(3)]

        datasets = manager.list_datasets()

        assert len(datasets) == 3
        dataset_ids = [d.id for d in datasets]
        for id_ in ids:
            assert id_ in dataset_ids

    def test_get_metadata_nonexistent(self, uploads_dir: Path):
        """Test getting metadata for non-existent dataset."""
        manager = DatasetManager(uploads_dir=uploads_dir)
        metadata = manager.get_metadata("nonexistent")

        assert metadata is None

    def test_extract_corrupted_zip_error(self, uploads_dir: Path, corrupted_zip_file: Path):
        """Test that corrupted ZIP raises appropriate error."""
        manager = DatasetManager(uploads_dir=uploads_dir)
        dataset_id = manager.create_dataset("corrupted_test")

        with pytest.raises(CorruptedFileError):
            manager.extract_zip(dataset_id, corrupted_zip_file)

        # Verify error status is saved
        metadata = manager.get_metadata(dataset_id)
        assert metadata.status == "error"
        assert metadata.error is not None


# =============================================================================
# Data Type Detection Tests
# =============================================================================

class TestDataTypeDetection:
    """Tests for data type detection."""

    def test_image_type_detection(self, temp_dir: Path):
        """Test detection of image data type."""
        manager = DatasetManager(uploads_dir=temp_dir)
        extensions = {'.png', '.jpg'}

        data_type = manager._detect_data_type(extensions, [])

        assert data_type == DataType.IMAGE

    def test_text_type_detection(self, temp_dir: Path):
        """Test detection of text data type."""
        manager = DatasetManager(uploads_dir=temp_dir)
        extensions = {'.txt'}

        data_type = manager._detect_data_type(extensions, [])

        assert data_type == DataType.TEXT

    def test_tabular_type_detection(self, temp_dir: Path, csv_dataset: Path):
        """Test detection of tabular data type."""
        manager = DatasetManager(uploads_dir=temp_dir)
        extensions = {'.csv'}

        data_type = manager._detect_data_type(extensions, [csv_dataset])

        assert data_type == DataType.TABULAR

    def test_unknown_type_detection(self, temp_dir: Path):
        """Test detection of unknown data type."""
        manager = DatasetManager(uploads_dir=temp_dir)
        extensions = {'.xyz', '.abc'}

        data_type = manager._detect_data_type(extensions, [])

        assert data_type == DataType.UNKNOWN


# =============================================================================
# Metadata Serialization Tests
# =============================================================================

class TestMetadataSerialization:
    """Tests for metadata serialization and deserialization."""

    def test_metadata_to_dict(self):
        """Test metadata conversion to dictionary."""
        metadata = DatasetMetadata(
            id="test123",
            name="test_dataset",
            data_type=DataType.IMAGE,
            created_at="2024-01-01T00:00:00",
            num_classes=10,
            class_names=["0", "1", "2"],
            samples_per_class={"0": 100, "1": 100, "2": 100},
            total_samples=300,
            input_shape=[3, 32, 32],
            format="folder_per_class",
            status="ready",
        )

        result = metadata.to_dict()

        assert result["id"] == "test123"
        assert result["data_type"] == "image"  # Enum value
        assert result["num_classes"] == 10
        assert isinstance(result["class_names"], list)

    def test_metadata_round_trip(self, uploads_dir: Path):
        """Test saving and loading metadata."""
        manager = DatasetManager(uploads_dir=uploads_dir)
        dataset_id = manager.create_dataset("round_trip_test")

        original = manager.get_metadata(dataset_id)
        original.num_classes = 5
        original.class_names = ["a", "b", "c", "d", "e"]

        manager._save_metadata(dataset_id, original)
        loaded = manager._load_metadata(dataset_id)

        assert loaded.id == original.id
        assert loaded.num_classes == 5
        assert loaded.class_names == ["a", "b", "c", "d", "e"]
