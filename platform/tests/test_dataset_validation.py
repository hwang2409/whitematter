"""
Unit tests for services/dataset_validation.py

Tests cover:
1. File size validation
2. ZIP security validation (path traversal, zip bombs, corruption)
3. Image magic-byte validation
4. Class folder structure validation
5. Extracted image integrity checks
"""

import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure platform/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.dataset_validation import (
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
    validate_image_file,
    validate_class_folder_structure,
    validate_extracted_images,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_FILES_IN_ZIP,
    IMAGE_EXTENSIONS,
)


# =============================================================================
# ZIP Validation Tests
# =============================================================================

class TestZIPValidation:
    """Tests for ZIP file security validation."""

    def test_valid_zip_passes_validation(self, valid_zip_file: Path):
        result = validate_zip_safety(valid_zip_file)

        assert "file_count" in result
        assert "total_uncompressed_size" in result
        assert "files" in result
        assert result["file_count"] > 0

    def test_path_traversal_detection(self, zip_with_path_traversal: Path):
        with pytest.raises(PathTraversalError) as exc_info:
            validate_zip_safety(zip_with_path_traversal)

        assert "path traversal" in str(exc_info.value).lower()

    def test_corrupted_zip_detection(self, corrupted_zip_file: Path):
        with pytest.raises(CorruptedFileError):
            validate_zip_safety(corrupted_zip_file)

    def test_empty_zip_validation(self, empty_zip_file: Path):
        result = validate_zip_safety(empty_zip_file)

        assert result["file_count"] == 0
        assert result["total_uncompressed_size"] == 0

    def test_file_count_limit(self, temp_dir: Path):
        zip_path = temp_dir / "many_files.zip"

        with patch('services.dataset_validation.MAX_FILES_IN_ZIP', 5):
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for i in range(10):
                    zf.writestr(f"file_{i}.txt", f"content {i}")

            with pytest.raises(ZipBombError) as exc_info:
                validate_zip_safety(zip_path)

            assert "too many files" in str(exc_info.value).lower()

    def test_absolute_path_detection(self, temp_dir: Path):
        zip_path = temp_dir / "absolute_path.zip"

        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("normal.txt", "normal content")

        with zipfile.ZipFile(zip_path, 'a') as zf:
            info = zipfile.ZipInfo("/etc/passwd")
            zf.writestr(info, "malicious")

        with pytest.raises(PathTraversalError):
            validate_zip_safety(zip_path)

    def test_file_size_validation(self, temp_dir: Path):
        small_file = temp_dir / "small.txt"
        small_file.write_text("small content")

        validate_file_size(small_file)

        with pytest.raises(FileSizeError):
            validate_file_size(small_file, max_size=5)

    def test_file_not_found_error(self, temp_dir: Path):
        non_existent = temp_dir / "does_not_exist.zip"

        with pytest.raises(FileNotFoundError):
            validate_file_size(non_existent)


# =============================================================================
# Class Folder Structure Validation Tests
# =============================================================================

class TestClassFolderStructureValidation:

    def test_valid_structure_returns_stats(self, folder_per_class_structure: Path):
        class_dirs = list(folder_per_class_structure.iterdir())
        result = validate_class_folder_structure(class_dirs)

        assert result["num_classes"] == 3
        assert result["total_samples"] == 15
        assert "cat" in result["class_stats"]
        assert result["class_stats"]["cat"] == 5

    def test_empty_class_folder_detection(self, temp_dir: Path):
        dataset_dir = temp_dir / "dataset"
        dataset_dir.mkdir()

        (dataset_dir / "class_a").mkdir()
        (dataset_dir / "class_a" / "sample.png").write_bytes(b'\x89PNG')
        (dataset_dir / "class_b").mkdir()

        class_dirs = [dataset_dir / "class_a", dataset_dir / "class_b"]

        with pytest.raises(ClassStructureError) as exc_info:
            validate_class_folder_structure(class_dirs)

        assert "empty" in str(exc_info.value).lower()

    def test_class_imbalance_warning(self, temp_dir: Path):
        dataset_dir = temp_dir / "dataset"
        dataset_dir.mkdir()

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
        with pytest.raises(ClassStructureError) as exc_info:
            validate_class_folder_structure([])

        assert "no class folders" in str(exc_info.value).lower()

    def test_multiple_empty_classes_error_message(self, temp_dir: Path):
        dataset_dir = temp_dir / "dataset"
        dataset_dir.mkdir()

        for i in range(7):
            (dataset_dir / f"class_{i}").mkdir()

        class_dirs = list(dataset_dir.iterdir())

        with pytest.raises(ClassStructureError) as exc_info:
            validate_class_folder_structure(class_dirs)

        error_msg = str(exc_info.value)
        assert "7" in error_msg or "empty" in error_msg.lower()
        assert "more" in error_msg.lower()

    def test_mixed_file_types_warning(self, temp_dir: Path):
        dataset_dir = temp_dir / "dataset"
        dataset_dir.mkdir()

        (dataset_dir / "class_a").mkdir()
        (dataset_dir / "class_a" / "file.png").write_bytes(b'\x89PNG')
        (dataset_dir / "class_a" / "file.txt").write_text("text")

        class_dirs = [dataset_dir / "class_a"]
        result = validate_class_folder_structure(class_dirs)

        assert len(result["warnings"]) > 0

    def test_low_sample_passes(self, temp_dir: Path):
        dataset_dir = temp_dir / "dataset"
        dataset_dir.mkdir()

        (dataset_dir / "class_a").mkdir()
        (dataset_dir / "class_a" / "single.png").write_bytes(b'\x89PNG')

        class_dirs = [dataset_dir / "class_a"]
        result = validate_class_folder_structure(class_dirs)
        assert result["total_samples"] == 1


# =============================================================================
# Image Validation Tests
# =============================================================================

class TestImageValidation:

    def test_valid_png_magic_bytes(self, temp_dir: Path):
        png_file = temp_dir / "test.png"
        png_file.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)

        is_valid, format_name = validate_image_magic_bytes(png_file)

        assert is_valid is True
        assert format_name == 'png'

    def test_valid_jpeg_magic_bytes(self, temp_dir: Path):
        jpeg_file = temp_dir / "test.jpg"
        jpeg_file.write_bytes(b'\xff\xd8\xff' + b'\x00' * 100)

        is_valid, format_name = validate_image_magic_bytes(jpeg_file)

        assert is_valid is True
        assert format_name == 'jpeg'

    def test_valid_gif_magic_bytes(self, temp_dir: Path):
        for magic in [b'GIF87a', b'GIF89a']:
            gif_file = temp_dir / "test.gif"
            gif_file.write_bytes(magic + b'\x00' * 100)

            is_valid, format_name = validate_image_magic_bytes(gif_file)

            assert is_valid is True
            assert format_name == 'gif'

    def test_valid_bmp_magic_bytes(self, temp_dir: Path):
        bmp_file = temp_dir / "test.bmp"
        bmp_file.write_bytes(b'BM' + b'\x00' * 100)

        is_valid, format_name = validate_image_magic_bytes(bmp_file)

        assert is_valid is True
        assert format_name == 'bmp'

    def test_invalid_magic_bytes(self, temp_dir: Path):
        fake_image = temp_dir / "fake.png"
        fake_image.write_bytes(b'This is not an image')

        is_valid, format_name = validate_image_magic_bytes(fake_image)

        assert is_valid is False
        assert format_name == 'unknown'

    def test_empty_file_handling(self, temp_dir: Path):
        empty_file = temp_dir / "empty.png"
        empty_file.write_bytes(b'')

        is_valid, format_name = validate_image_magic_bytes(empty_file)

        assert is_valid is False
        assert format_name == 'empty'

    def test_webp_detection(self, temp_dir: Path):
        webp_file = temp_dir / "test.webp"
        webp_file.write_bytes(b'RIFF\x00\x00\x00\x00WEBP' + b'\x00' * 100)

        is_valid, format_name = validate_image_magic_bytes(webp_file)

        assert is_valid is True
        assert format_name == 'webp'


# =============================================================================
# Exception Hierarchy Tests
# =============================================================================

class TestExceptionHierarchy:

    def test_all_exceptions_inherit_from_base(self):
        for exc_class in [FileSizeError, PathTraversalError, ZipBombError,
                          CorruptedFileError, ClassStructureError, InvalidImageError]:
            assert issubclass(exc_class, DatasetValidationError)

    def test_base_inherits_from_exception(self):
        assert issubclass(DatasetValidationError, Exception)
