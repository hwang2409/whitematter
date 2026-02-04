"""
Dataset Manager - handles dataset upload, extraction, validation, and processing.
Supports multiple dataset formats: folder-per-class, MNIST IDX, CSV, and more.
"""

import json
import logging
import os
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

# Configure module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# Validation Constants
# =============================================================================

# Maximum file size for uploads (1 GB)
MAX_UPLOAD_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB

# Maximum total extracted size (to prevent zip bombs)
MAX_EXTRACTED_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB

# Maximum number of files in a ZIP (to prevent resource exhaustion)
MAX_FILES_IN_ZIP = 100_000

# Magic bytes for common image formats
IMAGE_MAGIC_BYTES = {
    b'\x89PNG\r\n\x1a\n': 'png',           # PNG
    b'\xff\xd8\xff': 'jpeg',                # JPEG (various subtypes)
    b'GIF87a': 'gif',                       # GIF87a
    b'GIF89a': 'gif',                       # GIF89a
    b'BM': 'bmp',                           # BMP
    b'RIFF': 'webp',                        # WebP (needs additional check)
}

# Minimum number of samples per class for consistency check
MIN_SAMPLES_PER_CLASS = 1

# Maximum class imbalance ratio (largest class / smallest class)
MAX_CLASS_IMBALANCE_RATIO = 100


# =============================================================================
# Custom Exceptions
# =============================================================================

class DatasetValidationError(Exception):
    """Base exception for dataset validation errors."""
    pass


class FileSizeError(DatasetValidationError):
    """Raised when file size exceeds limits."""
    pass


class PathTraversalError(DatasetValidationError):
    """Raised when path traversal attack is detected."""
    pass


class InvalidImageError(DatasetValidationError):
    """Raised when an invalid image file is detected."""
    pass


class ClassStructureError(DatasetValidationError):
    """Raised when class folder structure is invalid."""
    pass


class CorruptedFileError(DatasetValidationError):
    """Raised when a file is corrupted or cannot be read."""
    pass


class ZipBombError(DatasetValidationError):
    """Raised when a potential zip bomb is detected."""
    pass


# =============================================================================
# Validation Helper Functions
# =============================================================================

def validate_file_size(file_path: Path, max_size: int = MAX_UPLOAD_SIZE_BYTES) -> None:
    """
    Validate that a file does not exceed the maximum allowed size.

    Args:
        file_path: Path to the file to validate
        max_size: Maximum allowed size in bytes (default: 1GB)

    Raises:
        FileSizeError: If file exceeds the maximum size
        FileNotFoundError: If file does not exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = file_path.stat().st_size
    if file_size > max_size:
        size_mb = file_size / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        raise FileSizeError(
            f"File size ({size_mb:.1f} MB) exceeds maximum allowed size ({max_mb:.1f} MB). "
            f"Please upload a smaller file or contact support for larger uploads."
        )


def validate_zip_safety(zip_path: Path) -> Dict[str, Any]:
    """
    Validate ZIP file for security issues including path traversal and zip bombs.

    Args:
        zip_path: Path to the ZIP file to validate

    Returns:
        Dictionary with validation results including file count and total size

    Raises:
        PathTraversalError: If ZIP contains path traversal attempts
        ZipBombError: If ZIP appears to be a zip bomb
        CorruptedFileError: If ZIP file is corrupted
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Check for corrupted ZIP
            bad_file = zf.testzip()
            if bad_file is not None:
                raise CorruptedFileError(f"ZIP file is corrupted. Bad file: {bad_file}")

            file_list = zf.namelist()
            file_count = len(file_list)

            # Check file count limit
            if file_count > MAX_FILES_IN_ZIP:
                raise ZipBombError(
                    f"ZIP contains too many files ({file_count:,}). "
                    f"Maximum allowed: {MAX_FILES_IN_ZIP:,}"
                )

            total_uncompressed_size = 0

            for file_info in zf.infolist():
                filename = file_info.filename

                # Check for path traversal attacks
                # Normalize the path and check for directory escape attempts
                normalized = os.path.normpath(filename)

                # Check for absolute paths
                if os.path.isabs(normalized):
                    raise PathTraversalError(
                        f"ZIP contains absolute path which is not allowed: {filename}"
                    )

                # Check for parent directory references
                if normalized.startswith('..') or '/../' in filename or filename.startswith('../'):
                    raise PathTraversalError(
                        f"ZIP contains path traversal attempt: {filename}. "
                        f"Paths containing '../' are not allowed for security reasons."
                    )

                # Check for suspicious path components
                parts = filename.replace('\\', '/').split('/')
                for part in parts:
                    if part == '..':
                        raise PathTraversalError(
                            f"ZIP contains path traversal attempt: {filename}"
                        )

                # Accumulate uncompressed size
                total_uncompressed_size += file_info.file_size

                # Check for zip bomb (high compression ratio)
                if file_info.compress_size > 0:
                    compression_ratio = file_info.file_size / file_info.compress_size
                    # Suspicious if ratio is > 100:1 for large files
                    if compression_ratio > 100 and file_info.file_size > 10 * 1024 * 1024:
                        raise ZipBombError(
                            f"Suspicious compression ratio detected for {filename}. "
                            f"This may indicate a zip bomb attack."
                        )

            # Check total extracted size
            if total_uncompressed_size > MAX_EXTRACTED_SIZE_BYTES:
                size_gb = total_uncompressed_size / (1024 * 1024 * 1024)
                max_gb = MAX_EXTRACTED_SIZE_BYTES / (1024 * 1024 * 1024)
                raise ZipBombError(
                    f"Total extracted size ({size_gb:.1f} GB) exceeds maximum ({max_gb:.1f} GB). "
                    f"This may indicate a zip bomb attack."
                )

            return {
                'file_count': file_count,
                'total_uncompressed_size': total_uncompressed_size,
                'files': file_list
            }

    except zipfile.BadZipFile as e:
        raise CorruptedFileError(f"Invalid or corrupted ZIP file: {e}")


def validate_image_magic_bytes(file_path: Path) -> Tuple[bool, str]:
    """
    Validate that a file is actually an image by checking magic bytes.

    Args:
        file_path: Path to the file to validate

    Returns:
        Tuple of (is_valid, detected_format)

    Raises:
        CorruptedFileError: If file cannot be read
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(12)  # Read enough bytes for all signatures

        if len(header) < 2:
            return False, 'empty'

        # Check against known magic bytes
        for magic, format_name in IMAGE_MAGIC_BYTES.items():
            if header.startswith(magic):
                # Special handling for WebP (RIFF header + WEBP)
                if magic == b'RIFF' and len(header) >= 12:
                    if header[8:12] == b'WEBP':
                        return True, 'webp'
                    continue
                return True, format_name

        return False, 'unknown'

    except IOError as e:
        raise CorruptedFileError(f"Cannot read file {file_path}: {e}")


def validate_image_file(file_path: Path, check_loadable: bool = True) -> Dict[str, Any]:
    """
    Comprehensively validate an image file.

    Args:
        file_path: Path to the image file
        check_loadable: Whether to attempt loading the image with PIL

    Returns:
        Dictionary with image information (format, size, dimensions if loadable)

    Raises:
        InvalidImageError: If file is not a valid image
        CorruptedFileError: If file is corrupted
    """
    # First check magic bytes
    is_valid, detected_format = validate_image_magic_bytes(file_path)

    if not is_valid:
        raise InvalidImageError(
            f"File {file_path.name} is not a valid image. "
            f"Expected image magic bytes but found: {detected_format}"
        )

    result = {
        'path': str(file_path),
        'format': detected_format,
        'size_bytes': file_path.stat().st_size
    }

    # Optionally try to load with PIL for deeper validation
    if check_loadable:
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                # This will raise if image is corrupted
                img.verify()

            # Re-open to get dimensions (verify() leaves file in unusable state)
            with Image.open(file_path) as img:
                result['width'] = img.width
                result['height'] = img.height
                result['mode'] = img.mode
                result['pil_format'] = img.format

        except Exception as e:
            raise CorruptedFileError(
                f"Image file {file_path.name} appears corrupted or cannot be loaded: {e}"
            )

    return result


def validate_class_folder_structure(
    class_dirs: List[Path],
    warn_on_imbalance: bool = True
) -> Dict[str, Any]:
    """
    Validate the consistency of class folder structure.

    Args:
        class_dirs: List of class directory paths
        warn_on_imbalance: Whether to include warnings for class imbalance

    Returns:
        Dictionary with validation results and statistics

    Raises:
        ClassStructureError: If structure is invalid
    """
    if not class_dirs:
        raise ClassStructureError("No class folders found in dataset")

    results = {
        'num_classes': len(class_dirs),
        'class_stats': {},
        'warnings': [],
        'total_samples': 0
    }

    samples_per_class = {}
    extensions_per_class = {}
    empty_classes = []

    for class_dir in class_dirs:
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        files = [f for f in class_dir.iterdir() if f.is_file()]
        num_files = len(files)

        samples_per_class[class_name] = num_files
        results['total_samples'] += num_files

        if num_files == 0:
            empty_classes.append(class_name)
        elif num_files < MIN_SAMPLES_PER_CLASS:
            results['warnings'].append(
                f"Class '{class_name}' has very few samples ({num_files})"
            )

        # Track file extensions per class
        extensions = set(f.suffix.lower() for f in files if f.suffix)
        extensions_per_class[class_name] = extensions

    # Check for empty classes
    if empty_classes:
        raise ClassStructureError(
            f"Found {len(empty_classes)} empty class folder(s): {', '.join(empty_classes[:5])}"
            + (f" and {len(empty_classes) - 5} more" if len(empty_classes) > 5 else "")
        )

    results['class_stats'] = samples_per_class

    # Check class imbalance
    if warn_on_imbalance and samples_per_class:
        counts = list(samples_per_class.values())
        min_count = min(counts)
        max_count = max(counts)

        if min_count > 0:
            imbalance_ratio = max_count / min_count
            if imbalance_ratio > MAX_CLASS_IMBALANCE_RATIO:
                min_class = min(samples_per_class, key=samples_per_class.get)
                max_class = max(samples_per_class, key=samples_per_class.get)
                results['warnings'].append(
                    f"Significant class imbalance detected (ratio: {imbalance_ratio:.1f}:1). "
                    f"'{max_class}' has {max_count} samples, '{min_class}' has {min_count}."
                )

    # Check extension consistency
    all_extensions = set()
    for exts in extensions_per_class.values():
        all_extensions.update(exts)

    # Warn if classes have different file types
    if len(all_extensions) > 1:
        image_exts = all_extensions & IMAGE_EXTENSIONS
        non_image_exts = all_extensions - IMAGE_EXTENSIONS

        if image_exts and non_image_exts:
            results['warnings'].append(
                f"Mixed file types found: images ({', '.join(image_exts)}) "
                f"and other files ({', '.join(non_image_exts)})"
            )

    return results


def validate_extracted_images(
    raw_dir: Path,
    sample_size: int = 10,
    validate_all: bool = False
) -> Dict[str, Any]:
    """
    Validate extracted image files for integrity.

    Args:
        raw_dir: Directory containing extracted files
        sample_size: Number of images to validate if not validating all
        validate_all: Whether to validate all images (slower but thorough)

    Returns:
        Dictionary with validation results

    Raises:
        InvalidImageError: If invalid images are found
    """
    results = {
        'total_checked': 0,
        'valid': 0,
        'invalid': [],
        'warnings': []
    }

    # Find all image files
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(raw_dir.rglob(f"*{ext}"))
        image_files.extend(raw_dir.rglob(f"*{ext.upper()}"))

    # Remove duplicates
    image_files = list(set(image_files))

    if not image_files:
        results['warnings'].append("No image files found in extracted content")
        return results

    # Determine which files to check
    if validate_all:
        files_to_check = image_files
    else:
        # Sample files from different parts of the directory structure
        import random
        files_to_check = random.sample(
            image_files,
            min(sample_size, len(image_files))
        )

    for file_path in files_to_check:
        results['total_checked'] += 1
        try:
            validate_image_file(file_path, check_loadable=True)
            results['valid'] += 1
        except (InvalidImageError, CorruptedFileError) as e:
            results['invalid'].append({
                'path': str(file_path),
                'error': str(e)
            })

    # If we found invalid images in sample, warn user
    if results['invalid']:
        invalid_count = len(results['invalid'])
        if not validate_all:
            # Estimate total invalid based on sample
            estimated_total = (invalid_count / results['total_checked']) * len(image_files)
            results['warnings'].append(
                f"Found {invalid_count} invalid images in sample. "
                f"Estimated ~{int(estimated_total)} invalid images in total."
            )
        else:
            results['warnings'].append(
                f"Found {invalid_count} invalid images out of {len(image_files)} total."
            )

    return results


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

    def extract_zip(
        self,
        dataset_id: str,
        zip_path: Path,
        validate_images: bool = True,
        validate_all_images: bool = False
    ) -> DatasetMetadata:
        """
        Extract ZIP file and analyze contents with comprehensive validation.

        Args:
            dataset_id: The dataset ID to extract into
            zip_path: Path to the ZIP file
            validate_images: Whether to validate image files (default: True)
            validate_all_images: Whether to validate ALL images vs a sample (default: False)

        Returns:
            DatasetMetadata with analysis results

        Raises:
            FileSizeError: If file exceeds size limit
            PathTraversalError: If ZIP contains path traversal attempts
            ZipBombError: If ZIP appears to be a zip bomb
            CorruptedFileError: If ZIP or contents are corrupted
            InvalidImageError: If image validation fails
            ClassStructureError: If class folder structure is invalid
        """
        dataset_dir = self.uploads_dir / dataset_id
        raw_dir = dataset_dir / "raw"

        metadata = self._load_metadata(dataset_id)
        metadata.status = "validating"
        self._save_metadata(dataset_id, metadata)

        validation_warnings = []

        try:
            # Step 1: Validate file size
            logger.info("Validating ZIP file size...")
            validate_file_size(zip_path)

            # Step 2: Validate ZIP safety (path traversal, zip bombs, corruption)
            logger.info("Validating ZIP file safety...")
            metadata.status = "validating"
            self._save_metadata(dataset_id, metadata)

            zip_info = validate_zip_safety(zip_path)
            logger.info(
                "ZIP validation passed: %d files, %.1f MB uncompressed",
                zip_info['file_count'],
                zip_info['total_uncompressed_size'] / (1024 * 1024)
            )

            # Step 3: Extract ZIP (now safe to do so)
            logger.info("Extracting ZIP file...")
            metadata.status = "extracting"
            self._save_metadata(dataset_id, metadata)

            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Extract with safe member filtering (extra safety)
                for member in zf.namelist():
                    # Double-check path safety during extraction
                    member_path = Path(member)
                    if member_path.is_absolute() or '..' in member_path.parts:
                        logger.warning("Skipping suspicious path: %s", member)
                        continue
                    zf.extract(member, raw_dir)

            # Handle nested folder (common when zipping a folder)
            contents = list(raw_dir.iterdir())
            if len(contents) == 1 and contents[0].is_dir():
                # Move contents up one level
                nested_dir = contents[0]
                for item in nested_dir.iterdir():
                    shutil.move(str(item), str(raw_dir / item.name))
                nested_dir.rmdir()

            # Step 4: Analyze structure
            logger.info("Analyzing dataset structure...")
            metadata.status = "analyzing"
            self._save_metadata(dataset_id, metadata)

            metadata = self._analyze_dataset(dataset_id, metadata)

            # Step 5: Validate class folder structure if folder-per-class format
            if metadata.format == DatasetFormat.FOLDER_PER_CLASS.value:
                logger.info("Validating class folder structure...")
                class_dirs = [raw_dir / name for name in metadata.class_names]
                try:
                    structure_result = validate_class_folder_structure(class_dirs)
                    validation_warnings.extend(structure_result.get('warnings', []))
                except ClassStructureError as e:
                    # Log but don't fail - user might want to fix it
                    logger.warning("Class structure issue: %s", e)
                    validation_warnings.append(str(e))

            # Step 6: Validate image files if this is an image dataset
            if validate_images and metadata.data_type == DataType.IMAGE:
                logger.info("Validating image files...")
                try:
                    image_result = validate_extracted_images(
                        raw_dir,
                        sample_size=20,
                        validate_all=validate_all_images
                    )
                    validation_warnings.extend(image_result.get('warnings', []))

                    if image_result['invalid']:
                        # Log invalid images but don't fail
                        for invalid in image_result['invalid'][:5]:
                            logger.warning(
                                "Invalid image: %s - %s",
                                invalid['path'],
                                invalid['error']
                            )
                except (InvalidImageError, CorruptedFileError) as e:
                    logger.warning("Image validation issue: %s", e)
                    validation_warnings.append(str(e))

            # Store any validation warnings in metadata
            if validation_warnings:
                # Store warnings in error field if there were issues (but dataset is still usable)
                metadata.error = "Warnings: " + "; ".join(validation_warnings[:3])
                if len(validation_warnings) > 3:
                    metadata.error += f" (+{len(validation_warnings) - 3} more)"

            metadata.status = "ready"
            self._save_metadata(dataset_id, metadata)

        except DatasetValidationError as e:
            # Handle our custom validation errors with clear messages
            logger.error("Dataset validation failed: %s", e)
            metadata.status = "error"
            metadata.error = str(e)
            self._save_metadata(dataset_id, metadata)
            raise

        except zipfile.BadZipFile as e:
            # Handle corrupted ZIP files
            logger.error("Invalid ZIP file: %s", e)
            metadata.status = "error"
            metadata.error = f"Invalid or corrupted ZIP file: {e}"
            self._save_metadata(dataset_id, metadata)
            raise CorruptedFileError(f"Invalid or corrupted ZIP file: {e}")

        except Exception as e:
            # Handle unexpected errors
            logger.exception("Unexpected error during extraction")
            metadata.status = "error"
            metadata.error = f"Unexpected error: {str(e)}"
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

        logger.debug("Detecting format in %s", raw_dir)
        logger.debug("Found %d files, %d dirs", len(files), len(dirs))
        for f in files:
            logger.debug("File: %s (is_file=%s)", f.name, f.is_file())

        # Check for MNIST IDX format by filename pattern
        idx_images = [f for f in files if
                      ('images' in f.name.lower() or 'image' in f.name.lower()) and
                      ('ubyte' in f.name.lower() or 'idx' in f.name.lower())]
        idx_labels = [f for f in files if
                      ('labels' in f.name.lower() or 'label' in f.name.lower()) and
                      ('ubyte' in f.name.lower() or 'idx' in f.name.lower())]

        logger.debug("IDX by name - images: %s, labels: %s", [f.name for f in idx_images], [f.name for f in idx_labels])

        # If filename detection didn't work, try reading magic numbers
        if not (idx_images and idx_labels):
            logger.debug("Trying magic number detection...")
            for f in files:
                is_idx, idx_type = self._is_idx_file(f)
                if is_idx:
                    logger.debug("%s is IDX type: %s", f.name, idx_type)
                    if idx_type == 'images':
                        idx_images.append(f)
                    elif idx_type == 'labels':
                        idx_labels.append(f)

        if idx_images and idx_labels:
            logger.debug("Detected MNIST IDX format!")
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
                logger.debug("Detected folder-per-class format")
                return DatasetFormat.FOLDER_PER_CLASS, {"class_dirs": dirs}

        # Check for CSV with labels
        csv_files = [f for f in files if f.suffix.lower() in {'.csv', '.tsv'}]
        if csv_files:
            return DatasetFormat.CSV_LABELS, {"csv_file": csv_files[0]}

        # Check for flat images
        image_files = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
        if image_files:
            return DatasetFormat.FLAT_IMAGES, {"image_files": image_files}

        logger.debug("Unknown format")
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

        logger.debug("Preview MNIST - images: %s, labels: %s", images_file, labels_file)

        if not images_file or not labels_file:
            logger.debug("Missing files for MNIST preview")
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
            logger.debug("Failed to preview MNIST IDX: %s", e, exc_info=True)

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
                        logger.warning("Failed to load preview image %s: %s", f, e)
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
                logger.warning("Failed to load preview image %s: %s", f, e)
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
            logger.warning("Failed to load metadata: %s", e)
            return None
