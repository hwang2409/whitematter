"""
Dataset validation utilities — security checks for uploads.

Zip bomb detection, path traversal prevention, magic-byte image validation,
class-folder structure checks, and image integrity sampling.
"""

import logging
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB
MAX_EXTRACTED_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
MAX_FILES_IN_ZIP = 100_000

IMAGE_MAGIC_BYTES = {
    b'\x89PNG\r\n\x1a\n': 'png',
    b'\xff\xd8\xff': 'jpeg',
    b'GIF87a': 'gif',
    b'GIF89a': 'gif',
    b'BM': 'bmp',
    b'RIFF': 'webp',
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

MIN_SAMPLES_PER_CLASS = 1
MAX_CLASS_IMBALANCE_RATIO = 100


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


def validate_file_size(file_path: Path, max_size: int = MAX_UPLOAD_SIZE_BYTES) -> None:
    """Validate that a file does not exceed the maximum allowed size."""
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

    Returns dict with file_count, total_uncompressed_size, files.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            bad_file = zf.testzip()
            if bad_file is not None:
                raise CorruptedFileError(f"ZIP file is corrupted. Bad file: {bad_file}")

            file_list = zf.namelist()
            file_count = len(file_list)

            if file_count > MAX_FILES_IN_ZIP:
                raise ZipBombError(
                    f"ZIP contains too many files ({file_count:,}). "
                    f"Maximum allowed: {MAX_FILES_IN_ZIP:,}"
                )

            total_uncompressed_size = 0

            for file_info in zf.infolist():
                filename = file_info.filename

                normalized = os.path.normpath(filename)

                if os.path.isabs(normalized):
                    raise PathTraversalError(
                        f"ZIP contains absolute path which is not allowed: {filename}"
                    )

                if normalized.startswith('..') or '/../' in filename or filename.startswith('../'):
                    raise PathTraversalError(
                        f"ZIP contains path traversal attempt: {filename}. "
                        f"Paths containing '../' are not allowed for security reasons."
                    )

                parts = filename.replace('\\', '/').split('/')
                for part in parts:
                    if part == '..':
                        raise PathTraversalError(
                            f"ZIP contains path traversal attempt: {filename}"
                        )

                total_uncompressed_size += file_info.file_size

                if file_info.compress_size > 0:
                    compression_ratio = file_info.file_size / file_info.compress_size
                    if compression_ratio > 100 and file_info.file_size > 10 * 1024 * 1024:
                        raise ZipBombError(
                            f"Suspicious compression ratio detected for {filename}. "
                            f"This may indicate a zip bomb attack."
                        )

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
    """Validate that a file is actually an image by checking magic bytes."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(12)

        if len(header) < 2:
            return False, 'empty'

        for magic, format_name in IMAGE_MAGIC_BYTES.items():
            if header.startswith(magic):
                if magic == b'RIFF' and len(header) >= 12:
                    if header[8:12] == b'WEBP':
                        return True, 'webp'
                    continue
                return True, format_name

        return False, 'unknown'

    except IOError as e:
        raise CorruptedFileError(f"Cannot read file {file_path}: {e}")


def validate_image_file(file_path: Path, check_loadable: bool = True) -> Dict[str, Any]:
    """Comprehensively validate an image file."""
    is_valid, detected_format = validate_image_magic_bytes(file_path)

    if not is_valid:
        raise InvalidImageError(
            f"File {file_path.name} is not a valid image. "
            f"Expected image magic bytes but found: {detected_format}"
        )

    result: Dict[str, Any] = {
        'path': str(file_path),
        'format': detected_format,
        'size_bytes': file_path.stat().st_size
    }

    if check_loadable:
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                img.verify()
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
    """Validate the consistency of class folder structure."""
    if not class_dirs:
        raise ClassStructureError("No class folders found in dataset")

    results: Dict[str, Any] = {
        'num_classes': len(class_dirs),
        'class_stats': {},
        'warnings': [],
        'total_samples': 0
    }

    samples_per_class: Dict[str, int] = {}
    extensions_per_class: Dict[str, set] = {}
    empty_classes: List[str] = []

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

        extensions = set(f.suffix.lower() for f in files if f.suffix)
        extensions_per_class[class_name] = extensions

    if empty_classes:
        raise ClassStructureError(
            f"Found {len(empty_classes)} empty class folder(s): {', '.join(empty_classes[:5])}"
            + (f" and {len(empty_classes) - 5} more" if len(empty_classes) > 5 else "")
        )

    results['class_stats'] = samples_per_class

    if warn_on_imbalance and samples_per_class:
        counts = list(samples_per_class.values())
        min_count = min(counts)
        max_count = max(counts)

        if min_count > 0:
            imbalance_ratio = max_count / min_count
            if imbalance_ratio > MAX_CLASS_IMBALANCE_RATIO:
                min_class = min(samples_per_class, key=samples_per_class.get)  # type: ignore[arg-type]
                max_class = max(samples_per_class, key=samples_per_class.get)  # type: ignore[arg-type]
                results['warnings'].append(
                    f"Significant class imbalance detected (ratio: {imbalance_ratio:.1f}:1). "
                    f"'{max_class}' has {max_count} samples, '{min_class}' has {min_count}."
                )

    all_extensions: set = set()
    for exts in extensions_per_class.values():
        all_extensions.update(exts)

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
    """Validate extracted image files for integrity."""
    results: Dict[str, Any] = {
        'total_checked': 0,
        'valid': 0,
        'invalid': [],
        'warnings': []
    }

    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(raw_dir.rglob(f"*{ext}"))
        image_files.extend(raw_dir.rglob(f"*{ext.upper()}"))

    image_files = list(set(image_files))

    if not image_files:
        results['warnings'].append("No image files found in extracted content")
        return results

    if validate_all:
        files_to_check = image_files
    else:
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

    if results['invalid']:
        invalid_count = len(results['invalid'])
        if not validate_all:
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
