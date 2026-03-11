"""
Whitematter Model Format Validation

Provides utilities for validating, reading, and writing whitematter model files.
Supports both the new format with enhanced header and backwards compatibility
with older model files.

File Format (v1):
    - magic (4 bytes): 0x574D4D4C ("WMML" in ASCII)
    - version (1 byte): format version number
    - arch_type (1 byte): architecture type identifier
    - param_count (8 bytes): total parameter count (uint64)
    - [parameter tensors...]

Legacy Format:
    - magic (4 bytes): 0x574D4D00 ("WMM\0" - old format)
    - num_params (4 bytes): number of parameter tensors
    - [parameter tensors...]
"""

import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple, Union, BinaryIO
import logging

logger = logging.getLogger(__name__)


# Magic number for new whitematter model format: "WMML" (WhiteMatter Model Language)
MODEL_MAGIC = 0x574D4D4C  # 'W' 'M' 'M' 'L' in ASCII

# Legacy magic number for backwards compatibility
LEGACY_MODEL_MAGIC = 0x574D4D00  # 'W' 'M' 'M' '\0'

# Tensor file magic number
TENSOR_MAGIC = 0x54454E53  # 'T' 'E' 'N' 'S'

# Current format version
FORMAT_VERSION = 1


class ArchType(IntEnum):
    """Model architecture type identifiers."""
    UNKNOWN = 0
    SIMPLE_CNN = 1      # Simple CNN for image classification
    VGG_STYLE = 2       # VGG-style deep CNN
    MNIST_CNN = 3       # MNIST-optimized CNN
    TRANSFORMER_LM = 4  # Transformer language model
    LSTM_LM = 5         # LSTM language model
    CUSTOM = 255        # Custom architecture


@dataclass
class ModelHeader:
    """Model file header information."""
    magic: int
    version: int
    arch_type: ArchType
    param_count: int

    @property
    def is_valid(self) -> bool:
        """Check if header has valid magic number."""
        return self.magic == MODEL_MAGIC

    @property
    def is_legacy(self) -> bool:
        """Check if this is a legacy format file."""
        return self.magic == LEGACY_MODEL_MAGIC


class ModelFormatError(Exception):
    """Raised when model file format validation fails."""
    pass


def write_model_header(
    file: BinaryIO,
    arch_type: Union[ArchType, int] = ArchType.CUSTOM,
    param_count: int = 0,
    version: int = FORMAT_VERSION
) -> int:
    """
    Write model header to a binary file.

    Args:
        file: Binary file object opened for writing
        arch_type: Architecture type identifier
        param_count: Total number of parameters in the model
        version: Format version number

    Returns:
        Number of bytes written

    Example:
        with open('model.bin', 'wb') as f:
            write_model_header(f, ArchType.SIMPLE_CNN, param_count=1000000)
            # Write parameter data...
    """
    if isinstance(arch_type, ArchType):
        arch_type_val = arch_type.value
    else:
        arch_type_val = int(arch_type)

    # Header format: magic(4B) + version(1B) + arch_type(1B) + param_count(8B)
    header = struct.pack(
        '<I B B Q',  # Little-endian: uint32, uint8, uint8, uint64
        MODEL_MAGIC,
        version,
        arch_type_val,
        param_count
    )

    file.write(header)
    return len(header)  # 14 bytes


def read_model_header(file: BinaryIO) -> ModelHeader:
    """
    Read model header from a binary file.

    Args:
        file: Binary file object opened for reading

    Returns:
        ModelHeader with parsed information

    Raises:
        ModelFormatError: If header cannot be read or is invalid

    Example:
        with open('model.bin', 'rb') as f:
            header = read_model_header(f)
            if header.is_valid:
                print(f"Model has {header.param_count} parameters")
    """
    # Read magic number first
    magic_bytes = file.read(4)
    if len(magic_bytes) < 4:
        raise ModelFormatError("File too small to contain valid header")

    magic = struct.unpack('<I', magic_bytes)[0]

    # Check for legacy format
    if magic == LEGACY_MODEL_MAGIC:
        # Legacy format: magic(4B) + num_params(4B)
        num_params_bytes = file.read(4)
        if len(num_params_bytes) < 4:
            raise ModelFormatError("Incomplete legacy header")
        num_params = struct.unpack('<I', num_params_bytes)[0]

        return ModelHeader(
            magic=magic,
            version=0,  # Legacy version
            arch_type=ArchType.UNKNOWN,
            param_count=num_params
        )

    # New format: read remaining header fields
    header_rest = file.read(10)  # version(1B) + arch_type(1B) + param_count(8B)
    if len(header_rest) < 10:
        raise ModelFormatError("Incomplete model header")

    version, arch_type_val, param_count = struct.unpack('<B B Q', header_rest)

    try:
        arch_type = ArchType(arch_type_val)
    except ValueError:
        arch_type = ArchType.UNKNOWN

    return ModelHeader(
        magic=magic,
        version=version,
        arch_type=arch_type,
        param_count=param_count
    )


def validate_model_file(
    path: Union[str, Path],
    expected_arch_type: Optional[ArchType] = None,
    expected_param_count: Optional[int] = None
) -> Tuple[bool, Optional[ModelHeader], Optional[str]]:
    """
    Validate a model file has correct format and optional expected values.

    Args:
        path: Path to the model file
        expected_arch_type: If provided, verify architecture type matches
        expected_param_count: If provided, verify parameter count matches

    Returns:
        Tuple of (is_valid, header, error_message)
        - is_valid: True if file is valid (new or legacy format)
        - header: ModelHeader if readable, None on read failure
        - error_message: Description of validation failure, None if valid

    Example:
        valid, header, error = validate_model_file('model.bin')
        if not valid:
            print(f"Invalid model: {error}")
        elif header.is_legacy:
            print("Using legacy format - consider re-saving")
        else:
            print(f"Valid model: {header.arch_type.name}, {header.param_count} params")
    """
    path = Path(path)

    if not path.exists():
        return False, None, f"Model file not found: {path}"

    if path.stat().st_size < 8:
        return False, None, f"Model file too small: {path} ({path.stat().st_size} bytes)"

    try:
        with open(path, 'rb') as f:
            header = read_model_header(f)
    except ModelFormatError as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, f"Failed to read model header: {e}"

    # Check magic number
    if not header.is_valid and not header.is_legacy:
        actual_magic = hex(header.magic)
        return False, header, (
            f"Invalid model format: magic number {actual_magic} does not match "
            f"expected {hex(MODEL_MAGIC)} or legacy {hex(LEGACY_MODEL_MAGIC)}"
        )

    # Validate architecture type if specified
    if expected_arch_type is not None and header.arch_type != expected_arch_type:
        return False, header, (
            f"Architecture mismatch: expected {expected_arch_type.name}, "
            f"got {header.arch_type.name}"
        )

    # Validate parameter count if specified
    if expected_param_count is not None and header.param_count != expected_param_count:
        return False, header, (
            f"Parameter count mismatch: expected {expected_param_count}, "
            f"got {header.param_count}"
        )

    return True, header, None


def get_arch_type_from_name(name: str) -> ArchType:
    """
    Get architecture type from a string name.

    Args:
        name: Architecture name (e.g., 'simple', 'vgg', 'mnist', 'transformer')

    Returns:
        Corresponding ArchType enum value
    """
    name_lower = name.lower()

    if 'vgg' in name_lower:
        return ArchType.VGG_STYLE
    elif 'mnist' in name_lower:
        return ArchType.MNIST_CNN
    elif 'transformer' in name_lower:
        return ArchType.TRANSFORMER_LM
    elif 'lstm' in name_lower:
        return ArchType.LSTM_LM
    elif 'simple' in name_lower or 'cnn' in name_lower:
        return ArchType.SIMPLE_CNN
    else:
        return ArchType.CUSTOM


def check_file_magic(path: Union[str, Path]) -> Optional[int]:
    """
    Read just the magic number from a file without full validation.

    Args:
        path: Path to the file

    Returns:
        Magic number as integer, or None if file cannot be read
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size < 4:
        return None

    try:
        with open(path, 'rb') as f:
            magic_bytes = f.read(4)
            if len(magic_bytes) < 4:
                return None
            return struct.unpack('<I', magic_bytes)[0]
    except Exception:
        return None


def is_whitematter_model(path: Union[str, Path]) -> bool:
    """
    Quick check if a file appears to be a whitematter model.

    Args:
        path: Path to the file

    Returns:
        True if file has a valid whitematter magic number
    """
    magic = check_file_magic(path)
    return magic in (MODEL_MAGIC, LEGACY_MODEL_MAGIC)


def format_header_info(header: ModelHeader) -> str:
    """
    Format header information as a human-readable string.

    Args:
        header: ModelHeader to format

    Returns:
        Formatted string describing the header
    """
    if header.is_legacy:
        return (
            f"Whitematter Model (Legacy Format)\n"
            f"  Parameters: {header.param_count}"
        )
    else:
        return (
            f"Whitematter Model v{header.version}\n"
            f"  Magic: {hex(header.magic)}\n"
            f"  Architecture: {header.arch_type.name}\n"
            f"  Parameters: {header.param_count:,}"
        )
