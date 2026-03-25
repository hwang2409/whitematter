import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple, Union, BinaryIO
import logging

logger = logging.getLogger(__name__)


MODEL_MAGIC = 0x574D4D4C
LEGACY_MODEL_MAGIC = 0x574D4D00
TENSOR_MAGIC = 0x54454E53
FORMAT_VERSION = 1


class ArchType(IntEnum):
    UNKNOWN = 0
    SIMPLE_CNN = 1      # Simple CNN for image classification
    VGG_STYLE = 2       # VGG-style deep CNN
    MNIST_CNN = 3       # MNIST-optimized CNN
    TRANSFORMER_LM = 4  # Transformer language model
    LSTM_LM = 5         # LSTM language model
    CUSTOM = 255        # Custom architecture


@dataclass
class ModelHeader:
    magic: int
    version: int
    arch_type: ArchType
    param_count: int

    @property
    def is_valid(self) -> bool:
        return self.magic == MODEL_MAGIC

    @property
    def is_legacy(self) -> bool:
        return self.magic == LEGACY_MODEL_MAGIC


class ModelFormatError(Exception):
    pass


def write_model_header(
    file: BinaryIO,
    arch_type: Union[ArchType, int] = ArchType.CUSTOM,
    param_count: int = 0,
    version: int = FORMAT_VERSION
) -> int:
    if isinstance(arch_type, ArchType):
        arch_type_val = arch_type.value
    else:
        arch_type_val = int(arch_type)

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
    magic_bytes = file.read(4)
    if len(magic_bytes) < 4:
        raise ModelFormatError("File too small to contain valid header")

    magic = struct.unpack('<I', magic_bytes)[0]

    if magic == LEGACY_MODEL_MAGIC:
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

    if not header.is_valid and not header.is_legacy:
        actual_magic = hex(header.magic)
        return False, header, (
            f"Invalid model format: magic number {actual_magic} does not match "
            f"expected {hex(MODEL_MAGIC)} or legacy {hex(LEGACY_MODEL_MAGIC)}"
        )

    if expected_arch_type is not None and header.arch_type != expected_arch_type:
        return False, header, (
            f"Architecture mismatch: expected {expected_arch_type.name}, "
            f"got {header.arch_type.name}"
        )

    if expected_param_count is not None and header.param_count != expected_param_count:
        return False, header, (
            f"Parameter count mismatch: expected {expected_param_count}, "
            f"got {header.param_count}"
        )

    return True, header, None


def get_arch_type_from_name(name: str) -> ArchType:
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
    magic = check_file_magic(path)
    return magic in (MODEL_MAGIC, LEGACY_MODEL_MAGIC)


def format_header_info(header: ModelHeader) -> str:
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
