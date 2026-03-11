"""
Import datasets from Hugging Face Hub.
Supports image classification (image + label columns) and text datasets.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

from datasets import load_dataset

logger = logging.getLogger(__name__)

# Reasonable limits to avoid OOM or abuse
MAX_IMAGE_ROWS = 50_000
MAX_TEXT_ROWS = 100_000
MAX_TEXT_CHARS = 10_000_000  # 10M chars

# Column names we look for
IMAGE_COLUMN_NAMES = ("image", "img", "images")
LABEL_COLUMN_NAMES = ("label", "labels", "label_id", "class", "class_label")
TEXT_COLUMN_NAMES = ("text", "content", "sentence", "paragraph")


class HuggingFaceImportError(Exception):
    """Raised when HF import fails."""

    pass


def _find_column(hf_features: Any, candidates: tuple[str, ...]) -> str | None:
    """Return first matching column name from dataset features."""
    if hf_features is None:
        return None
    names = list(hf_features.keys()) if hasattr(hf_features, "keys") else []
    for c in candidates:
        if c in names:
            return c
    return None


def _ensure_pil(image: Any) -> Any:
    """Return PIL Image. HF sometimes gives dict with bytes."""
    from PIL import Image as PILImage
    if hasattr(image, "save"):
        return image
    if isinstance(image, dict) and "bytes" in image:
        return PILImage.open(io.BytesIO(image["bytes"])).convert("RGB")
    if isinstance(image, bytes):
        return PILImage.open(io.BytesIO(image)).convert("RGB")
    raise HuggingFaceImportError("Unsupported image type in dataset")


def import_image_dataset(
    dataset_id: str,
    name: str,
    config: str | None = None,
    split: str = "train",
) -> dict[str, Any]:
    """
    Load an image classification dataset from HF and return dataset info
    using dataset_service (creates dataset + upload_zip).
    Expects columns for image and label. Builds folder-per-class ZIP.
    """
    from dependencies import dataset_service

    try:
        ds = load_dataset(
            dataset_id,
            config=config,
            split=split,
            trust_remote_code=True,
        )
    except Exception as e:
        raise HuggingFaceImportError(f"Failed to load dataset: {e}") from e

    if ds is None or len(ds) == 0:
        raise HuggingFaceImportError("Dataset is empty")

    features = ds.features if hasattr(ds, "features") else None
    image_col = _find_column(features, IMAGE_COLUMN_NAMES)
    label_col = _find_column(features, LABEL_COLUMN_NAMES)

    if not image_col:
        raise HuggingFaceImportError(
            "No image column found. Need one of: " + ", ".join(IMAGE_COLUMN_NAMES)
        )
    if not label_col:
        raise HuggingFaceImportError(
            "No label column found. Need one of: " + ", ".join(LABEL_COLUMN_NAMES)
        )

    n = min(len(ds), MAX_IMAGE_ROWS)
    if n < len(ds):
        logger.warning("Truncating HF image dataset to %d rows", MAX_IMAGE_ROWS)

    # Build class folders in memory and create ZIP
    buffer = io.BytesIO()
    class_to_idx: dict[str, int] = {}
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(n):
            row = ds[i]
            img = row.get(image_col)
            label_val = row.get(label_col)
            if img is None or label_val is None:
                continue
            try:
                pil_img = _ensure_pil(img)
            except Exception as e:
                logger.warning("Skip row %d: %s", i, e)
                continue
            if isinstance(label_val, int):
                class_name = str(label_val)
            else:
                class_name = str(label_val).replace("/", "_").replace("\\", "_") or "unknown"
            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(class_to_idx)
            arcname = f"{class_name}/img_{i:06d}.png"
            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format="PNG")
            zf.writestr(arcname, img_buffer.getvalue())

    if len(class_to_idx) == 0:
        raise HuggingFaceImportError("No valid image/label rows found")

    dataset_info = dataset_service.create_dataset(name)
    dataset_id_db = dataset_info["id"]
    zip_bytes = buffer.getvalue()
    dataset_service.upload_zip(
        dataset_id_db,
        zip_bytes,
        "huggingface_images.zip",
    )
    return dataset_service.get_dataset(dataset_id_db)


def import_text_dataset(
    dataset_id: str,
    name: str,
    config: str | None = None,
    split: str = "train",
    tokenizer_type: str = "character",
    seq_length: int = 128,
) -> dict[str, Any]:
    """
    Load a text dataset from HF and return dataset info
    using dataset_service (create_dataset + upload_text).
    """
    from dependencies import dataset_service

    try:
        ds = load_dataset(
            dataset_id,
            config=config,
            split=split,
            trust_remote_code=True,
        )
    except Exception as e:
        raise HuggingFaceImportError(f"Failed to load dataset: {e}") from e

    if ds is None or len(ds) == 0:
        raise HuggingFaceImportError("Dataset is empty")

    features = ds.features if hasattr(ds, "features") else None
    text_col = _find_column(features, TEXT_COLUMN_NAMES)
    if not text_col:
        raise HuggingFaceImportError(
            "No text column found. Need one of: " + ", ".join(TEXT_COLUMN_NAMES)
        )

    n = min(len(ds), MAX_TEXT_ROWS)
    if n < len(ds):
        logger.warning("Truncating HF text dataset to %d rows", MAX_TEXT_ROWS)

    parts: list[str] = []
    total_chars = 0
    for i in range(n):
        row = ds[i]
        val = row.get(text_col)
        if val is None:
            continue
        s = str(val).strip()
        if not s:
            continue
        parts.append(s)
        total_chars += len(s)
        if total_chars >= MAX_TEXT_CHARS:
            break

    if not parts:
        raise HuggingFaceImportError("No text content found in dataset")

    text_content = "\n\n".join(parts)
    if isinstance(text_content, str):
        text_bytes = text_content.encode("utf-8")
    else:
        text_bytes = text_content

    dataset_info = dataset_service.create_dataset(name)
    dataset_id_db = dataset_info["id"]
    dataset_service.upload_text(
        dataset_id=dataset_id_db,
        file_content=text_bytes,
        filename="huggingface_text.txt",
        tokenizer_type=tokenizer_type,
        seq_length=seq_length,
    )
    return dataset_service.get_dataset(dataset_id_db)


def import_from_huggingface(
    dataset_id: str,
    name: str | None = None,
    config: str | None = None,
    split: str = "train",
) -> dict[str, Any]:
    """
    Auto-detect dataset type (image vs text) and import.
    Returns dataset info dict from dataset_service.
    """
    display_name = name or dataset_id.replace("/", "_")[:200]
    try:
        ds = load_dataset(
            dataset_id,
            config=config,
            split=split,
            trust_remote_code=True,
        )
    except Exception as e:
        raise HuggingFaceImportError(f"Failed to load dataset: {e}") from e

    if ds is None or len(ds) == 0:
        raise HuggingFaceImportError("Dataset is empty")

    features = ds.features if hasattr(ds, "features") else None
    image_col = _find_column(features, IMAGE_COLUMN_NAMES)
    label_col = _find_column(features, LABEL_COLUMN_NAMES)
    text_col = _find_column(features, TEXT_COLUMN_NAMES)

    if image_col and label_col:
        return import_image_dataset(dataset_id, display_name, config=config, split=split)
    if text_col:
        return import_text_dataset(dataset_id, display_name, config=config, split=split)

    raise HuggingFaceImportError(
        "Could not detect dataset type. Need (image + label) for images or a text column for text."
    )
