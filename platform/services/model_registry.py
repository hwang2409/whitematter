"""
Model registry - file-based model management operations.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import HTTPException
from PIL import Image

from config import (
    MODELS_DIR, DATASETS,
    CIFAR10_MEAN, CIFAR10_STD, MNIST_MEAN, MNIST_STD,
)
from schemas import ModelMetadata
from services.model_format import (
    validate_model_file, format_header_info,
)

try:
    import whitematter as wm
except Exception as e:
    logging.warning(
        "whitematter extension could not be loaded; C++ model operations are disabled: %s",
        e,
    )
    wm = None

logger = logging.getLogger(__name__)

# In-memory cache of loaded whitematter Model objects
loaded_models: dict = {}


def get_model_path(model_id: str) -> Path:
    return MODELS_DIR / f"{model_id}.bin"


def get_metadata_path(model_id: str) -> Path:
    return MODELS_DIR / f"{model_id}.json"


def load_model_metadata(model_id: str) -> Optional[ModelMetadata]:
    path = get_metadata_path(model_id)
    if not path.exists():
        return None
    with open(path) as f:
        return ModelMetadata(**json.load(f))


def save_model_metadata(metadata: ModelMetadata):
    with open(get_metadata_path(metadata.id), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)


def list_all_models() -> list[ModelMetadata]:
    models = []
    for path in MODELS_DIR.glob("*.json"):
        if m := load_model_metadata(path.stem):
            models.append(m)
    return sorted(models, key=lambda m: m.created_at, reverse=True)


def get_loaded_model(model_id: str) -> "wm.Model":
    if wm is None:
        raise HTTPException(
            status_code=503,
            detail="whitematter extension is not available; model loading is disabled",
        )
    if model_id in loaded_models:
        return loaded_models[model_id]
    metadata = load_model_metadata(model_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    model_path = get_model_path(model_id)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model weights not found: {model_id}")

    is_valid, header, error = validate_model_file(model_path)
    if not is_valid:
        logger.error(f"Model validation failed for {model_id}: {error}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model file format: {error}"
        )

    if header and header.is_legacy:
        logger.info(f"Loading legacy format model: {model_id}")
    elif header:
        logger.info(f"Loading model {model_id}: {format_header_info(header)}")

    arch = "simple"
    if "vgg" in metadata.architecture.lower():
        arch = "vgg"
    elif "mnist" in metadata.architecture.lower():
        arch = "mnist"

    model = wm.Model()
    model.load(str(model_path), arch)
    loaded_models[model_id] = model
    return model


def preprocess_image(image: Image.Image, dataset: str) -> np.ndarray:
    info = DATASETS[dataset]
    c, h, w = info["input_shape"]
    image = image.resize((w, h), Image.Resampling.BILINEAR)

    if c == 1:
        image = image.convert('L')
        arr = np.array(image, dtype=np.float32).reshape(1, h, w) / 255.0
        mean, std = np.array(MNIST_MEAN).reshape(1,1,1), np.array(MNIST_STD).reshape(1,1,1)
    else:
        image = image.convert('RGB')
        arr = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mean, std = np.array(CIFAR10_MEAN).reshape(3,1,1), np.array(CIFAR10_STD).reshape(3,1,1)

    return np.ascontiguousarray((arr - mean) / std, dtype=np.float32)
