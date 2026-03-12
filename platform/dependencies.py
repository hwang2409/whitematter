"""
Shared FastAPI dependencies and application state.
"""

import asyncio
import json
import logging
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from fastapi import HTTPException
from PIL import Image
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import (
    PROJECT_ROOT, MODELS_DIR, DATA_DIR, UPLOADS_DIR, GENERATED_DIR,
    DATASETS, CIFAR10_MEAN, CIFAR10_STD, MNIST_MEAN, MNIST_STD,
)
from schemas import ModelMetadata, TrainStatus
from dataset_manager import DatasetManager, DataType
from codegen import CodeGenerator, compile_training_code
from services.dataset_service import DatasetService
from services.job_store import TrainingJobStore
from llm.service import get_llm_service
from model_format import (
    validate_model_file, ModelFormatError, is_whitematter_model,
    format_header_info, get_arch_type_from_name
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

# Shared rate limiter (single instance so all routes share counters)
limiter = Limiter(key_func=get_remote_address)

# Shared mutable state
loaded_models: dict = {}
training_jobs: TrainingJobStore = TrainingJobStore()

# WebSocket subscriber registry for real-time training updates
_ws_subscribers: Dict[str, list] = {}  # job_id -> list of asyncio.Queue
_ws_lock = threading.Lock()
_event_loop: Optional[asyncio.AbstractEventLoop] = None

# Service instances
dataset_manager = DatasetManager(uploads_dir=UPLOADS_DIR)
dataset_service = DatasetService()
code_generator = CodeGenerator()
llm_service = get_llm_service()


def capture_event_loop():
    """Capture the running event loop for cross-thread WebSocket notifications."""
    global _event_loop
    _event_loop = asyncio.get_running_loop()


def ensure_dirs():
    MODELS_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(exist_ok=True)
    GENERATED_DIR.mkdir(exist_ok=True)


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


def _get_job_snapshot(job_id: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON-safe dict from training_jobs."""
    j = training_jobs.get(job_id)
    if j is None:
        return None
    status = j["status"]
    if hasattr(status, 'value'):
        status = status.value
    return {
        "job_id": j["id"],
        "model_id": j["model_id"],
        "status": status,
        "epoch": j.get("epoch", 0),
        "total_epochs": j.get("total_epochs", 0),
        "loss": j.get("loss", 0.0),
        "accuracy": j.get("accuracy", 0.0),
        "message": j.get("message", ""),
    }


def notify_training_subscribers(job_id: str):
    """Push latest job snapshot to all WebSocket subscribers. Safe to call from background threads."""
    snapshot = _get_job_snapshot(job_id)
    if snapshot is None or _event_loop is None:
        return
    with _ws_lock:
        queues = list(_ws_subscribers.get(job_id, []))
    for q in queues:
        try:
            asyncio.run_coroutine_threadsafe(q.put(snapshot), _event_loop)
        except RuntimeError:
            pass  # event loop closed


def process_mnist_idx(raw_dir: Path, output_dir: Path, metadata) -> dict:
    """Process MNIST IDX format files into binary tensors."""
    import struct

    output_dir.mkdir(parents=True, exist_ok=True)

    all_items = list(raw_dir.iterdir())
    files = [f for f in all_items if f.is_file()]

    logger.debug("MNIST: Looking for IDX files in %s", raw_dir)
    logger.debug("MNIST: Found %d files: %s", len(files), [f.name for f in files])

    train_images_file = None
    train_labels_file = None
    test_images_file = None
    test_labels_file = None

    for f in files:
        if not f.is_file():
            continue
        name_lower = f.name.lower()
        if not ('ubyte' in name_lower or 'idx' in name_lower):
            continue

        if 'train' in name_lower and 'images' in name_lower:
            train_images_file = f
        elif 'train' in name_lower and 'labels' in name_lower:
            train_labels_file = f
        elif ('t10k' in name_lower or 'test' in name_lower) and 'images' in name_lower:
            test_images_file = f
        elif ('t10k' in name_lower or 'test' in name_lower) and 'labels' in name_lower:
            test_labels_file = f

    if not train_images_file:
        for f in files:
            if f.is_file() and 'images' in f.name.lower() and ('ubyte' in f.name.lower() or 'idx' in f.name.lower()):
                train_images_file = f
                break
    if not train_labels_file:
        for f in files:
            if f.is_file() and 'labels' in f.name.lower() and ('ubyte' in f.name.lower() or 'idx' in f.name.lower()):
                train_labels_file = f
                break

    logger.debug("MNIST: train_images: %s, train_labels: %s", train_images_file, train_labels_file)
    logger.debug("MNIST: test_images: %s, test_labels: %s", test_images_file, test_labels_file)

    if not train_images_file or not train_labels_file:
        raise ValueError(f"Could not find MNIST IDX files. Found files: {[f.name for f in files]}")

    def read_idx_images(filepath):
        with open(filepath, 'rb') as f:
            magic = struct.unpack('>I', f.read(4))[0]
            num_images = struct.unpack('>I', f.read(4))[0]
            rows = struct.unpack('>I', f.read(4))[0]
            cols = struct.unpack('>I', f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_images, 1, rows, cols).astype(np.float32) / 255.0
            return data, rows, cols

    def read_idx_labels(filepath):
        with open(filepath, 'rb') as f:
            magic = struct.unpack('>I', f.read(4))[0]
            num_labels = struct.unpack('>I', f.read(4))[0]
            labels = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32)
            return labels

    def save_tensor(path, data):
        TENSOR_MAGIC = 0x54454E53
        with open(path, 'wb') as f:
            f.write(struct.pack('I', TENSOR_MAGIC))
            f.write(struct.pack('I', len(data.shape)))
            for dim in data.shape:
                f.write(struct.pack('Q', dim))
            data = np.ascontiguousarray(data, dtype=np.float32)
            f.write(data.tobytes())

    train_images, rows, cols = read_idx_images(train_images_file)
    train_labels = read_idx_labels(train_labels_file)

    if test_images_file and test_labels_file:
        test_images, _, _ = read_idx_images(test_images_file)
        test_labels = read_idx_labels(test_labels_file)
    else:
        split_idx = int(len(train_images) * 0.8)
        indices = np.random.permutation(len(train_images))
        test_images = train_images[indices[split_idx:]]
        test_labels = train_labels[indices[split_idx:]]
        train_images = train_images[indices[:split_idx]]
        train_labels = train_labels[indices[:split_idx]]

    mean = [float(train_images.mean())]
    std = [float(max(train_images.std(), 1e-7))]

    train_images = (train_images - mean[0]) / std[0]
    test_images = (test_images - mean[0]) / std[0]

    save_tensor(output_dir / "train_images.bin", train_images)
    save_tensor(output_dir / "train_labels.bin", train_labels)
    save_tensor(output_dir / "test_images.bin", test_images)
    save_tensor(output_dir / "test_labels.bin", test_labels)

    config = {
        "target_size": [rows, cols],
        "channels": 1,
        "mean": mean,
        "std": std,
        "num_classes": metadata.num_classes,
        "class_names": metadata.class_names,
        "train_samples": len(train_images),
        "test_samples": len(test_images),
        "input_shape": [1, rows, cols]
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("MNIST: Processed %d train, %d test images", len(train_images), len(test_images))
    return config
