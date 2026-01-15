#!/usr/bin/env python3
"""
Whitematter Model Server v0.4.0
"""

import argparse
import json
import shutil
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import io
import uvicorn

from dataset_manager import DatasetManager, DataType
from preprocessing import ImageProcessor
from codegen import CodeGenerator, compile_training_code
from codegen.compiler import run_training
from llm.service import get_llm_service

try:
    import whitematter as wm
except ImportError:
    print("Error: whitematter module not found. Build with: pip install -e .")
    exit(1)

MODELS_DIR = Path("models")
DATA_DIR = Path("data")

DATASETS = {
    "cifar10": {
        "name": "CIFAR-10",
        "description": "60,000 32x32 color images in 10 classes",
        "input_shape": [3, 32, 32],
        "num_classes": 10,
        "classes": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    },
    "mnist": {
        "name": "MNIST",
        "description": "70,000 28x28 grayscale handwritten digits",
        "input_shape": [1, 28, 28],
        "num_classes": 10,
        "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    }
}

CIFAR10_MEAN, CIFAR10_STD = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
MNIST_MEAN, MNIST_STD = [0.1307], [0.3081]

LAYER_TYPES = {
    "conv2d": {"name": "Conv2d", "params": ["in_channels", "out_channels", "kernel_size", "stride", "padding"]},
    "batchnorm2d": {"name": "BatchNorm2d", "params": ["num_features"]},
    "relu": {"name": "ReLU", "params": []},
    "leakyrelu": {"name": "LeakyReLU", "params": ["negative_slope"]},
    "maxpool2d": {"name": "MaxPool2d", "params": ["kernel_size"]},
    "avgpool2d": {"name": "AvgPool2d", "params": ["kernel_size"]},
    "dropout": {"name": "Dropout", "params": ["p"]},
    "flatten": {"name": "Flatten", "params": []},
    "linear": {"name": "Linear", "params": ["in_features", "out_features"]}
}

OPTIMIZERS = {
    "sgd": {"name": "SGD", "params": {"learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0}},
    "adam": {"name": "Adam", "params": {"learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "weight_decay": 0.0}}
}

SCHEDULERS = {
    "none": {"name": "None", "params": {}},
    "step": {"name": "StepLR", "params": {"step_size": 10, "gamma": 0.1}},
    "cosine": {"name": "CosineAnnealing", "params": {"eta_min": 0.0}},
    "exponential": {"name": "ExponentialLR", "params": {"gamma": 0.95}}
}

AUGMENTATIONS = {
    "horizontal_flip": {"name": "Horizontal Flip", "params": {"p": 0.5}},
    "random_crop": {"name": "Random Crop", "params": {"padding": 4}},
    "color_jitter": {"name": "Color Jitter", "params": {"brightness": 0.2, "contrast": 0.2}},
    "normalize": {"name": "Normalize", "params": {}}
}

PRESET_ARCHITECTURES = {
    "simple_cnn_cifar10": {
        "name": "Simple CNN (CIFAR-10)", "dataset": "cifar10",
        "layers": [
            {"type": "conv2d", "params": {"in_channels": 3, "out_channels": 32, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 32}}, {"type": "relu", "params": {}},
            {"type": "maxpool2d", "params": {"kernel_size": 2}},
            {"type": "conv2d", "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 64}}, {"type": "relu", "params": {}},
            {"type": "maxpool2d", "params": {"kernel_size": 2}}, {"type": "flatten", "params": {}},
            {"type": "linear", "params": {"in_features": 4096, "out_features": 256}}, {"type": "relu", "params": {}},
            {"type": "dropout", "params": {"p": 0.5}},
            {"type": "linear", "params": {"in_features": 256, "out_features": 10}}
        ]
    },
    "vgg_cifar10": {
        "name": "VGG-style CNN (CIFAR-10)", "dataset": "cifar10",
        "layers": [
            {"type": "conv2d", "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 64}}, {"type": "relu", "params": {}},
            {"type": "conv2d", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 64}}, {"type": "relu", "params": {}},
            {"type": "maxpool2d", "params": {"kernel_size": 2}},
            {"type": "conv2d", "params": {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 128}}, {"type": "relu", "params": {}},
            {"type": "conv2d", "params": {"in_channels": 128, "out_channels": 128, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 128}}, {"type": "relu", "params": {}},
            {"type": "maxpool2d", "params": {"kernel_size": 2}},
            {"type": "conv2d", "params": {"in_channels": 128, "out_channels": 256, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 256}}, {"type": "relu", "params": {}},
            {"type": "conv2d", "params": {"in_channels": 256, "out_channels": 256, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 256}}, {"type": "relu", "params": {}},
            {"type": "maxpool2d", "params": {"kernel_size": 2}}, {"type": "flatten", "params": {}},
            {"type": "linear", "params": {"in_features": 4096, "out_features": 512}}, {"type": "relu", "params": {}},
            {"type": "dropout", "params": {"p": 0.5}},
            {"type": "linear", "params": {"in_features": 512, "out_features": 10}}
        ]
    },
    "simple_cnn_mnist": {
        "name": "Simple CNN (MNIST)", "dataset": "mnist",
        "layers": [
            {"type": "conv2d", "params": {"in_channels": 1, "out_channels": 16, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 16}}, {"type": "relu", "params": {}},
            {"type": "maxpool2d", "params": {"kernel_size": 2}},
            {"type": "conv2d", "params": {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 32}}, {"type": "relu", "params": {}},
            {"type": "maxpool2d", "params": {"kernel_size": 2}}, {"type": "flatten", "params": {}},
            {"type": "linear", "params": {"in_features": 1568, "out_features": 128}}, {"type": "relu", "params": {}},
            {"type": "linear", "params": {"in_features": 128, "out_features": 10}}
        ]
    }
}

# Pydantic Models
class LayerConfig(BaseModel):
    type: str
    params: Dict[str, Any] = {}

class OptimizerConfig(BaseModel):
    type: str = "sgd"
    params: Dict[str, float] = {}

class SchedulerConfig(BaseModel):
    type: str = "none"
    params: Dict[str, Any] = {}

class AugmentationConfig(BaseModel):
    type: str
    params: Dict[str, Any] = {}

class TrainRequest(BaseModel):
    dataset: str
    name: Optional[str] = None
    preset: Optional[str] = None
    layers: Optional[List[LayerConfig]] = None
    epochs: int = 10
    batch_size: int = 128
    optimizer: OptimizerConfig = Field(default_factory=lambda: OptimizerConfig(type="sgd", params={"learning_rate": 0.01, "momentum": 0.9}))
    scheduler: SchedulerConfig = Field(default_factory=lambda: SchedulerConfig(type="none"))
    augmentations: List[AugmentationConfig] = []

class TrainStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ModelMetadata(BaseModel):
    id: str
    name: str
    dataset: str
    architecture: str
    created_at: str
    epochs_trained: int
    best_accuracy: float
    status: TrainStatus
    training_history: list = []
    config: Dict[str, Any] = {}

loaded_models: dict = {}
training_jobs: dict = {}
dataset_manager = DatasetManager()
code_generator = CodeGenerator()
llm_service = get_llm_service()

GENERATED_DIR = Path("generated")

app = FastAPI(title="Whitematter Model Server", version="0.4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def ensure_dirs():
    MODELS_DIR.mkdir(exist_ok=True)
    Path("uploads").mkdir(exist_ok=True)
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

def get_loaded_model(model_id: str) -> wm.Model:
    if model_id in loaded_models:
        return loaded_models[model_id]
    metadata = load_model_metadata(model_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    model_path = get_model_path(model_id)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model weights not found: {model_id}")

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

def run_training(job_id: str, request: TrainRequest, metadata: ModelMetadata):
    import subprocess
    try:
        training_jobs[job_id]["status"] = TrainStatus.RUNNING
        metadata.status = TrainStatus.RUNNING
        save_model_metadata(metadata)

        if request.dataset == "cifar10":
            cmd, src = ["./cnn_cifar10", str(DATA_DIR)], Path("cnn_cifar10.bin")
        elif request.dataset == "mnist":
            cmd, src = ["./cnn_mnist", str(DATA_DIR)], Path("cnn_mnist.bin")
        else:
            raise ValueError(f"Unsupported dataset: {request.dataset}")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        training_jobs[job_id]["process"] = process

        for line in process.stdout:
            line = line.strip()
            training_jobs[job_id]["output"].append(line)

            if "Epoch" in line and "Loss:" in line:
                try:
                    parts = line.split("|")
                    epoch = int(parts[0].split()[1])
                    loss = float(parts[1].split(":")[1].strip())
                    acc = 0.0
                    for p in parts[2:]:
                        if "Test Acc:" in p or "Acc:" in p:
                            acc = float(p.split(":")[1].strip().rstrip('%'))
                            break

                    training_jobs[job_id].update({"epoch": epoch, "loss": loss, "accuracy": acc, "message": f"Epoch {epoch}: {acc:.2f}%"})
                    metadata.epochs_trained = epoch
                    metadata.best_accuracy = max(metadata.best_accuracy, acc)
                    metadata.training_history.append({"epoch": epoch, "loss": loss, "accuracy": acc})
                    save_model_metadata(metadata)
                except:
                    pass

            if training_jobs[job_id].get("cancelled"):
                process.terminate()
                break

        process.wait()

        if training_jobs[job_id].get("cancelled"):
            training_jobs[job_id]["status"] = metadata.status = TrainStatus.CANCELLED
        elif process.returncode == 0 and src.exists():
            shutil.copy(src, get_model_path(metadata.id))
            training_jobs[job_id]["status"] = metadata.status = TrainStatus.COMPLETED
            training_jobs[job_id]["message"] = f"Complete! Best: {metadata.best_accuracy:.2f}%"
        else:
            training_jobs[job_id]["status"] = metadata.status = TrainStatus.FAILED
        save_model_metadata(metadata)
    except Exception as e:
        training_jobs[job_id]["status"] = metadata.status = TrainStatus.FAILED
        training_jobs[job_id]["message"] = str(e)
        save_model_metadata(metadata)

# API Endpoints
@app.get("/")
async def root():
    return {"name": "Whitematter Model Server", "version": "0.4.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# ============= Dataset Upload Endpoints =============

@app.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None)
):
    """Upload a ZIP file containing labeled data (folder per class)."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    # Create dataset entry
    dataset_id = dataset_manager.create_dataset(name or file.filename.replace('.zip', ''))

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Extract and analyze
        metadata = dataset_manager.extract_zip(dataset_id, tmp_path)

        # If it's an image dataset, process it
        if metadata.data_type == DataType.IMAGE:
            processor = ImageProcessor(
                target_size=(metadata.input_shape[1], metadata.input_shape[2]),
                channels=metadata.input_shape[0]
            )
            raw_dir = dataset_manager.uploads_dir / dataset_id / "raw"
            processed_dir = dataset_manager.uploads_dir / dataset_id / "processed"
            config = processor.process_dataset(
                raw_dir, processed_dir, metadata.class_names
            )
            # Update metadata with processing info
            metadata.input_shape = config["input_shape"]
            metadata.status = "processed"
            dataset_manager._save_metadata(dataset_id, metadata)

        return {
            "dataset_id": dataset_id,
            "name": metadata.name,
            "data_type": metadata.data_type.value,
            "num_classes": metadata.num_classes,
            "class_names": metadata.class_names,
            "total_samples": metadata.total_samples,
            "input_shape": metadata.input_shape,
            "status": metadata.status
        }
    except Exception as e:
        # Clean up on error
        dataset_manager.delete_dataset(dataset_id)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

@app.get("/datasets")
async def list_uploaded_datasets():
    """List all uploaded datasets."""
    datasets = dataset_manager.list_datasets()
    return {"datasets": [d.to_dict() for d in datasets]}

@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset metadata."""
    metadata = dataset_manager.get_metadata(dataset_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return metadata.to_dict()

@app.get("/datasets/{dataset_id}/preview")
async def get_dataset_preview(dataset_id: str):
    """Get a preview of the dataset."""
    preview = dataset_manager.get_preview(dataset_id)
    if not preview:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return preview

@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    if not dataset_manager.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"message": f"Dataset {dataset_id} deleted"}

# ============= Architecture Design Endpoints =============

class DesignRequest(BaseModel):
    dataset_id: str
    prompt: str

class RefineRequest(BaseModel):
    architecture: Dict[str, Any]
    feedback: str

class CustomTrainRequest(BaseModel):
    dataset_id: str
    architecture: Dict[str, Any]
    name: Optional[str] = None

@app.post("/design/suggest")
async def suggest_architecture(request: DesignRequest):
    """Get LLM-suggested architecture for a dataset."""
    metadata = dataset_manager.get_metadata(request.dataset_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        result = llm_service.suggest_architecture(
            dataset_info=metadata.to_dict(),
            user_prompt=request.prompt
        )

        # Validate the suggested architecture
        validation = code_generator.validate_architecture(result["architecture"])

        return {
            "architecture": result["architecture"],
            "explanation": result["explanation"],
            "validation": validation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

@app.post("/design/validate")
async def validate_architecture(architecture: Dict[str, Any]):
    """Validate an architecture specification."""
    validation = code_generator.validate_architecture(architecture)
    return validation

@app.post("/design/refine")
async def refine_architecture(request: RefineRequest):
    """Refine an architecture based on feedback."""
    try:
        result = llm_service.refine_architecture(
            current_architecture=request.architecture,
            feedback=request.feedback
        )

        validation = code_generator.validate_architecture(result["architecture"])

        return {
            "architecture": result["architecture"],
            "explanation": result["explanation"],
            "validation": validation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

# ============= Custom Training Endpoints =============

def run_custom_training(job_id: str, request: CustomTrainRequest, metadata: ModelMetadata):
    """Background function to run custom model training."""
    try:
        training_jobs[job_id]["status"] = TrainStatus.RUNNING
        metadata.status = TrainStatus.RUNNING
        save_model_metadata(metadata)

        # Get dataset config
        dataset_meta = dataset_manager.get_metadata(request.dataset_id)
        processed_dir = dataset_manager.uploads_dir / request.dataset_id / "processed"
        config_path = processed_dir / "config.json"

        if not config_path.exists():
            raise FileNotFoundError("Dataset not processed")

        with open(config_path) as f:
            dataset_config = json.load(f)

        # Generate training code
        job_dir = GENERATED_DIR / job_id
        code_generator.generate(
            architecture=request.architecture,
            dataset_config=dataset_config,
            output_dir=job_dir
        )

        # Compile
        training_jobs[job_id]["message"] = "Compiling..."
        success, msg = compile_training_code(job_dir)
        if not success:
            raise RuntimeError(f"Compilation failed: {msg}")

        # Run training
        training_jobs[job_id]["message"] = "Training..."
        output_model = job_dir / "model.bin"

        process = run_training(
            generated_dir=job_dir,
            data_dir=processed_dir,
            output_model=output_model
        )
        training_jobs[job_id]["process"] = process

        # Stream output
        for line in process.stdout:
            line = line.strip()
            training_jobs[job_id]["output"].append(line)

            if "Epoch" in line and "Loss:" in line:
                try:
                    parts = line.split("|")
                    epoch = int(parts[0].split()[1])
                    loss = float(parts[1].split(":")[1].strip())
                    acc = 0.0
                    for p in parts[2:]:
                        if "Test Acc:" in p or "Acc:" in p:
                            acc = float(p.split(":")[1].strip().rstrip('%'))
                            break

                    training_jobs[job_id].update({
                        "epoch": epoch,
                        "loss": loss,
                        "accuracy": acc,
                        "message": f"Epoch {epoch}: {acc:.2f}%"
                    })
                    metadata.epochs_trained = epoch
                    metadata.best_accuracy = max(metadata.best_accuracy, acc)
                    metadata.training_history.append({"epoch": epoch, "loss": loss, "accuracy": acc})
                    save_model_metadata(metadata)
                except:
                    pass

            if training_jobs[job_id].get("cancelled"):
                process.terminate()
                break

        process.wait()

        if training_jobs[job_id].get("cancelled"):
            training_jobs[job_id]["status"] = metadata.status = TrainStatus.CANCELLED
        elif process.returncode == 0 and output_model.exists():
            shutil.copy(output_model, get_model_path(metadata.id))
            training_jobs[job_id]["status"] = metadata.status = TrainStatus.COMPLETED
            training_jobs[job_id]["message"] = f"Complete! Best: {metadata.best_accuracy:.2f}%"
        else:
            training_jobs[job_id]["status"] = metadata.status = TrainStatus.FAILED

        save_model_metadata(metadata)

    except Exception as e:
        training_jobs[job_id]["status"] = metadata.status = TrainStatus.FAILED
        training_jobs[job_id]["message"] = str(e)
        save_model_metadata(metadata)

@app.post("/train/custom")
async def start_custom_training(request: CustomTrainRequest):
    """Start training with a custom architecture on an uploaded dataset."""
    # Validate dataset exists
    dataset_meta = dataset_manager.get_metadata(request.dataset_id)
    if not dataset_meta:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset_meta.status != "processed":
        raise HTTPException(status_code=400, detail="Dataset not processed yet")

    # Validate architecture
    validation = code_generator.validate_architecture(request.architecture)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=f"Invalid architecture: {validation['errors']}")

    # Create job
    model_id = job_id = str(uuid.uuid4())[:8]
    model_name = request.name or f"custom_{dataset_meta.name}_{model_id}"

    metadata = ModelMetadata(
        id=model_id,
        name=model_name,
        dataset=f"custom:{request.dataset_id}",
        architecture="custom",
        created_at=datetime.now().isoformat(),
        epochs_trained=0,
        best_accuracy=0.0,
        status=TrainStatus.PENDING,
        training_history=[],
        config={
            "dataset_id": request.dataset_id,
            "architecture": request.architecture
        }
    )
    save_model_metadata(metadata)

    training_jobs[job_id] = {
        "id": job_id,
        "model_id": model_id,
        "status": TrainStatus.PENDING,
        "epoch": 0,
        "total_epochs": request.architecture.get("training", {}).get("epochs", 10),
        "loss": 0.0,
        "accuracy": 0.0,
        "message": "Starting...",
        "output": [],
        "cancelled": False
    }

    threading.Thread(target=run_custom_training, args=(job_id, request, metadata)).start()

    return {
        "job_id": job_id,
        "model_id": model_id,
        "message": "Custom training started",
        "architecture": request.architecture
    }

# ============= Built-in Dataset Endpoints =============

@app.get("/config/datasets")
async def list_datasets():
    result = []
    for k, v in DATASETS.items():
        available = (DATA_DIR / ("data_batch_1.bin" if k == "cifar10" else "train-images-idx3-ubyte")).exists()
        result.append({"id": k, "available": available, **v})
    return {"datasets": result}

@app.get("/config/layers")
async def list_layer_types():
    return {"layers": [{"id": k, **v} for k, v in LAYER_TYPES.items()]}

@app.get("/config/optimizers")
async def list_optimizers():
    return {"optimizers": [{"id": k, **v} for k, v in OPTIMIZERS.items()]}

@app.get("/config/schedulers")
async def list_schedulers():
    return {"schedulers": [{"id": k, **v} for k, v in SCHEDULERS.items()]}

@app.get("/config/augmentations")
async def list_augmentations():
    return {"augmentations": [{"id": k, **v} for k, v in AUGMENTATIONS.items()]}

@app.get("/config/presets")
async def list_presets():
    return {"presets": [{"id": k, "name": v["name"], "dataset": v["dataset"], "num_layers": len(v["layers"])} for k, v in PRESET_ARCHITECTURES.items()]}

@app.get("/config/presets/{preset_id}")
async def get_preset(preset_id: str):
    if preset_id not in PRESET_ARCHITECTURES:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"preset": {"id": preset_id, **PRESET_ARCHITECTURES[preset_id]}}

@app.get("/models")
async def list_models():
    return {"models": [m.model_dump() for m in list_all_models()]}

@app.get("/models/{model_id}")
async def get_model(model_id: str):
    if not (m := load_model_metadata(model_id)):
        raise HTTPException(status_code=404, detail="Model not found")
    return m.model_dump()

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    if not load_model_metadata(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    loaded_models.pop(model_id, None)
    for p in [get_model_path(model_id), get_metadata_path(model_id)]:
        p.unlink(missing_ok=True)
    return {"message": f"Model {model_id} deleted"}

@app.post("/train")
async def start_training(request: TrainRequest):
    if request.dataset not in DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {request.dataset}")

    if request.preset:
        if request.preset not in PRESET_ARCHITECTURES:
            raise HTTPException(status_code=400, detail=f"Unknown preset: {request.preset}")
        preset = PRESET_ARCHITECTURES[request.preset]
        if preset["dataset"] != request.dataset:
            raise HTTPException(status_code=400, detail=f"Preset {request.preset} is for {preset['dataset']}")
        architecture, layers = f"preset:{request.preset}", preset["layers"]
    elif request.layers:
        architecture, layers = "custom", [l.model_dump() for l in request.layers]
    else:
        raise HTTPException(status_code=400, detail="Either preset or layers must be provided")

    model_id = job_id = str(uuid.uuid4())[:8]
    config = {"dataset": request.dataset, "architecture": architecture, "layers": layers, "epochs": request.epochs,
              "batch_size": request.batch_size, "optimizer": request.optimizer.model_dump(),
              "scheduler": request.scheduler.model_dump(), "augmentations": [a.model_dump() for a in request.augmentations]}

    metadata = ModelMetadata(id=model_id, name=request.name or f"{request.dataset}_{model_id}", dataset=request.dataset,
                             architecture=architecture, created_at=datetime.now().isoformat(), epochs_trained=0,
                             best_accuracy=0.0, status=TrainStatus.PENDING, training_history=[], config=config)
    save_model_metadata(metadata)

    training_jobs[job_id] = {"id": job_id, "model_id": model_id, "status": TrainStatus.PENDING, "epoch": 0,
                            "total_epochs": request.epochs, "loss": 0.0, "accuracy": 0.0, "message": "Starting...",
                            "output": [], "cancelled": False}

    threading.Thread(target=run_training, args=(job_id, request, metadata)).start()
    return {"job_id": job_id, "model_id": model_id, "message": "Training started", "config": config}

@app.get("/train/{job_id}")
async def get_training_status(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    j = training_jobs[job_id]
    return {k: j[k] for k in ["id", "model_id", "status", "epoch", "total_epochs", "loss", "accuracy", "message"]}

@app.delete("/train/{job_id}")
async def cancel_training(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    j = training_jobs[job_id]
    if j["status"] not in [TrainStatus.PENDING, TrainStatus.RUNNING]:
        raise HTTPException(status_code=400, detail="Job not running")
    j["cancelled"] = True
    if "process" in j:
        j["process"].terminate()
    return {"message": "Training cancelled"}

@app.post("/predict")
async def predict(model_id: str, file: UploadFile = File(...)):
    if not (metadata := load_model_metadata(model_id)):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if metadata.status != TrainStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Model not ready")

    # Check if this is a custom model
    if metadata.dataset.startswith("custom:"):
        return await predict_custom_model(model_id, metadata, file)

    model = get_loaded_model(model_id)
    image = Image.open(io.BytesIO(await file.read()))
    input_tensor = preprocess_image(image, metadata.dataset)

    predicted_class = model.predict_class(input_tensor)
    probs = model.predict_proba(input_tensor).flatten().tolist()
    classes = DATASETS[metadata.dataset]["classes"]

    return {"model_id": model_id, "model_name": metadata.name, "predicted_class": predicted_class,
            "class_name": classes[predicted_class], "confidence": float(probs[predicted_class]),
            "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))}}

async def predict_custom_model(model_id: str, metadata: ModelMetadata, file: UploadFile):
    """Handle prediction for custom-trained models."""
    # Get the dataset config for preprocessing params
    dataset_id = metadata.dataset.replace("custom:", "")
    dataset_meta = dataset_manager.get_metadata(dataset_id)
    if not dataset_meta:
        raise HTTPException(status_code=400, detail="Original dataset not found")

    processed_dir = dataset_manager.uploads_dir / dataset_id / "processed"
    config_path = processed_dir / "config.json"

    if not config_path.exists():
        raise HTTPException(status_code=400, detail="Dataset config not found")

    with open(config_path) as f:
        config = json.load(f)

    # Load and preprocess image
    image = Image.open(io.BytesIO(await file.read()))
    target_size = (config["target_size"][0], config["target_size"][1])
    channels = config["channels"]

    if channels == 3:
        image = image.convert('RGB')
    else:
        image = image.convert('L')

    image = image.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0

    if channels == 3:
        arr = arr.transpose(2, 0, 1)
    else:
        arr = arr[np.newaxis, ...]

    # Apply normalization
    mean = np.array(config["mean"]).reshape(-1, 1, 1)
    std = np.array(config["std"]).reshape(-1, 1, 1)
    arr = (arr - mean) / std

    input_tensor = np.ascontiguousarray(arr, dtype=np.float32)

    # Load model if not cached
    model = get_loaded_model(model_id)
    predicted_class = model.predict_class(input_tensor)
    probs = model.predict_proba(input_tensor).flatten().tolist()
    classes = config["class_names"]

    return {
        "model_id": model_id,
        "model_name": metadata.name,
        "predicted_class": predicted_class,
        "class_name": classes[predicted_class],
        "confidence": float(probs[predicted_class]),
        "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))}
    }

# Convenience API endpoint
@app.post("/api/{model_id}/predict")
async def api_predict(model_id: str, file: UploadFile = File(...)):
    """Convenience endpoint for model prediction: /api/{model_id}/predict"""
    return await predict(model_id, file)

@app.get("/api/{model_id}/info")
async def api_model_info(model_id: str):
    """Get model information."""
    metadata = load_model_metadata(model_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Model not found")

    result = {
        "model_id": model_id,
        "name": metadata.name,
        "dataset": metadata.dataset,
        "architecture": metadata.architecture,
        "status": metadata.status,
        "accuracy": metadata.best_accuracy,
        "epochs_trained": metadata.epochs_trained
    }

    # Add class names
    if metadata.dataset.startswith("custom:"):
        dataset_id = metadata.dataset.replace("custom:", "")
        processed_dir = dataset_manager.uploads_dir / dataset_id / "processed"
        config_path = processed_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                result["classes"] = config["class_names"]
                result["input_shape"] = config["input_shape"]
    elif metadata.dataset in DATASETS:
        result["classes"] = DATASETS[metadata.dataset]["classes"]
        result["input_shape"] = DATASETS[metadata.dataset]["input_shape"]

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    global MODELS_DIR
    MODELS_DIR = Path(args.models_dir)
    ensure_dirs()

    print(f"Whitematter Model Server v0.3.0 at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
