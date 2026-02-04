#!/usr/bin/env python3
"""
Whitematter Model Server v0.5.0
Now with database and blob storage backend.
"""

import argparse
import json
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import io
import uvicorn

# Initialize database on import
from db import init_db, get_db, get_data_dir, get_blob_store
from db import Dataset, Model, JobStatus, ModelStatus, DatasetStatus
from services import DatasetService, ModelService, TrainingService
from codegen import CodeGenerator, compile_training_code
from llm.service import get_llm_service

try:
    import whitematter as wm
except ImportError:
    print("Warning: whitematter module not found. Inference will use subprocess.")
    wm = None

# Initialize database
init_db()

# Initialize services
dataset_service = DatasetService()
model_service = ModelService()
training_service = TrainingService()
code_generator = CodeGenerator()
llm_service = get_llm_service()
blob_store = get_blob_store()

# Built-in datasets configuration
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

# FastAPI app
app = FastAPI(title="Whitematter Model Server", version="0.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Pydantic models
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


# API Endpoints

@app.get("/")
async def root():
    return {"name": "Whitematter Model Server", "version": "0.5.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ============= Dataset Endpoints =============

@app.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None)
):
    """Upload a ZIP file containing labeled data."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    dataset_name = name or file.filename.replace('.zip', '')

    try:
        # Create dataset entry
        dataset = dataset_service.create_dataset(dataset_name)
        dataset_id = dataset['id']

        # Read file content
        content = await file.read()

        # Process the upload
        result = dataset_service.upload_zip(dataset_id, content, file.filename)

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/datasets")
async def list_datasets():
    """List all uploaded datasets."""
    datasets = dataset_service.list_datasets()
    return {"datasets": datasets}


@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset metadata."""
    dataset = dataset_service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@app.get("/datasets/{dataset_id}/preview")
async def get_dataset_preview(dataset_id: str):
    """Get a preview of the dataset."""
    preview = dataset_service.get_preview(dataset_id)
    if not preview:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return preview


@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    if not dataset_service.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"message": f"Dataset {dataset_id} deleted"}


# ============= Model Endpoints =============

@app.get("/models")
async def list_models():
    """List all models."""
    models = model_service.list_models()
    return {"models": models}


@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model details."""
    model = model_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Include training history
    model['training_history'] = model_service.get_training_history(model_id)
    return model


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model."""
    if not model_service.delete_model(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    return {"message": f"Model {model_id} deleted"}


# ============= Architecture Design Endpoints =============

@app.post("/design/suggest")
async def suggest_architecture(request: DesignRequest):
    """Get LLM-suggested architecture for a dataset."""
    dataset = dataset_service.get_dataset(request.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        result = llm_service.suggest_architecture(
            dataset_info=dataset,
            user_prompt=request.prompt
        )

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


# ============= Training Endpoints =============

@app.post("/train/custom")
async def start_custom_training(request: CustomTrainRequest):
    """Start training with a custom architecture on an uploaded dataset."""
    # Validate dataset exists
    dataset = dataset_service.get_dataset(request.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset['status'] != DatasetStatus.READY.value:
        raise HTTPException(status_code=400, detail="Dataset not processed yet")

    # Validate architecture
    validation = code_generator.validate_architecture(request.architecture)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=f"Invalid architecture: {validation['errors']}")

    # Create model
    model_name = request.name or f"custom_{dataset['name']}_{uuid.uuid4().hex[:8]}"
    model = model_service.create_model(
        name=model_name,
        dataset_id=request.dataset_id,
        dataset_name=dataset['name'],
        architecture_name="custom",
        architecture_config=request.architecture,
        training_config={
            "epochs": request.architecture.get("training", {}).get("epochs", 10),
            "batch_size": request.architecture.get("training", {}).get("batch_size", 32)
        }
    )

    # Start training job
    epochs = request.architecture.get("training", {}).get("epochs", 10)
    job = training_service.start_training(
        model_id=model['id'],
        total_epochs=epochs
    )

    return {
        "job_id": job['id'],
        "model_id": model['id'],
        "message": "Custom training started",
        "architecture": request.architecture
    }


@app.get("/train/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status."""
    job = training_service.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@app.delete("/train/{job_id}")
async def cancel_training(job_id: str):
    """Cancel a training job."""
    if not training_service.cancel_job(job_id):
        raise HTTPException(status_code=400, detail="Cannot cancel job")
    return {"message": "Training cancelled"}


@app.get("/train")
async def list_training_jobs():
    """List all active training jobs."""
    jobs = training_service.list_active_jobs()
    return {"jobs": jobs}


# ============= Prediction Endpoints =============

@app.post("/predict")
async def predict(model_id: str, file: UploadFile = File(...)):
    """Make a prediction using a trained model."""
    model = model_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    if model['status'] != ModelStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Model not ready")

    # Load image
    image = Image.open(io.BytesIO(await file.read()))

    # Get dataset config
    dataset_id = model.get('dataset_id')
    if dataset_id:
        dataset = dataset_service.get_dataset(dataset_id)
        if dataset:
            return await _predict_with_dataset_config(model, dataset, image)

    # Fallback to built-in dataset
    dataset_name = model.get('dataset', '')
    if dataset_name in DATASETS:
        return await _predict_builtin(model, dataset_name, image)

    raise HTTPException(status_code=400, detail="Cannot determine model configuration")


def _detect_architecture_type(model: Dict, config: Dict) -> Optional[str]:
    """
    Detect if the model architecture matches a known whitematter built-in type.

    Returns architecture type string ('vgg', 'simple', 'mnist') or None if unknown.
    """
    arch_config = model.get('architecture_config', {})
    layers = arch_config.get('layers', [])

    if not layers:
        return None

    # Get input shape from config
    channels = config.get("channels", 3)
    target_size = config.get("target_size", [32, 32])
    num_classes = config.get("num_classes", 10)

    # Count layer types
    conv_count = sum(1 for l in layers if l.get('type', '').lower() == 'conv2d')
    linear_count = sum(1 for l in layers if l.get('type', '').lower() == 'linear')

    # Check for MNIST architecture: 1 channel input, 2 conv layers
    if channels == 1 and target_size == [28, 28]:
        if conv_count == 2 and num_classes == 10:
            return "mnist"

    # Check for simple CIFAR architecture: 2 conv layers, 2 linear layers
    if channels == 3 and target_size == [32, 32]:
        if conv_count == 2 and linear_count == 2 and num_classes == 10:
            return "simple"
        # Check for VGG-style: 6 conv layers
        if conv_count == 6 and num_classes == 10:
            return "vgg"

    return None


def _run_direct_inference(weights_path: str, input_tensor: np.ndarray, arch_type: str) -> Dict:
    """
    Run inference directly using whitematter Python bindings.

    Args:
        weights_path: Path to model weights file
        input_tensor: Preprocessed input tensor [1, C, H, W]
        arch_type: Architecture type ('vgg', 'simple', 'mnist')

    Returns:
        Dict with predicted_class and probabilities
    """
    wm_model = wm.Model()
    wm_model.load(str(weights_path), arch_type)

    # Run inference - input should be [C, H, W] for single sample
    input_single = input_tensor[0]  # Remove batch dimension
    predicted_class = wm_model.predict_class(input_single)
    probs = wm_model.predict_proba(input_single).flatten().tolist()

    return {
        "predicted_class": predicted_class,
        "probabilities": probs
    }


def _run_subprocess_inference(infer_exe: Path, weights_path: str, input_tensor: np.ndarray) -> Dict:
    """
    Run inference using compiled subprocess executable (fallback method).

    Args:
        infer_exe: Path to compiled inference executable
        weights_path: Path to model weights file
        input_tensor: Preprocessed input tensor [1, C, H, W]

    Returns:
        Dict with predicted_class and probabilities
    """
    import struct
    import subprocess

    TENSOR_MAGIC = 0x54454E53
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
        tmp.write(struct.pack('I', TENSOR_MAGIC))
        tmp.write(struct.pack('I', len(input_tensor.shape)))
        for dim in input_tensor.shape:
            tmp.write(struct.pack('Q', dim))
        tmp.write(input_tensor.tobytes())
        input_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [str(infer_exe), str(weights_path), str(input_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Inference failed: {result.stderr}")

        # Parse output
        stdout_lines = result.stdout.strip().split('\n')
        json_line = next((line for line in stdout_lines if line.startswith('{')), None)
        if not json_line:
            raise HTTPException(status_code=500, detail="No output from inference")

        output = json.loads(json_line)
        return {
            "predicted_class": output["predicted_class"],
            "probabilities": output["probabilities"]
        }
    finally:
        input_path.unlink(missing_ok=True)


async def _predict_with_dataset_config(model: Dict, dataset: Dict, image: Image.Image) -> Dict:
    """Predict using custom dataset configuration."""
    # Get processed config from blob storage
    config_blob = blob_store.get(f"{dataset.get('processed_blob_prefix', '')}/config.json")
    if not config_blob:
        raise HTTPException(status_code=400, detail="Dataset config not found")

    config = json.loads(config_blob.decode())

    # Preprocess image
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

    # Normalize
    mean = np.array(config["mean"]).reshape(-1, 1, 1)
    std = np.array(config["std"]).reshape(-1, 1, 1)
    arr = (arr - mean) / std

    # Add batch dimension
    arr = arr[np.newaxis, ...]
    input_tensor = np.ascontiguousarray(arr, dtype=np.float32)

    # Get model weights path
    weights_path = model_service.get_weights_path(model['id'])
    if not weights_path:
        raise HTTPException(status_code=400, detail="Model weights not found")

    classes = config["class_names"]

    # Try direct inference with whitematter if available
    if wm is not None:
        arch_type = _detect_architecture_type(model, config)
        if arch_type is not None:
            try:
                output = _run_direct_inference(str(weights_path), input_tensor, arch_type)
                predicted_class = output["predicted_class"]
                probs = output["probabilities"]

                return {
                    "model_id": model['id'],
                    "model_name": model['name'],
                    "predicted_class": predicted_class,
                    "class_name": classes[predicted_class],
                    "confidence": float(probs[predicted_class]),
                    "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))}
                }
            except Exception as e:
                # Log the error and fall back to subprocess
                print(f"Direct inference failed, falling back to subprocess: {e}")

    # Fallback: use subprocess inference with compiled executable
    generated_dir = get_data_dir() / "generated" / model['id']
    infer_exe = generated_dir / "infer"

    if not infer_exe.exists():
        raise HTTPException(status_code=400, detail="Inference executable not found")

    output = _run_subprocess_inference(infer_exe, str(weights_path), input_tensor)
    predicted_class = output["predicted_class"]
    probs = output["probabilities"]

    return {
        "model_id": model['id'],
        "model_name": model['name'],
        "predicted_class": predicted_class,
        "class_name": classes[predicted_class],
        "confidence": float(probs[predicted_class]),
        "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))}
    }


async def _predict_builtin(model: Dict, dataset_name: str, image: Image.Image) -> Dict:
    """Predict using built-in dataset."""
    if wm is None:
        raise HTTPException(status_code=500, detail="whitematter module not available")

    info = DATASETS[dataset_name]
    c, h, w = info["input_shape"]
    image = image.resize((w, h), Image.Resampling.BILINEAR)

    if c == 1:
        image = image.convert('L')
        arr = np.array(image, dtype=np.float32).reshape(1, h, w) / 255.0
        mean, std = np.array(MNIST_MEAN).reshape(1, 1, 1), np.array(MNIST_STD).reshape(1, 1, 1)
    else:
        image = image.convert('RGB')
        arr = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mean, std = np.array(CIFAR10_MEAN).reshape(3, 1, 1), np.array(CIFAR10_STD).reshape(3, 1, 1)

    input_tensor = np.ascontiguousarray((arr - mean) / std, dtype=np.float32)

    # Load model
    weights_path = model_service.get_weights_path(model['id'])
    if not weights_path:
        raise HTTPException(status_code=400, detail="Model weights not found")

    wm_model = wm.Model()
    wm_model.load(str(weights_path), "simple")

    predicted_class = wm_model.predict_class(input_tensor)
    probs = wm_model.predict_proba(input_tensor).flatten().tolist()
    classes = info["classes"]

    return {
        "model_id": model['id'],
        "model_name": model['name'],
        "predicted_class": predicted_class,
        "class_name": classes[predicted_class],
        "confidence": float(probs[predicted_class]),
        "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))}
    }


@app.post("/api/{model_id}/predict")
async def api_predict(model_id: str, file: UploadFile = File(...)):
    """Convenience endpoint for model prediction."""
    return await predict(model_id, file)


@app.get("/api/{model_id}/info")
async def api_model_info(model_id: str):
    """Get model information."""
    model = model_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    result = {
        "model_id": model_id,
        "name": model['name'],
        "dataset": model.get('dataset', 'unknown'),
        "architecture": model.get('architecture', 'custom'),
        "status": model['status'],
        "accuracy": model.get('best_accuracy', 0),
        "epochs_trained": model.get('epochs_trained', 0)
    }

    # Add class names from dataset
    dataset_id = model.get('dataset_id')
    if dataset_id:
        dataset = dataset_service.get_dataset(dataset_id)
        if dataset:
            result["classes"] = dataset.get('class_names', [])
            result["input_shape"] = dataset.get('input_shape', [])

    return result


# ============= Config Endpoints =============

@app.get("/config/datasets")
async def list_builtin_datasets():
    """List built-in datasets."""
    return {"datasets": [{"id": k, **v} for k, v in DATASETS.items()]}


@app.get("/config/optimizers")
async def list_optimizers():
    """List available optimizers with their parameters."""
    optimizers = [
        {
            "name": "SGD",
            "description": "Stochastic Gradient Descent with optional momentum and weight decay",
            "parameters": {
                "lr": {"type": "float", "default": 0.01, "description": "Learning rate"},
                "momentum": {"type": "float", "default": 0.0, "description": "Momentum factor"},
                "weight_decay": {"type": "float", "default": 0.0, "description": "L2 regularization"},
                "nesterov": {"type": "bool", "default": False, "description": "Enable Nesterov momentum"}
            }
        },
        {
            "name": "Adam",
            "description": "Adaptive Moment Estimation optimizer",
            "parameters": {
                "lr": {"type": "float", "default": 0.001, "description": "Learning rate"},
                "betas": {"type": "tuple", "default": [0.9, 0.999], "description": "Coefficients for computing running averages"},
                "eps": {"type": "float", "default": 1e-8, "description": "Term added for numerical stability"},
                "weight_decay": {"type": "float", "default": 0.0, "description": "L2 regularization"}
            }
        },
        {
            "name": "AdamW",
            "description": "Adam with decoupled weight decay regularization",
            "parameters": {
                "lr": {"type": "float", "default": 0.001, "description": "Learning rate"},
                "betas": {"type": "tuple", "default": [0.9, 0.999], "description": "Coefficients for computing running averages"},
                "eps": {"type": "float", "default": 1e-8, "description": "Term added for numerical stability"},
                "weight_decay": {"type": "float", "default": 0.01, "description": "Decoupled weight decay"}
            }
        },
        {
            "name": "RMSprop",
            "description": "Root Mean Square Propagation optimizer",
            "parameters": {
                "lr": {"type": "float", "default": 0.01, "description": "Learning rate"},
                "alpha": {"type": "float", "default": 0.99, "description": "Smoothing constant"},
                "eps": {"type": "float", "default": 1e-8, "description": "Term added for numerical stability"},
                "weight_decay": {"type": "float", "default": 0.0, "description": "L2 regularization"},
                "momentum": {"type": "float", "default": 0.0, "description": "Momentum factor"}
            }
        }
    ]
    return {"optimizers": optimizers}


@app.get("/config/schedulers")
async def list_schedulers():
    """List available learning rate schedulers with their parameters."""
    schedulers = [
        {
            "name": "StepLR",
            "description": "Decays learning rate by gamma every step_size epochs",
            "parameters": {
                "step_size": {"type": "int", "required": True, "description": "Period of learning rate decay"},
                "gamma": {"type": "float", "default": 0.1, "description": "Multiplicative factor of learning rate decay"}
            }
        },
        {
            "name": "ExponentialLR",
            "description": "Decays learning rate by gamma every epoch",
            "parameters": {
                "gamma": {"type": "float", "required": True, "description": "Multiplicative factor of learning rate decay"}
            }
        },
        {
            "name": "CosineAnnealingLR",
            "description": "Anneals learning rate using a cosine schedule",
            "parameters": {
                "T_max": {"type": "int", "required": True, "description": "Maximum number of iterations"},
                "eta_min": {"type": "float", "default": 0.0, "description": "Minimum learning rate"}
            }
        },
        {
            "name": "ReduceLROnPlateau",
            "description": "Reduces learning rate when a metric has stopped improving",
            "parameters": {
                "mode": {"type": "str", "default": "min", "description": "One of 'min' or 'max'"},
                "factor": {"type": "float", "default": 0.1, "description": "Factor by which to reduce learning rate"},
                "patience": {"type": "int", "default": 10, "description": "Number of epochs with no improvement after which to reduce LR"},
                "threshold": {"type": "float", "default": 1e-4, "description": "Threshold for measuring improvement"},
                "min_lr": {"type": "float", "default": 0.0, "description": "Lower bound on the learning rate"}
            }
        }
    ]
    return {"schedulers": schedulers}


@app.get("/config/augmentations")
async def list_augmentations():
    """List available data augmentation transforms."""
    augmentations = [
        {
            "name": "random_flip",
            "description": "Randomly flip images horizontally or vertically",
            "parameters": {
                "horizontal": {"type": "bool", "default": True, "description": "Enable horizontal flipping"},
                "vertical": {"type": "bool", "default": False, "description": "Enable vertical flipping"},
                "p": {"type": "float", "default": 0.5, "description": "Probability of applying the transform"}
            }
        },
        {
            "name": "random_crop",
            "description": "Randomly crop a portion of the image and resize to original size",
            "parameters": {
                "size": {"type": "tuple", "required": True, "description": "Output size (height, width)"},
                "padding": {"type": "int", "default": 0, "description": "Padding on each border before cropping"},
                "scale": {"type": "tuple", "default": [0.08, 1.0], "description": "Range of size of the origin size cropped"},
                "ratio": {"type": "tuple", "default": [0.75, 1.33], "description": "Range of aspect ratio"}
            }
        },
        {
            "name": "pad",
            "description": "Pad the image on all sides",
            "parameters": {
                "padding": {"type": "int", "required": True, "description": "Padding on each side"},
                "fill": {"type": "int", "default": 0, "description": "Fill value for padding"},
                "padding_mode": {"type": "str", "default": "constant", "description": "Type of padding: constant, edge, reflect, symmetric"}
            }
        },
        {
            "name": "normalize",
            "description": "Normalize tensor with mean and standard deviation",
            "parameters": {
                "mean": {"type": "list", "required": True, "description": "Mean for each channel"},
                "std": {"type": "list", "required": True, "description": "Standard deviation for each channel"}
            }
        },
        {
            "name": "random_rotation",
            "description": "Randomly rotate the image",
            "parameters": {
                "degrees": {"type": "float", "required": True, "description": "Range of degrees to rotate (-degrees, +degrees)"},
                "fill": {"type": "int", "default": 0, "description": "Fill value for areas outside the rotated image"}
            }
        },
        {
            "name": "color_jitter",
            "description": "Randomly change brightness, contrast, saturation, and hue",
            "parameters": {
                "brightness": {"type": "float", "default": 0.0, "description": "Brightness jitter factor"},
                "contrast": {"type": "float", "default": 0.0, "description": "Contrast jitter factor"},
                "saturation": {"type": "float", "default": 0.0, "description": "Saturation jitter factor"},
                "hue": {"type": "float", "default": 0.0, "description": "Hue jitter factor"}
            }
        }
    ]
    return {"augmentations": augmentations}


@app.get("/config/layers")
async def list_layers():
    """List available neural network layers with their parameters."""
    layers = [
        {
            "name": "Linear",
            "category": "linear",
            "description": "Fully connected linear layer",
            "parameters": {
                "in_features": {"type": "int", "required": True, "description": "Size of input features"},
                "out_features": {"type": "int", "required": True, "description": "Size of output features"},
                "bias": {"type": "bool", "default": True, "description": "Include bias term"}
            }
        },
        {
            "name": "Conv2d",
            "category": "convolution",
            "description": "2D convolutional layer",
            "parameters": {
                "in_channels": {"type": "int", "required": True, "description": "Number of input channels"},
                "out_channels": {"type": "int", "required": True, "description": "Number of output channels"},
                "kernel_size": {"type": "int", "required": True, "description": "Size of the convolving kernel"},
                "stride": {"type": "int", "default": 1, "description": "Stride of the convolution"},
                "padding": {"type": "int", "default": 0, "description": "Zero-padding added to both sides"},
                "bias": {"type": "bool", "default": True, "description": "Include bias term"}
            }
        },
        {
            "name": "Conv1d",
            "category": "convolution",
            "description": "1D convolutional layer for sequence data",
            "parameters": {
                "in_channels": {"type": "int", "required": True, "description": "Number of input channels"},
                "out_channels": {"type": "int", "required": True, "description": "Number of output channels"},
                "kernel_size": {"type": "int", "required": True, "description": "Size of the convolving kernel"},
                "stride": {"type": "int", "default": 1, "description": "Stride of the convolution"},
                "padding": {"type": "int", "default": 0, "description": "Zero-padding added to both sides"}
            }
        },
        {
            "name": "ReLU",
            "category": "activation",
            "description": "Rectified Linear Unit activation function",
            "parameters": {
                "inplace": {"type": "bool", "default": False, "description": "Perform operation in-place"}
            }
        },
        {
            "name": "LeakyReLU",
            "category": "activation",
            "description": "Leaky ReLU activation with small negative slope",
            "parameters": {
                "negative_slope": {"type": "float", "default": 0.01, "description": "Slope for negative values"},
                "inplace": {"type": "bool", "default": False, "description": "Perform operation in-place"}
            }
        },
        {
            "name": "GELU",
            "category": "activation",
            "description": "Gaussian Error Linear Unit activation",
            "parameters": {}
        },
        {
            "name": "Sigmoid",
            "category": "activation",
            "description": "Sigmoid activation function",
            "parameters": {}
        },
        {
            "name": "Tanh",
            "category": "activation",
            "description": "Hyperbolic tangent activation function",
            "parameters": {}
        },
        {
            "name": "Softmax",
            "category": "activation",
            "description": "Softmax activation for probability distribution",
            "parameters": {
                "dim": {"type": "int", "default": -1, "description": "Dimension along which to compute softmax"}
            }
        },
        {
            "name": "BatchNorm2d",
            "category": "normalization",
            "description": "Batch normalization for 2D inputs (images)",
            "parameters": {
                "num_features": {"type": "int", "required": True, "description": "Number of features/channels"},
                "eps": {"type": "float", "default": 1e-5, "description": "Value added for numerical stability"},
                "momentum": {"type": "float", "default": 0.1, "description": "Momentum for running mean/variance"}
            }
        },
        {
            "name": "BatchNorm1d",
            "category": "normalization",
            "description": "Batch normalization for 1D inputs",
            "parameters": {
                "num_features": {"type": "int", "required": True, "description": "Number of features"},
                "eps": {"type": "float", "default": 1e-5, "description": "Value added for numerical stability"},
                "momentum": {"type": "float", "default": 0.1, "description": "Momentum for running mean/variance"}
            }
        },
        {
            "name": "LayerNorm",
            "category": "normalization",
            "description": "Layer normalization",
            "parameters": {
                "normalized_shape": {"type": "list", "required": True, "description": "Shape of input to normalize"},
                "eps": {"type": "float", "default": 1e-5, "description": "Value added for numerical stability"}
            }
        },
        {
            "name": "Dropout",
            "category": "regularization",
            "description": "Randomly zero out elements during training",
            "parameters": {
                "p": {"type": "float", "default": 0.5, "description": "Probability of element being zeroed"},
                "inplace": {"type": "bool", "default": False, "description": "Perform operation in-place"}
            }
        },
        {
            "name": "MaxPool2d",
            "category": "pooling",
            "description": "2D max pooling operation",
            "parameters": {
                "kernel_size": {"type": "int", "required": True, "description": "Size of the pooling window"},
                "stride": {"type": "int", "default": None, "description": "Stride of the pooling (defaults to kernel_size)"},
                "padding": {"type": "int", "default": 0, "description": "Zero-padding added to both sides"}
            }
        },
        {
            "name": "AvgPool2d",
            "category": "pooling",
            "description": "2D average pooling operation",
            "parameters": {
                "kernel_size": {"type": "int", "required": True, "description": "Size of the pooling window"},
                "stride": {"type": "int", "default": None, "description": "Stride of the pooling (defaults to kernel_size)"},
                "padding": {"type": "int", "default": 0, "description": "Zero-padding added to both sides"}
            }
        },
        {
            "name": "AdaptiveAvgPool2d",
            "category": "pooling",
            "description": "Adaptive average pooling to fixed output size",
            "parameters": {
                "output_size": {"type": "tuple", "required": True, "description": "Target output size (H, W)"}
            }
        },
        {
            "name": "Flatten",
            "category": "reshape",
            "description": "Flatten tensor to 1D (keeping batch dimension)",
            "parameters": {
                "start_dim": {"type": "int", "default": 1, "description": "First dimension to flatten"},
                "end_dim": {"type": "int", "default": -1, "description": "Last dimension to flatten"}
            }
        },
        {
            "name": "LSTM",
            "category": "recurrent",
            "description": "Long Short-Term Memory recurrent layer",
            "parameters": {
                "input_size": {"type": "int", "required": True, "description": "Size of input features"},
                "hidden_size": {"type": "int", "required": True, "description": "Size of hidden state"},
                "num_layers": {"type": "int", "default": 1, "description": "Number of stacked LSTM layers"},
                "batch_first": {"type": "bool", "default": False, "description": "Input/output tensors are (batch, seq, feature)"},
                "dropout": {"type": "float", "default": 0.0, "description": "Dropout probability between layers"},
                "bidirectional": {"type": "bool", "default": False, "description": "Use bidirectional LSTM"}
            }
        },
        {
            "name": "GRU",
            "category": "recurrent",
            "description": "Gated Recurrent Unit recurrent layer",
            "parameters": {
                "input_size": {"type": "int", "required": True, "description": "Size of input features"},
                "hidden_size": {"type": "int", "required": True, "description": "Size of hidden state"},
                "num_layers": {"type": "int", "default": 1, "description": "Number of stacked GRU layers"},
                "batch_first": {"type": "bool", "default": False, "description": "Input/output tensors are (batch, seq, feature)"},
                "dropout": {"type": "float", "default": 0.0, "description": "Dropout probability between layers"},
                "bidirectional": {"type": "bool", "default": False, "description": "Use bidirectional GRU"}
            }
        },
        {
            "name": "MultiHeadAttention",
            "category": "attention",
            "description": "Multi-head attention mechanism",
            "parameters": {
                "embed_dim": {"type": "int", "required": True, "description": "Total dimension of the model"},
                "num_heads": {"type": "int", "required": True, "description": "Number of attention heads"},
                "dropout": {"type": "float", "default": 0.0, "description": "Dropout probability"},
                "batch_first": {"type": "bool", "default": False, "description": "Input/output tensors are (batch, seq, feature)"}
            }
        },
        {
            "name": "Embedding",
            "category": "embedding",
            "description": "Embedding layer for discrete tokens",
            "parameters": {
                "num_embeddings": {"type": "int", "required": True, "description": "Size of vocabulary"},
                "embedding_dim": {"type": "int", "required": True, "description": "Size of embedding vectors"},
                "padding_idx": {"type": "int", "default": None, "description": "Index for padding token"}
            }
        },
        {
            "name": "ConvTranspose2d",
            "category": "convolution",
            "description": "2D transposed convolution (deconvolution)",
            "parameters": {
                "in_channels": {"type": "int", "required": True, "description": "Number of input channels"},
                "out_channels": {"type": "int", "required": True, "description": "Number of output channels"},
                "kernel_size": {"type": "int", "required": True, "description": "Size of the convolving kernel"},
                "stride": {"type": "int", "default": 1, "description": "Stride of the convolution"},
                "padding": {"type": "int", "default": 0, "description": "Zero-padding added to both sides"},
                "output_padding": {"type": "int", "default": 0, "description": "Additional size added to output shape"}
            }
        }
    ]
    return {"layers": layers}


@app.get("/design/help")
async def get_design_help():
    """Return helpful tips for model design."""
    tips = {
        "general": [
            "Start with a simple architecture and gradually add complexity",
            "Use batch normalization after convolutional layers to stabilize training",
            "Add dropout layers to prevent overfitting, especially in fully connected layers",
            "Match the output layer size to your number of classes"
        ],
        "image_classification": [
            "Use Conv2d -> BatchNorm2d -> ReLU as a basic building block",
            "Gradually increase channels while decreasing spatial dimensions",
            "Use MaxPool2d or strided convolutions to reduce spatial size",
            "End with AdaptiveAvgPool2d -> Flatten -> Linear for flexible input sizes",
            "Common channel progression: 32 -> 64 -> 128 -> 256"
        ],
        "sequence_models": [
            "LSTM and GRU are good for sequential data like text or time series",
            "Use bidirectional=True for tasks where future context is available",
            "Stack multiple layers for more complex patterns",
            "Add dropout between recurrent layers to reduce overfitting"
        ],
        "transformers": [
            "MultiHeadAttention is the core of transformer architectures",
            "Use LayerNorm before or after attention (pre-norm is often more stable)",
            "Combine attention with feedforward networks (Linear -> ReLU -> Linear)",
            "Positional encodings are essential for sequence order information"
        ],
        "training": [
            "AdamW is a good default optimizer for most tasks",
            "Start with a learning rate of 0.001 and adjust based on loss curves",
            "Use CosineAnnealingLR or ReduceLROnPlateau for learning rate scheduling",
            "Monitor validation loss to detect overfitting early"
        ],
        "augmentation": [
            "random_flip and random_crop are essential for image classification",
            "Use color_jitter for robustness to lighting changes",
            "normalize should match the dataset statistics",
            "More augmentation can help with smaller datasets"
        ],
        "common_mistakes": [
            "Forgetting to match tensor dimensions between layers",
            "Using too high a learning rate causing training instability",
            "Not using enough regularization (dropout, weight decay)",
            "Making the model too deep without skip connections"
        ]
    }
    return {
        "tips": tips,
        "recommended_architectures": {
            "small_image_classifier": {
                "description": "Simple CNN for small images (32x32)",
                "layers": ["Conv2d(3, 32, 3)", "BatchNorm2d(32)", "ReLU", "MaxPool2d(2)",
                          "Conv2d(32, 64, 3)", "BatchNorm2d(64)", "ReLU", "MaxPool2d(2)",
                          "Flatten", "Linear(64*6*6, 128)", "ReLU", "Dropout(0.5)", "Linear(128, num_classes)"]
            },
            "deeper_cnn": {
                "description": "Deeper CNN with more capacity",
                "layers": ["Conv2d -> BatchNorm2d -> ReLU blocks with increasing channels",
                          "Use residual connections for very deep networks",
                          "AdaptiveAvgPool2d(1, 1) before classifier"]
            },
            "simple_rnn": {
                "description": "Basic RNN for sequence classification",
                "layers": ["Embedding(vocab_size, embed_dim)", "LSTM(embed_dim, hidden_size, bidirectional=True)",
                          "Linear(hidden_size*2, num_classes)"]
            }
        }
    }


# ============= Worker Status =============

@app.get("/workers/status")
async def get_worker_status():
    """Get status of training workers."""
    pending = training_service.get_pending_count()
    active = training_service.list_active_jobs()

    return {
        "pending_jobs": pending,
        "active_jobs": len(active),
        "jobs": active
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print(f"Whitematter Model Server v0.5.0 at http://{args.host}:{args.port}")
    print(f"Data directory: {get_data_dir()}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
