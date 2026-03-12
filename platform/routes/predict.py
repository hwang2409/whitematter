"""
Prediction and inference endpoints (/predict and /api/*).
"""

import io
import json
import logging
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from auth.dependencies import get_current_user
from db.auth_models import User
from PIL import Image

from config import DATASETS, GENERATED_DIR
from schemas import (
    ModelMetadata, TrainStatus, GenerateRequest,
    PredictionResponse, GenerateTextResponse, ModelInfoResponse,
)
from dependencies import (
    dataset_service,
    load_model_metadata, get_model_path,
    get_loaded_model, preprocess_image,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(model_id: str, file: UploadFile = File(...), user: User = Depends(get_current_user)):
    if not (metadata := load_model_metadata(model_id)):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if metadata.status != TrainStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Model not ready")

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
    """Handle prediction for custom-trained models using subprocess inference."""
    from db import get_blob_store
    blob_store = get_blob_store()

    dataset_id = metadata.dataset.replace("custom:", "")
    db_dataset = dataset_service.get_dataset(dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=400, detail="Original dataset not found")

    processed_prefix = db_dataset.get('processed_blob_prefix')
    if not processed_prefix:
        raise HTTPException(status_code=400, detail="Dataset config not found")

    config_blob = blob_store.get(f"{processed_prefix}/config.json")
    if not config_blob:
        raise HTTPException(status_code=400, detail="Dataset config not found")
    config = json.loads(config_blob.decode())

    job_dir = GENERATED_DIR / model_id
    infer_exe = job_dir / "infer"
    model_bin = get_model_path(model_id)

    if not infer_exe.exists():
        if not (job_dir / "infer.cpp").exists():
            raise HTTPException(status_code=400, detail="Inference code not found - model may need retraining")
        result = subprocess.run(
            ["make", "infer"],
            cwd=job_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Failed to compile inference: {result.stderr}")

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

    mean = np.array(config["mean"]).reshape(-1, 1, 1)
    std = np.array(config["std"]).reshape(-1, 1, 1)
    arr = (arr - mean) / std

    arr = arr[np.newaxis, ...]
    input_tensor = np.ascontiguousarray(arr, dtype=np.float32)

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
            [str(infer_exe), str(model_bin), str(input_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Inference failed: {result.stderr}")

        stdout_lines = result.stdout.strip().split('\n')
        json_line = None
        for line in stdout_lines:
            if line.startswith('{'):
                json_line = line
                break
        if not json_line:
            raise HTTPException(status_code=500, detail=f"No JSON output from inference: {result.stdout}")
        output = json.loads(json_line)
        predicted_class = output["predicted_class"]
        probs = output["probabilities"]
        classes = config["class_names"]

        return {
            "model_id": model_id,
            "model_name": metadata.name,
            "predicted_class": predicted_class,
            "class_name": classes[predicted_class],
            "confidence": float(probs[predicted_class]),
            "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))}
        }
    finally:
        input_path.unlink(missing_ok=True)


@router.post("/api/{model_id}/predict", response_model=PredictionResponse)
async def api_predict(model_id: str, file: UploadFile = File(...), user: User = Depends(get_current_user)):
    """Convenience endpoint for model prediction: /api/{model_id}/predict"""
    return await predict(model_id, file, user)


@router.post("/api/{model_id}/generate", response_model=GenerateTextResponse)
async def generate_text(model_id: str, request: GenerateRequest, user: User = Depends(get_current_user)):
    """Generate text using a trained language model."""
    metadata = load_model_metadata(model_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Model not found")

    if metadata.status != TrainStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Model not ready")

    if not metadata.dataset.startswith("custom:"):
        raise HTTPException(status_code=400, detail="This model is not a language model")

    dataset_id = metadata.dataset.replace("custom:", "")

    from db import get_blob_store
    blob_store = get_blob_store()

    dataset_info = dataset_service.get_dataset(dataset_id)
    if not dataset_info:
        raise HTTPException(status_code=400, detail="Dataset not found")

    if dataset_info.get('data_type') != 'text':
        raise HTTPException(status_code=400, detail="Not a text dataset")

    processed_prefix = dataset_info.get('processed_blob_prefix')
    if not processed_prefix:
        raise HTTPException(status_code=400, detail="Dataset not processed")

    vocab_blob = blob_store.get(f"{processed_prefix}/vocabulary.json")
    if not vocab_blob:
        raise HTTPException(status_code=400, detail="Vocabulary not found")

    job_dir = GENERATED_DIR / model_id
    job_dir.mkdir(parents=True, exist_ok=True)
    infer_exe = job_dir / "infer"
    model_bin = get_model_path(model_id)

    if not infer_exe.exists():
        infer_blob = blob_store.get(f"models/{model_id}/infer")
        if not infer_blob:
            raise HTTPException(status_code=400, detail="Inference executable not found")
        infer_exe.write_bytes(infer_blob)
        import stat
        infer_exe.chmod(infer_exe.stat().st_mode | stat.S_IEXEC)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='wb') as tmp:
        tmp.write(vocab_blob)
        vocab_path = Path(tmp.name)

    # Validate prompt input
    prompt = request.prompt
    if not prompt or len(prompt) > 10_000:
        raise HTTPException(status_code=400, detail="Prompt must be 1-10000 characters")
    if '\x00' in prompt:
        raise HTTPException(status_code=400, detail="Prompt contains invalid characters")

    try:
        # Pass prompt via stdin to avoid shell injection through argv
        result = subprocess.run(
            [
                str(infer_exe),
                str(model_bin),
                str(vocab_path),
                "--stdin",
                str(request.max_tokens),
                str(request.temperature)
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Generation failed: {result.stderr}")

        return {
            "model_id": model_id,
            "prompt": prompt,
            "generated_text": result.stdout.strip(),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
    finally:
        vocab_path.unlink(missing_ok=True)


@router.get("/api/{model_id}/info", response_model=ModelInfoResponse)
async def api_model_info(model_id: str, user: User = Depends(get_current_user)):
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

    if metadata.dataset.startswith("custom:"):
        from db import get_blob_store
        blob_store = get_blob_store()
        dataset_id = metadata.dataset.replace("custom:", "")
        db_ds = dataset_service.get_dataset(dataset_id)
        if db_ds and db_ds.get('processed_blob_prefix'):
            config_blob = blob_store.get(f"{db_ds['processed_blob_prefix']}/config.json")
            if config_blob:
                config = json.loads(config_blob.decode())
                result["classes"] = config["class_names"]
                result["input_shape"] = config["input_shape"]
    elif metadata.dataset in DATASETS:
        result["classes"] = DATASETS[metadata.dataset]["classes"]
        result["input_shape"] = DATASETS[metadata.dataset]["input_shape"]

    return result
