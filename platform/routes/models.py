"""
Model management endpoints (list, get, delete, resume).
"""

import json
import logging
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from auth.dependencies import get_current_user
from db.auth_models import User

from config import GENERATED_DIR
from schemas import ModelMetadata, TrainStatus, ModelListResponse, DetailResponse, ResumeTrainingResponse
from dependencies import training_jobs, dataset_service, code_generator
from services.model_registry import (
    loaded_models, load_model_metadata, save_model_metadata,
    list_all_models, get_model_path, get_metadata_path,
)
from services.training_state import notify_training_subscribers
from codegen import compile_training_code
from codegen.compiler import run_training as run_custom_training_process
from routes.training import _monitor_process, _finalize_process, _fail_job

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
async def list_models(user: User = Depends(get_current_user)):
    return {"models": [m.model_dump() for m in list_all_models()]}


@router.get("/models/{model_id}", response_model=ModelMetadata)
async def get_model(model_id: str, user: User = Depends(get_current_user)):
    if not (m := load_model_metadata(model_id)):
        raise HTTPException(status_code=404, detail="Model not found")
    return m.model_dump()


@router.delete("/models/{model_id}", response_model=DetailResponse)
async def delete_model(model_id: str, user: User = Depends(get_current_user)):
    if not load_model_metadata(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    loaded_models.pop(model_id, None)
    for p in [get_model_path(model_id), get_metadata_path(model_id)]:
        p.unlink(missing_ok=True)
    return {"detail": f"Model {model_id} deleted"}


@router.post("/models/{model_id}/resume", response_model=ResumeTrainingResponse)
async def resume_training(model_id: str, user: User = Depends(get_current_user)):
    """Resume training a failed or cancelled model from its last checkpoint."""
    metadata = load_model_metadata(model_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Model not found")

    if metadata.status not in (TrainStatus.FAILED, TrainStatus.CANCELLED):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume model with status '{metadata.status}'. Only failed or cancelled models can be resumed."
        )

    config = metadata.config
    if not config or "architecture" not in config:
        raise HTTPException(status_code=400, detail="Model config missing architecture")

    weights_path = get_model_path(model_id)
    if not weights_path.exists():
        raise HTTPException(status_code=400, detail="No checkpoint found to resume from")

    dataset_id = config.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="Model config missing dataset_id")

    job_id = str(uuid.uuid4())[:8]

    architecture = config["architecture"]
    total_epochs = architecture.get("training", {}).get("epochs", 10)
    start_epoch = metadata.epochs_trained

    if start_epoch >= total_epochs:
        raise HTTPException(status_code=400, detail="Model already completed all epochs")

    metadata.status = TrainStatus.PENDING
    save_model_metadata(metadata)

    training_jobs[job_id] = {
        "id": job_id,
        "model_id": model_id,
        "status": "pending",
        "epoch": start_epoch,
        "total_epochs": total_epochs,
        "loss": 0.0,
        "accuracy": metadata.best_accuracy,
        "message": f"Resuming from epoch {start_epoch}...",
        "output": [],
        "cancelled": False
    }

    class ResumeRequest:
        def __init__(self, dataset_id, architecture):
            self.dataset_id = dataset_id
            self.architecture = architecture
            self.name = None

    request = ResumeRequest(dataset_id, architecture)

    threading.Thread(
        target=run_resume_training,
        args=(job_id, request, metadata, weights_path, start_epoch)
    ).start()

    return {
        "job_id": job_id,
        "model_id": model_id,
        "detail": f"Resuming training from epoch {start_epoch}",
        "start_epoch": start_epoch,
        "total_epochs": total_epochs
    }


def run_resume_training(job_id: str, request, metadata: ModelMetadata, weights_path: Path, start_epoch: int):
    """Background function to run resumed model training."""
    from db import get_blob_store
    blob_store = get_blob_store()

    try:
        training_jobs[job_id]["status"] = "running"
        notify_training_subscribers(job_id)
        metadata.status = TrainStatus.RUNNING
        save_model_metadata(metadata)

        db_dataset = dataset_service.get_dataset(request.dataset_id)
        if not db_dataset or not db_dataset.get('processed_blob_prefix'):
            raise FileNotFoundError("Dataset not processed")

        dataset_config = None
        config_blob = blob_store.get(f"{db_dataset['processed_blob_prefix']}/config.json")
        if config_blob:
            dataset_config = json.loads(config_blob.decode())

        if not dataset_config:
            raise FileNotFoundError("Dataset not processed")

        if request.architecture.get('data_type'):
            dataset_config['data_type'] = request.architecture['data_type']

        job_dir = GENERATED_DIR / job_id
        code_generator.generate(
            architecture=request.architecture,
            dataset_config=dataset_config,
            output_dir=job_dir
        )

        training_jobs[job_id]["status"] = "compiling"
        training_jobs[job_id]["message"] = "Compiling..."
        notify_training_subscribers(job_id)
        success, msg = compile_training_code(job_dir)
        if not success:
            raise RuntimeError(f"Compilation failed: {msg}")

        training_jobs[job_id]["status"] = "training"
        training_jobs[job_id]["message"] = f"Resuming from epoch {start_epoch}..."
        notify_training_subscribers(job_id)
        output_model = job_dir / "model.bin"

        # Extract processed data from blob storage to temp dir
        actual_data_dir = job_dir / "data"
        actual_data_dir.mkdir(exist_ok=True)
        blob_prefix = db_dataset['processed_blob_prefix']

        data_type = dataset_config.get('data_type', 'image')
        if data_type == 'text':
            blob_files = ['train_inputs.bin', 'train_targets.bin',
                         'test_inputs.bin', 'test_targets.bin',
                         'config.json', 'vocabulary.json']
        else:
            blob_files = ['train_images.bin', 'train_labels.bin',
                         'test_images.bin', 'test_labels.bin', 'config.json']

        for filename in blob_files:
            data = blob_store.get(f"{blob_prefix}/{filename}")
            if data:
                (actual_data_dir / filename).write_bytes(data)

        process = run_custom_training_process(
            generated_dir=job_dir,
            data_dir=actual_data_dir,
            output_model=output_model,
            resume_weights=weights_path,
            start_epoch=start_epoch
        )
        _monitor_process(job_id, process, metadata)
        _finalize_process(job_id, process, metadata, output_model)

    except Exception as e:
        _fail_job(job_id, metadata, e)
