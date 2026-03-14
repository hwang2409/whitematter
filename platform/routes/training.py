"""
Training endpoints including start, status, and cancel.
"""

import json
import logging
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from auth.dependencies import get_current_user
from db.auth_models import User

from config import PROJECT_ROOT, DATA_DIR, DATASETS, PRESET_ARCHITECTURES, GENERATED_DIR
from schemas import (
    TrainRequest, CustomTrainRequest, ModelMetadata, TrainStatus,
    TrainingJobResponse, DetailResponse,
)
from dependencies import training_jobs, dataset_service, code_generator, limiter
from services.model_registry import save_model_metadata, get_model_path
from codegen import compile_training_code
from codegen.compiler import run_training as run_custom_training_process

logger = logging.getLogger(__name__)

router = APIRouter()


def _monitor_process(job_id: str, process, metadata: ModelMetadata):
    """Read training process stdout, parse Epoch lines, update job dict + metadata."""
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

                training_jobs[job_id].update({
                    "epoch": epoch, "loss": loss, "accuracy": acc,
                    "message": f"Epoch {epoch}: {acc:.2f}%",
                })

                metadata.epochs_trained = epoch
                metadata.best_accuracy = max(metadata.best_accuracy, acc)
                metadata.training_history.append({"epoch": epoch, "loss": loss, "accuracy": acc})
                save_model_metadata(metadata)
                training_jobs.sync_to_db(job_id)
            except (ValueError, IndexError, KeyError) as e:
                logger.debug("Failed to parse training output line: %s - %s", line, e)

        if training_jobs[job_id].get("cancelled"):
            process.terminate()
            break
    process.wait()


def _finalize_process(job_id: str, process, metadata: ModelMetadata, model_src: Path):
    """Handle post-process status: cancelled / completed / failed."""
    if training_jobs[job_id].get("cancelled"):
        training_jobs[job_id]["status"] = "cancelled"
        metadata.status = TrainStatus.CANCELLED
    elif process.returncode == 0 and model_src.exists():
        shutil.copy(model_src, get_model_path(metadata.id))
        training_jobs[job_id]["status"] = "completed"
        metadata.status = TrainStatus.COMPLETED
        training_jobs[job_id]["message"] = f"Complete! Best: {metadata.best_accuracy:.2f}%"
    else:
        training_jobs[job_id]["status"] = "failed"
        metadata.status = TrainStatus.FAILED
        output_lines = training_jobs[job_id].get("output", [])
        if output_lines:
            training_jobs[job_id]["message"] = "\n".join(output_lines[-5:])

    save_model_metadata(metadata)
    training_jobs.sync_to_db(job_id)


def _fail_job(job_id: str, metadata: ModelMetadata, error: Exception):
    """Mark a training job as failed with an error message."""
    logger.exception("Training job %s failed", job_id)
    training_jobs[job_id]["status"] = "failed"
    metadata.status = TrainStatus.FAILED
    training_jobs[job_id]["message"] = str(error)

    save_model_metadata(metadata)
    training_jobs.sync_to_db(job_id)


def run_training(job_id: str, request: TrainRequest, metadata: ModelMetadata):
    import subprocess
    try:
        training_jobs[job_id]["status"] = "running"
    
        metadata.status = TrainStatus.RUNNING
        save_model_metadata(metadata)

        BUILD_DIR = PROJECT_ROOT / "build"
        if request.dataset == "cifar10":
            cmd, src = [str(BUILD_DIR / "cnn_cifar10"), str(DATA_DIR)], BUILD_DIR / "cnn_cifar10.bin"
        elif request.dataset == "mnist":
            cmd, src = [str(BUILD_DIR / "cnn_mnist"), str(DATA_DIR)], BUILD_DIR / "cnn_mnist.bin"
        else:
            raise ValueError(f"Unsupported dataset: {request.dataset}")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        _monitor_process(job_id, process, metadata)
        _finalize_process(job_id, process, metadata, src)
    except (ValueError, RuntimeError, OSError) as e:
        _fail_job(job_id, metadata, e)


def run_custom_training(job_id: str, request: CustomTrainRequest, metadata: ModelMetadata):
    """Background function to run custom model training."""
    from db import get_blob_store
    blob_store = get_blob_store()

    try:
        training_jobs[job_id]["status"] = "running"
    
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
    
        success, msg = compile_training_code(job_dir)
        if not success:
            raise RuntimeError(f"Compilation failed: {msg}")

        training_jobs[job_id]["status"] = "training"
        training_jobs[job_id]["message"] = "Training..."
    
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
            output_model=output_model
        )
        _monitor_process(job_id, process, metadata)
        _finalize_process(job_id, process, metadata, output_model)

    except (ValueError, RuntimeError, FileNotFoundError, OSError) as e:
        _fail_job(job_id, metadata, e)


@router.post("/train/custom", response_model=TrainingJobResponse)
@limiter.limit("5/hour")
async def start_custom_training(request: Request, req: CustomTrainRequest, user: User = Depends(get_current_user)):
    """Start training with a custom architecture on an uploaded dataset."""
    db_dataset = dataset_service.get_dataset(req.dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if db_dataset.get('status') != 'ready':
        raise HTTPException(status_code=400, detail="Dataset not processed yet")

    validation = code_generator.validate_architecture(req.architecture)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=f"Invalid architecture: {validation['errors']}")

    model_id = job_id = str(uuid.uuid4())[:8]
    dataset_name = db_dataset.get('name', 'unknown')
    model_name = req.name or f"custom_{dataset_name}_{model_id}"

    metadata = ModelMetadata(
        id=model_id,
        name=model_name,
        dataset=f"custom:{req.dataset_id}",
        architecture="custom",
        created_at=datetime.now().isoformat(),
        epochs_trained=0,
        best_accuracy=0.0,
        status=TrainStatus.PENDING,
        training_history=[],
        config={
            "dataset_id": req.dataset_id,
            "architecture": req.architecture
        }
    )
    save_model_metadata(metadata)

    training_jobs[job_id] = {
        "id": job_id,
        "model_id": model_id,
        "status": "pending",
        "epoch": 0,
        "total_epochs": req.architecture.get("training", {}).get("epochs", 10),
        "loss": 0.0,
        "accuracy": 0.0,
        "message": "Starting...",
        "output": [],
        "cancelled": False
    }

    threading.Thread(target=run_custom_training, args=(job_id, req, metadata)).start()

    job = training_jobs[job_id]
    return {
        "job_id": job_id,
        "model_id": model_id,
        "status": job["status"],
        "epoch": job["epoch"],
        "total_epochs": job["total_epochs"],
        "loss": job["loss"],
        "accuracy": job["accuracy"],
        "message": job["message"],
        "architecture": req.architecture
    }


@router.post("/train", response_model=TrainingJobResponse)
@limiter.limit("5/hour")
async def start_training(request: Request, req: TrainRequest, user: User = Depends(get_current_user)):
    if req.dataset not in DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {req.dataset}")

    if req.preset:
        if req.preset not in PRESET_ARCHITECTURES:
            raise HTTPException(status_code=400, detail=f"Unknown preset: {req.preset}")
        preset = PRESET_ARCHITECTURES[req.preset]
        if preset["dataset"] != req.dataset:
            raise HTTPException(status_code=400, detail=f"Preset {req.preset} is for {preset['dataset']}")
        architecture, layers = f"preset:{req.preset}", preset["layers"]
    elif req.layers:
        architecture, layers = "custom", [l.model_dump() for l in req.layers]
    else:
        raise HTTPException(status_code=400, detail="Either preset or layers must be provided")

    model_id = job_id = str(uuid.uuid4())[:8]
    config = {"dataset": req.dataset, "architecture": architecture, "layers": layers, "epochs": req.epochs,
              "batch_size": req.batch_size, "optimizer": req.optimizer.model_dump(),
              "scheduler": req.scheduler.model_dump(), "augmentations": [a.model_dump() for a in req.augmentations]}

    metadata = ModelMetadata(id=model_id, name=req.name or f"{req.dataset}_{model_id}", dataset=req.dataset,
                             architecture=architecture, created_at=datetime.now().isoformat(), epochs_trained=0,
                             best_accuracy=0.0, status=TrainStatus.PENDING, training_history=[], config=config)
    save_model_metadata(metadata)

    training_jobs[job_id] = {"id": job_id, "model_id": model_id, "status": "pending", "epoch": 0,
                            "total_epochs": req.epochs, "loss": 0.0, "accuracy": 0.0, "message": "Starting...",
                            "output": [], "cancelled": False}

    threading.Thread(target=run_training, args=(job_id, req, metadata)).start()

    job = training_jobs[job_id]
    return {
        "job_id": job_id,
        "model_id": model_id,
        "status": job["status"],
        "epoch": job["epoch"],
        "total_epochs": job["total_epochs"],
        "loss": job["loss"],
        "accuracy": job["accuracy"],
        "message": job["message"],
        "config": config
    }


@router.get("/train/{job_id}", response_model=TrainingJobResponse)
async def get_training_status(job_id: str, user: User = Depends(get_current_user)):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    j = training_jobs[job_id]
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
        "message": j.get("message", "")
    }


@router.delete("/train/{job_id}", response_model=DetailResponse)
async def cancel_training(job_id: str, user: User = Depends(get_current_user)):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    j = training_jobs[job_id]
    if j["status"] not in ["pending", "running", "compiling", "training"]:
        raise HTTPException(status_code=400, detail="Job not running")
    j["cancelled"] = True
    if "process" in j:
        j["process"].terminate()
    return {"detail": "Training cancelled"}
