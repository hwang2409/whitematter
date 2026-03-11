"""
Training endpoints including start, status, cancel, and WebSocket progress.
"""

import asyncio
import json
import logging
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from config import PROJECT_ROOT, DATA_DIR, DATASETS, PRESET_ARCHITECTURES, GENERATED_DIR
from schemas import TrainRequest, CustomTrainRequest, ModelMetadata, TrainStatus
from dependencies import (
    training_jobs, _ws_subscribers, _ws_lock,
    dataset_manager, dataset_service, code_generator,
    save_model_metadata, get_model_path,
    notify_training_subscribers, _get_job_snapshot,
)
from codegen import compile_training_code
from codegen.compiler import run_training as run_custom_training_process

logger = logging.getLogger(__name__)

router = APIRouter()


def run_training(job_id: str, request: TrainRequest, metadata: ModelMetadata):
    import subprocess
    try:
        training_jobs[job_id]["status"] = "running"
        notify_training_subscribers(job_id)
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
                    notify_training_subscribers(job_id)
                    metadata.epochs_trained = epoch
                    metadata.best_accuracy = max(metadata.best_accuracy, acc)
                    metadata.training_history.append({"epoch": epoch, "loss": loss, "accuracy": acc})
                    save_model_metadata(metadata)
                except (ValueError, IndexError, KeyError) as e:
                    logger.debug("Failed to parse training output line: %s - %s", line, e)

            if training_jobs[job_id].get("cancelled"):
                process.terminate()
                break

        process.wait()

        if training_jobs[job_id].get("cancelled"):
            training_jobs[job_id]["status"] = "cancelled"
            metadata.status = TrainStatus.CANCELLED
            notify_training_subscribers(job_id)
        elif process.returncode == 0 and src.exists():
            shutil.copy(src, get_model_path(metadata.id))
            training_jobs[job_id]["status"] = "completed"
            metadata.status = TrainStatus.COMPLETED
            training_jobs[job_id]["message"] = f"Complete! Best: {metadata.best_accuracy:.2f}%"
            notify_training_subscribers(job_id)
        else:
            training_jobs[job_id]["status"] = "failed"
            metadata.status = TrainStatus.FAILED
            notify_training_subscribers(job_id)
        save_model_metadata(metadata)
    except (ValueError, RuntimeError, OSError) as e:
        logger.exception("Training job %s failed", job_id)
        training_jobs[job_id]["status"] = "failed"
        metadata.status = TrainStatus.FAILED
        training_jobs[job_id]["message"] = str(e)
        notify_training_subscribers(job_id)
        save_model_metadata(metadata)


def run_custom_training(job_id: str, request: CustomTrainRequest, metadata: ModelMetadata):
    """Background function to run custom model training."""
    from db import get_blob_store
    blob_store = get_blob_store()

    try:
        training_jobs[job_id]["status"] = "running"
        notify_training_subscribers(job_id)
        metadata.status = TrainStatus.RUNNING
        save_model_metadata(metadata)

        dataset_meta = dataset_manager.get_metadata(request.dataset_id)
        processed_dir = dataset_manager.uploads_dir / request.dataset_id / "processed"
        config_path = processed_dir / "config.json"

        dataset_config = None
        data_from_blobs = False

        if config_path.exists():
            with open(config_path) as f:
                dataset_config = json.load(f)
        else:
            db_dataset = dataset_service.get_dataset(request.dataset_id)
            if db_dataset and db_dataset.get('processed_blob_prefix'):
                config_blob = blob_store.get(f"{db_dataset['processed_blob_prefix']}/config.json")
                if config_blob:
                    dataset_config = json.loads(config_blob.decode())
                    data_from_blobs = True

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
        training_jobs[job_id]["message"] = "Training..."
        notify_training_subscribers(job_id)
        output_model = job_dir / "model.bin"

        actual_data_dir = processed_dir
        temp_data_dir = None
        if data_from_blobs:
            db_dataset = dataset_service.get_dataset(request.dataset_id)
            if db_dataset and db_dataset.get('processed_blob_prefix'):
                temp_data_dir = job_dir / "data"
                temp_data_dir.mkdir(exist_ok=True)
                blob_prefix = db_dataset['processed_blob_prefix']

                data_type = dataset_config.get('data_type', 'image')
                if data_type == 'text':
                    files = ['train_inputs.bin', 'train_targets.bin',
                             'test_inputs.bin', 'test_targets.bin',
                             'config.json', 'vocabulary.json']
                else:
                    files = ['train_images.bin', 'train_labels.bin',
                             'test_images.bin', 'test_labels.bin', 'config.json']

                for filename in files:
                    data = blob_store.get(f"{blob_prefix}/{filename}")
                    if data:
                        (temp_data_dir / filename).write_bytes(data)

                actual_data_dir = temp_data_dir

        process = run_custom_training_process(
            generated_dir=job_dir,
            data_dir=actual_data_dir,
            output_model=output_model
        )
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
                        "epoch": epoch,
                        "loss": loss,
                        "accuracy": acc,
                        "message": f"Epoch {epoch}: {acc:.2f}%"
                    })
                    notify_training_subscribers(job_id)
                    metadata.epochs_trained = epoch
                    metadata.best_accuracy = max(metadata.best_accuracy, acc)
                    metadata.training_history.append({"epoch": epoch, "loss": loss, "accuracy": acc})
                    save_model_metadata(metadata)
                except (ValueError, IndexError, KeyError) as e:
                    logger.debug("Failed to parse custom training output line: %s - %s", line, e)

            if training_jobs[job_id].get("cancelled"):
                process.terminate()
                break

        process.wait()

        if training_jobs[job_id].get("cancelled"):
            logger.info("Custom training job %s was cancelled", job_id)
            training_jobs[job_id]["status"] = "cancelled"
            metadata.status = TrainStatus.CANCELLED
            notify_training_subscribers(job_id)
        elif process.returncode == 0 and output_model.exists():
            shutil.copy(output_model, get_model_path(metadata.id))
            training_jobs[job_id]["status"] = "completed"
            metadata.status = TrainStatus.COMPLETED
            training_jobs[job_id]["message"] = f"Complete! Best: {metadata.best_accuracy:.2f}%"
            notify_training_subscribers(job_id)
            logger.info("Custom training job %s completed with accuracy %.2f%%", job_id, metadata.best_accuracy)
        else:
            training_jobs[job_id]["status"] = "failed"
            metadata.status = TrainStatus.FAILED
            output_lines = training_jobs[job_id].get("output", [])
            error_msg = "\n".join(output_lines[-5:]) if output_lines else f"Training failed (exit code {process.returncode})"
            training_jobs[job_id]["message"] = error_msg
            notify_training_subscribers(job_id)
            logger.error("Custom training job %s failed: %s", job_id, error_msg)

        save_model_metadata(metadata)

    except (ValueError, RuntimeError, FileNotFoundError, OSError) as e:
        logger.exception("Custom training job %s encountered an error", job_id)
        training_jobs[job_id]["status"] = "failed"
        metadata.status = TrainStatus.FAILED
        training_jobs[job_id]["message"] = str(e)
        notify_training_subscribers(job_id)
        save_model_metadata(metadata)


@router.post("/train/custom")
async def start_custom_training(request: CustomTrainRequest):
    """Start training with a custom architecture on an uploaded dataset."""
    dataset_meta = dataset_manager.get_metadata(request.dataset_id)
    db_dataset = None
    if not dataset_meta:
        db_dataset = dataset_service.get_dataset(request.dataset_id)
        if not db_dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if db_dataset.get('status') != 'ready':
            raise HTTPException(status_code=400, detail="Dataset not processed yet")
    else:
        if dataset_meta.status not in ("processed", "ready"):
            raise HTTPException(status_code=400, detail="Dataset not processed yet")

    validation = code_generator.validate_architecture(request.architecture)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=f"Invalid architecture: {validation['errors']}")

    model_id = job_id = str(uuid.uuid4())[:8]
    dataset_name = dataset_meta.name if dataset_meta else db_dataset.get('name', 'unknown')
    model_name = request.name or f"custom_{dataset_name}_{model_id}"

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
        "status": "pending",
        "epoch": 0,
        "total_epochs": request.architecture.get("training", {}).get("epochs", 10),
        "loss": 0.0,
        "accuracy": 0.0,
        "message": "Starting...",
        "output": [],
        "cancelled": False
    }

    threading.Thread(target=run_custom_training, args=(job_id, request, metadata)).start()

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
        "architecture": request.architecture
    }


@router.post("/train")
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

    training_jobs[job_id] = {"id": job_id, "model_id": model_id, "status": "pending", "epoch": 0,
                            "total_epochs": request.epochs, "loss": 0.0, "accuracy": 0.0, "message": "Starting...",
                            "output": [], "cancelled": False}

    threading.Thread(target=run_training, args=(job_id, request, metadata)).start()

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


@router.websocket("/ws/train/{job_id}")
async def ws_training_status(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time training updates."""
    if job_id not in training_jobs:
        await websocket.close(code=4004, reason="Training job not found")
        return

    await websocket.accept()

    queue: asyncio.Queue = asyncio.Queue()
    with _ws_lock:
        _ws_subscribers.setdefault(job_id, []).append(queue)

    try:
        snapshot = _get_job_snapshot(job_id)
        if snapshot:
            await websocket.send_json(snapshot)

        while True:
            try:
                update = await asyncio.wait_for(queue.get(), timeout=5.0)
                await websocket.send_json(update)
                if update.get("status") in ("completed", "failed", "cancelled"):
                    await websocket.close(code=1000)
                    break
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        with _ws_lock:
            subs = _ws_subscribers.get(job_id, [])
            if queue in subs:
                subs.remove(queue)
            if not subs:
                _ws_subscribers.pop(job_id, None)


@router.get("/train/{job_id}")
async def get_training_status(job_id: str):
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


@router.delete("/train/{job_id}")
async def cancel_training(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    j = training_jobs[job_id]
    if j["status"] not in ["pending", "running", "compiling", "training"]:
        raise HTTPException(status_code=400, detail="Job not running")
    j["cancelled"] = True
    if "process" in j:
        j["process"].terminate()
    return {"message": "Training cancelled"}
