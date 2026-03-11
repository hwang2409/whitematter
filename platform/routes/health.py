"""
Health, config, and worker status endpoints.
"""

import logging

from fastapi import APIRouter

from config import DATASETS, LAYER_TYPES, OPTIMIZERS, SCHEDULERS, AUGMENTATIONS, PRESET_ARCHITECTURES, DATA_DIR
from dependencies import training_jobs

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def root():
    return {"name": "Whitematter Model Server", "version": "0.4.0"}


@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.get("/workers/status")
async def get_worker_status():
    """Get status of training workers and job queue."""
    try:
        from workers import get_job_queue
        queue = get_job_queue()
        pending = queue.get_pending_count()
        active = queue.get_active_jobs()
        logger.info("Retrieved worker status: %d pending, %d active jobs", pending, len(active))
        return {
            "pending_jobs": pending,
            "active_jobs": len(active),
            "jobs": active
        }
    except ImportError:
        logger.debug("Workers module not available, using legacy job tracking")
        return {
            "pending_jobs": 0,
            "active_jobs": len(training_jobs),
            "jobs": [
                {k: v for k, v in job.items() if k != 'process'}
                for job in training_jobs.values()
            ],
            "note": "Using legacy in-memory job tracking"
        }


@router.get("/config/datasets")
async def list_datasets():
    result = []
    for k, v in DATASETS.items():
        available = (DATA_DIR / ("data_batch_1.bin" if k == "cifar10" else "train-images-idx3-ubyte")).exists()
        result.append({"id": k, "available": available, **v})
    return {"datasets": result}


@router.get("/config/layers")
async def list_layer_types():
    return {"layers": [{"id": k, **v} for k, v in LAYER_TYPES.items()]}


@router.get("/config/optimizers")
async def list_optimizers():
    return {"optimizers": [{"id": k, **v} for k, v in OPTIMIZERS.items()]}


@router.get("/config/schedulers")
async def list_schedulers():
    return {"schedulers": [{"id": k, **v} for k, v in SCHEDULERS.items()]}


@router.get("/config/augmentations")
async def list_augmentations():
    return {"augmentations": [{"id": k, **v} for k, v in AUGMENTATIONS.items()]}


@router.get("/config/presets")
async def list_presets():
    return {"presets": [{"id": k, "name": v["name"], "dataset": v["dataset"], "num_layers": len(v["layers"])} for k, v in PRESET_ARCHITECTURES.items()]}


@router.get("/config/presets/{preset_id}")
async def get_preset(preset_id: str):
    if preset_id not in PRESET_ARCHITECTURES:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"preset": {"id": preset_id, **PRESET_ARCHITECTURES[preset_id]}}
