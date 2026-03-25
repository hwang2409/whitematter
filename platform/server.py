#!/usr/bin/env python3
"""
Whitematter Model Server v0.5.0

Now with database and blob storage backend.
Data is stored in ~/.whitematter/ instead of the project directory.

To migrate existing data, run: python migrate_to_db.py
To start workers for training: python run_worker.py --count 2
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root so CREDENTIAL_ENCRYPTION_KEY etc. are set before services import
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from db import init_db, get_data_dir
from config import ensure_dirs
from dependencies import limiter
from services.job_store import TrainingJobStore
from config import MODELS_DIR
from services.auth_service import AuthService

from routes.health import router as health_router
from routes.auth import router as auth_router
from routes.credentials import router as credentials_router
from routes.storage import router as storage_router
from routes.byoc_training import router as byoc_training_router
from routes.deploy import router as deploy_router
from routes.datasets import router as datasets_router
from routes.design import router as design_router
from routes.training import router as training_router
from routes.models import router as models_router
from routes.predict import router as predict_router
from routes.chat import router as chat_router
from routes.billing import router as billing_router

logger = logging.getLogger(__name__)

# Initialize database and clean up stale jobs
init_db()
TrainingJobStore.cleanup_stale_jobs()
logger.info("Data directory: %s", get_data_dir())

# Validate JWT secret at startup (fail fast)
try:
    AuthService()
except RuntimeError as e:
    raise SystemExit(f"FATAL: {e}") from e

app = FastAPI(title="Whitematter Model Server", version="0.5.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_allowed_origins = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(credentials_router)
app.include_router(storage_router)
app.include_router(byoc_training_router)
app.include_router(deploy_router)
app.include_router(datasets_router)
app.include_router(design_router)
app.include_router(training_router)
app.include_router(models_router)
app.include_router(predict_router)
app.include_router(chat_router)
app.include_router(billing_router)


_background_worker = None


@app.on_event("startup")
def _start_background_worker():
    """Start a training worker in a background daemon thread."""
    import threading
    from workers.worker import TrainingWorker

    global _background_worker
    _background_worker = TrainingWorker(worker_id="server-worker")

    def _run():
        # Can't register signal handlers from non-main thread, so just run the poll loop
        _background_worker.running = True
        _background_worker._stop_event.clear()
        logger.info("[%s] Background worker started", _background_worker.worker_id)
        while _background_worker.running and not _background_worker._stop_event.is_set():
            try:
                job = _background_worker.queue.claim(_background_worker.worker_id)
                if job:
                    _background_worker._process_job(job)
                else:
                    _background_worker._stop_event.wait(_background_worker.poll_interval)
            except Exception as e:
                logger.error("[%s] Error: %s", _background_worker.worker_id, e)
                _background_worker._stop_event.wait(_background_worker.poll_interval)

    t = threading.Thread(target=_run, daemon=True, name="training-worker")
    t.start()
    logger.info("Background training worker thread started")


@app.on_event("shutdown")
def _stop_background_worker():
    global _background_worker
    if _background_worker:
        _background_worker.stop()


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    import config
    config.MODELS_DIR = Path(args.models_dir)
    ensure_dirs()

    logger.info("Whitematter Model Server v0.5.0 at http://%s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
