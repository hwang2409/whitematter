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

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db import init_db, get_data_dir
from dependencies import capture_event_loop, ensure_dirs
from config import MODELS_DIR

from routes.health import router as health_router
from routes.auth import router as auth_router
from routes.credentials import router as credentials_router
from routes.storage import router as storage_router
from routes.datasets import router as datasets_router
from routes.design import router as design_router
from routes.training import router as training_router
from routes.models import router as models_router
from routes.predict import router as predict_router

logger = logging.getLogger(__name__)

# Initialize database
init_db()
logger.info("Data directory: %s", get_data_dir())

app = FastAPI(title="Whitematter Model Server", version="0.5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Register routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(credentials_router)
app.include_router(storage_router)
app.include_router(datasets_router)
app.include_router(design_router)
app.include_router(training_router)
app.include_router(models_router)
app.include_router(predict_router)


@app.on_event("startup")
async def _capture_event_loop():
    capture_event_loop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    import config
    config.MODELS_DIR = args.models_dir
    ensure_dirs()

    logger.info("Whitematter Model Server v0.5.0 at http://%s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
