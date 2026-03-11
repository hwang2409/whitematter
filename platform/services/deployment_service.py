"""
Deployment service: build artifact tarball and manage deployment state for one-click Deploy to API.
"""

import io
import json
import logging
import secrets
import tarfile
from pathlib import Path
from typing import Optional, Tuple

from db import get_db_session, get_blob_store
from db.auth_models import Deployment, DeploymentStatus
from db.models import Model
from config import GENERATED_DIR
from dependencies import load_model_metadata, get_model_path
from schemas import TrainStatus
from deploy.server_py_content import get_server_py_content

logger = logging.getLogger(__name__)


class DeploymentService:
    """Build deployment artifacts and serve them for EC2 bootstrap."""

    def __init__(self):
        self.blob_store = get_blob_store()

    def get_config_for_model(self, model_id: str, dataset_id: str) -> Optional[dict]:
        """Get dataset config (target_size, mean, std, channels, class_names) for a custom dataset."""
        from dependencies import dataset_manager
        from services.dataset_service import DatasetService
        ds = DatasetService()

        # File-based: dataset_manager.uploads_dir / dataset_id / "processed" / config.json
        processed_dir = dataset_manager.uploads_dir / dataset_id / "processed"
        config_path = processed_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)

        # DB/blob: dataset has processed_blob_prefix
        dataset_info = ds.get_dataset(dataset_id)
        if not dataset_info:
            return None
        prefix = dataset_info.get("processed_blob_prefix")
        if not prefix:
            return None
        config_blob = self.blob_store.get(f"{prefix}/config.json")
        if not config_blob:
            return None
        return json.loads(config_blob.decode("utf-8"))

    def get_model_and_infer_bytes(self, model_id: str) -> Tuple[Optional[bytes], Optional[bytes]]:
        """Return (model_weights_bytes, infer_binary_bytes). Prefer blob, then disk."""
        # Blob: models/{model_id}/weights.bin, models/{model_id}/infer
        weights_blob = self.blob_store.get(f"models/{model_id}/weights.bin")
        infer_blob = self.blob_store.get(f"models/{model_id}/infer")
        if weights_blob and infer_blob:
            return (weights_blob, infer_blob)

        # Disk: MODELS_DIR/{model_id}.bin, GENERATED_DIR/{model_id}/infer
        model_path = get_model_path(model_id)
        job_dir = GENERATED_DIR / model_id
        infer_exe = job_dir / "infer"
        if model_path.exists() and infer_exe.exists():
            return (model_path.read_bytes(), infer_exe.read_bytes())

        # Only weights in blob (e.g. worker stored weights but infer in job_dir)
        if weights_blob and infer_exe.exists():
            return (weights_blob, infer_exe.read_bytes())
        if model_path.exists() and infer_blob:
            return (model_path.read_bytes(), infer_blob)

        return (None, None)

    def build_artifact_tarball(self, model_id: str) -> Optional[bytes]:
        """
        Build a tarball containing model.bin, infer, config.json, server.py.
        Returns None if model/dataset not suitable (e.g. not image custom model).
        """
        metadata = load_model_metadata(model_id)
        dataset_id = None
        if metadata:
            if metadata.status != TrainStatus.COMPLETED:
                return None
            if not metadata.dataset.startswith("custom:"):
                return None  # Only custom (image) models for now
            dataset_id = metadata.dataset.replace("custom:", "")
        else:
            with get_db_session() as db:
                m = db.query(Model).filter_by(id=model_id).first()
                if not m or m.status != ModelStatus.COMPLETED.value:
                    return None
                dataset_id = m.dataset_id
        if not dataset_id:
            return None

        config = self.get_config_for_model(model_id, dataset_id)
        if not config:
            return None

        model_bytes, infer_bytes = self.get_model_and_infer_bytes(model_id)
        if not model_bytes or not infer_bytes:
            return None

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:") as tar:
            # model.bin
            ti = tarfile.TarInfo(name="model.bin")
            ti.size = len(model_bytes)
            tar.addfile(ti, io.BytesIO(model_bytes))
            # infer
            ti = tarfile.TarInfo(name="infer")
            ti.size = len(infer_bytes)
            ti.mode = 0o755
            tar.addfile(ti, io.BytesIO(infer_bytes))
            # config.json
            config_bytes = json.dumps(config).encode("utf-8")
            ti = tarfile.TarInfo(name="config.json")
            ti.size = len(config_bytes)
            tar.addfile(ti, io.BytesIO(config_bytes))
            # server.py
            server_py = get_server_py_content().encode("utf-8")
            ti = tarfile.TarInfo(name="server.py")
            ti.size = len(server_py)
            tar.addfile(ti, io.BytesIO(server_py))

        buf.seek(0)
        return buf.read()

    def create_deployment(self, user_id: str, model_id: str, region: str = "us-east-1") -> Optional[Deployment]:
        """Create a Deployment record with a one-time token. Caller launches EC2 and sets token in user-data."""
        if self.build_artifact_tarball(model_id) is None:
            return None

        token = secrets.token_urlsafe(32)
        with get_db_session() as db:
            d = Deployment(
                user_id=user_id,
                model_id=model_id,
                target_type="ec2",
                status=DeploymentStatus.PENDING.value,
                region=region,
                deployment_token=token,
            )
            db.add(d)
            db.flush()
            db.refresh(d)
            return d

    def get_artifacts(self, deployment_id: str, token: str) -> Optional[bytes]:
        """If token matches deployment, return tarball bytes. Token is consumed on set_live (callback)."""
        with get_db_session() as db:
            d = db.query(Deployment).filter_by(id=deployment_id).first()
            if not d or d.deployment_token is None or d.deployment_token != token:
                return None
            tarball = self.build_artifact_tarball(d.model_id)
            if tarball is None:
                return None
            d.status = DeploymentStatus.BOOTSTRAPPING.value
            db.commit()
            return tarball

    def set_live(self, deployment_id: str, token: str, endpoint_url: str) -> bool:
        """Callback from EC2: set deployment to LIVE, store endpoint_url, and consume token."""
        with get_db_session() as db:
            d = db.query(Deployment).filter_by(id=deployment_id).first()
            if not d:
                return False
            if d.deployment_token is None or d.deployment_token != token:
                return False
            d.status = DeploymentStatus.LIVE.value
            d.endpoint_url = endpoint_url.rstrip("/")
            d.deployment_token = None
            db.commit()
            return True

    def set_failed(self, deployment_id: str, error_message: str) -> None:
        with get_db_session() as db:
            d = db.query(Deployment).filter_by(id=deployment_id).first()
            if d:
                d.status = DeploymentStatus.FAILED.value
                d.error_message = error_message
                db.commit()
