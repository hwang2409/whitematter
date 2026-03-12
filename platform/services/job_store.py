"""
Training job store backed by the database.

Keeps an in-memory cache for fast real-time updates (WebSocket epoch streaming)
and persists state to the training_jobs DB table on key transitions.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from db import get_db_session, TrainingJob, JobStatus

logger = logging.getLogger(__name__)

# Statuses that indicate an active (in-progress) job
_ACTIVE_STATUSES = {"pending", "running", "compiling", "training", "queued"}


class TrainingJobStore:
    """Dict-like wrapper around training_jobs DB table with in-memory cache."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ---- dict-like interface used by existing code ----

    def __contains__(self, job_id: str) -> bool:
        return job_id in self._cache

    def __getitem__(self, job_id: str) -> Dict[str, Any]:
        return self._cache[job_id]

    def __setitem__(self, job_id: str, value: Dict[str, Any]) -> None:
        self._cache[job_id] = value
        self._persist(job_id, value)

    def get(self, job_id: str, default=None):
        return self._cache.get(job_id, default)

    def pop(self, job_id: str, *args):
        return self._cache.pop(job_id, *args)

    def keys(self):
        return self._cache.keys()

    def values(self):
        return self._cache.values()

    def items(self):
        return self._cache.items()

    def clear(self):
        self._cache.clear()

    def update(self, *args, **kwargs):
        self._cache.update(*args, **kwargs)

    # ---- persistence ----

    def _persist(self, job_id: str, data: Dict[str, Any]) -> None:
        """Write job state to DB."""
        try:
            with get_db_session() as db:
                job = db.query(TrainingJob).filter_by(id=job_id).first()
                if job is None:
                    job = TrainingJob(
                        id=job_id,
                        model_id=data.get("model_id", ""),
                        status=data.get("status", JobStatus.PENDING.value),
                        total_epochs=data.get("total_epochs", 0),
                        message=data.get("message"),
                    )
                    db.add(job)
                else:
                    job.status = data.get("status", job.status)
                    job.current_epoch = data.get("epoch", job.current_epoch)
                    job.total_epochs = data.get("total_epochs", job.total_epochs)
                    job.current_loss = data.get("loss", job.current_loss)
                    job.current_accuracy = data.get("accuracy", job.current_accuracy)
                    job.message = data.get("message", job.message)
                    job.updated_at = datetime.utcnow()

                    status = data.get("status", "")
                    if status == "running" and not job.started_at:
                        job.started_at = datetime.utcnow()
                    if status in ("completed", "failed", "cancelled"):
                        job.completed_at = datetime.utcnow()
        except Exception:
            logger.debug("Failed to persist training job %s", job_id, exc_info=True)

    def sync_to_db(self, job_id: str) -> None:
        """Explicitly sync current cache state to DB (call after frequent updates)."""
        data = self._cache.get(job_id)
        if data:
            self._persist(job_id, data)

    # ---- startup cleanup ----

    @staticmethod
    def cleanup_stale_jobs() -> int:
        """Mark any in-progress DB jobs as failed (e.g. after server crash). Returns count."""
        count = 0
        try:
            with get_db_session() as db:
                stale = (
                    db.query(TrainingJob)
                    .filter(TrainingJob.status.in_(list(_ACTIVE_STATUSES)))
                    .all()
                )
                for job in stale:
                    job.status = JobStatus.FAILED.value
                    job.message = "Server restarted while job was in progress"
                    job.completed_at = datetime.utcnow()
                    count += 1
            if count:
                logger.info("Marked %d stale training jobs as failed", count)
        except Exception:
            logger.debug("cleanup_stale_jobs failed", exc_info=True)
        return count
