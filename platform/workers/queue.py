"""
Job queue for training tasks.
Uses SQLite as the backing store for simplicity.
Can be swapped to Redis for production deployments.
"""
import uuid
import threading
from datetime import datetime
from typing import Optional, List, Callable
from dataclasses import dataclass

from db import get_db_session, TrainingJob, JobStatus


@dataclass
class JobMessage:
    """Message for a training job."""
    job_id: str
    model_id: str
    priority: int = 0


class JobQueue:
    """
    Simple job queue backed by the TrainingJob table.
    Workers poll for pending jobs and claim them atomically.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._listeners: List[Callable[[str], None]] = []

    def enqueue(
        self,
        model_id: str,
        total_epochs: int,
        priority: int = 0
    ) -> str:
        """
        Add a new training job to the queue.

        Args:
            model_id: ID of the model to train
            total_epochs: Number of epochs to train
            priority: Job priority (higher = more important)

        Returns:
            The job ID
        """
        job_id = uuid.uuid4().hex[:16]

        with get_db_session() as db:
            job = TrainingJob(
                id=job_id,
                model_id=model_id,
                status=JobStatus.PENDING.value,
                total_epochs=total_epochs,
                current_epoch=0,
                message=f"Queued with priority {priority}"
            )
            db.add(job)

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(job_id)
            except Exception:
                pass

        return job_id

    def claim(self, worker_id: str) -> Optional[JobMessage]:
        """
        Atomically claim the next available job.

        Args:
            worker_id: ID of the claiming worker

        Returns:
            JobMessage if a job was claimed, None otherwise
        """
        with self._lock:
            with get_db_session() as db:
                # Find oldest pending job
                job = (
                    db.query(TrainingJob)
                    .filter(TrainingJob.status == JobStatus.PENDING.value)
                    .filter(TrainingJob.worker_id.is_(None))
                    .order_by(TrainingJob.created_at)
                    .first()
                )

                if not job:
                    return None

                # Claim it
                job.worker_id = worker_id
                job.status = JobStatus.QUEUED.value
                job.message = f"Claimed by worker {worker_id}"

                return JobMessage(
                    job_id=job.id,
                    model_id=job.model_id,
                    priority=0
                )

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        message: Optional[str] = None,
        current_epoch: Optional[int] = None,
        current_loss: Optional[float] = None,
        current_accuracy: Optional[float] = None,
        error_message: Optional[str] = None
    ):
        """Update job status and progress."""
        with get_db_session() as db:
            job = db.query(TrainingJob).filter_by(id=job_id).first()
            if not job:
                return

            job.status = status.value
            job.updated_at = datetime.utcnow()

            if message is not None:
                job.message = message
            if current_epoch is not None:
                job.current_epoch = current_epoch
            if current_loss is not None:
                job.current_loss = current_loss
            if current_accuracy is not None:
                job.current_accuracy = current_accuracy
            if error_message is not None:
                job.error_message = error_message

            if status == JobStatus.TRAINING and job.started_at is None:
                job.started_at = datetime.utcnow()
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                job.completed_at = datetime.utcnow()

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job status and details."""
        with get_db_session() as db:
            job = db.query(TrainingJob).filter_by(id=job_id).first()
            if not job:
                return None

            return {
                'id': job.id,
                'model_id': job.model_id,
                'worker_id': job.worker_id,
                'status': job.status,
                'current_epoch': job.current_epoch,
                'total_epochs': job.total_epochs,
                'current_loss': job.current_loss,
                'current_accuracy': job.current_accuracy,
                'message': job.message,
                'error_message': job.error_message,
                'created_at': job.created_at.isoformat() if job.created_at else None,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None
            }

    def get_pending_count(self) -> int:
        """Get count of pending jobs."""
        with get_db_session() as db:
            return db.query(TrainingJob).filter(
                TrainingJob.status == JobStatus.PENDING.value
            ).count()

    def get_active_jobs(self) -> List[dict]:
        """Get all active (non-completed) jobs."""
        with get_db_session() as db:
            jobs = db.query(TrainingJob).filter(
                TrainingJob.status.in_([
                    JobStatus.PENDING.value,
                    JobStatus.QUEUED.value,
                    JobStatus.COMPILING.value,
                    JobStatus.TRAINING.value
                ])
            ).order_by(TrainingJob.created_at).all()

            return [
                {
                    'id': job.id,
                    'model_id': job.model_id,
                    'worker_id': job.worker_id,
                    'status': job.status,
                    'current_epoch': job.current_epoch,
                    'total_epochs': job.total_epochs,
                    'current_loss': job.current_loss,
                    'current_accuracy': job.current_accuracy,
                    'message': job.message
                }
                for job in jobs
            ]

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        with get_db_session() as db:
            job = db.query(TrainingJob).filter_by(id=job_id).first()
            if not job:
                return False

            if job.status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
                return False

            job.status = JobStatus.CANCELLED.value
            job.message = "Cancelled by user"
            job.completed_at = datetime.utcnow()
            return True

    def add_listener(self, callback: Callable[[str], None]):
        """Add a listener for new jobs."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[str], None]):
        """Remove a job listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)


# Singleton instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get the global job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue
