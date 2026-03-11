"""
Training service - handles training job operations.
"""
from typing import Optional, List, Dict, Any

from db import get_db_session, TrainingJob, JobStatus, Model, ModelStatus
from workers import get_job_queue


class TrainingService:
    """Service for managing training jobs."""

    def __init__(self):
        self.queue = get_job_queue()

    def start_training(
        self,
        model_id: str,
        total_epochs: int,
        priority: int = 0
    ) -> Dict:
        """Start a new training job for a model."""
        # Verify model exists
        with get_db_session() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            if not model:
                raise ValueError(f"Model {model_id} not found")

            # Update model status
            model.status = ModelStatus.TRAINING.value

        # Enqueue the job
        job_id = self.queue.enqueue(
            model_id=model_id,
            total_epochs=total_epochs,
            priority=priority
        )

        return self.get_job_status(job_id)

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get training job status."""
        return self.queue.get_job(job_id)

    def list_active_jobs(self) -> List[Dict]:
        """List all active training jobs."""
        return self.queue.get_active_jobs()

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        return self.queue.cancel_job(job_id)

    def get_pending_count(self) -> int:
        """Get count of pending jobs."""
        return self.queue.get_pending_count()

    def get_jobs_for_model(self, model_id: str) -> List[Dict]:
        """Get all jobs for a specific model."""
        with get_db_session() as db:
            jobs = (
                db.query(TrainingJob)
                .filter_by(model_id=model_id)
                .order_by(TrainingJob.created_at.desc())
                .all()
            )
            return [
                {
                    'id': j.id,
                    'model_id': j.model_id,
                    'worker_id': j.worker_id,
                    'status': j.status,
                    'current_epoch': j.current_epoch,
                    'total_epochs': j.total_epochs,
                    'current_loss': j.current_loss,
                    'current_accuracy': j.current_accuracy,
                    'message': j.message,
                    'error_message': j.error_message,
                    'created_at': j.created_at.isoformat() if j.created_at else None,
                    'started_at': j.started_at.isoformat() if j.started_at else None,
                    'completed_at': j.completed_at.isoformat() if j.completed_at else None
                }
                for j in jobs
            ]
