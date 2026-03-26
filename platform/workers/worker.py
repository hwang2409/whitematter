"""
Training worker that processes jobs from the queue.
Can be run as multiple instances for parallel training.

All persistent data is stored in the database - temp files are used
only during training and cleaned up afterwards.
"""
import logging
import uuid
import signal
import subprocess
import threading
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from db import (
    get_db_session, get_blob_store,
    Model, Dataset, TrainingJob, TrainingHistory,
    JobStatus, ModelStatus, DatasetStatus
)
from workers.queue import JobQueue, get_job_queue, JobMessage
from workers.training_executor import TrainingExecutor
from workers.output_parser import parse_training_line


class TrainingWorker:
    """
    Worker that claims and processes training jobs.
    Uses temp directories for all file operations - nothing persists to disk
    except the SQLite database.
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        poll_interval: float = 2.0,
        project_root: Optional[Path] = None
    ):
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.poll_interval = poll_interval
        self.queue = get_job_queue()
        self.blob_store = get_blob_store()
        self.running = False
        self._stop_event = threading.Event()
        self._current_process: Optional[subprocess.Popen] = None

        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.executor = TrainingExecutor(self.blob_store, self.project_root)

    def start(self):
        """Start the worker loop."""
        self.running = True
        self._stop_event.clear()

        logger.info("[%s] Starting worker...", self.worker_id)

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        while self.running and not self._stop_event.is_set():
            try:
                job = self.queue.claim(self.worker_id)
                if job:
                    self._process_job(job)
                else:
                    self._stop_event.wait(self.poll_interval)
            except Exception as e:
                logger.error("[%s] Error: %s", self.worker_id, e)
                self._stop_event.wait(self.poll_interval)

        logger.info("[%s] Worker stopped", self.worker_id)

    def stop(self):
        """Stop the worker gracefully."""
        logger.info("[%s] Stopping...", self.worker_id)
        self.running = False
        self._stop_event.set()

        if self._current_process:
            self._current_process.terminate()

    def _handle_signal(self, signum, frame):
        """Handle termination signals."""
        self.stop()

    def _process_job(self, job: JobMessage):
        """Process a single training job using temp directories."""
        logger.info("[%s] Processing job %s", self.worker_id, job.job_id)

        with tempfile.TemporaryDirectory(prefix=f"wm_job_{job.job_id}_") as temp_dir:
            job_dir = Path(temp_dir)

            try:
                # Get model and dataset info — extract plain data inside session
                # to avoid detached-instance errors after session closes.
                with get_db_session() as db:
                    model = db.query(Model).filter_by(id=job.model_id).first()
                    if not model:
                        raise ValueError(f"Model {job.model_id} not found")

                    dataset = None
                    dataset_info = None
                    if model.dataset_id:
                        dataset = db.query(Dataset).filter_by(id=model.dataset_id).first()
                    if dataset:
                        dataset_info = {
                            "processed_blob_prefix": dataset.processed_blob_prefix,
                            "input_shape": dataset.input_shape,
                            "num_classes": dataset.num_classes,
                            "class_names": dataset.class_names,
                        }

                    arch_config = model.architecture_config or {}
                    train_config = model.training_config or {}

                try:
                    self.queue.update_status(
                        job.job_id, JobStatus.COMPILING,
                        message="Generating training code..."
                    )
                    self.executor.generate_code(job_dir, dataset_info, arch_config, train_config)
                except Exception as e:
                    raise RuntimeError(f"[codegen] {str(e)[:500]}")

                try:
                    self.queue.update_status(
                        job.job_id, JobStatus.COMPILING, message="Compiling..."
                    )
                    success, compile_msg = self.executor.compile(job_dir)
                    if not success:
                        # Filter out warnings to surface the actual error
                        error_lines = [
                            line for line in compile_msg.splitlines()
                            if "warning:" not in line and line.strip()
                        ]
                        filtered = "\n".join(error_lines) if error_lines else compile_msg
                        raise RuntimeError(f"[compile] {filtered[:1000]}")
                except RuntimeError:
                    raise
                except Exception as e:
                    raise RuntimeError(f"[compile] {str(e)[:500]}")

                try:
                    self.queue.update_status(
                        job.job_id, JobStatus.TRAINING, message="Training started"
                    )
                    self._run_training(job, job_dir, dataset_info)
                except Exception as e:
                    raise RuntimeError(f"[train] {str(e)[:500]}")

                # Check if cancelled
                job_info = self.queue.get_job(job.job_id)
                if job_info and job_info['status'] == JobStatus.CANCELLED.value:
                    return

                try:
                    self._store_artifacts(job, job_dir)
                except Exception as e:
                    raise RuntimeError(f"[store] {str(e)[:500]}")

                self.queue.update_status(
                    job.job_id, JobStatus.COMPLETED, message="Training complete"
                )
                with get_db_session() as db:
                    model = db.query(Model).filter_by(id=job.model_id).first()
                    if model:
                        model.status = ModelStatus.COMPLETED.value

                logger.info("[%s] Job %s completed", self.worker_id, job.job_id)

            except Exception as e:
                logger.error("[%s] Job %s failed: %s", self.worker_id, job.job_id, e)
                self.queue.update_status(
                    job.job_id, JobStatus.FAILED,
                    message=str(e), error_message=str(e)
                )
                with get_db_session() as db:
                    model = db.query(Model).filter_by(id=job.model_id).first()
                    if model:
                        model.status = ModelStatus.FAILED.value

    def _run_training(self, job: JobMessage, job_dir: Path, dataset_info: Optional[dict]):
        """Run the training process and stream progress."""
        blob_prefix = dataset_info.get("processed_blob_prefix") if dataset_info else None
        if blob_prefix:
            data_dir = job_dir / "data"
            data_dir.mkdir(exist_ok=True)
            self.executor.extract_dataset_from_blobs(blob_prefix, data_dir)
        else:
            data_dir = job_dir / "data"

        self._current_process = self.executor.run_training(job_dir, data_dir)

        best_accuracy = 0.0
        final_loss = None

        try:
            for line in self._current_process.stdout:
                line = line.strip()

                # Check if job was cancelled
                job_info = self.queue.get_job(job.job_id)
                if job_info and job_info['status'] == JobStatus.CANCELLED.value:
                    self._current_process.terminate()
                    break

                metrics = parse_training_line(line)
                if metrics:
                    epoch = metrics["epoch"]
                    loss = metrics["loss"]
                    acc = metrics["accuracy"]

                    best_accuracy = max(best_accuracy, acc)
                    final_loss = loss

                    self.queue.update_status(
                        job.job_id, JobStatus.TRAINING,
                        message=f"Epoch {epoch}: {acc:.2f}%",
                        current_epoch=epoch,
                        current_loss=loss,
                        current_accuracy=acc
                    )

                    with get_db_session() as db:
                        history = TrainingHistory(
                            model_id=job.model_id,
                            job_id=job.job_id,
                            epoch=epoch,
                            loss=loss,
                            accuracy=acc
                        )
                        db.add(history)

                        m = db.query(Model).filter_by(id=job.model_id).first()
                        if m:
                            m.epochs_trained = epoch
                            m.best_accuracy = best_accuracy
                            m.final_loss = final_loss

            self._current_process.wait()
            if self._current_process.returncode != 0:
                raise RuntimeError(f"Training process exited with code {self._current_process.returncode}")

        finally:
            self._current_process = None

    def _store_artifacts(self, job: JobMessage, job_dir: Path):
        """Store model weights and inference binary in blob storage."""
        output_model = job_dir / "model.bin"
        if output_model.exists():
            blob_key = self.blob_store.put_file(
                output_model,
                key=f"models/{job.model_id}/weights.bin",
                content_type="application/octet-stream"
            )
            with get_db_session() as db:
                m = db.query(Model).filter_by(id=job.model_id).first()
                if m:
                    m.weights_blob_key = blob_key

        infer_exe = job_dir / "infer"
        if infer_exe.exists():
            self.blob_store.put_file(
                infer_exe,
                key=f"models/{job.model_id}/infer",
                content_type="application/octet-stream"
            )


def run_worker(worker_id: Optional[str] = None):
    """Entry point for running a worker."""
    worker = TrainingWorker(worker_id=worker_id)
    worker.start()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training worker")
    parser.add_argument("--id", help="Worker ID")
    args = parser.parse_args()
    run_worker(args.id)
