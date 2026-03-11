"""
Training worker that processes jobs from the queue.
Can be run as multiple instances for parallel training.

All persistent data is stored in the database - temp files are used
only during training and cleaned up afterwards.
"""
import os
import json
import logging
import uuid
import time
import signal
import shutil
import subprocess
import threading
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

from db import (
    get_db_session, get_blob_store,
    Model, Dataset, TrainingJob, TrainingHistory,
    JobStatus, ModelStatus, DatasetStatus
)
from workers.queue import JobQueue, get_job_queue, JobMessage


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

        # Project paths (for finding C++ headers)
        self.project_root = project_root or Path(__file__).parent.parent.parent

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

        # Use temp directory for all file operations - cleaned up automatically
        with tempfile.TemporaryDirectory(prefix=f"wm_job_{job.job_id}_") as temp_dir:
            job_dir = Path(temp_dir)

            try:
                # Get model and dataset info
                with get_db_session() as db:
                    model = db.query(Model).filter_by(id=job.model_id).first()
                    if not model:
                        raise ValueError(f"Model {job.model_id} not found")

                    dataset = None
                    if model.dataset_id:
                        dataset = db.query(Dataset).filter_by(id=model.dataset_id).first()

                    arch_config = model.architecture_config or {}
                    train_config = model.training_config or {}

                # Update status to compiling
                self.queue.update_status(
                    job.job_id,
                    JobStatus.COMPILING,
                    message="Generating training code..."
                )

                # Generate training code in temp directory
                self._generate_code(job_dir, model, dataset, arch_config, train_config)

                # Compile
                self.queue.update_status(
                    job.job_id,
                    JobStatus.COMPILING,
                    message="Compiling..."
                )

                success, compile_msg = self._compile(job_dir)
                if not success:
                    raise RuntimeError(f"Compilation failed: {compile_msg}")

                # Run training
                self.queue.update_status(
                    job.job_id,
                    JobStatus.TRAINING,
                    message="Training started"
                )

                self._run_training(job, job_dir, model, dataset)

                # Check if cancelled
                job_info = self.queue.get_job(job.job_id)
                if job_info and job_info['status'] == JobStatus.CANCELLED.value:
                    return

                # Store model weights in blob storage BEFORE temp dir is deleted
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

                # Store inference executable for text generation models
                infer_exe = job_dir / "infer"
                if infer_exe.exists():
                    self.blob_store.put_file(
                        infer_exe,
                        key=f"models/{job.model_id}/infer",
                        content_type="application/octet-stream"
                    )

                # Mark as completed
                self.queue.update_status(
                    job.job_id,
                    JobStatus.COMPLETED,
                    message="Training complete"
                )

                # Update model status
                with get_db_session() as db:
                    model = db.query(Model).filter_by(id=job.model_id).first()
                    if model:
                        model.status = ModelStatus.COMPLETED.value

                logger.info("[%s] Job %s completed", self.worker_id, job.job_id)

            except Exception as e:
                logger.error("[%s] Job %s failed: %s", self.worker_id, job.job_id, e)
                self.queue.update_status(
                    job.job_id,
                    JobStatus.FAILED,
                    message=str(e),
                    error_message=str(e)
                )
                with get_db_session() as db:
                    model = db.query(Model).filter_by(id=job.model_id).first()
                    if model:
                        model.status = ModelStatus.FAILED.value

        # Temp directory is automatically cleaned up here

    def _generate_code(
        self,
        job_dir: Path,
        model: Model,
        dataset: Optional[Dataset],
        arch_config: dict,
        train_config: dict
    ):
        """Generate C++ training code."""
        # Import here to avoid circular imports
        from codegen import CodeGenerator

        generator = CodeGenerator()

        # Build dataset config from database model
        if dataset:
            processed_blob_prefix = dataset.processed_blob_prefix
            if processed_blob_prefix:
                # Load config from blob storage
                config_blob = self.blob_store.get(f"{processed_blob_prefix}/config.json")
                if config_blob:
                    dataset_config = json.loads(config_blob.decode())
                else:
                    dataset_config = {
                        "input_shape": dataset.input_shape,
                        "num_classes": dataset.num_classes,
                        "class_names": dataset.class_names,
                        "mean": [0.5],
                        "std": [0.5]
                    }
            else:
                dataset_config = {
                    "input_shape": dataset.input_shape,
                    "num_classes": dataset.num_classes,
                    "class_names": dataset.class_names,
                    "mean": [0.5],
                    "std": [0.5]
                }
        else:
            dataset_config = train_config.get("dataset_config", {})

        generator.generate(
            architecture=arch_config,
            dataset_config=dataset_config,
            output_dir=job_dir
        )

    def _compile(self, job_dir: Path) -> tuple[bool, str]:
        """Compile the generated training code."""
        from codegen import compile_training_code
        return compile_training_code(job_dir)

    def _run_training(
        self,
        job: JobMessage,
        job_dir: Path,
        model: Model,
        dataset: Optional[Dataset]
    ):
        """Run the training process and stream progress."""
        train_exe = job_dir / "train"
        output_model = job_dir / "model.bin"

        # Get data directory
        if dataset and dataset.processed_blob_prefix:
            # Extract processed data from blob storage to temp dir
            data_dir = job_dir / "data"
            data_dir.mkdir(exist_ok=True)
            self._extract_dataset_from_blobs(dataset.processed_blob_prefix, data_dir)
        else:
            data_dir = job_dir / "data"

        cmd = [str(train_exe), str(data_dir), str(output_model)]

        self._current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

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

                # Parse training output
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

                        best_accuracy = max(best_accuracy, acc)
                        final_loss = loss

                        # Update queue status
                        self.queue.update_status(
                            job.job_id,
                            JobStatus.TRAINING,
                            message=f"Epoch {epoch}: {acc:.2f}%",
                            current_epoch=epoch,
                            current_loss=loss,
                            current_accuracy=acc
                        )

                        # Record training history
                        with get_db_session() as db:
                            history = TrainingHistory(
                                model_id=job.model_id,
                                job_id=job.job_id,
                                epoch=epoch,
                                loss=loss,
                                accuracy=acc
                            )
                            db.add(history)

                            # Update model stats
                            m = db.query(Model).filter_by(id=job.model_id).first()
                            if m:
                                m.epochs_trained = epoch
                                m.best_accuracy = best_accuracy
                                m.final_loss = final_loss

                    except Exception as e:
                        logger.warning("[%s] Parse error: %s", self.worker_id, e)

            self._current_process.wait()

        finally:
            self._current_process = None

    def _extract_dataset_from_blobs(self, blob_prefix: str, output_dir: Path):
        """Extract dataset files from blob storage."""
        # List files with the prefix and extract them
        # Image dataset files
        image_files = [
            "train_images.bin",
            "train_labels.bin",
            "test_images.bin",
            "test_labels.bin",
            "config.json"
        ]
        # Text dataset files
        text_files = [
            "train_inputs.bin",
            "train_targets.bin",
            "test_inputs.bin",
            "test_targets.bin",
            "config.json",
            "vocabulary.json"
        ]

        # Try to extract all possible files (will skip if not found)
        all_files = set(image_files + text_files)
        for filename in all_files:
            blob_key = f"{blob_prefix}/{filename}"
            data = self.blob_store.get(blob_key)
            if data:
                (output_dir / filename).write_bytes(data)


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
