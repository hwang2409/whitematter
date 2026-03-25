import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class TrainingExecutor:
    """Handles the execution pipeline: codegen -> compile -> train."""

    def __init__(self, blob_store, project_root: Path):
        self.blob_store = blob_store
        self.project_root = project_root

    def generate_code(
        self,
        job_dir: Path,
        dataset_info: Optional[dict],
        arch_config: dict,
        train_config: dict
    ):
        """Generate C++ training code.

        Args:
            dataset_info: Plain dict with keys: processed_blob_prefix, input_shape,
                          num_classes, class_names.  None if no dataset.
        """
        from codegen import CodeGenerator

        generator = CodeGenerator()

        if dataset_info:
            processed_blob_prefix = dataset_info.get("processed_blob_prefix")
            if processed_blob_prefix:
                config_blob = self.blob_store.get(f"{processed_blob_prefix}/config.json")
                if config_blob:
                    dataset_config = json.loads(config_blob.decode())
                else:
                    dataset_config = {
                        "input_shape": dataset_info.get("input_shape"),
                        "num_classes": dataset_info.get("num_classes"),
                        "class_names": dataset_info.get("class_names"),
                        "mean": [0.5],
                        "std": [0.5]
                    }
            else:
                dataset_config = {
                    "input_shape": dataset_info.get("input_shape"),
                    "num_classes": dataset_info.get("num_classes"),
                    "class_names": dataset_info.get("class_names"),
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

    def compile(self, job_dir: Path) -> tuple:
        """Compile the generated training code."""
        from codegen import compile_training_code
        return compile_training_code(job_dir)

    def run_training(self, job_dir: Path, data_dir: Path) -> subprocess.Popen:
        """Start the training process and return the Popen handle."""
        train_exe = job_dir / "train"
        output_model = job_dir / "model.bin"

        cmd = [str(train_exe), str(data_dir), str(output_model)]

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

    def extract_dataset_from_blobs(self, blob_prefix: str, output_dir: Path):
        """Extract dataset files from blob storage."""
        image_files = [
            "train_images.bin",
            "train_labels.bin",
            "test_images.bin",
            "test_labels.bin",
            "config.json"
        ]
        text_files = [
            "train_inputs.bin",
            "train_targets.bin",
            "test_inputs.bin",
            "test_targets.bin",
            "config.json",
            "vocabulary.json"
        ]

        all_files = set(image_files + text_files)
        for filename in all_files:
            blob_key = f"{blob_prefix}/{filename}"
            data = self.blob_store.get(blob_key)
            if data:
                (output_dir / filename).write_bytes(data)
