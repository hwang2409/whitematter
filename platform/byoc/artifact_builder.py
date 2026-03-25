import io
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

from codegen import CodeGenerator
from codegen.templates.makefile import generate_byoc_makefile

logger = logging.getLogger(__name__)


def build_training_tarball(
    model_config: dict,
    dataset_config: dict,
    blob_store,
    project_root: Path = None,
) -> bytes:
    """Build a self-contained tarball for BYOC training on EC2.

    The tarball contains generated training code, all core C++ source files,
    a standalone Makefile, and dataset binaries from the blob store.

    Returns the raw bytes of the tar archive.
    """
    if project_root is None:
        # Default: repo root is three levels up from platform/byoc/
        project_root = Path(__file__).resolve().parent.parent.parent

    core_dir = project_root / "core"

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "whitematter-training"
        staging.mkdir()

        # 1. Generate training code (train.cpp, infer.cpp, architecture.json, Makefile)
        generator = CodeGenerator()
        generator.generate(
            architecture=model_config,
            dataset_config=dataset_config,
            output_dir=staging,
        )

        # 2. Replace generated Makefile with BYOC standalone version
        makefile_path = staging / "Makefile"
        with open(makefile_path, "w") as f:
            f.write(generate_byoc_makefile())

        # 3. Copy all core C++ source files (preserving directory structure)
        dest_core = staging / "core"
        shutil.copytree(
            core_dir,
            dest_core,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

        # 4. Extract dataset from blob store into data/ subdirectory
        data_dir = staging / "data"
        data_dir.mkdir()

        dataset_id = dataset_config.get("dataset_id")
        if dataset_id:
            prefix = f"datasets/{dataset_id}/processed"
            blob_keys = blob_store.list_keys(prefix=prefix)
            for key in blob_keys:
                data = blob_store.get(key)
                if data is None:
                    continue
                # Strip prefix to get the filename (e.g. "config.json")
                filename = key.rsplit("/", 1)[-1]
                (data_dir / filename).write_bytes(data)
                logger.info("Extracted dataset blob %s -> data/%s", key, filename)

        # 5. Package everything into a tar archive
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:") as tar:
            tar.add(str(staging), arcname="whitematter-training")
        buf.seek(0)
        return buf.read()
