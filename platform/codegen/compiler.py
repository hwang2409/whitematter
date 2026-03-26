import subprocess
from pathlib import Path
from typing import Tuple

from config import PROJECT_ROOT


def compile_training_code(
    generated_dir: Path,
    timeout: int = 300
) -> Tuple[bool, str]:
    try:
        # Build the core library first (in the project root)
        lib_result = subprocess.run(
            ["make", "build/libwhitematter.a"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if lib_result.returncode != 0:
            return False, f"Library build failed:\n{lib_result.stderr}"

        # Override PROJECT_ROOT with absolute path so it works from any directory
        # (the worker runs in a temp dir, not under generated/{job_id}/)
        result = subprocess.run(
            ["make", "all", f"PROJECT_ROOT={PROJECT_ROOT}"],
            cwd=generated_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        with open(generated_dir / "build.log", 'w') as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)

        if result.returncode == 0:
            return True, "Compilation successful"
        else:
            return False, f"Compilation failed:\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "Compilation timed out"
    except Exception as e:
        return False, f"Compilation error: {str(e)}"


def run_training(
    generated_dir: Path,
    data_dir: Path,
    output_model: Path,
    resume_weights: Path = None,
    start_epoch: int = 0,
    timeout: int = 3600
) -> subprocess.Popen:
    train_exe = generated_dir / "train"

    if not train_exe.exists():
        raise FileNotFoundError(f"Training executable not found: {train_exe}")

    cmd = [str(train_exe), str(data_dir), str(output_model)]
    if resume_weights and resume_weights.exists():
        cmd.extend([str(resume_weights), str(start_epoch)])

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=generated_dir
    )

    return process
