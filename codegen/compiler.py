"""
Compiler - handles compilation of generated training code.
"""

import subprocess
from pathlib import Path
from typing import Tuple


def compile_training_code(
    generated_dir: Path,
    timeout: int = 300
) -> Tuple[bool, str]:
    """
    Compile generated training code.

    Args:
        generated_dir: Directory containing train.cpp and Makefile
        timeout: Max compilation time in seconds

    Returns:
        Tuple of (success, output_message)
    """
    try:
        # Run make
        result = subprocess.run(
            ["make", "train"],
            cwd=generated_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Save build log
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
    timeout: int = 3600  # 1 hour default
) -> subprocess.Popen:
    """
    Start training process.

    Args:
        generated_dir: Directory containing compiled 'train' executable
        data_dir: Directory containing processed dataset
        output_model: Path for output model file
        timeout: Not enforced here (handled by caller)

    Returns:
        Popen object for the training process
    """
    train_exe = generated_dir / "train"

    if not train_exe.exists():
        raise FileNotFoundError(f"Training executable not found: {train_exe}")

    process = subprocess.Popen(
        [str(train_exe), str(data_dir), str(output_model)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=generated_dir
    )

    return process
