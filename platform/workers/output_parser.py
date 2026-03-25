import logging
import re
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

ERROR_PATTERNS = [
    re.compile(r"cannot open|Cannot open", re.IGNORECASE),
    re.compile(r"invalid tensor|Invalid tensor", re.IGNORECASE),
    re.compile(r"segmentation fault|segfault", re.IGNORECASE),
    re.compile(r"^error:|Error:", re.IGNORECASE),
    re.compile(r"undefined reference", re.IGNORECASE),
    re.compile(r"fatal error", re.IGNORECASE),
    re.compile(r"out of memory|OOM|malloc", re.IGNORECASE),
    re.compile(r"nan|NaN detected", re.IGNORECASE),
    re.compile(r"shape mismatch|dimension mismatch", re.IGNORECASE),
]


def parse_error_line(line: str) -> dict | None:
    """
    Check if a training output line matches known error patterns.
    Returns {"type": "error", "message": <stripped line>} or None.
    """
    for pattern in ERROR_PATTERNS:
        if pattern.search(line):
            return {"type": "error", "message": line.strip()}
    return None


def parse_training_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a training output line like:
        Epoch 3 | Loss: 0.452 | Test Acc: 87.30%

    Returns dict with epoch, loss, accuracy, or None if line is not a training metric.
    """
    if "Epoch" not in line or "Loss:" not in line:
        return None

    try:
        parts = line.split("|")
        epoch = int(parts[0].split()[1])
        loss = float(parts[1].split(":")[1].strip())
        acc = 0.0

        for p in parts[2:]:
            if "Test Acc:" in p or "Acc:" in p:
                acc = float(p.split(":")[1].strip().rstrip('%'))
                break

        return {"epoch": epoch, "loss": loss, "accuracy": acc}
    except Exception as e:
        logger.warning("Parse error: %s (line: %s)", e, line)
        return None
