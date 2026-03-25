"""
Parse training process stdout lines into structured metrics.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


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
