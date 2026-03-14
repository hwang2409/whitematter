"""
Training state - WebSocket pub/sub and job snapshot helpers.
"""

import asyncio
import logging
import threading
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# WebSocket subscriber registry for real-time training updates
_ws_subscribers: Dict[str, list] = {}  # job_id -> list of asyncio.Queue
_ws_lock = threading.Lock()
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def capture_event_loop():
    """Capture the running event loop for cross-thread WebSocket notifications."""
    global _event_loop
    _event_loop = asyncio.get_running_loop()


def _get_job_snapshot(job_id: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON-safe dict from training_jobs."""
    # Late import to avoid circular dependency with dependencies.py
    from dependencies import training_jobs

    j = training_jobs.get(job_id)
    if j is None:
        return None
    status = j["status"]
    if hasattr(status, 'value'):
        status = status.value
    return {
        "job_id": j["id"],
        "model_id": j["model_id"],
        "status": status,
        "epoch": j.get("epoch", 0),
        "total_epochs": j.get("total_epochs", 0),
        "loss": j.get("loss", 0.0),
        "accuracy": j.get("accuracy", 0.0),
        "message": j.get("message", ""),
    }


def notify_training_subscribers(job_id: str):
    """Push latest job snapshot to all WebSocket subscribers. Safe to call from background threads."""
    snapshot = _get_job_snapshot(job_id)
    if snapshot is None or _event_loop is None:
        return
    with _ws_lock:
        queues = list(_ws_subscribers.get(job_id, []))
    for q in queues:
        try:
            asyncio.run_coroutine_threadsafe(q.put(snapshot), _event_loop)
        except RuntimeError:
            pass  # event loop closed
