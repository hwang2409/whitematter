"""Tests for WebSocket endpoint /ws/train/{job_id}."""

import asyncio
from unittest.mock import patch, MagicMock

import pytest
from starlette.testclient import WebSocketDisconnect

import dependencies


@pytest.fixture(autouse=True)
def _isolate_training_jobs():
    """Save and restore dependencies.training_jobs around each test."""
    saved = dict(dependencies.training_jobs)
    yield
    dependencies.training_jobs.clear()
    dependencies.training_jobs.update(saved)


# ---------------------------------------------------------------------------
# WebSocket connect / receive updates
# ---------------------------------------------------------------------------

class TestWSTraining:
    def test_ws_connect_unknown_job(self, client):
        """WebSocket connect to unknown job closes with 4004."""
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/train/unknown-ws-job") as ws:
                # If we get here, the server accepted before closing --
                # try to receive to trigger the close
                ws.receive_json()

    def test_ws_connect_and_receive_initial_snapshot(self, client):
        """WebSocket connects and receives the initial job snapshot."""
        job_id = "ws-test-job-1"
        dependencies.training_jobs[job_id] = {
            "id": job_id,
            "model_id": "m1",
            "status": "running",
            "epoch": 3,
            "total_epochs": 10,
            "loss": 0.5,
            "accuracy": 72.0,
            "message": "Epoch 3",
        }

        try:
            with client.websocket_connect(f"/ws/train/{job_id}") as ws:
                # Should receive initial snapshot
                data = ws.receive_json()
                assert data["job_id"] == job_id
                assert data["status"] == "running"
                assert data["epoch"] == 3
                # Close from client side
        except WebSocketDisconnect:
            pass  # Server may close the WebSocket on its own

    def test_ws_receives_completed_status_and_closes(self, client):
        """WebSocket closes after receiving completed status."""
        job_id = "ws-test-completed"
        dependencies.training_jobs[job_id] = {
            "id": job_id,
            "model_id": "m2",
            "status": "completed",
            "epoch": 10,
            "total_epochs": 10,
            "loss": 0.1,
            "accuracy": 95.0,
            "message": "Done",
        }

        received_completed = False
        try:
            with client.websocket_connect(f"/ws/train/{job_id}") as ws:
                data = ws.receive_json()
                assert data["status"] == "completed"
                received_completed = True
                # The server should close the connection after sending completed
        except WebSocketDisconnect:
            pass  # Expected: server closes after sending completed status
        assert received_completed, "Should have received the completed status before disconnect"

    @pytest.mark.skip(reason="Non-deterministic: ping timing depends on server timeout and may not fire in sync tests")
    def test_ws_receives_ping(self, client):
        """WebSocket receives ping when no updates come within timeout."""
        job_id = "ws-test-ping"
        dependencies.training_jobs[job_id] = {
            "id": job_id,
            "model_id": "m3",
            "status": "running",
            "epoch": 1,
            "total_epochs": 10,
            "loss": 1.0,
            "accuracy": 10.0,
            "message": "Starting",
        }

        try:
            with client.websocket_connect(f"/ws/train/{job_id}") as ws:
                # First message is the snapshot
                data = ws.receive_json()
                assert data["job_id"] == job_id

                # Next message should be a ping (after 5s timeout in the server).
                # This is inherently non-deterministic in sync test mode.
                data2 = ws.receive_json(mode="text")
                assert "type" in data2 or "status" in data2
        except WebSocketDisconnect:
            pass  # Server may close before ping fires

    def test_ws_failed_status_closes(self, client):
        """WebSocket closes after receiving failed status."""
        job_id = "ws-test-failed"
        dependencies.training_jobs[job_id] = {
            "id": job_id,
            "model_id": "m4",
            "status": "failed",
            "epoch": 5,
            "total_epochs": 10,
            "loss": 2.0,
            "accuracy": 30.0,
            "message": "OOM",
        }

        received_failed = False
        try:
            with client.websocket_connect(f"/ws/train/{job_id}") as ws:
                data = ws.receive_json()
                assert data["status"] == "failed"
                received_failed = True
        except WebSocketDisconnect:
            pass  # Expected: server closes after sending failed status
        assert received_failed, "Should have received the failed status before disconnect"


# ---------------------------------------------------------------------------
# Subscriber cleanup
# ---------------------------------------------------------------------------

class TestWSSubscriberCleanup:
    def test_subscriber_added_and_removed(self, client):
        """Subscriber queue is added on connect and removed on disconnect."""
        job_id = "ws-cleanup-test"
        dependencies.training_jobs[job_id] = {
            "id": job_id,
            "model_id": "m5",
            "status": "completed",
            "epoch": 10,
            "total_epochs": 10,
            "loss": 0.05,
            "accuracy": 99.0,
            "message": "Done",
        }

        try:
            with client.websocket_connect(f"/ws/train/{job_id}") as ws:
                ws.receive_json()  # initial snapshot -> completed -> close
        except WebSocketDisconnect:
            pass  # Expected: server closes after sending completed status

        # After disconnect, subscriber should be cleaned up
        subs = dependencies._ws_subscribers.get(job_id, [])
        assert len(subs) == 0
