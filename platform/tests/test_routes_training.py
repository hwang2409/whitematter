"""Tests for /train routes: start, status, cancel, custom training."""

from unittest.mock import patch, MagicMock

import pytest

import dependencies


@pytest.fixture(autouse=True)
def _isolate_training_jobs():
    """Save and restore dependencies.training_jobs around each test."""
    saved = dict(dependencies.training_jobs)
    yield
    dependencies.training_jobs.clear()
    dependencies.training_jobs.update(saved)


# ---------------------------------------------------------------------------
# POST /train
# ---------------------------------------------------------------------------

class TestStartTraining:
    def test_start_training_cifar10(self, client):
        """POST /train with cifar10 dataset starts a training job."""

        mock_threading = MagicMock()
        mock_thread_instance = MagicMock()
        mock_threading.Thread.return_value = mock_thread_instance

        with patch("routes.training.threading", mock_threading), \
             patch("routes.training.save_model_metadata"):

            response = client.post("/train", json={
                "dataset": "cifar10",
                "epochs": 5,
                "preset": "simple_cnn_cifar10",
            })

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "model_id" in data
        assert data["status"] == "pending"
        assert data["total_epochs"] == 5

    def test_start_training_unknown_dataset(self, client):
        """POST /train with unknown dataset returns 400."""
        response = client.post("/train", json={
            "dataset": "nonexistent_dataset",
            "epochs": 5,
        })
        assert response.status_code == 400
        assert "Unknown dataset" in response.json()["detail"]

    def test_start_training_no_layers_no_preset(self, client):
        """POST /train without preset or layers returns 400."""
        response = client.post("/train", json={
            "dataset": "cifar10",
            "epochs": 5,
        })
        assert response.status_code == 400
        assert "preset or layers" in response.json()["detail"].lower()

    def test_start_training_unknown_preset(self, client):
        """POST /train with unknown preset returns 400."""
        response = client.post("/train", json={
            "dataset": "cifar10",
            "epochs": 5,
            "preset": "nonexistent_preset_xyz"
        })
        assert response.status_code == 400

    def test_start_training_with_layers(self, client):
        """POST /train with custom layers starts training."""
        mock_threading = MagicMock()
        mock_thread_instance = MagicMock()
        mock_threading.Thread.return_value = mock_thread_instance

        with patch("routes.training.threading", mock_threading), \
             patch("routes.training.save_model_metadata"):

            response = client.post("/train", json={
                "dataset": "mnist",
                "epochs": 3,
                "layers": [
                    {"type": "conv2d", "params": {"in_channels": 1, "out_channels": 16, "kernel_size": 3, "padding": 1}},
                    {"type": "relu", "params": {}},
                    {"type": "flatten", "params": {}},
                    {"type": "linear", "params": {"in_features": 16 * 28 * 28, "out_features": 10}},
                ]
            })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"


# ---------------------------------------------------------------------------
# GET /train/{job_id}
# ---------------------------------------------------------------------------

class TestGetTrainingStatus:
    def test_get_status_existing_job(self, client):
        """GET /train/{job_id} returns status for existing job."""
        dependencies.training_jobs["test-job-1"] = {
            "id": "test-job-1",
            "model_id": "model-1",
            "status": "running",
            "epoch": 3,
            "total_epochs": 10,
            "loss": 0.5,
            "accuracy": 72.0,
            "message": "Epoch 3: 72.00%",
        }

        response = client.get("/train/test-job-1")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-1"
        assert data["status"] == "running"
        assert data["epoch"] == 3
        assert data["accuracy"] == 72.0

    def test_get_status_nonexistent_job(self, client):
        """GET /train/{job_id} returns 404 for unknown job."""
        response = client.get("/train/nonexistent-job-xyz")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /train/{job_id}
# ---------------------------------------------------------------------------

class TestCancelTraining:
    def test_cancel_running_job(self, client):
        """DELETE /train/{job_id} cancels a running job."""
        mock_process = MagicMock()
        dependencies.training_jobs["cancel-job"] = {
            "id": "cancel-job",
            "model_id": "model-c",
            "status": "running",
            "epoch": 2,
            "total_epochs": 10,
            "loss": 0.3,
            "accuracy": 50.0,
            "message": "",
            "cancelled": False,
            "process": mock_process,
        }

        response = client.delete("/train/cancel-job")
        assert response.status_code == 200
        assert dependencies.training_jobs["cancel-job"]["cancelled"] is True
        mock_process.terminate.assert_called_once()

    def test_cancel_completed_job(self, client):
        """DELETE /train/{job_id} on a completed job returns 400."""
        dependencies.training_jobs["done-job"] = {
            "id": "done-job",
            "model_id": "m",
            "status": "completed",
            "epoch": 10,
            "total_epochs": 10,
            "loss": 0.1,
            "accuracy": 95.0,
            "message": "",
            "cancelled": False,
        }

        response = client.delete("/train/done-job")
        assert response.status_code == 400
        assert "not running" in response.json()["detail"].lower()

    def test_cancel_nonexistent_job(self, client):
        """DELETE /train/{job_id} returns 404 for unknown job."""
        response = client.delete("/train/nonexistent-job")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /train/custom
# ---------------------------------------------------------------------------

class TestCustomTraining:
    def test_custom_training_dataset_not_found(self, client):
        """POST /train/custom with unknown dataset returns 404."""
        with patch.object(dependencies.dataset_service, "get_dataset", return_value=None):
            response = client.post("/train/custom", json={
                "dataset_id": "nonexistent-ds",
                "architecture": {
                    "layers": [{"type": "linear", "params": {"in_features": 10, "out_features": 5}}],
                    "training": {"epochs": 5}
                }
            })

        assert response.status_code == 404

    def test_custom_training_invalid_architecture(self, client):
        """POST /train/custom with invalid architecture returns 400."""
        mock_ds = {"id": "ds-1", "name": "testds", "status": "ready"}

        with patch.object(dependencies.dataset_service, "get_dataset", return_value=mock_ds), \
             patch.object(dependencies.code_generator, "validate_architecture",
                          return_value={"valid": False, "errors": ["No layers"], "warnings": []}):
            response = client.post("/train/custom", json={
                "dataset_id": "ds-1",
                "architecture": {"layers": []}
            })

        assert response.status_code == 400
        assert "Invalid architecture" in response.json()["detail"]

    def test_custom_training_success(self, client):
        """POST /train/custom starts a custom training job."""
        mock_ds = {"id": "ds-1", "name": "my_dataset", "status": "ready"}

        mock_threading = MagicMock()
        mock_thread_instance = MagicMock()
        mock_threading.Thread.return_value = mock_thread_instance

        with patch.object(dependencies.dataset_service, "get_dataset", return_value=mock_ds), \
             patch.object(dependencies.code_generator, "validate_architecture",
                          return_value={"valid": True, "errors": [], "warnings": []}), \
             patch("routes.training.save_model_metadata"), \
             patch("routes.training.threading", mock_threading):

            response = client.post("/train/custom", json={
                "dataset_id": "ds-1",
                "architecture": {
                    "layers": [
                        {"type": "linear", "params": {"in_features": 10, "out_features": 5}}
                    ],
                    "training": {"epochs": 5, "batch_size": 32}
                }
            })

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
