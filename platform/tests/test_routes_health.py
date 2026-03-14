"""Tests for GET /health and related health/config endpoints."""


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_root_returns_server_info(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Whitematter Model Server"
    assert "version" in data


def test_config_datasets(client):
    response = client.get("/config/datasets")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    ids = [d["id"] for d in data["datasets"]]
    assert "cifar10" in ids
    assert "mnist" in ids


def test_config_layers(client):
    response = client.get("/config/layers")
    assert response.status_code == 200
    data = response.json()
    assert "layers" in data
    assert len(data["layers"]) > 0


def test_config_optimizers(client):
    response = client.get("/config/optimizers")
    assert response.status_code == 200
    data = response.json()
    assert "optimizers" in data


def test_config_schedulers(client):
    response = client.get("/config/schedulers")
    assert response.status_code == 200
    assert "schedulers" in response.json()


def test_config_augmentations(client):
    response = client.get("/config/augmentations")
    assert response.status_code == 200
    assert "augmentations" in response.json()


def test_config_presets(client):
    response = client.get("/config/presets")
    assert response.status_code == 200
    assert "presets" in response.json()


def test_config_preset_not_found(client):
    response = client.get("/config/presets/nonexistent-preset-xyz")
    assert response.status_code == 404


def test_workers_status(client):
    from unittest.mock import patch, MagicMock

    mock_queue = MagicMock()
    mock_queue.get_pending_count.return_value = 0
    mock_queue.get_active_jobs.return_value = []

    with patch("routes.health.training_jobs", {}):
        # Force the ImportError path to use legacy in-memory tracking
        with patch.dict("sys.modules", {"workers": None}):
            response = client.get("/workers/status")
    assert response.status_code == 200
    data = response.json()
    assert "pending_jobs" in data
    assert "active_jobs" in data
