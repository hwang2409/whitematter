"""Tests for /predict and /api/{model_id}/* prediction routes."""

import io
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image

from schemas import ModelMetadata, TrainStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_bytes(w=32, h=32, mode="RGB"):
    """Create a minimal in-memory image and return its bytes."""
    img = Image.new(mode, (w, h), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _completed_model(model_id="m1", dataset="cifar10", arch="custom"):
    return ModelMetadata(
        id=model_id,
        name="test_model",
        dataset=dataset,
        architecture=arch,
        created_at="2025-01-01",
        epochs_trained=10,
        best_accuracy=90.0,
        status=TrainStatus.COMPLETED,
    )


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_model_not_found(self, client):
        """POST /predict with unknown model_id returns 404."""
        with patch("routes.predict.load_model_metadata", return_value=None):
            img = _make_image_bytes()
            response = client.post(
                "/predict?model_id=nonexistent",
                files={"file": ("test.png", img, "image/png")},
            )
        assert response.status_code == 404

    def test_predict_model_not_ready(self, client):
        """POST /predict with a model still training returns 400."""
        meta = _completed_model()
        meta.status = TrainStatus.RUNNING

        with patch("routes.predict.load_model_metadata", return_value=meta):
            response = client.post(
                "/predict?model_id=m1",
                files={"file": ("test.png", _make_image_bytes(), "image/png")},
            )
        assert response.status_code == 400
        assert "not ready" in response.json()["detail"].lower()

    def test_predict_success(self, client):
        """POST /predict with valid model returns prediction."""
        meta = _completed_model()
        mock_model = MagicMock()
        mock_model.predict_class.return_value = 3
        mock_model.predict_proba.return_value = np.array(
            [0.01] * 3 + [0.9] + [0.01] * 6, dtype=np.float32
        )

        with patch("routes.predict.load_model_metadata", return_value=meta), \
             patch("routes.predict.get_loaded_model", return_value=mock_model), \
             patch("routes.predict.preprocess_image",
                   return_value=np.zeros((1, 3, 32, 32), dtype=np.float32)):
            response = client.post(
                "/predict?model_id=m1",
                files={"file": ("test.png", _make_image_bytes(), "image/png")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "m1"
        assert data["predicted_class"] == 3
        assert "confidence" in data
        assert "probabilities" in data


# ---------------------------------------------------------------------------
# POST /api/{model_id}/predict
# ---------------------------------------------------------------------------

class TestApiPredict:
    def test_api_predict_delegates(self, client):
        """POST /api/{model_id}/predict calls the main predict function."""
        meta = _completed_model()
        mock_model = MagicMock()
        mock_model.predict_class.return_value = 0
        mock_model.predict_proba.return_value = np.array(
            [0.95] + [0.005] * 9, dtype=np.float32
        )

        with patch("routes.predict.load_model_metadata", return_value=meta), \
             patch("routes.predict.get_loaded_model", return_value=mock_model), \
             patch("routes.predict.preprocess_image",
                   return_value=np.zeros((1, 3, 32, 32), dtype=np.float32)):
            response = client.post(
                "/api/m1/predict",
                files={"file": ("test.png", _make_image_bytes(), "image/png")},
            )

        assert response.status_code == 200
        assert response.json()["predicted_class"] == 0


# ---------------------------------------------------------------------------
# POST /api/{model_id}/generate
# ---------------------------------------------------------------------------

class TestGenerateText:
    def test_generate_model_not_found(self, client):
        """POST /api/{model_id}/generate with unknown model returns 404."""
        with patch("routes.predict.load_model_metadata", return_value=None):
            response = client.post("/api/m1/generate", json={
                "prompt": "Hello",
                "max_tokens": 50,
                "temperature": 0.8
            })
        assert response.status_code == 404

    def test_generate_not_language_model(self, client):
        """POST /api/{model_id}/generate with image model returns 400."""
        meta = _completed_model(dataset="cifar10")
        with patch("routes.predict.load_model_metadata", return_value=meta):
            response = client.post("/api/m1/generate", json={
                "prompt": "Hello",
                "max_tokens": 50,
            })
        assert response.status_code == 400
        assert "not a language model" in response.json()["detail"].lower()

    def test_generate_model_not_ready(self, client):
        """POST /api/{model_id}/generate with training model returns 400."""
        meta = _completed_model(dataset="custom:ds1")
        meta.status = TrainStatus.PENDING
        with patch("routes.predict.load_model_metadata", return_value=meta):
            response = client.post("/api/m1/generate", json={
                "prompt": "Hello",
            })
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/{model_id}/info
# ---------------------------------------------------------------------------

class TestApiModelInfo:
    def test_model_info_found(self, client):
        """GET /api/{model_id}/info returns model metadata."""
        meta = _completed_model()

        with patch("routes.predict.load_model_metadata", return_value=meta):
            response = client.get("/api/m1/info")

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "m1"
        assert data["name"] == "test_model"
        assert data["dataset"] == "cifar10"
        assert data["accuracy"] == 90.0
        assert "classes" in data
        assert "input_shape" in data

    def test_model_info_not_found(self, client):
        """GET /api/{model_id}/info returns 404 for unknown model."""
        with patch("routes.predict.load_model_metadata", return_value=None):
            response = client.get("/api/nonexistent/info")

        assert response.status_code == 404
