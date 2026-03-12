"""Tests for /design routes: suggest, validate, refine, preview-code, help."""

from unittest.mock import patch, MagicMock

import dependencies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_ARCH = {
    "name": "simple_cnn",
    "data_type": "image",
    "input_shape": [3, 32, 32],
    "num_classes": 10,
    "layers": [
        {"type": "conv2d", "params": {"in_channels": 3, "out_channels": 32, "kernel_size": 3, "padding": 1}},
        {"type": "relu", "params": {}},
        {"type": "flatten", "params": {}},
        {"type": "linear", "params": {"in_features": 32 * 32 * 32, "out_features": 10}},
    ],
    "training": {
        "optimizer": {"type": "adam", "params": {"learning_rate": 0.001}},
        "epochs": 10,
        "batch_size": 64
    }
}


# ---------------------------------------------------------------------------
# POST /design/suggest
# ---------------------------------------------------------------------------

class TestSuggestArchitecture:
    def test_suggest_success(self, client):
        """POST /design/suggest returns architecture suggestion."""
        mock_dataset_info = {
            "id": "ds1",
            "data_type": "image",
            "input_shape": [3, 32, 32],
            "num_classes": 10,
            "class_names": ["a", "b"],
            "total_samples": 100,
        }

        mock_result = {
            "architecture": _SIMPLE_ARCH,
            "explanation": "A simple CNN.",
            "raw_response": "",
        }

        with patch.object(dependencies.dataset_service, "get_dataset",
                          return_value=mock_dataset_info), \
             patch.object(dependencies.llm_service, "suggest_architecture",
                          return_value=mock_result), \
             patch.object(dependencies.code_generator, "validate_architecture",
                          return_value={"valid": True, "errors": [], "warnings": []}):
            response = client.post("/design/suggest", json={
                "dataset_id": "ds1",
                "prompt": "Build a simple CNN"
            })

        assert response.status_code == 200
        data = response.json()
        assert "architecture" in data
        assert "explanation" in data
        assert "validation" in data
        assert data["validation"]["valid"] is True

    def test_suggest_dataset_not_found(self, client):
        """POST /design/suggest with unknown dataset returns 404."""
        with patch.object(dependencies.dataset_service, "get_dataset",
                          return_value=None):
            response = client.post("/design/suggest", json={
                "dataset_id": "nonexistent",
                "prompt": "CNN"
            })
        assert response.status_code == 404

    def test_suggest_llm_error(self, client):
        """POST /design/suggest with LLM failure returns 500."""
        mock_dataset_info = {
            "id": "ds1",
            "data_type": "image",
            "input_shape": [3, 32, 32],
            "num_classes": 10,
        }

        with patch.object(dependencies.dataset_service, "get_dataset",
                          return_value=mock_dataset_info), \
             patch.object(dependencies.llm_service, "suggest_architecture",
                          side_effect=RuntimeError("API down")):
            response = client.post("/design/suggest", json={
                "dataset_id": "ds1",
                "prompt": "CNN"
            })
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# POST /design/validate
# ---------------------------------------------------------------------------

class TestValidateArchitecture:
    def test_validate_valid_architecture(self, client):
        """POST /design/validate returns valid=True for good architecture."""
        with patch.object(dependencies.code_generator, "validate_architecture",
                          return_value={"valid": True, "errors": [], "warnings": []}):
            response = client.post("/design/validate", json=_SIMPLE_ARCH)

        assert response.status_code == 200
        assert response.json()["valid"] is True

    def test_validate_invalid_architecture(self, client):
        """POST /design/validate returns errors for bad architecture."""
        with patch.object(dependencies.code_generator, "validate_architecture",
                          return_value={"valid": False, "errors": ["Missing layers"], "warnings": []}):
            response = client.post("/design/validate", json={"layers": []})

        assert response.status_code == 200
        assert response.json()["valid"] is False
        assert len(response.json()["errors"]) > 0


# ---------------------------------------------------------------------------
# POST /design/refine
# ---------------------------------------------------------------------------

class TestRefineArchitecture:
    def test_refine_success(self, client):
        """POST /design/refine returns refined architecture."""
        refined = {
            "architecture": _SIMPLE_ARCH,
            "explanation": "Added dropout",
            "raw_response": ""
        }

        with patch.object(dependencies.llm_service, "refine_architecture", return_value=refined), \
             patch.object(dependencies.code_generator, "validate_architecture",
                          return_value={"valid": True, "errors": [], "warnings": []}):
            response = client.post("/design/refine", json={
                "architecture": _SIMPLE_ARCH,
                "feedback": "Add dropout"
            })

        assert response.status_code == 200
        data = response.json()
        assert "architecture" in data
        assert "explanation" in data

    def test_refine_llm_error(self, client):
        """POST /design/refine with LLM error returns 500."""
        with patch.object(dependencies.llm_service, "refine_architecture",
                          side_effect=ConnectionError("timeout")):
            response = client.post("/design/refine", json={
                "architecture": _SIMPLE_ARCH,
                "feedback": "Add dropout"
            })
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# POST /design/preview-code
# ---------------------------------------------------------------------------

class TestPreviewCode:
    def test_preview_code_success(self, client, temp_dir):
        """POST /design/preview-code generates C++ files."""

        with patch("routes.design._resolve_dataset_config_for_preview",
                   return_value={"data_type": "image", "num_classes": 10, "input_shape": [3, 32, 32]}), \
             patch("routes.design.tempfile.mkdtemp", return_value=str(temp_dir)), \
             patch.object(dependencies.code_generator, "generate") as mock_gen, \
             patch("routes.design.shutil.rmtree"):
            # Write mock files that the route will read
            (temp_dir / "train.cpp").write_text("// train code")
            (temp_dir / "infer.cpp").write_text("// infer code")

            response = client.post("/design/preview-code", json={
                "dataset_id": "ds1",
                "architecture": _SIMPLE_ARCH
            })

        assert response.status_code == 200
        data = response.json()
        assert "train_cpp" in data
        assert "infer_cpp" in data

    def test_preview_code_dataset_not_found(self, client):
        """POST /design/preview-code with unknown dataset returns 404."""
        with patch("routes.design._resolve_dataset_config_for_preview",
                   side_effect=ValueError("Dataset not found")):
            response = client.post("/design/preview-code", json={
                "dataset_id": "nonexistent",
                "architecture": _SIMPLE_ARCH
            })
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /design/help
# ---------------------------------------------------------------------------

class TestDesignHelp:
    def test_help_conv(self, client):
        """POST /design/help about conv returns relevant info."""
        response = client.post("/design/help", json={
            "message": "What is conv?"
        })
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "conv" in data["response"].lower()

    def test_help_dropout(self, client):
        """POST /design/help about dropout returns relevant info."""
        response = client.post("/design/help", json={
            "message": "Explain dropout please"
        })
        assert response.status_code == 200
        assert "dropout" in response.json()["response"].lower()

    def test_help_generic_question(self, client):
        """POST /design/help with generic question returns help."""
        response = client.post("/design/help", json={
            "message": "help me build a model"
        })
        assert response.status_code == 200
        assert "response" in response.json()

    def test_help_text_context(self, client):
        """POST /design/help with text context returns text-specific help."""
        response = client.post("/design/help", json={
            "message": "some random unique question xyz",
            "context": {"dataset_type": "text"}
        })
        assert response.status_code == 200
        resp = response.json()["response"]
        assert "text" in resp.lower() or "language" in resp.lower()

    def test_help_learning_rate(self, client):
        """POST /design/help about learning rate returns relevant info."""
        response = client.post("/design/help", json={
            "message": "What learning rate should I use?"
        })
        assert response.status_code == 200
        assert "learning" in response.json()["response"].lower()

    def test_help_epoch(self, client):
        """POST /design/help about epochs returns relevant info."""
        response = client.post("/design/help", json={
            "message": "How many epochs?"
        })
        assert response.status_code == 200
        assert "epoch" in response.json()["response"].lower()
