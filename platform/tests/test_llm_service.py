"""Tests for llm/service.py: MockLLMService, get_llm_service, LLMService."""

import os
from unittest.mock import patch, MagicMock

import pytest

from llm.service import LLMService, MockLLMService, get_llm_service


# ---------------------------------------------------------------------------
# MockLLMService
# ---------------------------------------------------------------------------

class TestMockLLMService:
    def test_suggest_image_architecture(self):
        """MockLLMService returns a valid image CNN architecture."""
        svc = MockLLMService()
        result = svc.suggest_architecture(
            dataset_info={
                "data_type": "image",
                "input_shape": [3, 32, 32],
                "num_classes": 10,
                "class_names": ["a"] * 10,
                "total_samples": 1000,
            },
            user_prompt="Build a CNN"
        )

        assert "architecture" in result
        assert "explanation" in result
        arch = result["architecture"]
        assert arch["data_type"] == "image"
        assert "layers" in arch
        assert len(arch["layers"]) > 0
        assert "training" in arch

    def test_suggest_text_architecture_transformer(self):
        """MockLLMService returns a transformer architecture for text by default."""
        svc = MockLLMService()
        result = svc.suggest_architecture(
            dataset_info={
                "data_type": "text",
                "input_shape": [128],
                "num_classes": 100,
            },
            user_prompt="Build a language model"
        )

        arch = result["architecture"]
        assert arch["data_type"] == "text"
        assert "layers" in arch
        # Should default to transformer
        layer_types = [l["type"] for l in arch["layers"]]
        assert "transformer" in layer_types or "embedding" in layer_types

    def test_suggest_text_architecture_lstm(self):
        """MockLLMService returns an LSTM architecture when requested."""
        svc = MockLLMService()
        result = svc.suggest_architecture(
            dataset_info={
                "data_type": "text",
                "input_shape": [128],
                "num_classes": 100,
            },
            user_prompt="Build an LSTM model"
        )

        arch = result["architecture"]
        layer_types = [l["type"] for l in arch["layers"]]
        assert "lstm" in layer_types

    def test_suggest_tabular_architecture(self):
        """MockLLMService returns an MLP for tabular data."""
        svc = MockLLMService()
        result = svc.suggest_architecture(
            dataset_info={
                "data_type": "tabular",
                "input_shape": [10],
                "num_classes": 3,
            },
            user_prompt="Classify tabular data"
        )

        arch = result["architecture"]
        assert arch["data_type"] == "tabular"
        layer_types = [l["type"] for l in arch["layers"]]
        assert "linear" in layer_types

    def test_refine_returns_same_architecture(self):
        """MockLLMService.refine_architecture returns architecture unchanged."""
        svc = MockLLMService()
        original = {"layers": [{"type": "linear", "params": {}}]}
        result = svc.refine_architecture(
            current_architecture=original,
            feedback="Add dropout"
        )
        assert result["architecture"] == original
        assert "explanation" in result

    def test_suggest_architecture_has_training_config(self):
        """MockLLMService architecture includes training config."""
        svc = MockLLMService()
        result = svc.suggest_architecture(
            dataset_info={
                "data_type": "image",
                "input_shape": [1, 28, 28],
                "num_classes": 10,
            },
            user_prompt="MNIST classifier"
        )
        training = result["architecture"]["training"]
        assert "optimizer" in training
        assert "epochs" in training
        assert "batch_size" in training


# ---------------------------------------------------------------------------
# get_llm_service
# ---------------------------------------------------------------------------

class TestGetLLMService:
    def test_returns_mock_when_no_api_key(self):
        """get_llm_service returns MockLLMService when ANTHROPIC_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove the key if present
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                svc = get_llm_service(api_key=None)
        assert isinstance(svc, MockLLMService)

    def test_returns_llm_service_with_api_key(self):
        """get_llm_service returns LLMService when API key is provided."""
        svc = get_llm_service(api_key="sk-test-key")
        assert isinstance(svc, LLMService)
        assert svc.api_key == "sk-test-key"


# ---------------------------------------------------------------------------
# LLMService._extract_json
# ---------------------------------------------------------------------------

class TestLLMServiceExtractJson:
    def setup_method(self):
        self.svc = LLMService(api_key="test-key")

    def test_extract_from_code_block(self):
        """Extract JSON from markdown code block."""
        text = 'Here is the architecture:\n```json\n{"layers": [{"type": "linear"}]}\n```\nDone.'
        result = self.svc._extract_json(text)
        assert result == {"layers": [{"type": "linear"}]}

    def test_extract_raw_json(self):
        """Extract JSON without code block."""
        text = 'Architecture: {"layers": [{"type": "relu"}]}'
        result = self.svc._extract_json(text)
        assert result == {"layers": [{"type": "relu"}]}

    def test_extract_invalid_json_raises(self):
        """Raise ValueError when no valid JSON found."""
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            self.svc._extract_json("No JSON here at all")


# ---------------------------------------------------------------------------
# LLMService._extract_explanation
# ---------------------------------------------------------------------------

class TestLLMServiceExtractExplanation:
    def setup_method(self):
        self.svc = LLMService(api_key="test-key")

    def test_explanation_before_json(self):
        """Extract text before JSON block."""
        text = 'This is the explanation.\n```json\n{"layers": []}\n```'
        result = self.svc._extract_explanation(text)
        assert "explanation" in result.lower()

    def test_no_explanation(self):
        """Return empty string if response starts with JSON."""
        text = '{"layers": []}'
        result = self.svc._extract_explanation(text)
        assert result == ""
