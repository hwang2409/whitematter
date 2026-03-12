"""Tests for codegen/generator.py: CodeGenerator."""

import json
import tempfile
from pathlib import Path

import pytest

from codegen.generator import CodeGenerator, LAYER_TEMPLATES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generator():
    return CodeGenerator()


@pytest.fixture
def output_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# Image architectures
# ---------------------------------------------------------------------------

class TestGenerateImageCode:
    def test_simple_cnn(self, generator, output_dir):
        """Generate code for a simple CNN architecture."""
        architecture = {
            "name": "simple_cnn",
            "layers": [
                {"type": "conv2d", "params": {"in_channels": 3, "out_channels": 32, "kernel_size": 3, "padding": 1}},
                {"type": "batchnorm2d", "params": {"num_features": 32}},
                {"type": "relu", "params": {}},
                {"type": "maxpool2d", "params": {"kernel_size": 2}},
                {"type": "flatten", "params": {}},
                {"type": "linear", "params": {"in_features": 32 * 16 * 16, "out_features": 10}},
            ],
            "training": {
                "optimizer": {"type": "adam", "params": {"learning_rate": 0.001}},
                "scheduler": {"type": "cosine", "params": {"T_max": 50}},
                "epochs": 10,
                "batch_size": 64,
            }
        }
        dataset_config = {
            "data_type": "image",
            "num_classes": 10,
            "input_shape": [3, 32, 32],
        }

        result = generator.generate(architecture, dataset_config, output_dir)

        # Check files were created
        assert (output_dir / "train.cpp").exists()
        assert (output_dir / "infer.cpp").exists()
        assert (output_dir / "Makefile").exists()
        assert (output_dir / "architecture.json").exists()

        # Check train.cpp has expected content
        train_code = (output_dir / "train.cpp").read_text()
        assert "Conv2d" in train_code
        assert "BatchNorm2d" in train_code
        assert "ReLU" in train_code
        assert "MaxPool2d" in train_code
        assert "Linear" in train_code
        assert "Adam" in train_code

    def test_with_dropout(self, generator, output_dir):
        """Generate code with dropout layer."""
        architecture = {
            "layers": [
                {"type": "linear", "params": {"in_features": 784, "out_features": 128}},
                {"type": "relu", "params": {}},
                {"type": "dropout", "params": {"p": 0.5}},
                {"type": "linear", "params": {"in_features": 128, "out_features": 10}},
            ],
            "training": {
                "optimizer": {"type": "sgd", "params": {"learning_rate": 0.01}},
                "epochs": 5,
                "batch_size": 32,
            }
        }
        dataset_config = {"data_type": "image", "num_classes": 10, "input_shape": [1, 28, 28]}

        generator.generate(architecture, dataset_config, output_dir)

        train_code = (output_dir / "train.cpp").read_text()
        assert "Dropout" in train_code
        assert "0.5f" in train_code

    def test_sgd_optimizer(self, generator, output_dir):
        """Generate code with SGD optimizer."""
        architecture = {
            "layers": [
                {"type": "linear", "params": {"in_features": 10, "out_features": 5}},
            ],
            "training": {
                "optimizer": {"type": "sgd", "params": {"learning_rate": 0.01, "momentum": 0.9}},
                "epochs": 5,
                "batch_size": 32,
            }
        }
        dataset_config = {"data_type": "image", "num_classes": 5, "input_shape": [1, 1, 10]}

        generator.generate(architecture, dataset_config, output_dir)

        train_code = (output_dir / "train.cpp").read_text()
        assert "SGD" in train_code


# ---------------------------------------------------------------------------
# Text architectures
# ---------------------------------------------------------------------------

class TestGenerateTextCode:
    def test_lstm_language_model(self, generator, output_dir):
        """Generate code for text language model (uses Transformer internally)."""
        architecture = {
            "name": "char_lstm",
            "layers": [
                {"type": "embedding", "params": {"num_embeddings": 100, "embedding_dim": 64}},
                {"type": "lstm", "params": {"input_size": 64, "hidden_size": 128}},
                {"type": "dropout", "params": {"p": 0.3}},
                {"type": "linear", "params": {"in_features": 128, "out_features": 100}},
            ],
            "training": {
                "optimizer": {"type": "adam", "params": {"learning_rate": 0.002}},
                "epochs": 30,
                "batch_size": 32,
            }
        }
        dataset_config = {
            "data_type": "text",
            "num_classes": 100,
            "input_shape": [128],
            "vocab_size": 100,
            "seq_length": 128,
        }

        generator.generate(architecture, dataset_config, output_dir)

        train_code = (output_dir / "train.cpp").read_text()
        assert "Embedding" in train_code
        # Text models use TransformerLM architecture internally
        assert "Transformer" in train_code


# ---------------------------------------------------------------------------
# Unknown layer types
# ---------------------------------------------------------------------------

class TestUnknownLayers:
    def test_unknown_layer_type_raises(self, generator):
        """Unknown layer type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown layer type"):
            generator._generate_layers([
                {"type": "unknown_layer_xyz", "params": {}}
            ])

    def test_all_known_layers_accepted(self, generator):
        """All layer types in LAYER_TEMPLATES are accepted."""
        for layer_type in LAYER_TEMPLATES:
            if layer_type == "transformer":
                continue  # transformer is skipped in _generate_layers
            # Build params based on template requirements
            params = {}
            if layer_type == "conv2d":
                params = {"in_channels": 3, "out_channels": 16, "kernel_size": 3}
            elif layer_type == "batchnorm2d":
                params = {"num_features": 16}
            elif layer_type == "layernorm":
                params = {"normalized_shape": 64}
            elif layer_type == "linear":
                params = {"in_features": 10, "out_features": 5}
            elif layer_type == "embedding":
                params = {"num_embeddings": 100, "embedding_dim": 64}
            elif layer_type == "lstm":
                params = {"input_size": 64, "hidden_size": 128}
            elif layer_type == "gru":
                params = {"input_size": 64, "hidden_size": 128}
            elif layer_type == "maxpool2d":
                params = {"kernel_size": 2}
            elif layer_type == "avgpool2d":
                params = {"kernel_size": 2}
            elif layer_type == "softmax":
                params = {"dim": -1}

            result = generator._generate_layers([{"type": layer_type, "params": params}])
            assert len(result) > 0


# ---------------------------------------------------------------------------
# Missing params
# ---------------------------------------------------------------------------

class TestMissingParams:
    def test_missing_required_param_raises(self, generator):
        """Missing required parameter raises ValueError."""
        with pytest.raises(ValueError, match="Missing required parameter"):
            generator._generate_layers([
                {"type": "conv2d", "params": {}}  # Missing in_channels, out_channels, kernel_size
            ])

    def test_default_params_applied(self, generator):
        """Default parameters are applied for optional params."""
        # conv2d has defaults for stride=1, padding=0
        result = generator._generate_layers([
            {"type": "conv2d", "params": {"in_channels": 3, "out_channels": 16, "kernel_size": 3}}
        ])
        assert "Conv2d(3, 16, 3, 1, 0)" in result

    def test_dropout_default_p(self, generator):
        """Dropout uses default p=0.5."""
        result = generator._generate_layers([
            {"type": "dropout", "params": {}}
        ])
        assert "0.5f" in result

    def test_parameter_aliases(self, generator):
        """Parameter aliases (vocab_size -> num_embeddings) work."""
        result = generator._generate_layers([
            {"type": "embedding", "params": {"vocab_size": 200, "embed_dim": 64}}
        ])
        assert "Embedding(200, 64)" in result


# ---------------------------------------------------------------------------
# Optimizer / scheduler generation
# ---------------------------------------------------------------------------

class TestOptimizerScheduler:
    def test_unknown_optimizer_raises(self, generator):
        """Unknown optimizer type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            generator._generate_optimizer({"type": "nonexistent", "params": {}})

    def test_adam_optimizer(self, generator):
        """Adam optimizer generates correct code."""
        code = generator._generate_optimizer({
            "type": "adam",
            "params": {"learning_rate": 0.001}
        })
        assert "Adam" in code
        assert "0.001f" in code

    def test_none_scheduler(self, generator):
        """None scheduler generates empty string."""
        code = generator._generate_scheduler({"type": "none", "params": {}})
        assert code == ""

    def test_step_scheduler(self, generator):
        """Step scheduler generates StepLR code."""
        code = generator._generate_scheduler({
            "type": "step",
            "params": {"step_size": 10, "gamma": 0.1}
        })
        assert "StepLR" in code

    def test_cosine_scheduler(self, generator):
        """Cosine scheduler generates CosineAnnealingLR code."""
        code = generator._generate_scheduler({
            "type": "cosine",
            "params": {"T_max": 50}
        })
        assert "CosineAnnealing" in code

    def test_unknown_scheduler_returns_empty(self, generator):
        """Unknown scheduler type returns empty string."""
        code = generator._generate_scheduler({"type": "unknown_sched", "params": {}})
        assert code == ""


# ---------------------------------------------------------------------------
# validate_architecture
# ---------------------------------------------------------------------------

class TestValidateArchitecture:
    def test_valid_architecture(self, generator):
        """Valid architecture passes validation."""
        arch = {
            "layers": [
                {"type": "linear", "params": {"in_features": 10, "out_features": 5}},
            ],
            "training": {"optimizer": {"type": "adam"}}
        }
        result = generator.validate_architecture(arch)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_missing_layers(self, generator):
        """Architecture without layers fails validation."""
        result = generator.validate_architecture({})
        assert result["valid"] is False
        assert any("layers" in e.lower() for e in result["errors"])

    def test_empty_layers(self, generator):
        """Architecture with empty layers fails validation."""
        result = generator.validate_architecture({"layers": []})
        assert result["valid"] is False

    def test_unknown_layer_type_validation(self, generator):
        """Architecture with unknown layer type fails validation."""
        result = generator.validate_architecture({
            "layers": [{"type": "super_magic_layer"}]
        })
        assert result["valid"] is False
        assert any("unknown" in e.lower() for e in result["errors"])

    def test_unknown_optimizer_validation(self, generator):
        """Architecture with unknown optimizer fails validation."""
        result = generator.validate_architecture({
            "layers": [{"type": "linear", "params": {}}],
            "training": {"optimizer": {"type": "super_opt"}}
        })
        assert result["valid"] is False

    def test_many_layers_warning(self, generator):
        """Architecture with >50 layers generates warning."""
        layers = [{"type": "relu", "params": {}} for _ in range(51)]
        result = generator.validate_architecture({"layers": layers})
        assert result["valid"] is True
        assert len(result["warnings"]) > 0

    def test_layer_missing_type(self, generator):
        """Layer without type field fails validation."""
        result = generator.validate_architecture({
            "layers": [{"params": {}}]
        })
        assert result["valid"] is False
        assert any("type" in e.lower() for e in result["errors"])
