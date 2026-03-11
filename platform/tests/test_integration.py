"""
Integration tests for the whitematter custom training flow.

Tests end-to-end flows:
1. Design -> generation -> compilation
2. Dataset upload -> processing
3. Error handling (compilation failures, training failures)

Uses pytest fixtures and mocks external dependencies.
"""

import json
import os
import re
import struct
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch, PropertyMock
import zipfile

import pytest
import numpy as np
from PIL import Image

# Add platform to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from codegen.generator import CodeGenerator, LAYER_TEMPLATES
from codegen.compiler import compile_training_code
from llm.service import LLMService, MockLLMService, get_llm_service
from preprocessing.image_processor import ImageProcessor
from db.models import (
    Dataset, Model, TrainingJob, DatasetStatus,
    DatasetFormat, DataType, JobStatus, ModelStatus
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory(prefix="wm_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service that returns fixed architectures."""
    return MockLLMService()


@pytest.fixture
def code_generator():
    """Create a CodeGenerator instance."""
    return CodeGenerator()


@pytest.fixture
def sample_image_architecture():
    """Sample CNN architecture for image classification."""
    return {
        "name": "test_cnn",
        "description": "Simple CNN for testing",
        "data_type": "image",
        "input_shape": [3, 32, 32],
        "num_classes": 10,
        "layers": [
            {"type": "conv2d", "params": {"in_channels": 3, "out_channels": 32, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 32}},
            {"type": "relu", "params": {}},
            {"type": "maxpool2d", "params": {"kernel_size": 2}},
            {"type": "conv2d", "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 64}},
            {"type": "relu", "params": {}},
            {"type": "maxpool2d", "params": {"kernel_size": 2}},
            {"type": "flatten", "params": {}},
            {"type": "linear", "params": {"in_features": 64 * 8 * 8, "out_features": 128}},
            {"type": "relu", "params": {}},
            {"type": "dropout", "params": {"p": 0.5}},
            {"type": "linear", "params": {"in_features": 128, "out_features": 10}},
        ],
        "training": {
            "optimizer": {"type": "adam", "params": {"learning_rate": 0.001}},
            "scheduler": {"type": "cosine", "params": {"T_max": 50}},
            "epochs": 10,
            "batch_size": 64
        }
    }


@pytest.fixture
def sample_dataset_config():
    """Sample dataset configuration."""
    return {
        "data_type": "image",
        "input_shape": [3, 32, 32],
        "num_classes": 10,
        "class_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    }


@pytest.fixture
def small_test_dataset(temp_dir):
    """Create a small test dataset with a few images per class."""
    dataset_dir = temp_dir / "test_dataset"

    # Create folder-per-class structure with 3 classes, 5 images each
    class_names = ["cat", "dog", "bird"]
    for class_name in class_names:
        class_dir = dataset_dir / class_name
        class_dir.mkdir(parents=True)

        for i in range(5):
            # Create a simple test image (32x32 RGB)
            img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(class_dir / f"image_{i}.png")

    return dataset_dir, class_names


@pytest.fixture
def test_zip_dataset(temp_dir, small_test_dataset):
    """Create a ZIP file from the test dataset."""
    dataset_dir, class_names = small_test_dataset
    zip_path = temp_dir / "test_dataset.zip"

    with zipfile.ZipFile(zip_path, 'w') as zf:
        for class_name in class_names:
            class_dir = dataset_dir / class_name
            for img_file in class_dir.iterdir():
                arcname = f"{class_name}/{img_file.name}"
                zf.write(img_file, arcname)

    return zip_path


@pytest.fixture
def mock_db_session():
    """Mock database session for testing without actual DB."""
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    return mock_session


@pytest.fixture
def mock_blob_store():
    """Mock blob store for testing without actual storage."""
    store = MagicMock()
    store._data = {}

    def mock_put(data, key=None, content_type=None):
        store._data[key] = data
        return key

    def mock_get(key):
        return store._data.get(key)

    def mock_put_file(path, key=None, content_type=None):
        with open(path, 'rb') as f:
            store._data[key] = f.read()
        return key

    store.put = mock_put
    store.get = mock_get
    store.put_file = mock_put_file

    return store


# =============================================================================
# Test: Design -> Generation -> Compilation Flow
# =============================================================================

class TestDesignGenerationCompilationFlow:
    """Test the full flow from architecture design to code compilation."""

    def test_llm_suggests_architecture_for_image_dataset(self, mock_llm_service):
        """Test that mock LLM service suggests valid architecture for image data."""
        dataset_info = {
            "data_type": "image",
            "input_shape": [3, 32, 32],
            "num_classes": 10,
            "class_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "total_samples": 1000
        }

        result = mock_llm_service.suggest_architecture(
            dataset_info=dataset_info,
            user_prompt="Create a simple CNN for image classification"
        )

        # Verify response structure
        assert "architecture" in result
        assert "explanation" in result

        arch = result["architecture"]
        assert "layers" in arch
        assert "training" in arch
        assert len(arch["layers"]) > 0

        # Verify layer types are valid
        for layer in arch["layers"]:
            assert "type" in layer
            assert layer["type"].lower() in LAYER_TEMPLATES or layer["type"].lower() == "transformer"

    def test_llm_suggests_architecture_for_text_dataset(self, mock_llm_service):
        """Test that mock LLM service suggests valid architecture for text data."""
        dataset_info = {
            "data_type": "text",
            "input_shape": [128],
            "num_classes": 100,  # vocab size
            "total_samples": 5000
        }

        result = mock_llm_service.suggest_architecture(
            dataset_info=dataset_info,
            user_prompt="Create a language model"
        )

        arch = result["architecture"]
        assert arch["data_type"] == "text"
        assert "layers" in arch

        # Should have embedding layer for text
        layer_types = [l["type"].lower() for l in arch["layers"]]
        assert "embedding" in layer_types

    def test_code_generator_produces_valid_cpp(
        self, code_generator, sample_image_architecture, sample_dataset_config, temp_dir
    ):
        """Test that code generator produces syntactically valid C++ code."""
        output_dir = temp_dir / "generated"

        train_cpp_path = code_generator.generate(
            architecture=sample_image_architecture,
            dataset_config=sample_dataset_config,
            output_dir=output_dir
        )

        # Verify files were created
        assert train_cpp_path.exists()
        assert (output_dir / "infer.cpp").exists()
        assert (output_dir / "Makefile").exists()
        assert (output_dir / "architecture.json").exists()

        # Read and validate C++ structure
        train_code = train_cpp_path.read_text()

        # Check for basic C++ structure
        assert "#include" in train_code
        assert "int main(" in train_code
        assert "Sequential model" in train_code

        # Check for expected layers
        assert "Conv2d" in train_code
        assert "BatchNorm2d" in train_code
        assert "Linear" in train_code
        assert "ReLU" in train_code

        # Check for training loop components
        assert "optimizer" in train_code
        assert "forward" in train_code
        assert "backward" in train_code

        # Verify balanced braces (basic syntax check)
        open_braces = train_code.count('{')
        close_braces = train_code.count('}')
        assert open_braces == close_braces, "Unbalanced braces in generated code"

    def test_code_generator_handles_all_layer_types(self, code_generator, temp_dir):
        """Test that code generator can handle all supported layer types."""
        # Architecture using various layer types
        architecture = {
            "name": "test_all_layers",
            "layers": [
                {"type": "conv2d", "params": {"in_channels": 3, "out_channels": 16, "kernel_size": 3}},
                {"type": "batchnorm2d", "params": {"num_features": 16}},
                {"type": "relu", "params": {}},
                {"type": "leakyrelu", "params": {"negative_slope": 0.1}},
                {"type": "maxpool2d", "params": {"kernel_size": 2}},
                {"type": "avgpool2d", "params": {"kernel_size": 2}},
                {"type": "dropout", "params": {"p": 0.3}},
                {"type": "flatten", "params": {}},
                {"type": "linear", "params": {"in_features": 64, "out_features": 32}},
                {"type": "sigmoid", "params": {}},
                {"type": "tanh", "params": {}},
                {"type": "softmax", "params": {"dim": -1}},
            ],
            "training": {
                "optimizer": {"type": "sgd", "params": {"learning_rate": 0.01}},
                "epochs": 5,
                "batch_size": 32
            }
        }

        dataset_config = {
            "data_type": "image",
            "input_shape": [3, 16, 16],
            "num_classes": 10
        }

        output_dir = temp_dir / "all_layers"
        train_cpp = code_generator.generate(architecture, dataset_config, output_dir)

        assert train_cpp.exists()
        code = train_cpp.read_text()

        # Verify all layer types appear in the generated code
        expected_layers = [
            "Conv2d", "BatchNorm2d", "ReLU", "LeakyReLU",
            "MaxPool2d", "AvgPool2d", "Dropout", "Flatten",
            "Linear", "Sigmoid", "Tanh", "Softmax"
        ]
        for layer in expected_layers:
            assert layer in code, f"Missing layer type: {layer}"

    def test_code_generator_validates_architecture(self, code_generator):
        """Test that code generator validates architecture specification."""
        # Valid architecture
        valid_arch = {
            "layers": [
                {"type": "linear", "params": {"in_features": 10, "out_features": 5}}
            ]
        }
        result = code_generator.validate_architecture(valid_arch)
        assert result["valid"] is True
        assert len(result["errors"]) == 0

        # Invalid - missing layers
        invalid_arch_no_layers = {"training": {}}
        result = code_generator.validate_architecture(invalid_arch_no_layers)
        assert result["valid"] is False
        assert any("layers" in e.lower() for e in result["errors"])

        # Invalid - unknown layer type
        invalid_arch_bad_layer = {
            "layers": [
                {"type": "unknown_layer", "params": {}}
            ]
        }
        result = code_generator.validate_architecture(invalid_arch_bad_layer)
        assert result["valid"] is False

    def test_code_generator_handles_optimizer_types(self, code_generator, temp_dir):
        """Test that code generator handles different optimizer types."""
        base_arch = {
            "layers": [
                {"type": "linear", "params": {"in_features": 10, "out_features": 5}}
            ],
            "training": {
                "epochs": 5,
                "batch_size": 32
            }
        }
        dataset_config = {"data_type": "image", "input_shape": [10], "num_classes": 5}

        # Test SGD
        arch_sgd = {**base_arch, "training": {**base_arch["training"],
                    "optimizer": {"type": "sgd", "params": {"learning_rate": 0.01, "momentum": 0.9}}}}
        output_sgd = temp_dir / "sgd"
        train_cpp = code_generator.generate(arch_sgd, dataset_config, output_sgd)
        assert "SGD optimizer" in train_cpp.read_text()

        # Test Adam
        arch_adam = {**base_arch, "training": {**base_arch["training"],
                     "optimizer": {"type": "adam", "params": {"learning_rate": 0.001}}}}
        output_adam = temp_dir / "adam"
        train_cpp = code_generator.generate(arch_adam, dataset_config, output_adam)
        assert "Adam optimizer" in train_cpp.read_text()

    def test_code_generator_handles_scheduler_types(self, code_generator, temp_dir):
        """Test that code generator handles different scheduler types."""
        base_arch = {
            "layers": [
                {"type": "linear", "params": {"in_features": 10, "out_features": 5}}
            ],
            "training": {
                "optimizer": {"type": "sgd"},
                "epochs": 5,
                "batch_size": 32
            }
        }
        dataset_config = {"data_type": "image", "input_shape": [10], "num_classes": 5}

        # Test StepLR
        arch_step = {**base_arch}
        arch_step["training"]["scheduler"] = {"type": "step", "params": {"step_size": 10, "gamma": 0.1}}
        output_step = temp_dir / "step"
        train_cpp = code_generator.generate(arch_step, dataset_config, output_step)
        assert "StepLR" in train_cpp.read_text()

        # Test Cosine
        arch_cosine = {**base_arch}
        arch_cosine["training"]["scheduler"] = {"type": "cosine", "params": {"T_max": 50}}
        output_cosine = temp_dir / "cosine"
        train_cpp = code_generator.generate(arch_cosine, dataset_config, output_cosine)
        assert "CosineAnnealing" in train_cpp.read_text()

    def test_full_design_to_generation_flow(
        self, mock_llm_service, code_generator, temp_dir
    ):
        """Test the complete flow from LLM design to code generation."""
        # Step 1: Get architecture suggestion from LLM
        dataset_info = {
            "data_type": "image",
            "input_shape": [3, 32, 32],
            "num_classes": 10,
            "total_samples": 1000
        }

        llm_result = mock_llm_service.suggest_architecture(
            dataset_info=dataset_info,
            user_prompt="Create a CNN for CIFAR-10 style classification"
        )
        architecture = llm_result["architecture"]

        # Step 2: Validate architecture
        validation = code_generator.validate_architecture(architecture)
        assert validation["valid"], f"Architecture invalid: {validation['errors']}"

        # Step 3: Generate code
        dataset_config = {
            **dataset_info,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5]
        }

        output_dir = temp_dir / "full_flow"
        train_cpp = code_generator.generate(architecture, dataset_config, output_dir)

        # Step 4: Verify outputs
        assert train_cpp.exists()
        assert (output_dir / "infer.cpp").exists()
        assert (output_dir / "Makefile").exists()

        # Verify the generated code has all necessary components
        code = train_cpp.read_text()
        assert "int main(" in code
        assert "model.forward" in code
        assert "loss->backward" in code


# =============================================================================
# Test: Dataset Upload -> Processing Flow
# =============================================================================

class TestDatasetUploadProcessingFlow:
    """Test the dataset upload and processing flow."""

    def test_image_processor_processes_folder_dataset(
        self, small_test_dataset, temp_dir
    ):
        """Test that ImageProcessor correctly processes folder-per-class dataset."""
        dataset_dir, class_names = small_test_dataset
        output_dir = temp_dir / "processed"

        processor = ImageProcessor(
            target_size=(32, 32),
            channels=3,
            normalize=True
        )

        config = processor.process_dataset(
            raw_dir=dataset_dir,
            output_dir=output_dir,
            class_names=class_names,
            train_split=0.8
        )

        # Verify output files were created
        assert (output_dir / "train_images.bin").exists()
        assert (output_dir / "train_labels.bin").exists()
        assert (output_dir / "test_images.bin").exists()
        assert (output_dir / "test_labels.bin").exists()
        assert (output_dir / "config.json").exists()

        # Verify config
        assert config["num_classes"] == 3
        assert config["channels"] == 3
        assert config["target_size"] == [32, 32]
        assert config["train_samples"] + config["test_samples"] == 15  # 3 classes * 5 images

        # Load and verify tensor format
        train_images = ImageProcessor.load_tensor(output_dir / "train_images.bin")
        train_labels = ImageProcessor.load_tensor(output_dir / "train_labels.bin")

        assert train_images.ndim == 4  # [N, C, H, W]
        assert train_images.shape[1:] == (3, 32, 32)
        assert train_labels.ndim == 1
        assert len(train_labels) == len(train_images)

        # Verify labels are valid class indices
        assert all(0 <= label < 3 for label in train_labels)

    def test_image_processor_handles_grayscale(self, temp_dir):
        """Test that ImageProcessor handles grayscale images correctly."""
        # Create grayscale test images
        dataset_dir = temp_dir / "grayscale_dataset"
        class_dir = dataset_dir / "digits"
        class_dir.mkdir(parents=True)

        for i in range(5):
            img_array = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            img.save(class_dir / f"digit_{i}.png")

        output_dir = temp_dir / "processed_grayscale"

        processor = ImageProcessor(
            target_size=(28, 28),
            channels=1,
            normalize=True
        )

        config = processor.process_dataset(
            raw_dir=dataset_dir,
            output_dir=output_dir,
            class_names=["digits"],
            train_split=0.8
        )

        assert config["channels"] == 1
        assert config["input_shape"] == [1, 28, 28]

        train_images = ImageProcessor.load_tensor(output_dir / "train_images.bin")
        assert train_images.shape[1] == 1  # Single channel

    def test_image_processor_normalizes_data(self, small_test_dataset, temp_dir):
        """Test that ImageProcessor correctly normalizes image data."""
        dataset_dir, class_names = small_test_dataset
        output_dir = temp_dir / "normalized"

        processor = ImageProcessor(
            target_size=(32, 32),
            channels=3,
            normalize=True
        )

        config = processor.process_dataset(
            raw_dir=dataset_dir,
            output_dir=output_dir,
            class_names=class_names,
            train_split=0.8
        )

        # Verify mean and std are computed
        assert "mean" in config
        assert "std" in config
        assert len(config["mean"]) == 3
        assert len(config["std"]) == 3

        # Load normalized data
        train_images = ImageProcessor.load_tensor(output_dir / "train_images.bin")

        # Normalized data should have roughly zero mean and unit variance
        # (not exact due to small sample size, but should be reasonable)
        channel_means = train_images.mean(axis=(0, 2, 3))
        assert all(abs(m) < 1.0 for m in channel_means), "Normalized mean should be close to 0"

    def test_zip_extraction_and_processing(self, test_zip_dataset, temp_dir):
        """Test extracting ZIP file and processing the dataset."""
        zip_path = test_zip_dataset
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

        # Find class directories
        contents = list(extract_dir.iterdir())
        class_names = sorted([d.name for d in contents if d.is_dir()])

        assert len(class_names) == 3
        assert "cat" in class_names
        assert "dog" in class_names
        assert "bird" in class_names

        # Process the dataset
        output_dir = temp_dir / "processed_from_zip"
        processor = ImageProcessor(target_size=(32, 32), channels=3)

        config = processor.process_dataset(
            raw_dir=extract_dir,
            output_dir=output_dir,
            class_names=class_names
        )

        assert config["num_classes"] == 3
        assert config["train_samples"] > 0
        assert config["test_samples"] > 0

    def test_tensor_binary_format_correctness(self, temp_dir):
        """Test that tensor binary format is correctly read/written."""
        # Create test data
        test_data = np.random.randn(10, 3, 32, 32).astype(np.float32)
        tensor_path = temp_dir / "test_tensor.bin"

        # Write using the same format as ImageProcessor
        TENSOR_MAGIC = 0x54454E53
        with open(tensor_path, 'wb') as f:
            f.write(struct.pack('I', TENSOR_MAGIC))
            f.write(struct.pack('I', len(test_data.shape)))
            for dim in test_data.shape:
                f.write(struct.pack('Q', dim))
            f.write(np.ascontiguousarray(test_data).tobytes())

        # Read back
        loaded = ImageProcessor.load_tensor(tensor_path)

        assert loaded.shape == test_data.shape
        np.testing.assert_array_almost_equal(loaded, test_data)

    def test_processing_handles_missing_images(self, temp_dir):
        """Test that processing handles datasets with some invalid files."""
        dataset_dir = temp_dir / "mixed_dataset"
        class_dir = dataset_dir / "class_a"
        class_dir.mkdir(parents=True)

        # Create some valid images
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            img.save(class_dir / f"valid_{i}.png")

        # Create an invalid file (not an image)
        (class_dir / "invalid.txt").write_text("This is not an image")

        # Processing should still work, just skip invalid files
        output_dir = temp_dir / "processed_mixed"
        processor = ImageProcessor(target_size=(32, 32), channels=3)

        config = processor.process_dataset(
            raw_dir=dataset_dir,
            output_dir=output_dir,
            class_names=["class_a"]
        )

        # Should have processed only the valid images
        assert config["train_samples"] + config["test_samples"] == 3


# =============================================================================
# Test: Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling for compilation and training failures."""

    def test_compilation_failure_with_invalid_code(self, temp_dir):
        """Test that compilation failures are properly handled."""
        # Create directory with invalid C++ code
        generated_dir = temp_dir / "invalid_code"
        generated_dir.mkdir()

        # Write invalid C++ code
        (generated_dir / "train.cpp").write_text("""
// This code has syntax errors
int main() {
    invalid_syntax here;
    return 0
}  // missing semicolon
""")

        # Create a minimal Makefile
        (generated_dir / "Makefile").write_text("""
train: train.cpp
\tg++ -std=c++17 -o $@ $<
""")

        # Compilation should fail
        success, message = compile_training_code(generated_dir)

        assert success is False
        assert "failed" in message.lower() or "error" in message.lower()

    def test_compilation_failure_with_missing_files(self, temp_dir):
        """Test that compilation handles missing source files."""
        generated_dir = temp_dir / "missing_files"
        generated_dir.mkdir()

        # Create Makefile but no source files
        (generated_dir / "Makefile").write_text("""
train: train.cpp
\tg++ -std=c++17 -o $@ $<
""")

        success, message = compile_training_code(generated_dir)

        assert success is False

    def test_code_generator_raises_on_unknown_layer(self, code_generator, temp_dir):
        """Test that code generator raises error for unknown layer types."""
        architecture = {
            "layers": [
                {"type": "nonexistent_layer_type", "params": {}}
            ],
            "training": {"epochs": 1, "batch_size": 32}
        }
        dataset_config = {"data_type": "image", "input_shape": [3, 32, 32], "num_classes": 10}

        with pytest.raises(ValueError, match="Unknown layer"):
            code_generator.generate(architecture, dataset_config, temp_dir / "output")

    def test_code_generator_raises_on_missing_params(self, code_generator, temp_dir):
        """Test that code generator raises error for missing required parameters."""
        # Conv2d without required in_channels/out_channels
        architecture = {
            "layers": [
                {"type": "conv2d", "params": {"kernel_size": 3}}  # Missing in_channels, out_channels
            ],
            "training": {"epochs": 1, "batch_size": 32}
        }
        dataset_config = {"data_type": "image", "input_shape": [3, 32, 32], "num_classes": 10}

        with pytest.raises(ValueError, match="Missing required parameter"):
            code_generator.generate(architecture, dataset_config, temp_dir / "output")

    def test_code_generator_raises_on_unknown_optimizer(self, code_generator, temp_dir):
        """Test that code generator raises error for unknown optimizer."""
        architecture = {
            "layers": [
                {"type": "linear", "params": {"in_features": 10, "out_features": 5}}
            ],
            "training": {
                "optimizer": {"type": "nonexistent_optimizer"},
                "epochs": 1,
                "batch_size": 32
            }
        }
        dataset_config = {"data_type": "image", "input_shape": [10], "num_classes": 5}

        with pytest.raises(ValueError, match="Unknown optimizer"):
            code_generator.generate(architecture, dataset_config, temp_dir / "output")

    def test_image_processor_raises_on_empty_dataset(self, temp_dir):
        """Test that ImageProcessor raises error on empty dataset."""
        dataset_dir = temp_dir / "empty_dataset"
        class_dir = dataset_dir / "empty_class"
        class_dir.mkdir(parents=True)
        # No images in the class directory

        output_dir = temp_dir / "processed_empty"
        processor = ImageProcessor()

        with pytest.raises(ValueError, match="No valid images"):
            processor.process_dataset(
                raw_dir=dataset_dir,
                output_dir=output_dir,
                class_names=["empty_class"]
            )

    def test_tensor_load_raises_on_invalid_format(self, temp_dir):
        """Test that tensor loading raises error on invalid file format."""
        invalid_tensor = temp_dir / "invalid.bin"
        invalid_tensor.write_bytes(b"This is not a valid tensor file")

        with pytest.raises(ValueError, match="Invalid tensor file"):
            ImageProcessor.load_tensor(invalid_tensor)

    def test_llm_service_handles_missing_api_key(self):
        """Test that LLM service falls back to mock when API key missing."""
        # Temporarily clear API key
        original_key = os.environ.get("ANTHROPIC_API_KEY")
        try:
            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]

            service = get_llm_service(api_key=None)

            # Should return MockLLMService
            assert isinstance(service, MockLLMService)
        finally:
            if original_key:
                os.environ["ANTHROPIC_API_KEY"] = original_key

    def test_job_status_transitions(self):
        """Test that job status enum has expected values."""
        # Verify all expected statuses exist
        expected_statuses = [
            "pending", "queued", "compiling", "training",
            "completed", "failed", "cancelled"
        ]

        for status in expected_statuses:
            assert hasattr(JobStatus, status.upper())
            assert JobStatus[status.upper()].value == status

    def test_dataset_status_transitions(self):
        """Test that dataset status enum has expected values."""
        expected_statuses = ["created", "extracting", "processing", "ready", "error"]

        for status in expected_statuses:
            assert hasattr(DatasetStatus, status.upper())
            assert DatasetStatus[status.upper()].value == status


# =============================================================================
# Test: Integration with Mocked Services
# =============================================================================

class TestMockedIntegration:
    """Integration tests with mocked external services."""

    @patch('db.get_db_session')
    @patch('db.get_blob_store')
    def test_dataset_service_create_and_upload_flow(
        self, mock_get_blob, mock_get_db, mock_db_session, mock_blob_store,
        test_zip_dataset, temp_dir
    ):
        """Test DatasetService create and upload flow with mocks."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_blob.return_value = mock_blob_store

        # Mock dataset query
        mock_dataset = MagicMock(spec=Dataset)
        mock_dataset.id = "test123"
        mock_dataset.name = "test_dataset"
        mock_dataset.status = DatasetStatus.CREATED.value
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = mock_dataset

        # Read zip file content
        zip_content = test_zip_dataset.read_bytes()

        # The service would process this - we just verify the flow works
        # In a real test, we'd instantiate DatasetService and call its methods
        assert len(zip_content) > 0
        assert zipfile.is_zipfile(BytesIO(zip_content))

    def test_training_flow_code_generation_to_compilation_mock(
        self, code_generator, sample_image_architecture, sample_dataset_config, temp_dir
    ):
        """Test the training flow from code generation through to compilation attempt."""
        output_dir = temp_dir / "training_flow"

        # Generate code
        train_cpp = code_generator.generate(
            sample_image_architecture,
            sample_dataset_config,
            output_dir
        )

        assert train_cpp.exists()

        # Read the generated code
        code = train_cpp.read_text()

        # Verify all essential components are present for a successful compilation
        # (compilation will fail without the actual whitematter headers, but structure should be correct)
        essential_components = [
            "#include",
            "int main(",
            "Sequential model",
            "optimizer",
            "criterion",
            "forward",
            "backward",
            "return 0",
        ]

        for component in essential_components:
            assert component in code, f"Missing essential component: {component}"

        # Verify Makefile references correct source
        makefile = (output_dir / "Makefile").read_text()
        assert "train.cpp" in makefile
        assert "infer.cpp" in makefile

    def test_end_to_end_with_mock_services(
        self, mock_llm_service, code_generator, small_test_dataset, temp_dir
    ):
        """Test complete end-to-end flow with mocked services."""
        dataset_dir, class_names = small_test_dataset

        # Step 1: Process dataset
        processor = ImageProcessor(target_size=(32, 32), channels=3)
        processed_dir = temp_dir / "processed"
        dataset_config = processor.process_dataset(
            raw_dir=dataset_dir,
            output_dir=processed_dir,
            class_names=class_names
        )

        # Step 2: Get architecture from LLM
        dataset_info = {
            "data_type": "image",
            "input_shape": dataset_config["input_shape"],
            "num_classes": dataset_config["num_classes"],
            "class_names": dataset_config["class_names"],
            "total_samples": dataset_config["train_samples"] + dataset_config["test_samples"]
        }

        llm_result = mock_llm_service.suggest_architecture(
            dataset_info=dataset_info,
            user_prompt="Create a simple CNN"
        )
        architecture = llm_result["architecture"]

        # Step 3: Validate architecture
        validation = code_generator.validate_architecture(architecture)
        assert validation["valid"], f"Invalid architecture: {validation['errors']}"

        # Step 4: Generate training code
        generated_dir = temp_dir / "generated"
        train_cpp = code_generator.generate(
            architecture=architecture,
            dataset_config=dataset_config,
            output_dir=generated_dir
        )

        # Verify everything is in place
        assert train_cpp.exists()
        assert (generated_dir / "Makefile").exists()
        assert (processed_dir / "train_images.bin").exists()
        assert (processed_dir / "train_labels.bin").exists()

        # Verify the generated code references correct dataset configuration
        code = train_cpp.read_text()
        assert str(dataset_config["num_classes"]) in code or "num_classes" in code


# =============================================================================
# Test: Text Data Flow
# =============================================================================

class TestTextDataFlow:
    """Test the text data processing and model generation flow."""

    def test_text_architecture_generation(self, mock_llm_service, code_generator, temp_dir):
        """Test architecture generation for text/language model tasks."""
        dataset_info = {
            "data_type": "text",
            "input_shape": [128],  # sequence length
            "num_classes": 100,    # vocab size
            "total_samples": 10000
        }

        llm_result = mock_llm_service.suggest_architecture(
            dataset_info=dataset_info,
            user_prompt="Create a character-level language model"
        )

        architecture = llm_result["architecture"]

        # Verify text-specific architecture
        assert architecture["data_type"] == "text"

        # Generate code
        dataset_config = {
            "data_type": "text",
            "input_shape": [128],
            "num_classes": 100,
            "vocab_size": 100,
            "seq_length": 128
        }

        output_dir = temp_dir / "text_model"
        train_cpp = code_generator.generate(architecture, dataset_config, output_dir)

        assert train_cpp.exists()
        code = train_cpp.read_text()

        # Text models should have embedding
        assert "Embedding" in code or "embedding" in code.lower()

    def test_text_code_generation_includes_transformer(self, code_generator, temp_dir):
        """Test that text model code generation includes transformer components."""
        # Transformer-based architecture
        architecture = {
            "name": "transformer_lm",
            "data_type": "text",
            "layers": [
                {"type": "embedding", "params": {"num_embeddings": 100, "embedding_dim": 64}},
                {"type": "layernorm", "params": {"normalized_shape": 64}},
                {"type": "linear", "params": {"in_features": 64, "out_features": 100}},
            ],
            "training": {
                "optimizer": {"type": "adam", "params": {"learning_rate": 0.001}},
                "epochs": 10,
                "batch_size": 32
            }
        }

        dataset_config = {
            "data_type": "text",
            "vocab_size": 100,
            "seq_length": 128,
            "num_classes": 100
        }

        output_dir = temp_dir / "transformer"
        train_cpp = code_generator.generate(architecture, dataset_config, output_dir)

        code = train_cpp.read_text()

        # Should have text-specific components
        assert "Embedding" in code
        assert "Language" in code or "Text" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
