"""
Code Generator - generates C++ training code from architecture JSON.
Uses template-based generation for safety.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from .layers import LAYER_TEMPLATES, generate_layers, fill_template
from .training_config import (
    OPTIMIZER_TEMPLATES,
    SCHEDULER_TEMPLATES,
    generate_optimizer,
    generate_scheduler,
)
from .templates.image_train import generate_image_training_code
from .templates.image_infer import generate_image_inference_code
from .templates.text_train import generate_text_training_code
from .templates.text_infer import generate_text_inference_code
from .templates.makefile import generate_makefile


class CodeGenerator:
    """Generate C++ training code from architecture specification."""

    def __init__(self, templates_dir: Path = None):
        self.templates_dir = templates_dir or Path(__file__).parent / "templates"

    def generate(
        self,
        architecture: Dict[str, Any],
        dataset_config: Dict[str, Any],
        output_dir: Path
    ) -> Path:
        """
        Generate training code from architecture and dataset config.

        Args:
            architecture: Architecture JSON with layers and training config
            dataset_config: Dataset preprocessing config (from image_processor)
            output_dir: Where to write generated files

        Returns:
            Path to generated train.cpp
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Detect data type
        data_type = dataset_config.get("data_type", "image")

        # Generate layer code
        layers_code = self._generate_layers(architecture["layers"])

        # Generate optimizer code
        training = architecture.get("training", {})
        optimizer_code = self._generate_optimizer(training.get("optimizer", {}))
        scheduler_code = self._generate_scheduler(training.get("scheduler", {}))

        # Get training params
        epochs = training.get("epochs", 10)
        batch_size = training.get("batch_size", 64)

        # Generate full training code based on data type
        if data_type == "text":
            train_code = self._generate_text_training_code(
                layers_code=layers_code,
                optimizer_code=optimizer_code,
                scheduler_code=scheduler_code,
                epochs=epochs,
                batch_size=batch_size,
                dataset_config=dataset_config
            )
        else:
            train_code = self._generate_training_code(
                layers_code=layers_code,
                optimizer_code=optimizer_code,
                scheduler_code=scheduler_code,
                epochs=epochs,
                batch_size=batch_size,
                dataset_config=dataset_config
            )

        # Write files
        train_cpp = output_dir / "train.cpp"
        with open(train_cpp, 'w') as f:
            f.write(train_code)

        # Generate inference code
        if data_type == "text":
            infer_code = self._generate_text_inference_code(
                layers_code=layers_code,
                dataset_config=dataset_config
            )
        else:
            infer_code = self._generate_inference_code(
                layers_code=layers_code,
                dataset_config=dataset_config
            )
        infer_cpp = output_dir / "infer.cpp"
        with open(infer_cpp, 'w') as f:
            f.write(infer_code)

        # Generate Makefile
        makefile = output_dir / "Makefile"
        with open(makefile, 'w') as f:
            f.write(self._generate_makefile())

        # Copy architecture for reference
        with open(output_dir / "architecture.json", 'w') as f:
            json.dump(architecture, f, indent=2)

        return train_cpp

    def _generate_layers(self, layers: List[Dict[str, Any]]) -> str:
        """Generate C++ layer initialization code."""
        return generate_layers(layers)

    def _fill_template(self, template: str, params: dict, layer_type: str) -> str:
        """Fill template with parameters, using defaults where needed."""
        return fill_template(template, params, layer_type)

    def _generate_optimizer(self, optimizer_config: dict) -> str:
        """Generate optimizer initialization code."""
        return generate_optimizer(optimizer_config)

    def _generate_scheduler(self, scheduler_config: dict) -> str:
        """Generate scheduler initialization code."""
        return generate_scheduler(scheduler_config)

    def _generate_training_code(
        self,
        layers_code: str,
        optimizer_code: str,
        scheduler_code: str,
        epochs: int,
        batch_size: int,
        dataset_config: dict
    ) -> str:
        """Generate complete training code."""
        return generate_image_training_code(
            layers_code=layers_code,
            optimizer_code=optimizer_code,
            scheduler_code=scheduler_code,
            epochs=epochs,
            batch_size=batch_size,
            dataset_config=dataset_config
        )

    def _generate_inference_code(
        self,
        layers_code: str,
        dataset_config: dict
    ) -> str:
        """Generate inference-only code for the model."""
        return generate_image_inference_code(
            layers_code=layers_code,
            dataset_config=dataset_config
        )

    def _generate_text_training_code(
        self,
        layers_code: str,
        optimizer_code: str,
        scheduler_code: str,
        epochs: int,
        batch_size: int,
        dataset_config: dict
    ) -> str:
        """Generate text model training code for language modeling using Transformer."""
        return generate_text_training_code(
            layers_code=layers_code,
            optimizer_code=optimizer_code,
            scheduler_code=scheduler_code,
            epochs=epochs,
            batch_size=batch_size,
            dataset_config=dataset_config
        )

    def _generate_text_inference_code(
        self,
        layers_code: str,
        dataset_config: dict
    ) -> str:
        """Generate text generation inference code using Transformer."""
        return generate_text_inference_code(
            layers_code=layers_code,
            dataset_config=dataset_config
        )

    def _generate_makefile(self) -> str:
        """Generate Makefile for compiling training and inference code."""
        return generate_makefile()

    def validate_architecture(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an architecture specification.

        Returns dict with 'valid', 'errors', 'warnings' keys.
        """
        errors = []
        warnings = []

        # Check required fields
        if "layers" not in architecture:
            errors.append("Missing 'layers' field")
        else:
            layers = architecture["layers"]
            if not layers:
                errors.append("No layers specified")
            else:
                # Validate each layer
                for i, layer in enumerate(layers):
                    if "type" not in layer:
                        errors.append(f"Layer {i}: missing 'type' field")
                    elif layer["type"].lower() not in LAYER_TEMPLATES:
                        errors.append(f"Layer {i}: unknown type '{layer['type']}'")

        # Check training config
        training = architecture.get("training", {})
        optimizer = training.get("optimizer", {})
        if optimizer.get("type", "sgd").lower() not in OPTIMIZER_TEMPLATES:
            errors.append(f"Unknown optimizer: {optimizer.get('type')}")

        # Parameter count warning
        # (simplified - real implementation would compute actual params)
        if len(architecture.get("layers", [])) > 50:
            warnings.append("Architecture has many layers, training may be slow")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
