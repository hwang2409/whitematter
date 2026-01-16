"""
LLM Service - Claude API integration for architecture design.
"""

import json
import os
import re
from typing import Dict, Any, Optional

from .prompts import ARCHITECTURE_SYSTEM_PROMPT, REFINEMENT_PROMPT


class LLMService:
    """Interface to Claude API for architecture suggestions."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    def suggest_architecture(
        self,
        dataset_info: Dict[str, Any],
        user_prompt: str,
        model: str = "claude-sonnet-4-20250514"
    ) -> Dict[str, Any]:
        """
        Get architecture suggestion from Claude.

        Args:
            dataset_info: Dataset metadata (data_type, input_shape, num_classes, etc.)
            user_prompt: User's description of what they want
            model: Claude model to use

        Returns:
            Dict with 'architecture' (JSON) and 'explanation' (text)
        """
        # Build user message with dataset context
        user_message = f"""Dataset Information:
- Data type: {dataset_info.get('data_type', 'unknown')}
- Input shape: {dataset_info.get('input_shape', [])}
- Number of classes: {dataset_info.get('num_classes', 0)}
- Class names: {dataset_info.get('class_names', [])}
- Total samples: {dataset_info.get('total_samples', 0)}

User Request:
{user_prompt}

Please design an appropriate neural network architecture for this dataset and task."""

        # Call Claude API
        response = self.client.messages.create(
            model=model,
            max_tokens=4096,
            system=ARCHITECTURE_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        # Extract response
        response_text = response.content[0].text

        # Parse JSON from response
        architecture = self._extract_json(response_text)

        return {
            "architecture": architecture,
            "explanation": self._extract_explanation(response_text),
            "raw_response": response_text
        }

    def refine_architecture(
        self,
        current_architecture: Dict[str, Any],
        feedback: str,
        model: str = "claude-sonnet-4-20250514"
    ) -> Dict[str, Any]:
        """
        Refine an existing architecture based on user feedback.

        Args:
            current_architecture: Current architecture JSON
            feedback: User's feedback/changes requested
            model: Claude model to use

        Returns:
            Dict with 'architecture' (JSON) and 'explanation' (text)
        """
        user_message = REFINEMENT_PROMPT.format(
            current_architecture=json.dumps(current_architecture, indent=2),
            feedback=feedback
        )

        response = self.client.messages.create(
            model=model,
            max_tokens=4096,
            system=ARCHITECTURE_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        response_text = response.content[0].text
        architecture = self._extract_json(response_text)

        return {
            "architecture": architecture,
            "explanation": self._extract_explanation(response_text),
            "raw_response": response_text
        }

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from response text."""
        # Try to find JSON in code blocks
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not extract valid JSON from response")

    def _extract_explanation(self, text: str) -> str:
        """Extract explanation text (everything before JSON)."""
        # Find where JSON starts
        json_start = text.find('```json')
        if json_start == -1:
            json_start = text.find('{')

        if json_start > 0:
            return text[:json_start].strip()
        return ""


# Fallback for when Claude API is not available
class MockLLMService:
    """Mock LLM service for testing without API key."""

    def suggest_architecture(
        self,
        dataset_info: Dict[str, Any],
        user_prompt: str,
        model: str = None
    ) -> Dict[str, Any]:
        """Return a simple default architecture based on data type."""
        data_type = dataset_info.get('data_type', 'image')
        input_shape = dataset_info.get('input_shape', [3, 32, 32])
        num_classes = dataset_info.get('num_classes', 10)

        if data_type == 'image':
            c, h, w = input_shape
            # Simple CNN
            architecture = {
                "name": "simple_cnn",
                "description": "Simple CNN for image classification",
                "data_type": "image",
                "input_shape": input_shape,
                "num_classes": num_classes,
                "layers": [
                    {"type": "conv2d", "params": {"in_channels": c, "out_channels": 32, "kernel_size": 3, "padding": 1}},
                    {"type": "batchnorm2d", "params": {"num_features": 32}},
                    {"type": "relu", "params": {}},
                    {"type": "maxpool2d", "params": {"kernel_size": 2}},
                    {"type": "conv2d", "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": 1}},
                    {"type": "batchnorm2d", "params": {"num_features": 64}},
                    {"type": "relu", "params": {}},
                    {"type": "maxpool2d", "params": {"kernel_size": 2}},
                    {"type": "flatten", "params": {}},
                    {"type": "linear", "params": {"in_features": 64 * (h // 4) * (w // 4), "out_features": 128}},
                    {"type": "relu", "params": {}},
                    {"type": "dropout", "params": {"p": 0.5}},
                    {"type": "linear", "params": {"in_features": 128, "out_features": num_classes}},
                ],
                "training": {
                    "optimizer": {"type": "adam", "params": {"learning_rate": 0.001}},
                    "scheduler": {"type": "cosine", "params": {"T_max": 50}},
                    "epochs": 50,
                    "batch_size": 64
                }
            }
        elif data_type == 'text':
            seq_len = input_shape[0] if input_shape else 256
            architecture = {
                "name": "simple_lstm",
                "description": "Simple LSTM for text classification",
                "data_type": "text",
                "input_shape": [seq_len],
                "num_classes": num_classes,
                "layers": [
                    {"type": "embedding", "params": {"num_embeddings": 10000, "embedding_dim": 128}},
                    {"type": "lstm", "params": {"input_size": 128, "hidden_size": 256}},
                    {"type": "dropout", "params": {"p": 0.5}},
                    {"type": "linear", "params": {"in_features": 256, "out_features": num_classes}},
                ],
                "training": {
                    "optimizer": {"type": "adam", "params": {"learning_rate": 0.001}},
                    "scheduler": {"type": "none", "params": {}},
                    "epochs": 20,
                    "batch_size": 32
                }
            }
        else:  # tabular
            num_features = input_shape[0] if input_shape else 10
            architecture = {
                "name": "simple_mlp",
                "description": "Simple MLP for tabular classification",
                "data_type": "tabular",
                "input_shape": [num_features],
                "num_classes": num_classes,
                "layers": [
                    {"type": "linear", "params": {"in_features": num_features, "out_features": 128}},
                    {"type": "relu", "params": {}},
                    {"type": "dropout", "params": {"p": 0.3}},
                    {"type": "linear", "params": {"in_features": 128, "out_features": 64}},
                    {"type": "relu", "params": {}},
                    {"type": "linear", "params": {"in_features": 64, "out_features": num_classes}},
                ],
                "training": {
                    "optimizer": {"type": "adam", "params": {"learning_rate": 0.001}},
                    "scheduler": {"type": "none", "params": {}},
                    "epochs": 100,
                    "batch_size": 32
                }
            }

        return {
            "architecture": architecture,
            "explanation": f"Generated a simple {data_type} classification model based on the dataset characteristics.",
            "raw_response": ""
        }

    def refine_architecture(self, current_architecture, feedback, model=None):
        """Just return the current architecture unchanged."""
        return {
            "architecture": current_architecture,
            "explanation": "Mock service: architecture unchanged",
            "raw_response": ""
        }


def get_llm_service(api_key: Optional[str] = None) -> LLMService:
    """Get LLM service, falling back to mock if no API key."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return LLMService(api_key=key)
    else:
        print("Warning: ANTHROPIC_API_KEY not set, using mock LLM service")
        return MockLLMService()
