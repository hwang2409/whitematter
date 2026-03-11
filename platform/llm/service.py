"""
LLM Service - Claude API integration for architecture design.
"""

import json
import logging
import os
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

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
            seq_len = input_shape[0] if input_shape else 128
            vocab_size = num_classes  # For text, num_classes = vocab_size
            embedding_dim = 64
            hidden_size = 128

            # Check for keywords in user prompt to decide architecture
            # Default to Transformer architecture for text (better performance)
            use_lstm = 'lstm' in user_prompt.lower() or 'rnn' in user_prompt.lower()

            if use_lstm:
                # Legacy LSTM-based language model (only if explicitly requested)
                architecture = {
                    "name": "char_lstm",
                    "description": "Character-level LSTM for language modeling",
                    "data_type": "text",
                    "input_shape": [seq_len],
                    "num_classes": vocab_size,
                    "vocab_size": vocab_size,
                    "seq_length": seq_len,
                    "layers": [
                        {"type": "embedding", "params": {"num_embeddings": vocab_size, "embedding_dim": embedding_dim}},
                        {"type": "lstm", "params": {"input_size": embedding_dim, "hidden_size": hidden_size}},
                        {"type": "dropout", "params": {"p": 0.3}},
                        {"type": "linear", "params": {"in_features": hidden_size, "out_features": vocab_size}},
                    ],
                    "training": {
                        "optimizer": {"type": "adam", "params": {"learning_rate": 0.002}},
                        "scheduler": {"type": "step", "params": {"step_size": 10, "gamma": 0.5}},
                        "epochs": 30,
                        "batch_size": 32
                    }
                }
            else:
                # Transformer architecture (default - better convergence)
                num_heads = 4
                num_layers = 4
                ff_dim = embedding_dim * 2
                # Ensure embed_dim is divisible by num_heads
                if embedding_dim % num_heads != 0:
                    embedding_dim = (embedding_dim // num_heads + 1) * num_heads
                    ff_dim = embedding_dim * 2

                architecture = {
                    "name": "char_transformer",
                    "description": f"Transformer language model ({num_layers} layers, {num_heads} heads)",
                    "data_type": "text",
                    "input_shape": [seq_len],
                    "num_classes": vocab_size,
                    "vocab_size": vocab_size,
                    "seq_length": seq_len,
                    "layers": [
                        {"type": "embedding", "params": {"num_embeddings": vocab_size, "embedding_dim": embedding_dim}},
                        {"type": "transformer", "params": {"embed_dim": embedding_dim, "num_heads": num_heads, "num_layers": num_layers, "ff_dim": ff_dim}},
                        {"type": "layernorm", "params": {"normalized_shape": embedding_dim}},
                        {"type": "linear", "params": {"in_features": embedding_dim, "out_features": vocab_size}},
                    ],
                    "training": {
                        "optimizer": {"type": "adam", "params": {"learning_rate": 0.001, "beta1": 0.9, "beta2": 0.98}},
                        "scheduler": {"type": "cosine", "params": {"T_max": 50}},
                        "epochs": 50,
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
        logger.warning("ANTHROPIC_API_KEY not set, using mock LLM service")
        return MockLLMService()
