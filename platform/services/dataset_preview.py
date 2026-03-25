import base64
import json
import logging
import struct
from io import BytesIO
from typing import Dict, List

import numpy as np
from PIL import Image

from db import Dataset, DatasetFormat, DataType

logger = logging.getLogger(__name__)


class DatasetPreviewService:
    """Generates previews from processed dataset data in blob storage."""

    def __init__(self, blob_store):
        self.blob_store = blob_store

    def get_preview(self, dataset: Dataset, dataset_dict: Dict, num_samples: int = 18) -> Dict:
        """Get dataset preview with sample images or text."""
        result = {
            "metadata": dataset_dict,
            "samples": []
        }

        if dataset.data_type == DataType.TEXT.value:
            result["text_preview"] = self._preview_text_from_blobs(dataset)
        elif dataset.format == DatasetFormat.MNIST_IDX.value:
            result["samples"] = self._preview_mnist_from_blobs(dataset, num_samples)
        elif dataset.processed_blob_prefix:
            result["samples"] = self._preview_from_processed(dataset, num_samples)

        return result

    def _preview_mnist_from_blobs(self, dataset: Dataset, num_samples: int) -> List[Dict]:
        """Generate preview from processed MNIST data in blob storage."""
        samples = []

        if not dataset.processed_blob_prefix:
            return samples

        images_blob = self.blob_store.get(f"{dataset.processed_blob_prefix}/test_images.bin")
        labels_blob = self.blob_store.get(f"{dataset.processed_blob_prefix}/test_labels.bin")

        if not images_blob or not labels_blob:
            return samples

        try:
            TENSOR_MAGIC = 0x54454E53
            images_data = BytesIO(images_blob)
            labels_data = BytesIO(labels_blob)

            magic = struct.unpack('I', images_data.read(4))[0]
            if magic != TENSOR_MAGIC:
                return samples

            ndims = struct.unpack('I', images_data.read(4))[0]
            shape = [struct.unpack('Q', images_data.read(8))[0] for _ in range(ndims)]

            n_to_read = min(num_samples * 2, shape[0])
            image_size = np.prod(shape[1:])
            images = np.frombuffer(
                images_data.read(int(n_to_read * image_size * 4)),
                dtype=np.float32
            ).reshape(n_to_read, *shape[1:])

            magic = struct.unpack('I', labels_data.read(4))[0]
            ndims = struct.unpack('I', labels_data.read(4))[0]
            _ = [struct.unpack('Q', labels_data.read(8))[0] for _ in range(ndims)]
            labels = np.frombuffer(
                labels_data.read(n_to_read * 4),
                dtype=np.float32
            ).astype(int)

            config_blob = self.blob_store.get(f"{dataset.processed_blob_prefix}/config.json")
            if config_blob:
                config = json.loads(config_blob.decode())
                mean = config.get('mean', [0.5])
                std = config.get('std', [0.5])
            else:
                mean, std = [0.5], [0.5]

            for i in range(min(num_samples, n_to_read)):
                img_data = images[i]
                img_data = (img_data * std[0] + mean[0]) * 255
                img_data = np.clip(img_data, 0, 255).astype(np.uint8)

                if img_data.shape[0] == 1:
                    img = Image.fromarray(img_data[0], mode='L')
                else:
                    img = Image.fromarray(img_data.transpose(1, 2, 0))

                img = img.resize((64, 64), Image.Resampling.NEAREST)

                buffer = BytesIO()
                img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                class_names = dataset.class_names or []
                label = class_names[labels[i]] if labels[i] < len(class_names) else str(labels[i])

                samples.append({
                    "image": img_base64,
                    "label": label
                })

        except Exception as e:
            logger.error("Error generating preview: %s", e)

        return samples

    def _preview_from_processed(self, dataset: Dataset, num_samples: int) -> List[Dict]:
        """Generic preview from processed binary data."""
        return self._preview_mnist_from_blobs(dataset, num_samples)

    def _preview_text_from_blobs(self, dataset: Dataset) -> Dict:
        """Generate preview from processed text data in blob storage."""
        preview = {
            "vocab_size": 0,
            "sample_text": "",
            "sample_tokens": [],
            "tokenizer_type": "character"
        }

        if not dataset.processed_blob_prefix:
            return preview

        vocab_blob = self.blob_store.get(f"{dataset.processed_blob_prefix}/vocabulary.json")
        config_blob = self.blob_store.get(f"{dataset.processed_blob_prefix}/config.json")

        if vocab_blob:
            vocab_data = json.loads(vocab_blob.decode())
            preview["vocab_size"] = vocab_data.get("vocab_size", 0)
            preview["tokenizer_type"] = vocab_data.get("tokenizer_type", "character")

            idx_to_token = vocab_data.get("idx_to_token", {})
            special_tokens = set(vocab_data.get("special_tokens", {}).values())
            sample_tokens = []
            for idx in range(len(idx_to_token)):
                if idx not in special_tokens:
                    token = idx_to_token.get(str(idx), "")
                    if token:
                        sample_tokens.append(token)
                    if len(sample_tokens) >= 50:
                        break
            preview["sample_tokens"] = sample_tokens

        if config_blob:
            config = json.loads(config_blob.decode())
            preview["seq_length"] = config.get("seq_length", 128)
            preview["train_sequences"] = config.get("train_sequences", 0)
            preview["test_sequences"] = config.get("test_sequences", 0)
            preview["total_tokens"] = config.get("total_tokens", 0)
            preview["unique_tokens"] = config.get("unique_tokens", 0)

        if dataset.raw_blob_key:
            raw_text = self.blob_store.get(dataset.raw_blob_key)
            if raw_text:
                try:
                    text = raw_text.decode('utf-8')[:500]
                except:
                    text = raw_text.decode('latin-1')[:500]
                preview["sample_text"] = text

        return preview
