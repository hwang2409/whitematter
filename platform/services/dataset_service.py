"""
Dataset service - handles dataset operations using database and blob storage.
"""
import json
import logging
import struct
import tempfile
import zipfile
import base64
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

import numpy as np
from PIL import Image

from db import (
    get_db_session, get_blob_store,
    Dataset, DatasetStatus, DatasetFormat, DataType
)
from preprocessing import TextProcessor, TextVocabulary, TokenizerType

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}


class DatasetService:
    """Service for managing datasets with database and blob storage."""

    def __init__(self):
        self.blob_store = get_blob_store()

    def create_dataset(self, name: str) -> Dataset:
        """Create a new dataset entry in the database."""
        dataset_id = uuid.uuid4().hex[:16]

        with get_db_session() as db:
            dataset = Dataset(
                id=dataset_id,
                name=name,
                status=DatasetStatus.CREATED.value,
                created_at=datetime.utcnow()
            )
            db.add(dataset)
            db.flush()

            # Return a detached copy
            return self._dataset_to_dict(dataset)

    def upload_zip(self, dataset_id: str, file_content: bytes, filename: str) -> Dict:
        """
        Upload and process a ZIP file for a dataset.
        Returns the updated dataset info.
        """
        # Store raw zip in blob storage
        raw_blob_key = f"datasets/{dataset_id}/raw/{filename}"
        self.blob_store.put(file_content, key=raw_blob_key, content_type="application/zip")

        # Update dataset with raw blob reference
        with get_db_session() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            dataset.raw_blob_key = raw_blob_key
            dataset.status = DatasetStatus.EXTRACTING.value

        # Extract and analyze in temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_dir = temp_path / "raw"
            processed_dir = temp_path / "processed"
            raw_dir.mkdir()
            processed_dir.mkdir()

            # Extract ZIP
            zip_buffer = BytesIO(file_content)
            with zipfile.ZipFile(zip_buffer, 'r') as zf:
                zf.extractall(raw_dir)

            # Handle nested folder
            contents = list(raw_dir.iterdir())
            if len(contents) == 1 and contents[0].is_dir():
                nested_dir = contents[0]
                import shutil
                for item in nested_dir.iterdir():
                    shutil.move(str(item), str(raw_dir / item.name))
                nested_dir.rmdir()

            # Detect format and analyze
            format_type, format_info = self._detect_format(raw_dir)

            # Analyze based on format
            analysis = self._analyze_dataset(format_type, format_info, raw_dir)

            # Process the dataset
            if analysis['data_type'] == DataType.IMAGE.value:
                config = self._process_images(
                    raw_dir, processed_dir,
                    format_type, format_info, analysis
                )
            else:
                config = analysis

            # Upload processed files to blob storage
            processed_prefix = f"datasets/{dataset_id}/processed"
            for file_path in processed_dir.iterdir():
                if file_path.is_file():
                    blob_key = f"{processed_prefix}/{file_path.name}"
                    self.blob_store.put_file(file_path, key=blob_key)

            # Update database
            with get_db_session() as db:
                dataset = db.query(Dataset).filter_by(id=dataset_id).first()
                dataset.data_type = analysis['data_type']
                dataset.format = format_type.value
                dataset.num_classes = analysis['num_classes']
                dataset.class_names = analysis['class_names']
                dataset.samples_per_class = analysis.get('samples_per_class', {})
                dataset.total_samples = analysis['total_samples']
                dataset.input_shape = config.get('input_shape', analysis.get('input_shape', []))
                dataset.train_samples = config.get('train_samples', 0)
                dataset.test_samples = config.get('test_samples', 0)
                dataset.processed_blob_prefix = processed_prefix
                dataset.status = DatasetStatus.READY.value

                return self._dataset_to_dict(dataset)

    def upload_text(
        self,
        dataset_id: str,
        file_content: bytes,
        filename: str,
        tokenizer_type: str = "character",
        seq_length: int = 128
    ) -> Dict:
        """
        Upload and process a text file for language model training.
        Returns the updated dataset info.
        """
        # Store raw text in blob storage
        raw_blob_key = f"datasets/{dataset_id}/raw/{filename}"
        self.blob_store.put(file_content, key=raw_blob_key, content_type="text/plain")

        # Update dataset with raw blob reference
        with get_db_session() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            dataset.raw_blob_key = raw_blob_key
            dataset.status = DatasetStatus.PROCESSING.value

        # Process text in temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            processed_dir = temp_path / "processed"
            processed_dir.mkdir()

            # Decode text content
            try:
                text = file_content.decode('utf-8')
            except UnicodeDecodeError:
                text = file_content.decode('latin-1')

            # Clean text (basic normalization)
            text = text.strip()

            # Create text processor
            tok_type = TokenizerType(tokenizer_type)
            processor = TextProcessor(
                tokenizer_type=tok_type,
                seq_length=seq_length,
                min_freq=1
            )

            # Build vocabulary
            vocab = processor.build_vocabulary(text)

            # Get stats
            if tok_type == TokenizerType.CHARACTER:
                total_tokens = len(text)
            else:
                total_tokens = len(text.split())

            text_stats = {
                'total_tokens': total_tokens,
                'unique_tokens': vocab.vocab_size - len(vocab.special_tokens)
            }

            # Process into sequences
            train_in, train_tgt, test_in, test_tgt = processor.process_text(text, vocab)

            # Save processed data
            config = processor.save_processed_data(
                processed_dir,
                train_in, train_tgt, test_in, test_tgt,
                vocab, text_stats
            )

            # Upload processed files to blob storage
            processed_prefix = f"datasets/{dataset_id}/processed"
            for file_path in processed_dir.iterdir():
                if file_path.is_file():
                    blob_key = f"{processed_prefix}/{file_path.name}"
                    self.blob_store.put_file(file_path, key=blob_key)

            # Update database
            with get_db_session() as db:
                dataset = db.query(Dataset).filter_by(id=dataset_id).first()
                dataset.data_type = DataType.TEXT.value
                dataset.format = DatasetFormat.TEXT_FILE.value
                dataset.num_classes = vocab.vocab_size  # For text, num_classes = vocab_size
                dataset.class_names = list(vocab.token_to_idx.keys())[:100]  # First 100 tokens
                dataset.total_samples = total_tokens
                dataset.train_samples = len(train_in)
                dataset.test_samples = len(test_in)
                dataset.input_shape = [seq_length]
                dataset.preprocessing_config = {
                    'tokenizer_type': tokenizer_type,
                    'seq_length': seq_length,
                    'vocab_size': vocab.vocab_size,
                    'special_tokens': vocab.special_tokens
                }
                dataset.processed_blob_prefix = processed_prefix
                dataset.status = DatasetStatus.READY.value

                return self._dataset_to_dict(dataset)

    def get_dataset(self, dataset_id: str) -> Optional[Dict]:
        """Get dataset by ID."""
        with get_db_session() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            if not dataset:
                return None
            return self._dataset_to_dict(dataset)

    def list_datasets(self) -> List[Dict]:
        """List all datasets."""
        with get_db_session() as db:
            datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
            return [self._dataset_to_dict(d) for d in datasets]

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and its associated blobs."""
        with get_db_session() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            if not dataset:
                return False

            # Delete blobs
            if dataset.raw_blob_key:
                self.blob_store.delete(dataset.raw_blob_key)
            if dataset.processed_blob_prefix:
                # Delete all processed files (image datasets)
                for suffix in ['train_images.bin', 'train_labels.bin',
                              'test_images.bin', 'test_labels.bin', 'config.json']:
                    self.blob_store.delete(f"{dataset.processed_blob_prefix}/{suffix}")
                # Delete text dataset files
                for suffix in ['train_inputs.bin', 'train_targets.bin',
                              'test_inputs.bin', 'test_targets.bin', 'vocabulary.json']:
                    self.blob_store.delete(f"{dataset.processed_blob_prefix}/{suffix}")

            db.delete(dataset)
            return True

    def get_preview(self, dataset_id: str, num_samples: int = 18) -> Optional[Dict]:
        """Get dataset preview with sample images or text."""
        with get_db_session() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            if not dataset:
                return None

            result = {
                "metadata": self._dataset_to_dict(dataset),
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

        # Load a small portion of test images for preview
        images_blob = self.blob_store.get(f"{dataset.processed_blob_prefix}/test_images.bin")
        labels_blob = self.blob_store.get(f"{dataset.processed_blob_prefix}/test_labels.bin")

        if not images_blob or not labels_blob:
            return samples

        try:
            # Parse tensor format
            TENSOR_MAGIC = 0x54454E53
            images_data = BytesIO(images_blob)
            labels_data = BytesIO(labels_blob)

            # Read images header
            magic = struct.unpack('I', images_data.read(4))[0]
            if magic != TENSOR_MAGIC:
                return samples

            ndims = struct.unpack('I', images_data.read(4))[0]
            shape = [struct.unpack('Q', images_data.read(8))[0] for _ in range(ndims)]

            # Read only first num_samples*2 images (to get variety)
            n_to_read = min(num_samples * 2, shape[0])
            image_size = np.prod(shape[1:])
            images = np.frombuffer(
                images_data.read(int(n_to_read * image_size * 4)),
                dtype=np.float32
            ).reshape(n_to_read, *shape[1:])

            # Read labels header
            magic = struct.unpack('I', labels_data.read(4))[0]
            ndims = struct.unpack('I', labels_data.read(4))[0]
            _ = [struct.unpack('Q', labels_data.read(8))[0] for _ in range(ndims)]
            labels = np.frombuffer(
                labels_data.read(n_to_read * 4),
                dtype=np.float32
            ).astype(int)

            # Get samples from different classes
            config_blob = self.blob_store.get(f"{dataset.processed_blob_prefix}/config.json")
            if config_blob:
                config = json.loads(config_blob.decode())
                mean = config.get('mean', [0.5])
                std = config.get('std', [0.5])
            else:
                mean, std = [0.5], [0.5]

            # Denormalize and convert to images
            for i in range(min(num_samples, n_to_read)):
                img_data = images[i]
                # Denormalize
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

        # Load vocabulary
        vocab_blob = self.blob_store.get(f"{dataset.processed_blob_prefix}/vocabulary.json")
        config_blob = self.blob_store.get(f"{dataset.processed_blob_prefix}/config.json")

        if vocab_blob:
            vocab_data = json.loads(vocab_blob.decode())
            preview["vocab_size"] = vocab_data.get("vocab_size", 0)
            preview["tokenizer_type"] = vocab_data.get("tokenizer_type", "character")

            # Show sample tokens (first 50 non-special tokens)
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

        # Try to get original text from raw blob for sample
        if dataset.raw_blob_key:
            raw_text = self.blob_store.get(dataset.raw_blob_key)
            if raw_text:
                try:
                    text = raw_text.decode('utf-8')[:500]  # First 500 chars
                except:
                    text = raw_text.decode('latin-1')[:500]
                preview["sample_text"] = text

        return preview

    def _detect_format(self, raw_dir: Path) -> Tuple[DatasetFormat, Dict]:
        """Detect dataset format."""
        all_items = list(raw_dir.iterdir())
        files = [f for f in all_items if f.is_file()]
        dirs = [d for d in all_items if d.is_dir()]

        # Check for MNIST IDX format
        idx_images = [f for f in files if
                      ('images' in f.name.lower() or 'image' in f.name.lower()) and
                      ('ubyte' in f.name.lower() or 'idx' in f.name.lower())]
        idx_labels = [f for f in files if
                      ('labels' in f.name.lower() or 'label' in f.name.lower()) and
                      ('ubyte' in f.name.lower() or 'idx' in f.name.lower())]

        if idx_images and idx_labels:
            train_images = [f for f in idx_images if 'train' in f.name.lower()]
            train_labels = [f for f in idx_labels if 'train' in f.name.lower()]
            return DatasetFormat.MNIST_IDX, {
                "images_file": train_images[0] if train_images else idx_images[0],
                "labels_file": train_labels[0] if train_labels else idx_labels[0],
                "all_images": idx_images,
                "all_labels": idx_labels
            }

        # Check for folder-per-class
        if dirs:
            has_samples = any(any(d.iterdir()) for d in dirs)
            if has_samples:
                return DatasetFormat.FOLDER_PER_CLASS, {"class_dirs": dirs}

        # Flat images
        image_files = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
        if image_files:
            return DatasetFormat.FLAT_IMAGES, {"image_files": image_files}

        return DatasetFormat.UNKNOWN, {"files": files, "dirs": dirs}

    def _analyze_dataset(self, format_type: DatasetFormat, format_info: Dict, raw_dir: Path) -> Dict:
        """Analyze dataset and return metadata."""
        if format_type == DatasetFormat.MNIST_IDX:
            return self._analyze_mnist_idx(format_info)
        elif format_type == DatasetFormat.FOLDER_PER_CLASS:
            return self._analyze_folder_per_class(format_info)
        elif format_type == DatasetFormat.FLAT_IMAGES:
            return self._analyze_flat_images(format_info)
        else:
            return {
                'data_type': DataType.UNKNOWN.value,
                'num_classes': 0,
                'class_names': [],
                'total_samples': 0,
                'input_shape': []
            }

    def _analyze_mnist_idx(self, info: Dict) -> Dict:
        """Analyze MNIST IDX format."""
        images_file = info["images_file"]
        labels_file = info["labels_file"]

        with open(images_file, 'rb') as f:
            struct.unpack('>I', f.read(4))  # magic
            num_images = struct.unpack('>I', f.read(4))[0]
            rows = struct.unpack('>I', f.read(4))[0]
            cols = struct.unpack('>I', f.read(4))[0]

        with open(labels_file, 'rb') as f:
            struct.unpack('>I', f.read(4))  # magic
            struct.unpack('>I', f.read(4))  # num_labels
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        unique_labels = sorted(set(labels))
        class_names = [str(l) for l in unique_labels]
        samples_per_class = {str(l): int(np.sum(labels == l)) for l in unique_labels}

        return {
            'data_type': DataType.IMAGE.value,
            'num_classes': len(unique_labels),
            'class_names': class_names,
            'samples_per_class': samples_per_class,
            'total_samples': num_images,
            'input_shape': [1, rows, cols]
        }

    def _analyze_folder_per_class(self, info: Dict) -> Dict:
        """Analyze folder-per-class format."""
        class_dirs = info["class_dirs"]
        class_names = sorted([d.name for d in class_dirs])
        samples_per_class = {}
        sample_files = []

        for class_dir in class_dirs:
            files = [f for f in class_dir.iterdir() if f.is_file()]
            samples_per_class[class_dir.name] = len(files)
            sample_files.extend(files[:5])

        # Determine input shape from sample
        input_shape = [3, 32, 32]  # Default
        if sample_files:
            try:
                img = Image.open(sample_files[0])
                w, h = img.size
                channels = 3 if img.mode == 'RGB' else 1
                target = min(max(w, h, 32), 224)
                if target <= 32:
                    target = 32
                elif target <= 64:
                    target = 64
                elif target <= 128:
                    target = 128
                else:
                    target = 224
                input_shape = [channels, target, target]
            except:
                pass

        return {
            'data_type': DataType.IMAGE.value,
            'num_classes': len(class_names),
            'class_names': class_names,
            'samples_per_class': samples_per_class,
            'total_samples': sum(samples_per_class.values()),
            'input_shape': input_shape
        }

    def _analyze_flat_images(self, info: Dict) -> Dict:
        """Analyze flat images folder."""
        image_files = info["image_files"]

        input_shape = [3, 32, 32]
        if image_files:
            try:
                img = Image.open(image_files[0])
                w, h = img.size
                channels = 3 if img.mode == 'RGB' else 1
                target = min(max(w, h, 32), 224)
                input_shape = [channels, target, target]
            except:
                pass

        return {
            'data_type': DataType.IMAGE.value,
            'num_classes': 0,
            'class_names': [],
            'total_samples': len(image_files),
            'input_shape': input_shape
        }

    def _process_images(
        self, raw_dir: Path, output_dir: Path,
        format_type: DatasetFormat, format_info: Dict,
        analysis: Dict
    ) -> Dict:
        """Process images into binary tensor format."""
        if format_type == DatasetFormat.MNIST_IDX:
            return self._process_mnist_idx(raw_dir, output_dir, format_info, analysis)
        elif format_type == DatasetFormat.FOLDER_PER_CLASS:
            return self._process_folder_per_class(raw_dir, output_dir, format_info, analysis)
        else:
            return analysis

    def _process_mnist_idx(self, raw_dir: Path, output_dir: Path, info: Dict, analysis: Dict) -> Dict:
        """Process MNIST IDX format to binary tensors."""
        def read_idx_images(filepath):
            with open(filepath, 'rb') as f:
                struct.unpack('>I', f.read(4))
                num_images = struct.unpack('>I', f.read(4))[0]
                rows = struct.unpack('>I', f.read(4))[0]
                cols = struct.unpack('>I', f.read(4))[0]
                data = np.frombuffer(f.read(), dtype=np.uint8)
                return data.reshape(num_images, 1, rows, cols).astype(np.float32) / 255.0, rows, cols

        def read_idx_labels(filepath):
            with open(filepath, 'rb') as f:
                struct.unpack('>I', f.read(4))
                struct.unpack('>I', f.read(4))
                return np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32)

        def save_tensor(path, data):
            TENSOR_MAGIC = 0x54454E53
            with open(path, 'wb') as f:
                f.write(struct.pack('I', TENSOR_MAGIC))
                f.write(struct.pack('I', len(data.shape)))
                for dim in data.shape:
                    f.write(struct.pack('Q', dim))
                f.write(np.ascontiguousarray(data, dtype=np.float32).tobytes())

        # Find train/test files
        all_images = info.get("all_images", [info["images_file"]])
        all_labels = info.get("all_labels", [info["labels_file"]])

        train_img_file = next((f for f in all_images if 'train' in f.name.lower()), all_images[0])
        train_lbl_file = next((f for f in all_labels if 'train' in f.name.lower()), all_labels[0])
        test_img_file = next((f for f in all_images if 't10k' in f.name.lower() or 'test' in f.name.lower()), None)
        test_lbl_file = next((f for f in all_labels if 't10k' in f.name.lower() or 'test' in f.name.lower()), None)

        train_images, rows, cols = read_idx_images(train_img_file)
        train_labels = read_idx_labels(train_lbl_file)

        if test_img_file and test_lbl_file:
            test_images, _, _ = read_idx_images(test_img_file)
            test_labels = read_idx_labels(test_lbl_file)
        else:
            # Split 80/20
            split_idx = int(len(train_images) * 0.8)
            indices = np.random.permutation(len(train_images))
            test_images = train_images[indices[split_idx:]]
            test_labels = train_labels[indices[split_idx:]]
            train_images = train_images[indices[:split_idx]]
            train_labels = train_labels[indices[:split_idx]]

        # Compute normalization stats
        mean = [float(train_images.mean())]
        std = [float(max(train_images.std(), 1e-7))]

        # Normalize
        train_images = (train_images - mean[0]) / std[0]
        test_images = (test_images - mean[0]) / std[0]

        # Save tensors
        save_tensor(output_dir / "train_images.bin", train_images)
        save_tensor(output_dir / "train_labels.bin", train_labels)
        save_tensor(output_dir / "test_images.bin", test_images)
        save_tensor(output_dir / "test_labels.bin", test_labels)

        config = {
            "target_size": [rows, cols],
            "channels": 1,
            "mean": mean,
            "std": std,
            "num_classes": analysis['num_classes'],
            "class_names": analysis['class_names'],
            "train_samples": len(train_images),
            "test_samples": len(test_images),
            "input_shape": [1, rows, cols]
        }

        with open(output_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        return config

    def _process_folder_per_class(self, raw_dir: Path, output_dir: Path, info: Dict, analysis: Dict) -> Dict:
        """Process folder-per-class format to binary tensors."""
        from preprocessing import ImageProcessor

        class_dirs = info["class_dirs"]
        class_names = analysis['class_names']
        input_shape = analysis['input_shape']

        processor = ImageProcessor(
            target_size=(input_shape[1], input_shape[2]),
            channels=input_shape[0]
        )

        return processor.process_dataset(raw_dir, output_dir, class_names)

    def _dataset_to_dict(self, dataset: Dataset) -> Dict:
        """Convert dataset ORM object to dictionary."""
        return {
            'id': dataset.id,
            'name': dataset.name,
            'data_type': dataset.data_type,
            'format': dataset.format,
            'status': dataset.status,
            'error_message': dataset.error_message,
            'num_classes': dataset.num_classes,
            'total_samples': dataset.total_samples,
            'train_samples': dataset.train_samples,
            'test_samples': dataset.test_samples,
            'input_shape': dataset.input_shape,
            'class_names': dataset.class_names,
            'samples_per_class': dataset.samples_per_class,
            'preprocessing_config': dataset.preprocessing_config,
            'processed_blob_prefix': dataset.processed_blob_prefix,
            'raw_blob_key': dataset.raw_blob_key,
            'created_at': dataset.created_at.isoformat() if dataset.created_at else None,
            'updated_at': dataset.updated_at.isoformat() if dataset.updated_at else None
        }
