import json
import logging
import struct
import tempfile
import zipfile
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np

from db import (
    get_db_session, get_blob_store,
    Dataset, DatasetStatus, DatasetFormat, DataType
)
from preprocessing import TextProcessor, TokenizerType
from services.dataset_analyzer import detect_format, analyze_dataset
from services.dataset_preview import DatasetPreviewService

logger = logging.getLogger(__name__)


class DatasetService:
    """Service for managing datasets with database and blob storage."""

    def __init__(self):
        self.blob_store = get_blob_store()
        self._preview_service = DatasetPreviewService(self.blob_store)

    def create_dataset(self, name: str) -> Dict:
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
            return self._dataset_to_dict(dataset)

    def upload_zip(self, dataset_id: str, file_content: bytes, filename: str) -> Dict:
        """Upload and process a ZIP file for a dataset."""
        raw_blob_key = f"datasets/{dataset_id}/raw/{filename}"
        self.blob_store.put(file_content, key=raw_blob_key, content_type="application/zip")

        with get_db_session() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            dataset.raw_blob_key = raw_blob_key
            dataset.status = DatasetStatus.EXTRACTING.value

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_dir = temp_path / "raw"
            processed_dir = temp_path / "processed"
            raw_dir.mkdir()
            processed_dir.mkdir()

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

            format_type, format_info = detect_format(raw_dir)
            analysis = analyze_dataset(format_type, format_info, raw_dir)

            if analysis['data_type'] == DataType.IMAGE.value:
                config = self._process_images(
                    raw_dir, processed_dir,
                    format_type, format_info, analysis
                )
            else:
                config = analysis

            processed_prefix = f"datasets/{dataset_id}/processed"
            for file_path in processed_dir.iterdir():
                if file_path.is_file():
                    blob_key = f"{processed_prefix}/{file_path.name}"
                    self.blob_store.put_file(file_path, key=blob_key)

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
        """Upload and process a text file for language model training."""
        raw_blob_key = f"datasets/{dataset_id}/raw/{filename}"
        self.blob_store.put(file_content, key=raw_blob_key, content_type="text/plain")

        with get_db_session() as db:
            dataset = db.query(Dataset).filter_by(id=dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            dataset.raw_blob_key = raw_blob_key
            dataset.status = DatasetStatus.PROCESSING.value

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            processed_dir = temp_path / "processed"
            processed_dir.mkdir()

            try:
                text = file_content.decode('utf-8')
            except UnicodeDecodeError:
                text = file_content.decode('latin-1')

            text = text.strip()

            tok_type = TokenizerType(tokenizer_type)
            processor = TextProcessor(
                tokenizer_type=tok_type,
                seq_length=seq_length,
                min_freq=1
            )

            vocab = processor.build_vocabulary(text)

            if tok_type == TokenizerType.CHARACTER:
                total_tokens = len(text)
            else:
                total_tokens = len(text.split())

            text_stats = {
                'total_tokens': total_tokens,
                'unique_tokens': vocab.vocab_size - len(vocab.special_tokens)
            }

            train_in, train_tgt, test_in, test_tgt = processor.process_text(text, vocab)

            config = processor.save_processed_data(
                processed_dir,
                train_in, train_tgt, test_in, test_tgt,
                vocab, text_stats
            )

            processed_prefix = f"datasets/{dataset_id}/processed"
            for file_path in processed_dir.iterdir():
                if file_path.is_file():
                    blob_key = f"{processed_prefix}/{file_path.name}"
                    self.blob_store.put_file(file_path, key=blob_key)

            with get_db_session() as db:
                dataset = db.query(Dataset).filter_by(id=dataset_id).first()
                dataset.data_type = DataType.TEXT.value
                dataset.format = DatasetFormat.TEXT_FILE.value
                dataset.num_classes = vocab.vocab_size
                dataset.class_names = list(vocab.token_to_idx.keys())[:100]
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

            if dataset.raw_blob_key:
                self.blob_store.delete(dataset.raw_blob_key)
            if dataset.processed_blob_prefix:
                for suffix in ['train_images.bin', 'train_labels.bin',
                              'test_images.bin', 'test_labels.bin', 'config.json']:
                    self.blob_store.delete(f"{dataset.processed_blob_prefix}/{suffix}")
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
            dataset_dict = self._dataset_to_dict(dataset)
            return self._preview_service.get_preview(dataset, dataset_dict, num_samples)

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
        import struct

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
            split_idx = int(len(train_images) * 0.8)
            indices = np.random.permutation(len(train_images))
            test_images = train_images[indices[split_idx:]]
            test_labels = train_labels[indices[split_idx:]]
            train_images = train_images[indices[:split_idx]]
            train_labels = train_labels[indices[:split_idx]]

        mean = [float(train_images.mean())]
        std = [float(max(train_images.std(), 1e-7))]

        train_images = (train_images - mean[0]) / std[0]
        test_images = (test_images - mean[0]) / std[0]

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
