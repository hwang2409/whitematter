"""
SQLAlchemy database models for whitematter platform.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime,
    ForeignKey, JSON, Enum, Boolean, LargeBinary
)
from sqlalchemy.orm import relationship, declarative_base
import enum

Base = declarative_base()


class DatasetStatus(str, enum.Enum):
    CREATED = "created"
    EXTRACTING = "extracting"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class DatasetFormat(str, enum.Enum):
    FOLDER_PER_CLASS = "folder_per_class"
    MNIST_IDX = "mnist_idx"
    CSV_LABELS = "csv_labels"
    FLAT_IMAGES = "flat_images"
    TEXT_FILE = "text_file"
    UNKNOWN = "unknown"


class DataType(str, enum.Enum):
    IMAGE = "image"
    TEXT = "text"
    TABULAR = "tabular"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    QUEUED = "queued"
    COMPILING = "compiling"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelStatus(str, enum.Enum):
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class Dataset(Base):
    """Dataset metadata and configuration."""
    __tablename__ = "datasets"

    id = Column(String(32), primary_key=True)
    name = Column(String(255), nullable=False)
    data_type = Column(String(20), default=DataType.IMAGE.value)
    format = Column(String(30), default=DatasetFormat.UNKNOWN.value)
    status = Column(String(20), default=DatasetStatus.CREATED.value)
    error_message = Column(Text, nullable=True)

    # Dataset statistics
    num_classes = Column(Integer, default=0)
    total_samples = Column(Integer, default=0)
    train_samples = Column(Integer, default=0)
    test_samples = Column(Integer, default=0)

    # Shape and class info stored as JSON
    input_shape = Column(JSON, default=list)
    class_names = Column(JSON, default=list)
    samples_per_class = Column(JSON, default=dict)

    # Preprocessing config
    preprocessing_config = Column(JSON, nullable=True)

    # Blob references
    raw_blob_key = Column(String(255), nullable=True)  # Key to raw uploaded file
    processed_blob_prefix = Column(String(255), nullable=True)  # Prefix for processed files

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    models = relationship("Model", back_populates="dataset_rel")


class Model(Base):
    """Trained model metadata."""
    __tablename__ = "models"

    id = Column(String(32), primary_key=True)
    name = Column(String(255), nullable=False)
    dataset_id = Column(String(32), ForeignKey("datasets.id"), nullable=True)
    dataset_name = Column(String(255), nullable=True)  # For display (includes preset names)
    architecture_name = Column(String(255), nullable=True)
    status = Column(String(20), default=ModelStatus.TRAINING.value)

    # Training results
    epochs_trained = Column(Integer, default=0)
    best_accuracy = Column(Float, default=0.0)
    final_loss = Column(Float, nullable=True)

    # Configuration stored as JSON
    architecture_config = Column(JSON, nullable=True)
    training_config = Column(JSON, nullable=True)

    # Blob references
    weights_blob_key = Column(String(255), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    dataset_rel = relationship("Dataset", back_populates="models")
    training_jobs = relationship("TrainingJob", back_populates="model")
    training_history = relationship("TrainingHistory", back_populates="model", order_by="TrainingHistory.epoch")


class TrainingJob(Base):
    """Training job tracking."""
    __tablename__ = "training_jobs"

    id = Column(String(32), primary_key=True)
    model_id = Column(String(32), ForeignKey("models.id"), nullable=False)
    worker_id = Column(String(64), nullable=True)  # Which worker is handling this

    status = Column(String(20), default=JobStatus.PENDING.value)
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=0)
    current_loss = Column(Float, nullable=True)
    current_accuracy = Column(Float, nullable=True)
    message = Column(Text, nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)

    # Blob references for artifacts
    generated_code_blob_key = Column(String(255), nullable=True)
    build_log_blob_key = Column(String(255), nullable=True)
    training_log_blob_key = Column(String(255), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    model = relationship("Model", back_populates="training_jobs")


class TrainingHistory(Base):
    """Per-epoch training metrics."""
    __tablename__ = "training_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(32), ForeignKey("models.id"), nullable=False)
    job_id = Column(String(32), ForeignKey("training_jobs.id"), nullable=True)

    epoch = Column(Integer, nullable=False)
    loss = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    learning_rate = Column(Float, nullable=True)

    recorded_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    model = relationship("Model", back_populates="training_history")


class BlobMetadata(Base):
    """Metadata and storage for blob objects - everything in the database."""
    __tablename__ = "blob_metadata"

    key = Column(String(255), primary_key=True)
    content_type = Column(String(100), nullable=True)
    size_bytes = Column(Integer, default=0)
    checksum = Column(String(64), nullable=True)  # SHA256

    # Actual binary data stored in database
    data = Column(LargeBinary, nullable=True)

    # For cleanup
    reference_count = Column(Integer, default=1)

    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
