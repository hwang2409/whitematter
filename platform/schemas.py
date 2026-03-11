"""
Pydantic request/response models for the Whitematter Model Server.
"""

from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


class LayerConfig(BaseModel):
    type: str
    params: Dict[str, Any] = {}

class OptimizerConfig(BaseModel):
    type: str = "sgd"
    params: Dict[str, float] = {}

class SchedulerConfig(BaseModel):
    type: str = "none"
    params: Dict[str, Any] = {}

class AugmentationConfig(BaseModel):
    type: str
    params: Dict[str, Any] = {}

class TrainRequest(BaseModel):
    dataset: str
    name: Optional[str] = None
    preset: Optional[str] = None
    layers: Optional[List[LayerConfig]] = None
    epochs: int = 10
    batch_size: int = 128
    optimizer: OptimizerConfig = Field(default_factory=lambda: OptimizerConfig(type="sgd", params={"learning_rate": 0.01, "momentum": 0.9}))
    scheduler: SchedulerConfig = Field(default_factory=lambda: SchedulerConfig(type="none"))
    augmentations: List[AugmentationConfig] = []

class TrainStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ModelMetadata(BaseModel):
    id: str
    name: str
    dataset: str
    architecture: str
    created_at: str
    epochs_trained: int
    best_accuracy: float
    status: TrainStatus
    training_history: list = []
    config: Dict[str, Any] = {}

class DesignRequest(BaseModel):
    dataset_id: str
    prompt: str

class RefineRequest(BaseModel):
    architecture: Dict[str, Any]
    feedback: str

class PreviewCodeRequest(BaseModel):
    dataset_id: str
    architecture: Dict[str, Any]

class CustomTrainRequest(BaseModel):
    dataset_id: str
    architecture: Dict[str, Any]
    name: Optional[str] = None

class DesignHelpRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
