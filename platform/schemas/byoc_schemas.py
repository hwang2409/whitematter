"""BYOC training schemas."""
from pydantic import BaseModel


class LaunchRequest(BaseModel):
    model_config_data: dict  # renamed to avoid pydantic conflict
    dataset_config: dict
    instance_type: str = "g4dn.xlarge"
    region: str | None = None


class JobStatusResponse(BaseModel):
    id: str
    status: str
    instance_type: str
    instance_id: str | None = None
    region: str
    started_at: str | None = None
    completed_at: str | None = None
    metrics: dict | None = None
    error_message: str | None = None


class HeartbeatRequest(BaseModel):
    pass


class LogRequest(BaseModel):
    message: str


class MetricsRequest(BaseModel):
    epoch: int
    loss: float
    accuracy: float | None = None


class CompleteRequest(BaseModel):
    metrics: dict


class FailRequest(BaseModel):
    error_message: str
