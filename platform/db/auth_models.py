import enum
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, ForeignKey,
    JSON, LargeBinary, UniqueConstraint
)
from sqlalchemy.orm import relationship
from .models import Base


def gen_uuid():
    return uuid.uuid4().hex


class User(Base):
    __tablename__ = "users"
    id = Column(String(32), primary_key=True, default=gen_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)
    oauth_provider = Column(String(50), nullable=True)
    oauth_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    aws_credentials = relationship("AWSCredential", back_populates="user", uselist=False)
    byoc_jobs = relationship("ByocTrainingJob", back_populates="user")
    architectures = relationship("ModelArchitecture", back_populates="user")
    deployments = relationship("Deployment", back_populates="user")

    __table_args__ = (
        UniqueConstraint("oauth_provider", "oauth_id", name="uq_oauth"),
    )


class AWSCredential(Base):
    __tablename__ = "aws_credentials"
    id = Column(String(32), primary_key=True, default=gen_uuid)
    user_id = Column(String(32), ForeignKey("users.id"), unique=True, nullable=False)
    encrypted_access_key = Column(LargeBinary, nullable=False)
    encrypted_secret_key = Column(LargeBinary, nullable=False)
    default_region = Column(String(30), default="us-east-1")
    default_instance_type = Column(String(30), default="g4dn.xlarge")
    # S3-compatible storage: when set, use this endpoint (R2, B2, MinIO, etc.)
    endpoint_url = Column(String(512), nullable=True)
    provider = Column(String(20), nullable=True)  # 'aws' | 'r2' | 'b2' | 'custom'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="aws_credentials")


class ByocJobStatus(str, enum.Enum):
    PENDING = "pending"
    LAUNCHING = "launching"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class ByocTrainingJob(Base):
    __tablename__ = "byoc_training_jobs"
    id = Column(String(32), primary_key=True, default=gen_uuid)
    user_id = Column(String(32), ForeignKey("users.id"), nullable=False)
    model_config = Column(JSON, nullable=False)
    dataset_config = Column(JSON, nullable=False)
    instance_type = Column(String(30), default="g4dn.xlarge")
    instance_id = Column(String(30), nullable=True)
    region = Column(String(30), default="us-east-1")
    status = Column(String(20), default=ByocJobStatus.PENDING.value)
    job_token_hash = Column(String(128), nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    metrics = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    user = relationship("User", back_populates="byoc_jobs")


class ModelArchitecture(Base):
    __tablename__ = "model_architectures"
    id = Column(String(32), primary_key=True, default=gen_uuid)
    user_id = Column(String(32), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    config = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="architectures")


class DeploymentStatus(str, enum.Enum):
    PENDING = "pending"
    LAUNCHING = "launching"
    BOOTSTRAPPING = "bootstrapping"
    LIVE = "live"
    FAILED = "failed"
    TERMINATED = "terminated"


class Deployment(Base):
    """One-click deploy: model + infer binary on EC2 inference instance."""
    __tablename__ = "deployments"
    id = Column(String(32), primary_key=True, default=gen_uuid)
    user_id = Column(String(32), ForeignKey("users.id"), nullable=False)
    model_id = Column(String(32), nullable=False)
    target_type = Column(String(20), default="ec2")
    status = Column(String(20), default=DeploymentStatus.PENDING.value)
    instance_id = Column(String(30), nullable=True)
    endpoint_url = Column(String(512), nullable=True)
    region = Column(String(30), default="us-east-1")
    deployment_token = Column(String(128), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="deployments")