"""Chat conversation and message models."""
import uuid
from datetime import datetime
from sqlalchemy import Boolean, Column, String, Text, DateTime, ForeignKey, JSON
from .models import Base


def gen_short_uuid():
    return uuid.uuid4().hex[:16]


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String(16), primary_key=True, default=gen_short_uuid)
    user_id = Column(String(32), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=True)
    phase = Column(String(30), default="greeting")
    architecture = Column(JSON, nullable=True)
    dataset_id = Column(String(32), ForeignKey("datasets.id"), nullable=True)
    model_id = Column(String(32), ForeignKey("models.id"), nullable=True)
    training_job_id = Column(String(32), nullable=True)
    is_starred = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    id = Column(String(16), primary_key=True, default=gen_short_uuid)
    conversation_id = Column(String(16), ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user", "assistant", "system"
    content = Column(Text, nullable=False)
    message_type = Column(String(30), default="text")
    metadata_ = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
