from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class ChatMessageRequest(BaseModel):
    content: str
    attachments: Optional[List[str]] = None


class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    message_type: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: str


class ConversationResponse(BaseModel):
    id: str
    title: Optional[str] = None
    phase: str
    architecture: Optional[Dict[str, Any]] = None
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None
    training_job_id: Optional[str] = None
    is_starred: bool = False
    created_at: str
    updated_at: str


class ConversationUpdateRequest(BaseModel):
    title: Optional[str] = None
    is_starred: Optional[bool] = None


class ConversationListResponse(BaseModel):
    conversations: List[ConversationResponse]


class ConversationDetailResponse(BaseModel):
    conversation: ConversationResponse
    messages: List[ChatMessageResponse]
