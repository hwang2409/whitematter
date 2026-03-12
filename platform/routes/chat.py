"""Chat conversation routes with SSE streaming."""
import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from db.database import get_db
from db.auth_models import User
from auth.dependencies import get_current_user
from schemas.chat_schemas import (
    ChatMessageRequest,
    ConversationResponse,
    ConversationListResponse,
    ConversationDetailResponse,
    ChatMessageResponse,
)
from services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])
chat_service = ChatService()


def _conv_to_response(conv) -> ConversationResponse:
    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        phase=conv.phase,
        architecture=conv.architecture,
        dataset_id=conv.dataset_id,
        model_id=conv.model_id,
        training_job_id=conv.training_job_id,
        created_at=conv.created_at.isoformat(),
        updated_at=conv.updated_at.isoformat(),
    )


def _msg_to_response(msg) -> ChatMessageResponse:
    return ChatMessageResponse(
        id=msg.id,
        role=msg.role,
        content=msg.content,
        message_type=msg.message_type,
        metadata=msg.metadata_,
        created_at=msg.created_at.isoformat(),
    )


@router.post("/conversations", response_model=ConversationResponse, status_code=201)
def create_conversation(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new conversation."""
    conv = chat_service.create_conversation(db, user)
    return _conv_to_response(conv)


@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all conversations for the current user."""
    convs = chat_service.get_conversations(db, user)
    return ConversationListResponse(
        conversations=[_conv_to_response(c) for c in convs]
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a conversation with all messages."""
    try:
        conv, messages = chat_service.get_conversation_with_messages(
            db, conversation_id, user
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationDetailResponse(
        conversation=_conv_to_response(conv),
        messages=[_msg_to_response(m) for m in messages],
    )


@router.post("/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    body: ChatMessageRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Send a message in a conversation.
    Returns SSE stream with real-time response chunks.
    """
    # Verify conversation exists and belongs to user
    from db.chat_models import Conversation
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return StreamingResponse(
        chat_service.process_message(db, conversation_id, user, body.content),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations/{conversation_id}/training")
def get_training_status(
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get training status for a conversation's active training job."""
    from db.chat_models import Conversation
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    status = chat_service.get_training_status(db, conv)
    if not status:
        raise HTTPException(status_code=404, detail="No active training job")

    return status


@router.get("/conversations/{conversation_id}/training/stream")
async def stream_training_progress(
    conversation_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Stream training progress as SSE events."""
    from db.chat_models import Conversation

    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    async def event_stream():
        while True:
            status = chat_service.get_training_status(db, conv)
            if status:
                yield f"data: {json.dumps(status)}\n\n"
                if status.get("status") in ("completed", "failed", "cancelled"):
                    break
            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
