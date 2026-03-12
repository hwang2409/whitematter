"""Chat service — conversation state machine and orchestrator."""
import json
import logging
import os
from enum import Enum
from typing import AsyncIterator, Optional, Dict, Any, List

from sqlalchemy.orm import Session

from db.chat_models import Conversation, ConversationMessage
from db.auth_models import User

logger = logging.getLogger(__name__)


class ConversationPhase(str, Enum):
    GREETING = "greeting"
    EXPLORING = "exploring"
    ARCHITECTURE = "architecture"
    DATA_NEEDED = "data_needed"
    READY_TO_TRAIN = "ready"
    TRAINING = "training"
    COMPLETED = "completed"
    PREDICTING = "predicting"


GREETING_MESSAGE = (
    "Hey! I help you build and train neural networks. "
    "Try a demo or describe what you want to build."
)


CHAT_SYSTEM_PROMPT_TEMPLATE = """You are an expert ML assistant for the WhiteMatter neural network framework.

{layer_catalog}

## Current Conversation State
- Phase: {phase}
- Dataset uploaded: {has_dataset}
{dataset_info}
{architecture_info}

## Your Role
1. Help users describe their ML problem clearly
2. Ask clarifying questions about their data and goals
3. When you have enough info, suggest a model architecture
4. Guide them through dataset upload if needed
5. Help them understand training results

## Important Behaviors
- Be concise and helpful. Don't lecture.
- When the user seems ready (they've described data type, task, rough expectations), generate an architecture.
- To suggest an architecture, output a JSON block wrapped in ```json``` with the WhiteMatter architecture format.
- If the user says "train it", "let's go", "looks good", or similar confirmation, respond with [ACTION:START_TRAINING]
- If the user wants to tweak the architecture, update it and output new JSON.
- If you need a dataset and none is uploaded, ask the user to upload one.
- Keep your responses SHORT. 2-3 sentences max for conversational turns.
"""


class ChatService:
    """Orchestrates chat conversations and manages phase transitions."""

    def __init__(self):
        self._llm_client = None

    @property
    def llm_client(self):
        if self._llm_client is None:
            try:
                import anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if api_key:
                    self._llm_client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                logger.warning("anthropic package not installed")
        return self._llm_client

    def create_conversation(self, db: Session, user: User) -> Conversation:
        """Create a new conversation with greeting."""
        conv = Conversation(user_id=user.id)
        db.add(conv)
        db.flush()

        # Add greeting message
        greeting = ConversationMessage(
            conversation_id=conv.id,
            role="assistant",
            content=GREETING_MESSAGE,
            message_type="text",
        )
        db.add(greeting)
        db.commit()
        db.refresh(conv)
        return conv

    def get_conversations(self, db: Session, user: User) -> list[Conversation]:
        """Get all conversations for a user, most recent first."""
        return (
            db.query(Conversation)
            .filter(Conversation.user_id == user.id)
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    def get_conversation_with_messages(
        self, db: Session, conversation_id: str, user: User
    ) -> tuple[Conversation, list[ConversationMessage]]:
        """Get a conversation and all its messages."""
        conv = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
            .first()
        )
        if not conv:
            raise ValueError("Conversation not found")

        messages = (
            db.query(ConversationMessage)
            .filter(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at.asc())
            .all()
        )
        return conv, messages

    def _build_system_prompt(self, conv: Conversation) -> str:
        """Build the system prompt with current conversation context."""
        from llm.prompts import ARCHITECTURE_SYSTEM_PROMPT

        dataset_info = ""
        if conv.dataset_id:
            dataset_info = f"- Dataset ID: {conv.dataset_id}"

        architecture_info = ""
        if conv.architecture:
            architecture_info = f"- Current architecture: {json.dumps(conv.architecture, indent=2)}"

        return CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            layer_catalog=ARCHITECTURE_SYSTEM_PROMPT,
            phase=conv.phase,
            has_dataset="Yes" if conv.dataset_id else "No",
            dataset_info=dataset_info,
            architecture_info=architecture_info,
        )

    def _build_messages(self, db_messages: list[ConversationMessage]) -> list[dict]:
        """Convert DB messages to Claude API format."""
        messages = []
        for msg in db_messages:
            if msg.role in ("user", "assistant"):
                messages.append({"role": msg.role, "content": msg.content})
        return messages

    def _detect_phase_transition(
        self, conv: Conversation, assistant_response: str
    ) -> Optional[str]:
        """Detect if the assistant's response triggers a phase transition."""
        if "[ACTION:START_TRAINING]" in assistant_response:
            if conv.architecture and conv.dataset_id:
                return ConversationPhase.TRAINING.value
            elif conv.architecture:
                return ConversationPhase.DATA_NEEDED.value

        # Check if response contains architecture JSON
        if "```json" in assistant_response and conv.phase in (
            ConversationPhase.GREETING.value,
            ConversationPhase.EXPLORING.value,
            ConversationPhase.ARCHITECTURE.value,
        ):
            return ConversationPhase.ARCHITECTURE.value

        # Move from greeting to exploring on first user message
        if conv.phase == ConversationPhase.GREETING.value:
            return ConversationPhase.EXPLORING.value

        return None

    def start_training(self, db: Session, conv: Conversation, user: User) -> Optional[str]:
        """
        Enqueue a training job for the conversation's architecture + dataset.
        Returns the job_id or None if training can't start.
        """
        if not conv.architecture or not conv.dataset_id:
            return None

        from services.training_service import TrainingService
        training_service = TrainingService()

        try:
            result = training_service.start_custom_training(
                db=db,
                user_id=user.id,
                dataset_id=conv.dataset_id,
                architecture=conv.architecture,
                name=conv.architecture.get("name", "chat_model"),
            )
            job_id = result["job_id"]
            model_id = result["model_id"]

            conv.training_job_id = job_id
            conv.model_id = model_id
            conv.phase = ConversationPhase.TRAINING.value
            db.commit()

            # Add training-started message
            msg = ConversationMessage(
                conversation_id=conv.id,
                role="assistant",
                content=f"Training started! Job ID: {job_id}. I'll update you on progress.",
                message_type="training_progress",
                metadata_={"job_id": job_id, "model_id": model_id, "status": "started"},
            )
            db.add(msg)
            db.commit()

            return job_id
        except Exception as e:
            logger.exception("Failed to start training: %s", e)
            return None

    def get_training_status(self, db: Session, conv: Conversation) -> Optional[dict]:
        """Get current training status for a conversation's active job."""
        if not conv.training_job_id:
            return None

        from services.job_store import TrainingJobStore
        store = TrainingJobStore()
        job = store.get(conv.training_job_id)
        if not job:
            return None

        status = job.get("status", "unknown")
        if hasattr(status, "value"):
            status = status.value

        result = {
            "job_id": conv.training_job_id,
            "status": status,
            "epoch": job.get("epoch", 0),
            "total_epochs": job.get("total_epochs", 0),
            "loss": job.get("loss", 0.0),
            "accuracy": job.get("accuracy", 0.0),
            "message": job.get("message", ""),
        }

        # Check if completed and update conversation phase
        if status in ("completed", "failed", "cancelled"):
            if status == "completed":
                conv.phase = ConversationPhase.COMPLETED.value
            db.commit()

        return result

    def _extract_architecture(self, text: str) -> Optional[dict]:
        """Extract architecture JSON from assistant response."""
        import re
        match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
        if match:
            try:
                arch = json.loads(match.group(1))
                if "layers" in arch and "name" in arch:
                    return arch
            except json.JSONDecodeError:
                pass
        return None

    async def process_message(
        self,
        db: Session,
        conversation_id: str,
        user: User,
        content: str,
    ) -> AsyncIterator[str]:
        """
        Process a user message and stream the assistant response.

        Yields SSE-formatted strings: "data: {...}\n\n"
        """
        conv = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
            .first()
        )
        if not conv:
            raise ValueError("Conversation not found")

        # Save user message
        user_msg = ConversationMessage(
            conversation_id=conversation_id,
            role="user",
            content=content,
            message_type="text",
        )
        db.add(user_msg)

        # Set title from first user message
        if not conv.title:
            conv.title = content[:100]

        db.commit()

        # Get full message history
        messages = (
            db.query(ConversationMessage)
            .filter(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at.asc())
            .all()
        )

        # Build Claude API request
        system_prompt = self._build_system_prompt(conv)
        api_messages = self._build_messages(messages)

        # Determine which model to use
        use_sonnet = conv.phase in (
            ConversationPhase.EXPLORING.value,
        ) and any(
            kw in content.lower()
            for kw in ["suggest", "design", "architect", "build", "create"]
        )
        model = "claude-sonnet-4-20250514" if use_sonnet else "claude-haiku-4-5-20250514"

        # Stream response from Claude
        full_response = ""

        if self.llm_client:
            try:
                with self.llm_client.messages.stream(
                    model=model,
                    max_tokens=2048,
                    system=[{
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    messages=api_messages,
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        yield f"data: {json.dumps({'type': 'chunk', 'content': text})}\n\n"
            except Exception as e:
                logger.exception("Claude API error")
                full_response = f"I'm sorry, I encountered an error: {str(e)}. Please try again."
                yield f"data: {json.dumps({'type': 'chunk', 'content': full_response})}\n\n"
        else:
            # Fallback when no API key
            full_response = "I'm running in demo mode without an API key. I can still help you explore the interface! Try uploading a dataset or selecting a quick start template."
            yield f"data: {json.dumps({'type': 'chunk', 'content': full_response})}\n\n"

        # Clean action markers from displayed text
        clean_response = full_response.replace("[ACTION:START_TRAINING]", "").strip()

        # Detect phase transitions
        new_phase = self._detect_phase_transition(conv, full_response)
        if new_phase:
            conv.phase = new_phase

        # Auto-start training if phase transitioned to TRAINING
        if new_phase == ConversationPhase.TRAINING.value:
            job_id = self.start_training(db, conv, user)
            if job_id:
                yield f"data: {json.dumps({'type': 'training_started', 'job_id': job_id})}\n\n"

        # Extract architecture if present
        arch = self._extract_architecture(full_response)
        if arch:
            conv.architecture = arch
            conv.phase = ConversationPhase.ARCHITECTURE.value

        # Determine message type
        msg_type = "text"
        metadata = None
        if arch:
            msg_type = "architecture"
            metadata = {"architecture": arch}

        # Save assistant message
        assistant_msg = ConversationMessage(
            conversation_id=conversation_id,
            role="assistant",
            content=clean_response,
            message_type=msg_type,
            metadata_=metadata,
        )
        db.add(assistant_msg)
        db.commit()

        # Send done event with full message
        done_data = {
            "type": "done",
            "message": {
                "id": assistant_msg.id,
                "role": "assistant",
                "content": clean_response,
                "message_type": msg_type,
                "metadata": metadata,
                "created_at": assistant_msg.created_at.isoformat(),
            },
            "phase": conv.phase,
        }
        yield f"data: {json.dumps(done_data)}\n\n"
        yield "data: [DONE]\n\n"
