"""Chat service — conversation state machine and orchestrator."""
import asyncio
import json
import logging
import os
import shutil
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Optional, Dict, Any, List

from sqlalchemy.orm import Session

from db.chat_models import Conversation, ConversationMessage
from db.auth_models import User

logger = logging.getLogger(__name__)

PRESETS_DIR = Path(__file__).resolve().parent.parent.parent / "presets"


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

## Dataset Tools
You can search HuggingFace Hub and import datasets directly:
- Use search_datasets when the user wants to find a dataset
- Use import_dataset when the user picks one to use
- After importing, tell the user what was imported (type, sample count, classes)

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
        self._async_llm_client = None

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

    @property
    def async_llm_client(self):
        if self._async_llm_client is None:
            try:
                import anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if api_key:
                    self._async_llm_client = anthropic.AsyncAnthropic(api_key=api_key)
            except ImportError:
                logger.warning("anthropic package not installed")
        return self._async_llm_client

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
        from services.model_service import ModelService
        training_service = TrainingService()
        model_service = ModelService()

        try:
            # Create model entry
            arch = conv.architecture
            training_config = {}
            if isinstance(arch.get("trainingConfig"), str):
                # Parse "SGD lr=0.01, batch_size=64, epochs=5" style strings
                training_config = {"raw": arch["trainingConfig"]}
            elif isinstance(arch.get("trainingConfig"), dict):
                training_config = arch["trainingConfig"]

            total_epochs = training_config.get("epochs", 5)
            if isinstance(total_epochs, str):
                try:
                    total_epochs = int(total_epochs)
                except ValueError:
                    total_epochs = 5

            model_info = model_service.create_model(
                name=arch.get("name", "chat_model"),
                dataset_id=conv.dataset_id,
                architecture_name=arch.get("name"),
                architecture_config=arch,
                training_config=training_config,
            )
            model_id = model_info["id"]

            # Start training job
            result = training_service.start_training(
                model_id=model_id,
                total_epochs=total_epochs,
            )
            job_id = result["id"]

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
                metadata_={"job_id": job_id, "model_id": model_id, "conversation_id": conv.id, "status": "started"},
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

        # Query the TrainingJob DB table directly — TrainingJobStore's in-memory
        # cache is per-instance and empty when instantiated fresh.
        from db import TrainingJob
        job_row = db.query(TrainingJob).filter_by(id=conv.training_job_id).first()
        if not job_row:
            return None

        status = job_row.status
        if hasattr(status, "value"):
            status = status.value

        result = {
            "job_id": conv.training_job_id,
            "status": status,
            "epoch": job_row.current_epoch or 0,
            "total_epochs": job_row.total_epochs or 0,
            "loss": job_row.current_loss or 0.0,
            "accuracy": job_row.current_accuracy or 0.0,
            "message": job_row.message or "",
        }

        # Check if completed and update conversation phase
        if status in ("completed", "failed", "cancelled"):
            if status == "completed":
                conv.phase = ConversationPhase.COMPLETED.value
                conv.model_id = job_row.model_id or conv.model_id

                # Persist a training_complete message if one doesn't exist yet
                existing = (
                    db.query(ConversationMessage)
                    .filter(
                        ConversationMessage.conversation_id == conv.id,
                        ConversationMessage.message_type == "training_complete",
                    )
                    .first()
                )
                if not existing:
                    elapsed = 0
                    if job_row.started_at and job_row.completed_at:
                        elapsed = (job_row.completed_at - job_row.started_at).total_seconds()
                    complete_msg = ConversationMessage(
                        conversation_id=conv.id,
                        role="assistant",
                        content="Training complete!",
                        message_type="training_complete",
                        metadata_={
                            "model_id": job_row.model_id or "",
                            "accuracy": job_row.current_accuracy or 0,
                            "params": "unknown",
                            "training_time": f"{elapsed:.0f}s",
                            "architecture": "",
                            "dataset_name": "",
                        },
                    )
                    db.add(complete_msg)
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

    def _handle_quick_start_mnist(
        self, db: Session, conversation: Conversation, user: User
    ) -> dict:
        """Load pre-bundled MNIST dataset and return CNN architecture info."""
        from services.dataset_service import DatasetService

        dataset_service = DatasetService()
        dataset = dataset_service.create_dataset("mnist-quickstart")
        dataset_id = dataset["id"]

        # Copy preset .bin files into blob storage
        mnist_preset_dir = PRESETS_DIR / "mnist"
        dest_dir = Path(__file__).resolve().parent.parent / "uploads" / dataset_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(mnist_preset_dir), str(dest_dir), dirs_exist_ok=True)

        # Link dataset to conversation
        conversation.dataset_id = dataset_id
        conversation.phase = ConversationPhase.READY_TO_TRAIN.value

        architecture = {
            "name": "MNIST CNN",
            "description": "A simple CNN for handwritten digit classification",
            "layers": "Conv2d(1,16,3) → ReLU → MaxPool → Conv2d(16,32,3) → ReLU → MaxPool → FC(32*5*5, 128) → ReLU → FC(128, 10)",
            "trainingConfig": "SGD lr=0.01, batch_size=64, epochs=5",
            "params": "~207K",
            "dataset": "MNIST (60K train, 10K test)",
        }
        conversation.architecture = architecture
        db.commit()

        return architecture

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

        # Quick Start MNIST detection
        if (
            "quick start" in content.lower() or "mnist" in content.lower()
        ) and conv.phase == ConversationPhase.GREETING.value:
            architecture = self._handle_quick_start_mnist(db, conv, user)
            msg = ConversationMessage(
                conversation_id=conv.id,
                role="assistant",
                content="Great! I've loaded MNIST for you. Here's a simple CNN for digit classification:",
                message_type="architecture",
                metadata_=architecture,
            )
            db.add(msg)
            db.commit()
            yield f"data: {json.dumps({'type': 'architecture', 'content': msg.content, 'metadata': architecture})}\n\n"
            yield "data: [DONE]\n\n"
            return

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
        model = "claude-sonnet-4-20250514" if use_sonnet else "claude-haiku-4-5-20251001"

        # Stream response from Claude
        full_response = ""

        if self.async_llm_client:
            from services.chat_tools import CHAT_TOOLS, execute_tool_async, summarize_tool_result

            MAX_TOOL_ROUNDS = 3
            HEARTBEAT_INTERVAL = 5  # seconds between keep-alive events
            try:
                for tool_round in range(MAX_TOOL_ROUNDS + 1):
                    async with self.async_llm_client.messages.stream(
                        model=model,
                        max_tokens=2048,
                        system=[{
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }],
                        messages=api_messages,
                        tools=CHAT_TOOLS,
                    ) as stream:
                        async for text in stream.text_stream:
                            full_response += text
                            yield f"data: {json.dumps({'type': 'chunk', 'content': text})}\n\n"
                        final_message = await stream.get_final_message()

                    if final_message.stop_reason != "tool_use":
                        break

                    tool_blocks = [b for b in final_message.content if b.type == "tool_use"]
                    if not tool_blocks or tool_round >= MAX_TOOL_ROUNDS:
                        break

                    # Execute tools async with heartbeat to prevent socket timeout
                    tool_results: Dict[str, Any] = {}
                    for tool_block in tool_blocks:
                        yield f"data: {json.dumps({'type': 'tool_use', 'name': tool_block.name})}\n\n"

                        # Run tool in thread, send heartbeats while waiting
                        task = asyncio.ensure_future(
                            execute_tool_async(tool_block.name, tool_block.input)
                        )
                        elapsed = 0
                        while not task.done():
                            await asyncio.sleep(1)
                            elapsed += 1
                            if elapsed % HEARTBEAT_INTERVAL == 0:
                                progress_msg = (
                                    f"Downloading dataset... ({elapsed}s)"
                                    if tool_block.name == "import_dataset"
                                    else f"Working... ({elapsed}s)"
                                )
                                yield f"data: {json.dumps({'type': 'tool_progress', 'name': tool_block.name, 'message': progress_msg, 'elapsed': elapsed})}\n\n"
                        result = task.result()
                        tool_results[tool_block.id] = result

                        # Side effect: import_dataset updates conversation
                        if tool_block.name == "import_dataset" and result.get("success"):
                            conv.dataset_id = result["dataset_id"]
                            conv.phase = (
                                ConversationPhase.ARCHITECTURE.value
                                if conv.architecture
                                else ConversationPhase.DATA_NEEDED.value
                            )
                            db.commit()

                        yield f"data: {json.dumps({'type': 'tool_result', 'name': tool_block.name, 'summary': summarize_tool_result(tool_block.name, result)})}\n\n"

                    # Append assistant + tool_result messages for next round
                    api_messages.append({
                        "role": "assistant",
                        "content": [b.model_dump() for b in final_message.content],
                    })
                    api_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": b.id,
                                "content": json.dumps(tool_results[b.id]),
                            }
                            for b in tool_blocks
                        ],
                    })
                    # Don't reset full_response — accumulate across rounds so
                    # phase detection (e.g. [ACTION:START_TRAINING]) sees all text.
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
