"""
Model service - handles model operations using database and blob storage.
"""
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from db import (
    get_db_session, get_blob_store,
    Model, TrainingHistory, ModelStatus
)


class ModelService:
    """Service for managing models with database and blob storage."""

    def __init__(self):
        self.blob_store = get_blob_store()

    def create_model(
        self,
        name: str,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        architecture_name: Optional[str] = None,
        architecture_config: Optional[Dict] = None,
        training_config: Optional[Dict] = None
    ) -> Dict:
        """Create a new model entry."""
        model_id = uuid.uuid4().hex[:16]

        with get_db_session() as db:
            model = Model(
                id=model_id,
                name=name,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                architecture_name=architecture_name,
                architecture_config=architecture_config,
                training_config=training_config,
                status=ModelStatus.TRAINING.value,
                created_at=datetime.utcnow()
            )
            db.add(model)
            db.flush()

            return self._model_to_dict(model)

    def get_model(self, model_id: str) -> Optional[Dict]:
        """Get model by ID."""
        with get_db_session() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            if not model:
                return None
            return self._model_to_dict(model)

    def list_models(self) -> List[Dict]:
        """List all models."""
        with get_db_session() as db:
            models = db.query(Model).order_by(Model.created_at.desc()).all()
            return [self._model_to_dict(m) for m in models]

    def update_model(
        self,
        model_id: str,
        status: Optional[str] = None,
        epochs_trained: Optional[int] = None,
        best_accuracy: Optional[float] = None,
        final_loss: Optional[float] = None,
        weights_blob_key: Optional[str] = None
    ) -> Optional[Dict]:
        """Update model fields."""
        with get_db_session() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            if not model:
                return None

            if status is not None:
                model.status = status
            if epochs_trained is not None:
                model.epochs_trained = epochs_trained
            if best_accuracy is not None:
                model.best_accuracy = best_accuracy
            if final_loss is not None:
                model.final_loss = final_loss
            if weights_blob_key is not None:
                model.weights_blob_key = weights_blob_key

            model.updated_at = datetime.utcnow()

            return self._model_to_dict(model)

    def delete_model(self, model_id: str) -> bool:
        """Delete a model and its associated blobs."""
        with get_db_session() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            if not model:
                return False

            # Delete weights blob
            if model.weights_blob_key:
                self.blob_store.delete(model.weights_blob_key)

            # Delete training history
            db.query(TrainingHistory).filter_by(model_id=model_id).delete()

            db.delete(model)
            return True

    def get_training_history(self, model_id: str) -> List[Dict]:
        """Get training history for a model."""
        with get_db_session() as db:
            history = (
                db.query(TrainingHistory)
                .filter_by(model_id=model_id)
                .order_by(TrainingHistory.epoch)
                .all()
            )
            return [
                {
                    'epoch': h.epoch,
                    'loss': h.loss,
                    'accuracy': h.accuracy,
                    'learning_rate': h.learning_rate,
                    'recorded_at': h.recorded_at.isoformat() if h.recorded_at else None
                }
                for h in history
            ]

    def add_training_record(
        self,
        model_id: str,
        epoch: int,
        loss: float,
        accuracy: float,
        learning_rate: Optional[float] = None,
        job_id: Optional[str] = None
    ):
        """Add a training history record."""
        with get_db_session() as db:
            record = TrainingHistory(
                model_id=model_id,
                job_id=job_id,
                epoch=epoch,
                loss=loss,
                accuracy=accuracy,
                learning_rate=learning_rate,
                recorded_at=datetime.utcnow()
            )
            db.add(record)

    def get_weights_path(self, model_id: str) -> Optional[str]:
        """Get the blob path for model weights."""
        with get_db_session() as db:
            model = db.query(Model).filter_by(id=model_id).first()
            if not model or not model.weights_blob_key:
                return None
            return self.blob_store.get_path(model.weights_blob_key)

    def _model_to_dict(self, model: Model) -> Dict:
        """Convert model ORM object to dictionary."""
        return {
            'id': model.id,
            'name': model.name,
            'dataset_id': model.dataset_id,
            'dataset': model.dataset_name or (model.dataset_id if model.dataset_id else 'unknown'),
            'architecture': model.architecture_name or 'custom',
            'status': model.status,
            'epochs_trained': model.epochs_trained,
            'best_accuracy': model.best_accuracy,
            'final_loss': model.final_loss,
            'architecture_config': model.architecture_config,
            'training_config': model.training_config,
            'weights_blob_key': model.weights_blob_key,
            'created_at': model.created_at.isoformat() if model.created_at else None,
            'updated_at': model.updated_at.isoformat() if model.updated_at else None
        }
