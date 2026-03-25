"""
Shared FastAPI dependencies and application state.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from codegen import CodeGenerator
from services.dataset_service import DatasetService
from services.job_store import TrainingJobStore
from llm.service import get_llm_service

# Shared rate limiter (single instance so all routes share counters)
limiter = Limiter(key_func=get_remote_address)

# Shared mutable state
training_jobs: TrainingJobStore = TrainingJobStore()

# Service instances
dataset_service = DatasetService()
code_generator = CodeGenerator()
llm_service = get_llm_service()
