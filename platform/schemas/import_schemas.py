"""Request/response schemas for dataset import (URL, Hugging Face)."""
from pydantic import BaseModel, Field, HttpUrl


class ImportFromUrlRequest(BaseModel):
    """Import a dataset from a public HTTPS URL (ZIP or TXT)."""

    url: HttpUrl
    name: str | None = Field(None, min_length=1, max_length=255)


class ImportFromHuggingFaceRequest(BaseModel):
    """Import a dataset from Hugging Face Hub by dataset ID."""

    dataset_id: str = Field(..., min_length=2, max_length=255)
    name: str | None = Field(None, min_length=1, max_length=255)
    config: str | None = None
    split: str = "train"
