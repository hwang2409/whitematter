"""
Dataset upload and management endpoints.

All dataset storage is via dataset_service (database-backed).
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form, Request

from schemas import DatasetListResponse, DetailResponse
from dependencies import dataset_service, limiter
from services.url_fetcher import fetch_for_import, URLFetchError
from schemas.import_schemas import ImportFromUrlRequest, ImportFromHuggingFaceRequest
from auth.dependencies import get_current_user
from db.auth_models import User

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/datasets/upload")
@limiter.limit("10/hour")
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
):
    """Upload a ZIP file containing labeled data (folder per class)."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    dataset_name = name or file.filename.replace('.zip', '')
    dataset_info = dataset_service.create_dataset(dataset_name)
    dataset_id = dataset_info['id']

    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            tmp.write(chunk)
        tmp_path = Path(tmp.name)

    try:
        zip_content = tmp_path.read_bytes()
        result = dataset_service.upload_zip(
            dataset_id=dataset_id,
            file_content=zip_content,
            filename=file.filename,
        )
        logger.info("Successfully uploaded dataset %s (%s)", dataset_id, dataset_name)
        return result
    except (ValueError, IOError, OSError) as e:
        logger.exception("Failed to upload dataset %s", dataset_id)
        dataset_service.delete_dataset(dataset_id)
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/datasets/upload/text")
async def upload_text_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    tokenizer_type: str = Form("character"),
    seq_length: int = Form(128),
    user: User = Depends(get_current_user),
):
    """Upload a text file for language model training."""
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="File must be a .txt file")

    if tokenizer_type not in ("character", "word"):
        raise HTTPException(status_code=400, detail="tokenizer_type must be 'character' or 'word'")

    dataset_name = name or file.filename.replace('.txt', '')
    dataset_info = dataset_service.create_dataset(dataset_name)
    dataset_id = dataset_info['id']

    try:
        chunks = []
        while chunk := await file.read(1024 * 1024):
            chunks.append(chunk)
        content = b"".join(chunks)
        result = dataset_service.upload_text(
            dataset_id=dataset_id,
            file_content=content,
            filename=file.filename,
            tokenizer_type=tokenizer_type,
            seq_length=seq_length
        )

        logger.info("Successfully uploaded text dataset %s (%s)", dataset_id, dataset_name)
        return {
            "id": result['id'],
            "name": result['name'],
            "data_type": result['data_type'],
            "format": result['format'],
            "status": result['status'],
            "total_samples": result['total_samples'],
            "train_samples": result['train_samples'],
            "test_samples": result['test_samples'],
            "input_shape": result['input_shape'],
            "num_classes": result['num_classes'],
            "preprocessing_config": dataset_service.get_dataset(dataset_id).get('preprocessing_config', {})
        }
    except (ValueError, IOError, OSError) as e:
        logger.exception("Failed to upload text dataset %s", dataset_id)
        dataset_service.delete_dataset(dataset_id)
        raise HTTPException(status_code=400, detail=str(e)) from e


def _infer_name_from_url(url: str, suggested_filename: str | None) -> str:
    """Derive a dataset name from URL or filename."""
    if suggested_filename:
        base = suggested_filename
        for ext in (".zip", ".txt", ".tar.gz", ".tar"):
            if base.endswith(ext):
                base = base[: -len(ext)]
                break
        if base:
            return base[:200]
    path = urlparse(url).path.rstrip("/")
    name = path.split("/")[-1] if path else "imported"
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name[:200] or "imported"


@router.post("/datasets/import/url")
async def import_dataset_from_url(
    body: ImportFromUrlRequest,
    user: User = Depends(get_current_user),
):
    """
    Import a dataset from a public HTTPS URL.
    Supports ZIP (folder-per-class images or MNIST IDX) and TXT (text for language models).
    """
    url_str = str(body.url)
    try:
        result = fetch_for_import(url_str)
    except URLFetchError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    name = (body.name or _infer_name_from_url(url_str, result.suggested_filename)).strip() or "imported"
    content = result.content
    suggested = (result.suggested_filename or "").lower()

    # ZIP
    if suggested.endswith(".zip") or (result.content_type or "").startswith("application/zip"):
        dataset_info = dataset_service.create_dataset(name)
        dataset_id = dataset_info["id"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            result_dict = dataset_service.upload_zip(
                dataset_id=dataset_id,
                file_content=tmp_path.read_bytes(),
                filename=result.suggested_filename or "imported.zip",
            )
            logger.info("Imported dataset from URL %s -> %s", url_str[:80], dataset_id)
            return result_dict
        except (ValueError, IOError, OSError) as e:
            logger.exception("Failed to import dataset from URL %s", dataset_id)
            dataset_service.delete_dataset(dataset_id)
            raise HTTPException(status_code=400, detail=str(e)) from e
        finally:
            tmp_path.unlink(missing_ok=True)

    # TXT
    if suggested.endswith(".txt") or (result.content_type or "").startswith("text/"):
        dataset_info = dataset_service.create_dataset(name)
        dataset_id = dataset_info["id"]
        filename = result.suggested_filename or "imported.txt"
        try:
            result_dict = dataset_service.upload_text(
                dataset_id=dataset_id,
                file_content=content,
                filename=filename,
                tokenizer_type="character",
                seq_length=128,
            )
            logger.info("Imported text dataset from URL %s -> %s", url_str[:80], dataset_id)
            return result_dict
        except (ValueError, IOError, OSError) as e:
            logger.exception("Failed to import text dataset from URL %s", dataset_id)
            dataset_service.delete_dataset(dataset_id)
            raise HTTPException(status_code=400, detail=str(e)) from e

    raise HTTPException(
        status_code=400,
        detail="URL must point to a ZIP file (images) or a TXT file (text). Check Content-Type or filename.",
    )


@router.post("/datasets/import/huggingface")
async def import_dataset_from_huggingface(
    body: ImportFromHuggingFaceRequest,
    user: User = Depends(get_current_user),
):
    """
    Import a dataset from Hugging Face Hub by dataset ID (e.g. 'username/dataset-name').
    Auto-detects image classification (image + label columns) or text datasets.
    """
    from services.huggingface_import import import_from_huggingface, HuggingFaceImportError

    name = (body.name or body.dataset_id.replace("/", "_"))[:200].strip() or "hf_import"
    try:
        result = import_from_huggingface(
            dataset_id=body.dataset_id,
            name=name,
            config=body.config,
            split=body.split,
        )
    except HuggingFaceImportError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    logger.info("Imported HF dataset %s -> %s", body.dataset_id, result.get("id"))
    return result


@router.get("/datasets", response_model=DatasetListResponse)
async def list_uploaded_datasets(user: User = Depends(get_current_user)):
    """List all uploaded datasets."""
    return {"datasets": dataset_service.list_datasets()}


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str, user: User = Depends(get_current_user)):
    """Get dataset metadata."""
    db_dataset = dataset_service.get_dataset(dataset_id)
    if db_dataset:
        return db_dataset
    raise HTTPException(status_code=404, detail="Dataset not found")


@router.get("/datasets/{dataset_id}/preview")
async def get_dataset_preview(dataset_id: str, user: User = Depends(get_current_user)):
    """Get a preview of the dataset."""
    db_preview = dataset_service.get_preview(dataset_id)
    if db_preview:
        return db_preview
    raise HTTPException(status_code=404, detail="Dataset not found")


@router.delete("/datasets/{dataset_id}", response_model=DetailResponse)
async def delete_dataset(dataset_id: str, user: User = Depends(get_current_user)):
    """Delete a dataset."""
    if dataset_service.delete_dataset(dataset_id):
        return {"detail": f"Dataset {dataset_id} deleted"}
    raise HTTPException(status_code=404, detail="Dataset not found")
