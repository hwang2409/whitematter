"""
Dataset upload and management endpoints.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, File, UploadFile, HTTPException, Form

from dataset_manager import DataType
from preprocessing import ImageProcessor
from dependencies import (
    dataset_manager, dataset_service, process_mnist_idx,
)
from services.url_fetcher import fetch_for_import, URLFetchError
from schemas.import_schemas import ImportFromUrlRequest, ImportFromHuggingFaceRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None)
):
    """Upload a ZIP file containing labeled data (folder per class)."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    dataset_id = dataset_manager.create_dataset(name or file.filename.replace('.zip', ''))

    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        metadata = dataset_manager.extract_zip(dataset_id, tmp_path)

        if metadata.data_type == DataType.IMAGE:
            raw_dir = dataset_manager.uploads_dir / dataset_id / "raw"
            processed_dir = dataset_manager.uploads_dir / dataset_id / "processed"

            if metadata.format == "mnist_idx":
                config = process_mnist_idx(raw_dir, processed_dir, metadata)
            else:
                processor = ImageProcessor(
                    target_size=(metadata.input_shape[1], metadata.input_shape[2]),
                    channels=metadata.input_shape[0]
                )
                config = processor.process_dataset(
                    raw_dir, processed_dir, metadata.class_names
                )

            metadata.input_shape = config["input_shape"]
            metadata.status = "ready"
            dataset_manager._save_metadata(dataset_id, metadata)

        logger.info("Successfully uploaded dataset %s (%s)", dataset_id, metadata.name)
        return {
            "id": dataset_id,
            "name": metadata.name,
            "data_type": metadata.data_type.value,
            "format": metadata.format,
            "num_classes": metadata.num_classes,
            "class_names": metadata.class_names,
            "total_samples": metadata.total_samples,
            "input_shape": metadata.input_shape,
            "created_at": metadata.created_at,
            "status": metadata.status
        }
    except (ValueError, IOError, OSError) as e:
        logger.exception("Failed to upload dataset %s", dataset_id)
        dataset_manager.delete_dataset(dataset_id)
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/datasets/upload/text")
async def upload_text_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    tokenizer_type: str = Form("character"),
    seq_length: int = Form(128)
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
        content = await file.read()
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
async def import_dataset_from_url(body: ImportFromUrlRequest):
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

    # ZIP: use legacy dataset_manager path (same as file upload)
    if suggested.endswith(".zip") or (result.content_type or "").startswith("application/zip"):
        dataset_id = dataset_manager.create_dataset(name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            metadata = dataset_manager.extract_zip(dataset_id, tmp_path)
            if metadata.data_type == DataType.IMAGE:
                raw_dir = dataset_manager.uploads_dir / dataset_id / "raw"
                processed_dir = dataset_manager.uploads_dir / dataset_id / "processed"
                if metadata.format == "mnist_idx":
                    config = process_mnist_idx(raw_dir, processed_dir, metadata)
                else:
                    processor = ImageProcessor(
                        target_size=(metadata.input_shape[1], metadata.input_shape[2]),
                        channels=metadata.input_shape[0],
                    )
                    config = processor.process_dataset(
                        raw_dir, processed_dir, metadata.class_names
                    )
                metadata.input_shape = config["input_shape"]
                metadata.status = "ready"
                dataset_manager._save_metadata(dataset_id, metadata)
            logger.info("Imported dataset from URL %s -> %s (%s)", url_str[:80], dataset_id, metadata.name)
            return {
                "id": dataset_id,
                "name": metadata.name,
                "data_type": metadata.data_type.value,
                "format": metadata.format,
                "num_classes": metadata.num_classes,
                "class_names": metadata.class_names,
                "total_samples": metadata.total_samples,
                "input_shape": metadata.input_shape,
                "created_at": metadata.created_at,
                "status": metadata.status,
            }
        except (ValueError, IOError, OSError) as e:
            logger.exception("Failed to import dataset from URL %s", dataset_id)
            dataset_manager.delete_dataset(dataset_id)
            raise HTTPException(status_code=400, detail=str(e)) from e
        finally:
            tmp_path.unlink(missing_ok=True)
    # TXT: use dataset_service
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
async def import_dataset_from_huggingface(body: ImportFromHuggingFaceRequest):
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


@router.get("/datasets")
async def list_uploaded_datasets():
    """List all uploaded datasets (from both legacy manager and database)."""
    legacy_datasets = dataset_manager.list_datasets()
    legacy_list = [d.to_dict() for d in legacy_datasets]

    db_datasets = dataset_service.list_datasets()

    all_ids = set(d.get('id') for d in legacy_list)
    combined = legacy_list.copy()
    for d in db_datasets:
        if d.get('id') not in all_ids:
            combined.append(d)

    return {"datasets": combined}


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset metadata."""
    metadata = dataset_manager.get_metadata(dataset_id)
    if metadata:
        return metadata.to_dict()

    db_dataset = dataset_service.get_dataset(dataset_id)
    if db_dataset:
        return db_dataset

    raise HTTPException(status_code=404, detail="Dataset not found")


@router.get("/datasets/{dataset_id}/preview")
async def get_dataset_preview(dataset_id: str):
    """Get a preview of the dataset."""
    preview = dataset_manager.get_preview(dataset_id)
    if preview:
        return preview

    db_preview = dataset_service.get_preview(dataset_id)
    if db_preview:
        return db_preview

    raise HTTPException(status_code=404, detail="Dataset not found")


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    if dataset_manager.delete_dataset(dataset_id):
        return {"message": f"Dataset {dataset_id} deleted"}

    if dataset_service.delete_dataset(dataset_id):
        return {"message": f"Dataset {dataset_id} deleted"}

    raise HTTPException(status_code=404, detail="Dataset not found")
