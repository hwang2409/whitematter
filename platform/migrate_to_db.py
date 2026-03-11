#!/usr/bin/env python3
"""
Migration script to move existing data from file-based storage to database.

This script:
1. Migrates existing datasets from uploads/ to blob storage
2. Migrates existing models from models/ to blob storage
3. Creates database entries for all migrated data

Run this once before switching to server_v2.py
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

from db import init_db, get_db_session, get_blob_store
from db import Dataset, Model, DatasetStatus, DatasetFormat, DataType, ModelStatus


def migrate_datasets(uploads_dir: Path, verbose: bool = True):
    """Migrate datasets from uploads directory to database."""
    if not uploads_dir.exists():
        print(f"No uploads directory found at {uploads_dir}")
        return

    blob_store = get_blob_store()
    migrated = 0
    skipped = 0

    for dataset_dir in uploads_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_id = dataset_dir.name
        metadata_path = dataset_dir / "metadata.json"

        if not metadata_path.exists():
            if verbose:
                print(f"  Skipping {dataset_id}: no metadata.json")
            skipped += 1
            continue

        # Check if already migrated
        with get_db_session() as db:
            existing = db.query(Dataset).filter_by(id=dataset_id).first()
            if existing:
                if verbose:
                    print(f"  Skipping {dataset_id}: already in database")
                skipped += 1
                continue

        # Load metadata
        with open(metadata_path) as f:
            meta = json.load(f)

        if verbose:
            print(f"  Migrating dataset: {meta.get('name', dataset_id)}")

        # Upload processed files to blob storage
        processed_dir = dataset_dir / "processed"
        processed_prefix = f"datasets/{dataset_id}/processed"

        if processed_dir.exists():
            for file_path in processed_dir.iterdir():
                if file_path.is_file():
                    blob_key = f"{processed_prefix}/{file_path.name}"
                    blob_store.put_file(file_path, key=blob_key)
                    if verbose:
                        print(f"    Uploaded {file_path.name}")

        # Upload raw files
        raw_dir = dataset_dir / "raw"
        if raw_dir.exists():
            # Create a zip of raw files
            raw_zip = dataset_dir / "raw.zip"
            if not raw_zip.exists():
                shutil.make_archive(str(raw_zip.with_suffix('')), 'zip', raw_dir)
            raw_blob_key = f"datasets/{dataset_id}/raw/original.zip"
            blob_store.put_file(raw_zip, key=raw_blob_key)

        # Create database entry
        with get_db_session() as db:
            dataset = Dataset(
                id=dataset_id,
                name=meta.get('name', dataset_id),
                data_type=meta.get('data_type', DataType.IMAGE.value),
                format=meta.get('format', DatasetFormat.UNKNOWN.value),
                status=meta.get('status', DatasetStatus.READY.value),
                num_classes=meta.get('num_classes', 0),
                total_samples=meta.get('total_samples', 0),
                input_shape=meta.get('input_shape', []),
                class_names=meta.get('class_names', []),
                samples_per_class=meta.get('samples_per_class', {}),
                raw_blob_key=raw_blob_key if raw_dir.exists() else None,
                processed_blob_prefix=processed_prefix if processed_dir.exists() else None,
                created_at=datetime.fromisoformat(meta.get('created_at', datetime.utcnow().isoformat()))
            )
            db.add(dataset)

        migrated += 1

    print(f"Datasets: {migrated} migrated, {skipped} skipped")
    return migrated


def migrate_models(models_dir: Path, verbose: bool = True):
    """Migrate models from models directory to database."""
    if not models_dir.exists():
        print(f"No models directory found at {models_dir}")
        return

    blob_store = get_blob_store()
    migrated = 0
    skipped = 0

    for metadata_path in models_dir.glob("*.json"):
        model_id = metadata_path.stem
        weights_path = models_dir / f"{model_id}.bin"

        # Check if already migrated
        with get_db_session() as db:
            existing = db.query(Model).filter_by(id=model_id).first()
            if existing:
                if verbose:
                    print(f"  Skipping {model_id}: already in database")
                skipped += 1
                continue

        # Load metadata
        with open(metadata_path) as f:
            meta = json.load(f)

        if verbose:
            print(f"  Migrating model: {meta.get('name', model_id)}")

        # Upload weights to blob storage
        weights_blob_key = None
        if weights_path.exists():
            weights_blob_key = f"models/{model_id}/weights.bin"
            blob_store.put_file(weights_path, key=weights_blob_key)
            if verbose:
                print(f"    Uploaded weights ({weights_path.stat().st_size / 1024:.1f} KB)")

        # Determine dataset reference
        dataset_name = meta.get('dataset', '')
        dataset_id = None
        if dataset_name.startswith('custom:'):
            dataset_id = dataset_name.replace('custom:', '')

        # Create database entry
        with get_db_session() as db:
            model = Model(
                id=model_id,
                name=meta.get('name', model_id),
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                architecture_name=meta.get('architecture', 'unknown'),
                status=meta.get('status', ModelStatus.COMPLETED.value),
                epochs_trained=meta.get('epochs_trained', 0),
                best_accuracy=meta.get('best_accuracy', 0.0),
                architecture_config=meta.get('config', {}).get('architecture'),
                training_config=meta.get('config', {}),
                weights_blob_key=weights_blob_key,
                created_at=datetime.fromisoformat(meta.get('created_at', datetime.utcnow().isoformat()))
            )
            db.add(model)

            # Migrate training history
            for hist in meta.get('training_history', []):
                from db import TrainingHistory
                record = TrainingHistory(
                    model_id=model_id,
                    epoch=hist.get('epoch', 0),
                    loss=hist.get('loss', 0.0),
                    accuracy=hist.get('accuracy', 0.0)
                )
                db.add(record)

        migrated += 1

    print(f"Models: {migrated} migrated, {skipped} skipped")
    return migrated


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Migrate data to database")
    parser.add_argument("--uploads-dir", default="../uploads", help="Path to uploads directory")
    parser.add_argument("--models-dir", default="../models", help="Path to models directory")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    uploads_dir = Path(args.uploads_dir)
    if not uploads_dir.is_absolute():
        uploads_dir = project_root / uploads_dir.name

    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = project_root / models_dir.name

    verbose = not args.quiet

    print("Initializing database...")
    init_db()

    print(f"\nMigrating datasets from {uploads_dir}...")
    migrate_datasets(uploads_dir, verbose)

    print(f"\nMigrating models from {models_dir}...")
    migrate_models(models_dir, verbose)

    print("\nMigration complete!")
    print(f"Data is now stored in: ~/.whitematter/")
    print("\nYou can now use server_v2.py instead of server.py")
    print("Run workers with: python run_worker.py --count 2")


if __name__ == "__main__":
    main()
