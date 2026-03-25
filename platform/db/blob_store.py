import hashlib
import tempfile
from pathlib import Path
from typing import Optional, BinaryIO, Union
from datetime import datetime

from .database import get_db_session
from .models import BlobMetadata


class BlobStore:
    def __init__(self):
        pass

    def _compute_hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def put(
        self,
        data: Union[bytes, BinaryIO],
        key: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> str:
        if hasattr(data, 'read'):
            content = data.read()
        else:
            content = data

        content_hash = self._compute_hash(content)
        blob_key = key or content_hash

        with get_db_session() as db:
            existing = db.query(BlobMetadata).filter_by(key=blob_key).first()
            if existing:
                existing.reference_count += 1
                existing.last_accessed = datetime.utcnow()
                # Update data if it's different (shouldn't happen with hash keys)
                if existing.data != content:
                    existing.data = content
                    existing.size_bytes = len(content)
                    existing.checksum = content_hash
            else:
                blob = BlobMetadata(
                    key=blob_key,
                    content_type=content_type,
                    size_bytes=len(content),
                    checksum=content_hash,
                    data=content,
                    reference_count=1
                )
                db.add(blob)

        return blob_key

    def put_file(
        self,
        file_path: Path,
        key: Optional[str] = None,
        content_type: Optional[str] = None,
        move: bool = False
    ) -> str:
        file_path = Path(file_path)
        with open(file_path, 'rb') as f:
            content = f.read()

        result = self.put(content, key=key, content_type=content_type)

        # If move was requested, delete the source file
        if move and file_path.exists():
            file_path.unlink()

        return result

    def get(self, key: str) -> Optional[bytes]:
        with get_db_session() as db:
            blob = db.query(BlobMetadata).filter_by(key=key).first()
            if not blob:
                return None

            blob.last_accessed = datetime.utcnow()
            return blob.data

    def get_path(self, key: str) -> Optional[Path]:
        """Create a temp file with blob contents for APIs that need file paths."""
        data = self.get(key)
        if data is None:
            return None

        # Create a temp file with the data
        # The caller is responsible for cleanup
        suffix = Path(key).suffix or '.bin'
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(data)
        tmp.close()
        return Path(tmp.name)

    def exists(self, key: str) -> bool:
        with get_db_session() as db:
            return db.query(BlobMetadata).filter_by(key=key).first() is not None

    def delete(self, key: str) -> bool:
        with get_db_session() as db:
            blob = db.query(BlobMetadata).filter_by(key=key).first()
            if not blob:
                return False

            blob.reference_count -= 1
            if blob.reference_count <= 0:
                db.delete(blob)
                return True

        return False

    def get_metadata(self, key: str) -> Optional[dict]:
        with get_db_session() as db:
            blob = db.query(BlobMetadata).filter_by(key=key).first()
            if not blob:
                return None
            return {
                'key': blob.key,
                'content_type': blob.content_type,
                'size_bytes': blob.size_bytes,
                'checksum': blob.checksum,
                'reference_count': blob.reference_count,
                'created_at': blob.created_at.isoformat(),
                'last_accessed': blob.last_accessed.isoformat()
            }

    def list_keys(self, prefix: Optional[str] = None) -> list:
        with get_db_session() as db:
            query = db.query(BlobMetadata.key)
            if prefix:
                query = query.filter(BlobMetadata.key.like(f"{prefix}%"))
            return [row[0] for row in query.all()]

    def get_total_size(self) -> int:
        with get_db_session() as db:
            from sqlalchemy import func
            result = db.query(func.sum(BlobMetadata.size_bytes)).scalar()
            return result or 0

    def cleanup_orphans(self) -> int:
        with get_db_session() as db:
            result = db.query(BlobMetadata).filter(
                BlobMetadata.reference_count <= 0
            ).delete()
            return result


# Singleton instance
_blob_store: Optional[BlobStore] = None


def get_blob_store() -> BlobStore:
    global _blob_store
    if _blob_store is None:
        _blob_store = BlobStore()
    return _blob_store
