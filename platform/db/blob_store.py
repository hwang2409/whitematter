"""
Blob storage service - stores all binary data in the SQLite database.
No files are written to the local filesystem.
"""
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, BinaryIO, Union
from datetime import datetime

from .database import get_db_session
from .models import BlobMetadata


class BlobStore:
    """
    Database-backed blob storage.
    All binary data is stored directly in SQLite using LargeBinary columns.
    Files are content-addressable using SHA256 hashes for deduplication.
    """

    def __init__(self):
        pass  # No filesystem initialization needed

    def _compute_hash(self, data: bytes) -> str:
        """Compute SHA256 hash of data."""
        return hashlib.sha256(data).hexdigest()

    def put(
        self,
        data: Union[bytes, BinaryIO],
        key: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> str:
        """
        Store data in blob storage (database).

        Args:
            data: Bytes or file-like object to store
            key: Optional custom key (defaults to content hash)
            content_type: MIME type of the content

        Returns:
            The blob key
        """
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
        """
        Store a file in blob storage (database).

        Args:
            file_path: Path to the file to store
            key: Optional custom key (defaults to content hash)
            content_type: MIME type of the content
            move: If True, delete the file after storing (ignored for DB storage)

        Returns:
            The blob key
        """
        file_path = Path(file_path)
        with open(file_path, 'rb') as f:
            content = f.read()

        result = self.put(content, key=key, content_type=content_type)

        # If move was requested, delete the source file
        if move and file_path.exists():
            file_path.unlink()

        return result

    def get(self, key: str) -> Optional[bytes]:
        """
        Retrieve data from blob storage.

        Args:
            key: The blob key

        Returns:
            The blob data or None if not found
        """
        with get_db_session() as db:
            blob = db.query(BlobMetadata).filter_by(key=key).first()
            if not blob:
                return None

            blob.last_accessed = datetime.utcnow()
            return blob.data

    def get_path(self, key: str) -> Optional[Path]:
        """
        Get a temporary file path for a blob.
        Creates a temp file with the blob contents for APIs that need file paths.

        Args:
            key: The blob key

        Returns:
            Path to temp file or None if blob not found
        """
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
        """Check if a blob exists."""
        with get_db_session() as db:
            return db.query(BlobMetadata).filter_by(key=key).first() is not None

    def delete(self, key: str) -> bool:
        """
        Delete a blob, decrementing reference count.
        Only removes data when reference count reaches 0.

        Args:
            key: The blob key

        Returns:
            True if blob was fully deleted, False otherwise
        """
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
        """Get metadata for a blob (without the data)."""
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
        """List all blob keys, optionally filtered by prefix."""
        with get_db_session() as db:
            query = db.query(BlobMetadata.key)
            if prefix:
                query = query.filter(BlobMetadata.key.like(f"{prefix}%"))
            return [row[0] for row in query.all()]

    def get_total_size(self) -> int:
        """Get total size of all blobs in bytes."""
        with get_db_session() as db:
            from sqlalchemy import func
            result = db.query(func.sum(BlobMetadata.size_bytes)).scalar()
            return result or 0

    def cleanup_orphans(self) -> int:
        """
        Remove blobs with zero reference count.
        Returns count of blobs removed.
        """
        with get_db_session() as db:
            result = db.query(BlobMetadata).filter(
                BlobMetadata.reference_count <= 0
            ).delete()
            return result


# Singleton instance
_blob_store: Optional[BlobStore] = None


def get_blob_store() -> BlobStore:
    """Get the global blob store instance."""
    global _blob_store
    if _blob_store is None:
        _blob_store = BlobStore()
    return _blob_store
