import logging
import os
from pathlib import Path
from typing import Optional, Union, BinaryIO

logger = logging.getLogger(__name__)


class StorageService:
    """
    Unified storage interface.
    Uses R2/S3 when credentials are configured, falls back to SQLite BlobStore.
    """

    def __init__(self):
        self._r2_client = None
        self._blob_store = None
        self.use_r2 = bool(os.environ.get("R2_ACCESS_KEY_ID"))
        self.bucket = os.environ.get("R2_BUCKET", "whitematter")

    @property
    def r2(self):
        if self._r2_client is None and self.use_r2:
            import boto3
            self._r2_client = boto3.client(
                "s3",
                endpoint_url=os.environ["R2_ENDPOINT"],
                aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                region_name="auto",
            )
        return self._r2_client

    @property
    def blob_store(self):
        if self._blob_store is None:
            from db.blob_store import get_blob_store
            self._blob_store = get_blob_store()
        return self._blob_store

    def put(
        self,
        key: str,
        data: Union[bytes, BinaryIO],
        content_type: Optional[str] = None,
    ) -> str:
        """
        Store data. Returns the key.
        Uses R2 in production, BlobStore in development.
        """
        if hasattr(data, "read"):
            content = data.read()
        else:
            content = data

        if self.use_r2:
            try:
                extra = {}
                if content_type:
                    extra["ContentType"] = content_type
                self.r2.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=content,
                    **extra,
                )
                logger.debug("R2 put: %s (%d bytes)", key, len(content))
                return key
            except Exception as e:
                logger.error("R2 put failed for %s: %s, falling back to BlobStore", key, e)

        # Fallback to SQLite BlobStore
        return self.blob_store.put(content, key=key, content_type=content_type)

    def put_file(
        self,
        file_path: Path,
        key: str,
        content_type: Optional[str] = None,
    ) -> str:
        """Store a file. Returns the key."""
        with open(file_path, "rb") as f:
            return self.put(key, f, content_type=content_type)

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve data by key. Returns None if not found."""
        if self.use_r2:
            try:
                resp = self.r2.get_object(Bucket=self.bucket, Key=key)
                return resp["Body"].read()
            except self.r2.exceptions.NoSuchKey:
                logger.debug("R2 key not found: %s", key)
                return None
            except Exception as e:
                logger.error("R2 get failed for %s: %s, trying BlobStore", key, e)

        return self.blob_store.get(key)

    def delete(self, key: str) -> bool:
        """Delete data by key."""
        if self.use_r2:
            try:
                self.r2.delete_object(Bucket=self.bucket, Key=key)
                return True
            except Exception as e:
                logger.error("R2 delete failed for %s: %s", key, e)
                return False

        return self.blob_store.delete(key)

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        if self.use_r2:
            try:
                self.r2.head_object(Bucket=self.bucket, Key=key)
                return True
            except Exception:
                return False

        return self.blob_store.exists(key)

    def list_keys(self, prefix: Optional[str] = None) -> list[str]:
        """List keys, optionally filtered by prefix."""
        if self.use_r2:
            try:
                params = {"Bucket": self.bucket}
                if prefix:
                    params["Prefix"] = prefix
                resp = self.r2.list_objects_v2(**params)
                return [obj["Key"] for obj in resp.get("Contents", [])]
            except Exception as e:
                logger.error("R2 list failed: %s", e)
                return []

        return self.blob_store.list_keys(prefix=prefix)

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> Optional[str]:
        """Generate a presigned download URL (R2 only)."""
        if not self.use_r2:
            return None
        try:
            url = self.r2.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            return url
        except Exception as e:
            logger.error("Presigned URL failed for %s: %s", key, e)
            return None


# Singleton
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
