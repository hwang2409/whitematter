"""
Secure URL fetcher for dataset import.
HTTPS only, size limits, timeouts. No redirects to other schemes.
"""
from __future__ import annotations

import re
from typing import NamedTuple
from urllib.parse import urlparse

import httpx

from dataset_manager import MAX_UPLOAD_SIZE_BYTES

# 1 GB max for URL fetch (align with upload limit)
MAX_FETCH_BYTES = MAX_UPLOAD_SIZE_BYTES
FETCH_TIMEOUT_SECONDS = 120.0

# Only HTTPS
ALLOWED_SCHEMES = frozenset({"https"})

# Optional: allowlist of host patterns (e.g. r"^.*\.s3\.amazonaws\.com$"). Empty = allow all HTTPS.
ALLOWED_HOST_PATTERNS: tuple[str, ...] = ()


class FetchResult(NamedTuple):
    """Result of a successful fetch."""

    content: bytes
    content_type: str | None
    suggested_filename: str | None


class URLFetchError(Exception):
    """Raised when URL fetch fails validation or request."""

    pass


def _allowed_host(host: str) -> bool:
    if not ALLOWED_HOST_PATTERNS:
        return True
    return any(re.match(p, host) for p in ALLOWED_HOST_PATTERNS)


def fetch_for_import(url: str) -> FetchResult:
    """
    Fetch content from a URL for dataset import.

    - Only HTTPS.
    - Optional host allowlist (if configured).
    - Respects Content-Length when present; otherwise streams up to MAX_FETCH_BYTES.
    - Returns content type and suggested filename (from Content-Disposition or URL path).

    Raises:
        URLFetchError: On invalid URL, scheme, host, size, or request failure.
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise URLFetchError("Invalid URL: missing scheme or host")

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise URLFetchError("Only HTTPS URLs are allowed")

    if not _allowed_host(parsed.netloc):
        raise URLFetchError("URL host is not allowed for import")

    with httpx.Client(
        follow_redirects=True,
        timeout=FETCH_TIMEOUT_SECONDS,
        limits=httpx.Limits(max_keepalive_connections=2),
    ) as client:
        try:
            response = client.get(
                url,
                headers={"User-Agent": "Whitematter-Dataset-Import/1.0"},
            )
        except httpx.HTTPError as e:
            raise URLFetchError(f"Request failed: {e}") from e

    if response.status_code != 200:
        raise URLFetchError(f"URL returned status {response.status_code}")

    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            size = int(content_length)
            if size > MAX_FETCH_BYTES:
                raise URLFetchError(
                    f"Content length ({size:,} bytes) exceeds maximum ({MAX_FETCH_BYTES:,} bytes)"
                )
        except ValueError:
            pass

    content = response.content
    if len(content) > MAX_FETCH_BYTES:
        raise URLFetchError(
            f"Download size ({len(content):,} bytes) exceeds maximum ({MAX_FETCH_BYTES:,} bytes)"
        )

    content_type = response.headers.get("Content-Type")
    if content_type and ";" in content_type:
        content_type = content_type.split(";", 1)[0].strip()

    suggested_filename = None
    disp = response.headers.get("Content-Disposition")
    if disp and "filename=" in disp:
        # Simple extraction: filename="..."; or filename*=...
        for part in disp.split(";"):
            part = part.strip()
            if part.lower().startswith("filename="):
                raw = part.split("=", 1)[1].strip().strip('"')
                if raw:
                    suggested_filename = raw
                break
    if not suggested_filename and parsed.path:
        name = parsed.path.rstrip("/").split("/")[-1]
        if name and "." in name:
            suggested_filename = name

    return FetchResult(
        content=content,
        content_type=content_type,
        suggested_filename=suggested_filename,
    )
