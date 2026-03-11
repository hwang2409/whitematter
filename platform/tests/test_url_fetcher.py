"""Tests for URL fetcher (dataset import)."""
import pytest

from services.url_fetcher import fetch_for_import, URLFetchError


def test_rejects_http():
    with pytest.raises(URLFetchError, match="Only HTTPS"):
        fetch_for_import("http://example.com/dataset.zip")


def test_rejects_invalid_url():
    with pytest.raises(URLFetchError, match="Invalid URL"):
        fetch_for_import("not-a-url")
    with pytest.raises(URLFetchError, match="missing"):
        fetch_for_import("https://")
