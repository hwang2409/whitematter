"""Tests for /datasets routes: upload, list, get, delete, import."""

import io
import zipfile
from unittest.mock import patch, MagicMock

import dependencies


# ---------------------------------------------------------------------------
# Helper to create a minimal zip in memory
# ---------------------------------------------------------------------------

def _make_zip_bytes(filenames=None):
    """Create a minimal zip file in memory. Returns bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in (filenames or ["cat/img1.png", "dog/img1.png"]):
            zf.writestr(name, b"fake image data")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# POST /datasets/upload
# ---------------------------------------------------------------------------

class TestUploadDataset:
    def test_upload_non_zip(self, client):
        """Upload a non-ZIP file returns 400."""
        response = client.post(
            "/datasets/upload",
            files={"file": ("data.csv", b"col1,col2\n1,2", "text/csv")},
        )
        assert response.status_code == 400
        assert "ZIP" in response.json()["detail"]

    def test_upload_zip_success(self, client):
        """Upload a valid zip file succeeds."""
        zip_bytes = _make_zip_bytes()

        mock_result = {
            "id": "ds-123",
            "name": "testdata",
            "data_type": "image",
            "format": "folder_per_class",
            "status": "ready",
            "num_classes": 2,
            "class_names": ["cat", "dog"],
            "total_samples": 2,
            "input_shape": [3, 32, 32],
        }

        with patch.object(dependencies.dataset_service, "create_dataset",
                          return_value={"id": "ds-123"}), \
             patch.object(dependencies.dataset_service, "upload_zip",
                          return_value=mock_result):

            response = client.post(
                "/datasets/upload",
                files={"file": ("data.zip", zip_bytes, "application/zip")},
                data={"name": "testdata"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "ds-123"
        assert data["name"] == "testdata"
        assert data["status"] == "ready"


# ---------------------------------------------------------------------------
# GET /datasets
# ---------------------------------------------------------------------------

class TestListDatasets:
    def test_list_datasets(self, client):
        """GET /datasets returns combined list."""
        db_ds1 = {"id": "d1", "name": "ds1", "status": "ready"}
        db_ds2 = {"id": "d2", "name": "ds2", "status": "ready"}

        with patch.object(dependencies.dataset_service, "list_datasets",
                          return_value=[db_ds1, db_ds2]):
            response = client.get("/datasets")

        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        ids = [d["id"] for d in data["datasets"]]
        assert "d1" in ids
        assert "d2" in ids

    def test_list_datasets_empty(self, client):
        """GET /datasets with no datasets returns empty list."""
        with patch.object(dependencies.dataset_service, "list_datasets",
                          return_value=[]):
            response = client.get("/datasets")

        assert response.status_code == 200
        assert response.json()["datasets"] == []


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}
# ---------------------------------------------------------------------------

class TestGetDataset:
    def test_get_dataset_from_manager(self, client):
        """GET /datasets/{id} returns dataset from service."""
        mock_ds = {"id": "ds-abc", "name": "test", "status": "ready"}

        with patch.object(dependencies.dataset_service, "get_dataset",
                          return_value=mock_ds):
            response = client.get("/datasets/ds-abc")

        assert response.status_code == 200
        assert response.json()["id"] == "ds-abc"

    def test_get_dataset_from_service(self, client):
        """GET /datasets/{id} returns dataset from service."""
        with patch.object(dependencies.dataset_service, "get_dataset",
                          return_value={"id": "ds-db", "name": "from_db"}):
            response = client.get("/datasets/ds-db")

        assert response.status_code == 200
        assert response.json()["id"] == "ds-db"

    def test_get_dataset_not_found(self, client):
        """GET /datasets/{id} returns 404 if not found anywhere."""
        with patch.object(dependencies.dataset_service, "get_dataset",
                          return_value=None):
            response = client.get("/datasets/nonexistent")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /datasets/{dataset_id}
# ---------------------------------------------------------------------------

class TestDeleteDataset:
    def test_delete_from_manager(self, client):
        """DELETE /datasets/{id} deletes via dataset_service."""
        with patch.object(dependencies.dataset_service, "delete_dataset",
                          return_value=True):
            response = client.delete("/datasets/ds-del")

        assert response.status_code == 200
        assert "deleted" in response.json()["detail"].lower()

    def test_delete_from_service(self, client):
        """DELETE /datasets/{id} deletes via dataset_service."""
        with patch.object(dependencies.dataset_service, "delete_dataset",
                          return_value=True):
            response = client.delete("/datasets/ds-del2")

        assert response.status_code == 200

    def test_delete_not_found(self, client):
        """DELETE /datasets/{id} returns 404 if not found."""
        with patch.object(dependencies.dataset_service, "delete_dataset",
                          return_value=False):
            response = client.delete("/datasets/nonexistent")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /datasets/import/url
# ---------------------------------------------------------------------------

class TestImportFromUrl:
    def test_import_url_unsupported_format(self, client):
        """Import from URL with unsupported content type returns 400."""
        mock_result = MagicMock()
        mock_result.content = b"binary data"
        mock_result.content_type = "application/octet-stream"
        mock_result.suggested_filename = "file.bin"

        with patch("routes.datasets.fetch_for_import", return_value=mock_result):
            response = client.post("/datasets/import/url", json={
                "url": "https://example.com/file.bin"
            })

        assert response.status_code == 400

    def test_import_url_fetch_error(self, client):
        """Import from URL with fetch failure returns 400."""
        from services.url_fetcher import URLFetchError

        with patch("routes.datasets.fetch_for_import", side_effect=URLFetchError("Timeout")):
            response = client.post("/datasets/import/url", json={
                "url": "https://example.com/data.zip"
            })

        assert response.status_code == 400


# ---------------------------------------------------------------------------
# POST /datasets/import/huggingface
# ---------------------------------------------------------------------------

class TestImportFromHuggingFace:
    def test_import_hf_success(self, client):
        """Import from HuggingFace succeeds."""
        import sys
        mock_result = {"id": "hf-123", "name": "hf_dataset", "status": "ready"}

        # The route uses a lazy import: from services.huggingface_import import ...
        mock_module = MagicMock()
        mock_module.import_from_huggingface.return_value = mock_result
        mock_module.HuggingFaceImportError = type("HuggingFaceImportError", (Exception,), {})

        with patch.dict(sys.modules, {"services.huggingface_import": mock_module}):
            response = client.post("/datasets/import/huggingface", json={
                "dataset_id": "user/dataset",
                "split": "train"
            })

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "hf-123"

    def test_import_hf_error(self, client):
        """Import from HuggingFace with error returns 400."""
        import sys
        HFError = type("HuggingFaceImportError", (Exception,), {})

        mock_module = MagicMock()
        mock_module.import_from_huggingface.side_effect = HFError("Dataset not found")
        mock_module.HuggingFaceImportError = HFError

        with patch.dict(sys.modules, {"services.huggingface_import": mock_module}):
            response = client.post("/datasets/import/huggingface", json={
                "dataset_id": "nonexistent/dataset",
                "split": "train"
            })

        assert response.status_code == 400

    def test_import_hf_missing_dataset_id(self, client):
        """Import from HuggingFace without dataset_id returns 422."""
        response = client.post("/datasets/import/huggingface", json={})
        assert response.status_code == 422
