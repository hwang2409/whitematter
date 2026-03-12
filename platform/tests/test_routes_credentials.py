"""Tests for /credentials/aws routes: store, get, update, delete."""

from unittest.mock import patch, MagicMock
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_cred(user_id="test-user-123"):
    cred = MagicMock()
    cred.id = "cred-1"
    cred.user_id = user_id
    cred.encrypted_access_key = b"encrypted_ak"
    cred.encrypted_secret_key = b"encrypted_sk"
    cred.default_region = "us-east-1"
    cred.default_instance_type = "g4dn.xlarge"
    cred.endpoint_url = None
    cred.provider = None
    cred.created_at = datetime(2025, 1, 1)
    return cred


# ---------------------------------------------------------------------------
# POST /credentials/aws
# ---------------------------------------------------------------------------

class TestStoreCredentials:
    def test_store_success(self, client, _override_db):
        """POST /credentials/aws stores credentials."""
        mock_session = _override_db
        # No existing credentials
        mock_session.query.return_value.filter.return_value.first.return_value = None

        def _refresh(c):
            c.created_at = datetime(2025, 6, 1)
            c.default_region = "us-east-1"
            c.default_instance_type = "g4dn.xlarge"
            c.endpoint_url = None
            c.provider = None

        mock_session.refresh = _refresh

        mock_cred_service = MagicMock()
        mock_cred_service.encrypt.return_value = b"encrypted"

        with patch("routes.credentials._cred_service", return_value=mock_cred_service):
            response = client.post("/credentials/aws", json={
                "access_key": "AKIAIOSFODNN7EXAMPLE",
                "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            })

        assert response.status_code == 201

    def test_store_duplicate(self, client, _override_db):
        """POST /credentials/aws with existing creds returns 409."""
        existing_cred = _mock_cred()
        mock_session = _override_db
        mock_session.query.return_value.filter.return_value.first.return_value = existing_cred

        response = client.post("/credentials/aws", json={
            "access_key": "AKIAIOSFODNN7EXAMPLE",
            "secret_key": "secret",
        })

        assert response.status_code == 409


# ---------------------------------------------------------------------------
# GET /credentials/aws
# ---------------------------------------------------------------------------

class TestGetCredentials:
    def test_get_no_credentials(self, client, _override_db):
        """GET /credentials/aws when none exist returns has_credentials=False."""
        mock_session = _override_db
        mock_session.query.return_value.filter.return_value.first.return_value = None

        response = client.get("/credentials/aws")

        assert response.status_code == 200
        assert response.json()["has_credentials"] is False

    def test_get_with_credentials(self, client, _override_db):
        """GET /credentials/aws returns credential info when present."""
        cred = _mock_cred()
        mock_session = _override_db
        mock_session.query.return_value.filter.return_value.first.return_value = cred

        response = client.get("/credentials/aws")

        assert response.status_code == 200
        data = response.json()
        assert data["has_credentials"] is True
        assert data["default_region"] == "us-east-1"


# ---------------------------------------------------------------------------
# PUT /credentials/aws
# ---------------------------------------------------------------------------

class TestUpdateCredentials:
    def test_update_not_found(self, client, _override_db):
        """PUT /credentials/aws when none exist returns 404."""
        mock_session = _override_db
        mock_session.query.return_value.filter.return_value.first.return_value = None

        response = client.put("/credentials/aws", json={
            "access_key": "AKIANEW",
            "secret_key": "newsecret",
        })

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /credentials/aws
# ---------------------------------------------------------------------------

class TestDeleteCredentials:
    def test_delete_not_found(self, client, _override_db):
        """DELETE /credentials/aws when none exist returns 404."""
        mock_session = _override_db
        mock_session.query.return_value.filter.return_value.first.return_value = None

        response = client.delete("/credentials/aws")

        assert response.status_code == 404
