"""Tests for /deploy routes: create, list, get, terminate deployments."""

from unittest.mock import patch, MagicMock
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_deployment(
    dep_id="dep-1",
    model_id="m1",
    status="pending",
    instance_id=None,
    endpoint_url=None,
):
    d = MagicMock()
    d.id = dep_id
    d.model_id = model_id
    d.target_type = "ec2"
    d.status = status
    d.instance_id = instance_id
    d.endpoint_url = endpoint_url
    d.region = "us-east-1"
    d.error_message = None
    d.created_at = datetime(2025, 1, 1)
    d.updated_at = datetime(2025, 1, 1)
    d.user_id = "test-user-123"
    return d


def _mock_cred():
    cred = MagicMock()
    cred.user_id = "test-user-123"
    cred.encrypted_access_key = b"enc_key"
    cred.encrypted_secret_key = b"enc_secret"
    cred.default_region = "us-east-1"
    return cred


# ---------------------------------------------------------------------------
# POST /deploy
# ---------------------------------------------------------------------------

class TestCreateDeployment:
    def test_create_deployment_unsupported_target(self, client):
        """POST /deploy with unsupported target returns 400."""
        response = client.post("/deploy", json={
            "model_id": "m1",
            "target": "lambda"
        })

        assert response.status_code == 400
        assert "ec2" in response.json()["detail"].lower()

    def test_create_deployment_no_credentials(self, client, _override_db):
        """POST /deploy without AWS credentials returns 400."""
        dep = _mock_deployment()
        mock_session = _override_db

        # Deployment query returns dep, then AWSCredential query returns None
        mock_session.query.return_value.filter_by.return_value.first.side_effect = [
            dep,   # Deployment lookup
            None,  # AWSCredential lookup
        ]

        with patch("routes.deploy.deployment_service") as mock_deploy_svc:
            mock_deploy_svc.create_deployment.return_value = dep

            response = client.post("/deploy", json={
                "model_id": "m1",
                "target": "ec2"
            })

        assert response.status_code == 400


# ---------------------------------------------------------------------------
# GET /deploy/deployments
# ---------------------------------------------------------------------------

class TestListDeployments:
    def test_list_deployments_empty(self, client, _override_db):
        """GET /deploy/deployments returns empty list."""
        mock_session = _override_db
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.all.return_value = []

        response = client.get("/deploy/deployments")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_deployments_with_results(self, client, _override_db):
        """GET /deploy/deployments returns deployment list."""
        dep = _mock_deployment(status="live", endpoint_url="http://1.2.3.4:8080")
        mock_session = _override_db

        mock_session.query.return_value.filter_by.return_value.order_by.return_value.all.return_value = [dep]

        response = client.get("/deploy/deployments")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "dep-1"
        assert data[0]["status"] == "live"


# ---------------------------------------------------------------------------
# GET /deploy/deployments/{deployment_id}
# ---------------------------------------------------------------------------

class TestGetDeployment:
    def test_get_deployment_not_found(self, client, _override_db):
        """GET /deploy/deployments/{id} returns 404 if not found."""
        mock_session = _override_db
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        response = client.get("/deploy/deployments/nonexistent")

        assert response.status_code == 404

    def test_get_deployment_success(self, client, _override_db):
        """GET /deploy/deployments/{id} returns deployment details."""
        dep = _mock_deployment(status="live", endpoint_url="http://1.2.3.4:8080")
        mock_session = _override_db

        mock_session.query.return_value.filter_by.return_value.first.return_value = dep

        response = client.get("/deploy/deployments/dep-1")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "dep-1"
        assert data["endpoint_url"] == "http://1.2.3.4:8080"


# ---------------------------------------------------------------------------
# DELETE /deploy/deployments/{deployment_id}
# ---------------------------------------------------------------------------

class TestTerminateDeployment:
    def test_terminate_not_found(self, client, _override_db):
        """DELETE /deploy/deployments/{id} returns 404 if not found."""
        mock_session = _override_db
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        response = client.delete("/deploy/deployments/nonexistent")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /deploy/callback
# ---------------------------------------------------------------------------

class TestDeploymentCallback:
    def test_callback_missing_auth(self, client):
        """POST /deploy/callback without auth header returns 401."""
        response = client.post("/deploy/callback", json={
            "deployment_id": "dep-1",
            "endpoint_url": "http://1.2.3.4:8080"
        })
        assert response.status_code == 401

    def test_callback_success(self, client):
        """POST /deploy/callback with valid token sets deployment live."""
        with patch("routes.deploy.deployment_service") as mock_svc:
            mock_svc.set_live.return_value = True

            response = client.post(
                "/deploy/callback",
                json={
                    "deployment_id": "dep-1",
                    "endpoint_url": "http://1.2.3.4:8080"
                },
                headers={"Authorization": "Bearer valid-token"}
            )

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_callback_invalid_token(self, client):
        """POST /deploy/callback with invalid token returns 401."""
        with patch("routes.deploy.deployment_service") as mock_svc:
            mock_svc.set_live.return_value = False

            response = client.post(
                "/deploy/callback",
                json={
                    "deployment_id": "dep-1",
                    "endpoint_url": "http://1.2.3.4:8080"
                },
                headers={"Authorization": "Bearer bad-token"}
            )

        assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /deploy/artifacts/{deployment_id}
# ---------------------------------------------------------------------------

class TestGetArtifacts:
    def test_artifacts_found(self, client):
        """GET /deploy/artifacts/{id} returns tarball if valid."""
        with patch("routes.deploy.deployment_service") as mock_svc:
            mock_svc.get_artifacts.return_value = b"fake tarball data"

            response = client.get("/deploy/artifacts/dep-1?token=valid-tok")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-tar"

    def test_artifacts_not_found(self, client):
        """GET /deploy/artifacts/{id} returns 404 for invalid token."""
        with patch("routes.deploy.deployment_service") as mock_svc:
            mock_svc.get_artifacts.return_value = None

            response = client.get("/deploy/artifacts/dep-1?token=bad-tok")

        assert response.status_code == 404
