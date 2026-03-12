"""Tests for /auth routes: register, login, me, refresh."""

from unittest.mock import MagicMock
from datetime import datetime

import pytest

from db.database import get_db
from auth.dependencies import get_current_user


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_user(user_id="u1", email="new@test.com", pw_hash="$2b$12$hash"):
    user = MagicMock()
    user.id = user_id
    user.email = email
    user.password_hash = pw_hash
    user.oauth_provider = None
    user.created_at = datetime(2025, 1, 1)
    return user


@pytest.fixture
def db_session(app):
    """Override get_db dependency with a mock session."""
    mock_session = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_session
    yield mock_session
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def user_override(app):
    """Override get_current_user to return a mock user."""
    mock_user = _make_user(user_id="test-user-123", email="test@test.com")
    mock_user.created_at = MagicMock()
    mock_user.created_at.isoformat.return_value = "2025-01-01T00:00:00"
    app.dependency_overrides[get_current_user] = lambda: mock_user
    yield mock_user
    app.dependency_overrides.pop(get_current_user, None)


# ---------------------------------------------------------------------------
# POST /auth/register
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_success(self, client, db_session):
        """Register a new user successfully."""
        db_session.query.return_value.filter.return_value.first.return_value = None

        def _refresh(u):
            u.id = "new-uuid"
            u.email = "new@test.com"
        db_session.refresh = _refresh

        response = client.post("/auth/register", json={
            "email": "new@test.com",
            "password": "StrongP@ss1"
        })

        assert response.status_code == 201
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_register_duplicate_email(self, client, db_session):
        """Register with existing email returns 409."""
        existing_user = _make_user()
        db_session.query.return_value.filter.return_value.first.return_value = existing_user

        response = client.post("/auth/register", json={
            "email": "new@test.com",
            "password": "StrongP@ss1"
        })

        assert response.status_code == 409
        assert "already registered" in response.json()["detail"]

    def test_register_invalid_email(self, client):
        """Register with invalid email returns 422."""
        response = client.post("/auth/register", json={
            "email": "not-an-email",
            "password": "StrongP@ss1"
        })
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------

class TestLogin:
    def test_login_success(self, client, db_session):
        """Login with correct credentials."""
        from services.auth_service import AuthService
        svc = AuthService()
        hashed = svc.hash_password("correct-password")
        user = _make_user(pw_hash=hashed)

        db_session.query.return_value.filter.return_value.first.return_value = user

        response = client.post("/auth/login", json={
            "email": "new@test.com",
            "password": "correct-password"
        })

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_login_wrong_password(self, client, db_session):
        """Login with wrong password returns 401."""
        from services.auth_service import AuthService
        svc = AuthService()
        hashed = svc.hash_password("correct-password")
        user = _make_user(pw_hash=hashed)

        db_session.query.return_value.filter.return_value.first.return_value = user

        response = client.post("/auth/login", json={
            "email": "new@test.com",
            "password": "wrong-password"
        })

        assert response.status_code == 401

    def test_login_nonexistent_user(self, client, db_session):
        """Login with unknown email returns 401."""
        db_session.query.return_value.filter.return_value.first.return_value = None

        response = client.post("/auth/login", json={
            "email": "nobody@test.com",
            "password": "whatever"
        })

        assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /auth/me
# ---------------------------------------------------------------------------

class TestMe:
    def test_me_valid_token(self, client, user_override):
        """GET /auth/me with valid token returns user info."""
        response = client.get("/auth/me", headers={
            "Authorization": "Bearer fake-token"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-user-123"
        assert data["email"] == "test@test.com"

    def test_me_invalid_token(self, client, app):
        """GET /auth/me with invalid token returns 401/403."""
        # Remove the autouse auth override so real auth validation is used
        app.dependency_overrides.pop(get_current_user, None)
        try:
            response = client.get("/auth/me", headers={
                "Authorization": "Bearer invalid.token.here"
            })
            assert response.status_code in (401, 403)
        finally:
            # Restore for other tests (the autouse fixture will also restore)
            pass

    def test_me_no_token(self, client, app):
        """GET /auth/me with no auth header returns 401/403."""
        # Remove the autouse auth override so real auth validation is used
        app.dependency_overrides.pop(get_current_user, None)
        try:
            response = client.get("/auth/me")
            assert response.status_code in (401, 403)
        finally:
            pass


# ---------------------------------------------------------------------------
# POST /auth/refresh
# ---------------------------------------------------------------------------

class TestRefresh:
    def test_refresh_valid_token(self, client, db_session, refresh_token, mock_user):
        """POST /auth/refresh with valid refresh token returns new tokens."""
        db_session.query.return_value.filter.return_value.first.return_value = mock_user

        response = client.post("/auth/refresh", json={
            "refresh_token": refresh_token
        })

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_refresh_invalid_token(self, client):
        """POST /auth/refresh with invalid token returns 401."""
        response = client.post("/auth/refresh", json={
            "refresh_token": "invalid-refresh-token"
        })

        assert response.status_code == 401

    def test_refresh_with_access_token(self, client, db_session, auth_token, mock_user):
        """Using an access token for refresh should fail (wrong type)."""
        db_session.query.return_value.filter.return_value.first.return_value = mock_user

        response = client.post("/auth/refresh", json={
            "refresh_token": auth_token
        })

        assert response.status_code == 401
        assert "Invalid token type" in response.json()["detail"]
