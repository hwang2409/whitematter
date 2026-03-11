"""Tests for auth service."""
import pytest
from services.auth_service import AuthService


@pytest.fixture
def auth_service():
    return AuthService(jwt_secret="test-secret-key-min-32-chars-long")


def test_hash_and_verify_password(auth_service):
    hashed = auth_service.hash_password("mypassword")
    assert auth_service.verify_password("mypassword", hashed)
    assert not auth_service.verify_password("wrong", hashed)


def test_create_and_decode_access_token(auth_service):
    token = auth_service.create_access_token(user_id="abc123", email="test@test.com")
    payload = auth_service.decode_token(token)
    assert payload["sub"] == "abc123"
    assert payload["email"] == "test@test.com"


def test_create_and_decode_refresh_token(auth_service):
    token = auth_service.create_refresh_token(user_id="abc123")
    payload = auth_service.decode_token(token)
    assert payload["sub"] == "abc123"
    assert payload["type"] == "refresh"


def test_expired_token_raises(auth_service):
    from datetime import timedelta
    token = auth_service.create_access_token(
        user_id="abc123", email="test@test.com",
        expires_delta=timedelta(seconds=-1)
    )
    with pytest.raises(Exception):
        auth_service.decode_token(token)
