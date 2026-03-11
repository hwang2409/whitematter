"""Tests for credential encryption service."""
import os
import pytest
from cryptography.fernet import Fernet
from services.credential_service import CredentialService


@pytest.fixture
def cred_service():
    key = Fernet.generate_key().decode()
    return CredentialService(encryption_key=key)


def test_encrypt_decrypt_roundtrip(cred_service):
    encrypted = cred_service.encrypt("AKIAIOSFODNN7EXAMPLE")
    decrypted = cred_service.decrypt(encrypted)
    assert decrypted == "AKIAIOSFODNN7EXAMPLE"


def test_encrypted_differs_from_plaintext(cred_service):
    encrypted = cred_service.encrypt("mysecret")
    assert encrypted != b"mysecret"


def test_different_encryptions_differ(cred_service):
    e1 = cred_service.encrypt("same")
    e2 = cred_service.encrypt("same")
    assert e1 != e2  # Fernet uses random IV
