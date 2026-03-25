import os
from cryptography.fernet import Fernet


class CredentialService:
    def __init__(self, encryption_key: str | None = None):
        key = encryption_key or os.environ.get("CREDENTIAL_ENCRYPTION_KEY")
        if not key:
            raise ValueError("CREDENTIAL_ENCRYPTION_KEY must be set")
        self.fernet = Fernet(key.encode() if isinstance(key, str) else key)

    def encrypt(self, plaintext: str) -> bytes:
        return self.fernet.encrypt(plaintext.encode())

    def decrypt(self, ciphertext: bytes) -> str:
        return self.fernet.decrypt(ciphertext).decode()
