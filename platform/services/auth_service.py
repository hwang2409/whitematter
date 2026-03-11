"""Authentication service: password hashing and JWT tokens."""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from passlib.context import CryptContext
from jose import jwt, JWTError, ExpiredSignatureError


class AuthService:
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7

    def __init__(self, jwt_secret: Optional[str] = None):
        self.jwt_secret = jwt_secret or os.environ.get("JWT_SECRET", "dev-secret-change-me")
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def hash_password(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def verify_password(self, plain: str, hashed: str) -> bool:
        return self.pwd_context.verify(plain, hashed)

    def create_access_token(self, user_id: str, email: str,
                            expires_delta: Optional[timedelta] = None) -> str:
        expire = datetime.now(timezone.utc) + (
            expires_delta or timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        return jwt.encode(
            {"sub": user_id, "email": email, "exp": expire, "type": "access"},
            self.jwt_secret, algorithm=self.ALGORITHM
        )

    def create_refresh_token(self, user_id: str,
                             expires_delta: Optional[timedelta] = None) -> str:
        expire = datetime.now(timezone.utc) + (
            expires_delta or timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS)
        )
        return jwt.encode(
            {"sub": user_id, "exp": expire, "type": "refresh"},
            self.jwt_secret, algorithm=self.ALGORITHM
        )

    def decode_token(self, token: str) -> dict:
        try:
            return jwt.decode(token, self.jwt_secret, algorithms=[self.ALGORITHM])
        except ExpiredSignatureError:
            raise ValueError("Token has expired")
        except JWTError:
            raise ValueError("Invalid token")
