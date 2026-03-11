# Whitematter AWS Deployment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy whitematter as a public platform with BYOC training and pip-installable library.

**Architecture:** Extend existing FastAPI + React app with auth, AWS credential management, S3 proxy, BYOC EC2 provisioning, and production Docker deployment. PostgreSQL replaces SQLite for multi-user concurrency.

**Tech Stack:** FastAPI, SQLAlchemy + Alembic, PostgreSQL, boto3, pybind11, React + TypeScript, Docker, cibuildwheel

**Spec:** `docs/superpowers/specs/2026-03-11-aws-deployment-design.md`

---

## Chunk 1: Database & Auth Foundation

### Task 1: PostgreSQL + Alembic Setup

**Files:**
- Modify: `platform/requirements.txt`
- Modify: `platform/db/database.py`
- Create: `platform/alembic.ini`
- Create: `platform/migrations/env.py`
- Create: `platform/migrations/script.mako`
- Create: `platform/migrations/versions/.gitkeep`

- [ ] **Step 1: Add dependencies to requirements.txt**

Add to `platform/requirements.txt`:
```
psycopg2-binary>=2.9.9
alembic>=1.13.0
passlib[bcrypt]>=1.7.4
python-jose[cryptography]>=3.3.0
cryptography>=41.0.0
httpx>=0.25.0
boto3>=1.34.0
slowapi>=0.1.9
```

- [ ] **Step 2: Update database.py to support DATABASE_URL env var**

Modify `platform/db/database.py` — replace the hardcoded SQLite path logic with:
- Read `DATABASE_URL` from env (defaults to existing SQLite path for dev)
- If starts with `postgresql://`, use PostgreSQL engine
- If starts with `sqlite://`, use existing SQLite config
- Keep existing `get_db()` and `get_db_session()` unchanged

- [ ] **Step 3: Initialize Alembic**

Run:
```bash
cd platform && python -m alembic init migrations
```

- [ ] **Step 4: Configure alembic.ini**

Set `sqlalchemy.url` to use `DATABASE_URL` env var. Update `script_location = migrations`.

- [ ] **Step 5: Configure migrations/env.py**

Import all models from `platform/db/models.py`, set `target_metadata = Base.metadata`. Read `DATABASE_URL` from env.

- [ ] **Step 6: Generate initial migration from existing models**

Run:
```bash
cd platform && python -m alembic revision --autogenerate -m "initial schema"
```

- [ ] **Step 7: Test migration against local PostgreSQL (Docker)**

Run:
```bash
docker run -d --name wm-postgres -e POSTGRES_DB=whitematter -e POSTGRES_USER=whitematter -e POSTGRES_PASSWORD=test -p 5433:5432 postgres:16-alpine
DATABASE_URL=postgresql://whitematter:test@localhost:5433/whitematter python -m alembic upgrade head
```

- [ ] **Step 8: Commit**

```bash
git add platform/requirements.txt platform/db/database.py platform/alembic.ini platform/migrations/
git commit -m "feat: add PostgreSQL support and Alembic migrations"
```

---

### Task 2: User Model + Auth Tables

**Files:**
- Modify: `platform/db/models.py`
- Create: `platform/db/auth_models.py`

- [ ] **Step 1: Create auth_models.py with User, AWSCredential, ByocTrainingJob, ModelArchitecture**

Create `platform/db/auth_models.py`:
```python
"""Auth and BYOC database models."""
import enum
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, ForeignKey,
    JSON, LargeBinary, UniqueConstraint
)
from sqlalchemy.orm import relationship
from platform.db.models import Base


def gen_uuid():
    return uuid.uuid4().hex


class User(Base):
    __tablename__ = "users"
    id = Column(String(32), primary_key=True, default=gen_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)
    oauth_provider = Column(String(50), nullable=True)
    oauth_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    aws_credentials = relationship("AWSCredential", back_populates="user", uselist=False)
    byoc_jobs = relationship("ByocTrainingJob", back_populates="user")
    architectures = relationship("ModelArchitecture", back_populates="user")

    __table_args__ = (
        UniqueConstraint("oauth_provider", "oauth_id", name="uq_oauth"),
    )


class AWSCredential(Base):
    __tablename__ = "aws_credentials"
    id = Column(String(32), primary_key=True, default=gen_uuid)
    user_id = Column(String(32), ForeignKey("users.id"), unique=True, nullable=False)
    encrypted_access_key = Column(LargeBinary, nullable=False)
    encrypted_secret_key = Column(LargeBinary, nullable=False)
    default_region = Column(String(30), default="us-east-1")
    default_instance_type = Column(String(30), default="g4dn.xlarge")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="aws_credentials")


class ByocJobStatus(str, enum.Enum):
    PENDING = "pending"
    LAUNCHING = "launching"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class ByocTrainingJob(Base):
    __tablename__ = "byoc_training_jobs"
    id = Column(String(32), primary_key=True, default=gen_uuid)
    user_id = Column(String(32), ForeignKey("users.id"), nullable=False)
    model_config = Column(JSON, nullable=False)
    dataset_config = Column(JSON, nullable=False)
    instance_type = Column(String(30), default="g4dn.xlarge")
    instance_id = Column(String(30), nullable=True)
    region = Column(String(30), default="us-east-1")
    status = Column(String(20), default=ByocJobStatus.PENDING.value)
    job_token_hash = Column(String(128), nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    metrics = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    user = relationship("User", back_populates="byoc_jobs")


class ModelArchitecture(Base):
    __tablename__ = "model_architectures"
    id = Column(String(32), primary_key=True, default=gen_uuid)
    user_id = Column(String(32), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    config = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="architectures")
```

- [ ] **Step 2: Import auth_models in migrations/env.py**

Add to `migrations/env.py`:
```python
from platform.db.auth_models import User, AWSCredential, ByocTrainingJob, ModelArchitecture
```

- [ ] **Step 3: Generate migration for new tables**

Run:
```bash
cd platform && python -m alembic revision --autogenerate -m "add auth and byoc tables"
```

- [ ] **Step 4: Apply migration and verify**

Run:
```bash
DATABASE_URL=postgresql://whitematter:test@localhost:5433/whitematter python -m alembic upgrade head
```

- [ ] **Step 5: Commit**

```bash
git add platform/db/auth_models.py platform/migrations/
git commit -m "feat: add User, AWSCredential, ByocTrainingJob, ModelArchitecture models"
```

---

### Task 3: Auth Service (JWT + Password)

**Files:**
- Create: `platform/services/auth_service.py`
- Create: `platform/auth/__init__.py`
- Create: `platform/auth/dependencies.py`
- Create: `platform/tests/test_auth_service.py`

- [ ] **Step 1: Write test_auth_service.py**

```python
"""Tests for auth service."""
import pytest
from platform.services.auth_service import AuthService

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd platform && python -m pytest tests/test_auth_service.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement auth_service.py**

Create `platform/services/auth_service.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd platform && python -m pytest tests/test_auth_service.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Create auth dependencies (get_current_user)**

Create `platform/auth/__init__.py` (empty) and `platform/auth/dependencies.py`:
```python
"""FastAPI auth dependencies."""
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from platform.db.database import get_db
from platform.db.auth_models import User
from platform.services.auth_service import AuthService

security = HTTPBearer()
auth_service = AuthService()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    try:
        payload = auth_service.decode_token(credentials.credentials)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")

    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user
```

- [ ] **Step 6: Commit**

```bash
git add platform/services/auth_service.py platform/auth/ platform/tests/test_auth_service.py
git commit -m "feat: add auth service with JWT and password hashing"
```

---

### Task 4: Auth Routes

**Files:**
- Create: `platform/routes/auth.py`
- Create: `platform/schemas/auth_schemas.py`
- Modify: `platform/server.py`

- [ ] **Step 1: Create auth schemas**

Create `platform/schemas/` directory and `platform/schemas/__init__.py` (empty).

Create `platform/schemas/auth_schemas.py`:
```python
"""Auth request/response schemas."""
from pydantic import BaseModel, EmailStr


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class RefreshRequest(BaseModel):
    refresh_token: str

class UserResponse(BaseModel):
    id: str
    email: str
    oauth_provider: str | None = None
    created_at: str
```

- [ ] **Step 2: Create auth routes**

Create `platform/routes/auth.py`:
```python
"""Authentication routes."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from platform.db.database import get_db
from platform.db.auth_models import User
from platform.services.auth_service import AuthService
from platform.schemas.auth_schemas import (
    RegisterRequest, LoginRequest, TokenResponse, RefreshRequest, UserResponse
)
from platform.auth.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])
auth_service = AuthService()


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=req.email,
        password_hash=auth_service.hash_password(req.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return TokenResponse(
        access_token=auth_service.create_access_token(user.id, user.email),
        refresh_token=auth_service.create_refresh_token(user.id),
    )


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not user.password_hash:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not auth_service.verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return TokenResponse(
        access_token=auth_service.create_access_token(user.id, user.email),
        refresh_token=auth_service.create_refresh_token(user.id),
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(req: RefreshRequest, db: Session = Depends(get_db)):
    try:
        payload = auth_service.decode_token(req.refresh_token)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return TokenResponse(
        access_token=auth_service.create_access_token(user.id, user.email),
        refresh_token=auth_service.create_refresh_token(user.id),
    )


@router.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(get_current_user)):
    return UserResponse(
        id=user.id,
        email=user.email,
        oauth_provider=user.oauth_provider,
        created_at=user.created_at.isoformat(),
    )
```

- [ ] **Step 3: Register auth router in server.py**

Add to `platform/server.py` imports:
```python
from routes.auth import router as auth_router
```

Add after existing router includes:
```python
app.include_router(auth_router)
```

- [ ] **Step 4: Commit**

```bash
git add platform/routes/auth.py platform/schemas/ platform/server.py
git commit -m "feat: add auth routes (register, login, refresh, me)"
```

---

## Chunk 2: AWS Credentials & S3 Management

### Task 5: Credential Encryption Service

**Files:**
- Create: `platform/services/credential_service.py`
- Create: `platform/tests/test_credential_service.py`

- [ ] **Step 1: Write test**

Create `platform/tests/test_credential_service.py`:
```python
"""Tests for credential encryption service."""
import os
import pytest
from cryptography.fernet import Fernet
from platform.services.credential_service import CredentialService

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
```

- [ ] **Step 2: Run test — expect fail**

Run: `cd platform && python -m pytest tests/test_credential_service.py -v`

- [ ] **Step 3: Implement credential_service.py**

```python
"""AWS credential encryption service."""
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
```

- [ ] **Step 4: Run tests — expect pass**

Run: `cd platform && python -m pytest tests/test_credential_service.py -v`

- [ ] **Step 5: Commit**

```bash
git add platform/services/credential_service.py platform/tests/test_credential_service.py
git commit -m "feat: add credential encryption service (Fernet)"
```

---

### Task 6: AWS Credential Routes

**Files:**
- Create: `platform/routes/credentials.py`
- Create: `platform/schemas/credential_schemas.py`
- Modify: `platform/server.py`

- [ ] **Step 1: Create credential schemas**

Create `platform/schemas/credential_schemas.py`:
```python
"""AWS credential schemas."""
from pydantic import BaseModel


class AWSCredentialRequest(BaseModel):
    access_key: str
    secret_key: str
    default_region: str = "us-east-1"
    default_instance_type: str = "g4dn.xlarge"

class AWSCredentialResponse(BaseModel):
    has_credentials: bool
    default_region: str | None = None
    default_instance_type: str | None = None
    created_at: str | None = None
```

- [ ] **Step 2: Create credential routes**

Create `platform/routes/credentials.py`:
```python
"""AWS credential management routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from platform.db.database import get_db
from platform.db.auth_models import User, AWSCredential
from platform.auth.dependencies import get_current_user
from platform.services.credential_service import CredentialService
from platform.schemas.credential_schemas import AWSCredentialRequest, AWSCredentialResponse

router = APIRouter(prefix="/credentials", tags=["credentials"])
cred_service = CredentialService()


@router.post("/aws", response_model=AWSCredentialResponse, status_code=201)
def store_credentials(req: AWSCredentialRequest,
                      user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    existing = db.query(AWSCredential).filter(AWSCredential.user_id == user.id).first()
    if existing:
        raise HTTPException(status_code=409, detail="Credentials already exist. Use PUT to update.")

    cred = AWSCredential(
        user_id=user.id,
        encrypted_access_key=cred_service.encrypt(req.access_key),
        encrypted_secret_key=cred_service.encrypt(req.secret_key),
        default_region=req.default_region,
        default_instance_type=req.default_instance_type,
    )
    db.add(cred)
    db.commit()
    db.refresh(cred)

    return AWSCredentialResponse(
        has_credentials=True,
        default_region=cred.default_region,
        default_instance_type=cred.default_instance_type,
        created_at=cred.created_at.isoformat(),
    )


@router.get("/aws", response_model=AWSCredentialResponse)
def get_credentials(user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    cred = db.query(AWSCredential).filter(AWSCredential.user_id == user.id).first()
    if not cred:
        return AWSCredentialResponse(has_credentials=False)
    return AWSCredentialResponse(
        has_credentials=True,
        default_region=cred.default_region,
        default_instance_type=cred.default_instance_type,
        created_at=cred.created_at.isoformat(),
    )


@router.put("/aws", response_model=AWSCredentialResponse)
def update_credentials(req: AWSCredentialRequest,
                       user: User = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    cred = db.query(AWSCredential).filter(AWSCredential.user_id == user.id).first()
    if not cred:
        raise HTTPException(status_code=404, detail="No credentials found")

    cred.encrypted_access_key = cred_service.encrypt(req.access_key)
    cred.encrypted_secret_key = cred_service.encrypt(req.secret_key)
    cred.default_region = req.default_region
    cred.default_instance_type = req.default_instance_type
    db.commit()
    db.refresh(cred)

    return AWSCredentialResponse(
        has_credentials=True,
        default_region=cred.default_region,
        default_instance_type=cred.default_instance_type,
        created_at=cred.created_at.isoformat(),
    )


@router.delete("/aws", status_code=204)
def delete_credentials(user: User = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    cred = db.query(AWSCredential).filter(AWSCredential.user_id == user.id).first()
    if not cred:
        raise HTTPException(status_code=404, detail="No credentials found")
    db.delete(cred)
    db.commit()
```

- [ ] **Step 3: Register in server.py**

Add `from routes.credentials import router as credentials_router` and `app.include_router(credentials_router)`.

- [ ] **Step 4: Commit**

```bash
git add platform/routes/credentials.py platform/schemas/credential_schemas.py platform/server.py
git commit -m "feat: add AWS credential CRUD routes"
```

---

### Task 7: S3 Proxy Service + Routes

**Files:**
- Create: `platform/services/s3_service.py`
- Create: `platform/routes/storage.py`
- Create: `platform/schemas/storage_schemas.py`
- Modify: `platform/server.py`

- [ ] **Step 1: Create S3 service**

Create `platform/services/s3_service.py`:
```python
"""S3 proxy service — operates on user's S3 using their credentials."""
import boto3
from typing import BinaryIO
from platform.services.credential_service import CredentialService


class S3Service:
    def __init__(self, cred_service: CredentialService | None = None):
        self.cred_service = cred_service or CredentialService()

    def _get_client(self, encrypted_access_key: bytes, encrypted_secret_key: bytes, region: str):
        return boto3.client(
            "s3",
            aws_access_key_id=self.cred_service.decrypt(encrypted_access_key),
            aws_secret_access_key=self.cred_service.decrypt(encrypted_secret_key),
            region_name=region,
        )

    def list_buckets(self, encrypted_ak: bytes, encrypted_sk: bytes, region: str) -> list[dict]:
        client = self._get_client(encrypted_ak, encrypted_sk, region)
        resp = client.list_buckets()
        return [{"name": b["Name"], "created": b["CreationDate"].isoformat()} for b in resp["Buckets"]]

    def create_bucket(self, encrypted_ak: bytes, encrypted_sk: bytes, region: str, bucket_name: str):
        client = self._get_client(encrypted_ak, encrypted_sk, region)
        config = {"LocationConstraint": region} if region != "us-east-1" else {}
        create_args = {"Bucket": bucket_name}
        if config:
            create_args["CreateBucketConfiguration"] = config
        client.create_bucket(**create_args)

    def list_objects(self, encrypted_ak: bytes, encrypted_sk: bytes, region: str,
                     bucket: str, prefix: str = "") -> list[dict]:
        client = self._get_client(encrypted_ak, encrypted_sk, region)
        resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
        items = []
        for p in resp.get("CommonPrefixes", []):
            items.append({"key": p["Prefix"], "type": "folder"})
        for obj in resp.get("Contents", []):
            items.append({
                "key": obj["Key"],
                "type": "file",
                "size": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
            })
        return items

    def upload_file(self, encrypted_ak: bytes, encrypted_sk: bytes, region: str,
                    bucket: str, key: str, file_obj: BinaryIO):
        client = self._get_client(encrypted_ak, encrypted_sk, region)
        client.upload_fileobj(file_obj, bucket, key)

    def delete_object(self, encrypted_ak: bytes, encrypted_sk: bytes, region: str,
                      bucket: str, key: str):
        client = self._get_client(encrypted_ak, encrypted_sk, region)
        client.delete_object(Bucket=bucket, Key=key)

    def generate_presigned_url(self, encrypted_ak: bytes, encrypted_sk: bytes, region: str,
                               bucket: str, key: str, expires_in: int = 3600) -> str:
        client = self._get_client(encrypted_ak, encrypted_sk, region)
        return client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in
        )
```

- [ ] **Step 2: Create storage schemas**

Create `platform/schemas/storage_schemas.py`:
```python
"""S3 storage schemas."""
from pydantic import BaseModel


class CreateBucketRequest(BaseModel):
    name: str
    region: str | None = None

class DeleteObjectRequest(BaseModel):
    bucket: str
    key: str

class PresignedUrlRequest(BaseModel):
    bucket: str
    key: str
    expires_in: int = 3600

class PresignedUrlResponse(BaseModel):
    url: str
```

- [ ] **Step 3: Create storage routes**

Create `platform/routes/storage.py`:
```python
"""S3 storage management routes."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from platform.db.database import get_db
from platform.db.auth_models import User, AWSCredential
from platform.auth.dependencies import get_current_user
from platform.services.s3_service import S3Service
from platform.schemas.storage_schemas import (
    CreateBucketRequest, DeleteObjectRequest, PresignedUrlRequest, PresignedUrlResponse
)

router = APIRouter(prefix="/storage", tags=["storage"])
s3_service = S3Service()


def _get_aws_cred(user: User, db: Session) -> AWSCredential:
    cred = db.query(AWSCredential).filter(AWSCredential.user_id == user.id).first()
    if not cred:
        raise HTTPException(status_code=400, detail="No AWS credentials configured")
    return cred


@router.get("/buckets")
def list_buckets(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    cred = _get_aws_cred(user, db)
    return s3_service.list_buckets(cred.encrypted_access_key, cred.encrypted_secret_key, cred.default_region)


@router.post("/buckets", status_code=201)
def create_bucket(req: CreateBucketRequest,
                  user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    cred = _get_aws_cred(user, db)
    region = req.region or cred.default_region
    s3_service.create_bucket(cred.encrypted_access_key, cred.encrypted_secret_key, region, req.name)
    return {"name": req.name, "region": region}


@router.get("/buckets/{bucket}/objects")
def list_objects(bucket: str, prefix: str = "",
                 user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    cred = _get_aws_cred(user, db)
    return s3_service.list_objects(
        cred.encrypted_access_key, cred.encrypted_secret_key, cred.default_region, bucket, prefix
    )


@router.post("/upload")
async def upload_file(bucket: str = Form(...), key: str = Form(...),
                      file: UploadFile = File(...),
                      user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    cred = _get_aws_cred(user, db)
    s3_service.upload_file(
        cred.encrypted_access_key, cred.encrypted_secret_key, cred.default_region,
        bucket, key, file.file
    )
    return {"bucket": bucket, "key": key, "size": file.size}


@router.delete("/objects", status_code=204)
def delete_object(req: DeleteObjectRequest,
                  user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    cred = _get_aws_cred(user, db)
    s3_service.delete_object(
        cred.encrypted_access_key, cred.encrypted_secret_key, cred.default_region, req.bucket, req.key
    )


@router.post("/presigned-url", response_model=PresignedUrlResponse)
def get_presigned_url(req: PresignedUrlRequest,
                      user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    cred = _get_aws_cred(user, db)
    url = s3_service.generate_presigned_url(
        cred.encrypted_access_key, cred.encrypted_secret_key, cred.default_region,
        req.bucket, req.key, req.expires_in
    )
    return PresignedUrlResponse(url=url)
```

- [ ] **Step 4: Register in server.py**

Add `from routes.storage import router as storage_router` and `app.include_router(storage_router)`.

- [ ] **Step 5: Commit**

```bash
git add platform/services/s3_service.py platform/routes/storage.py platform/schemas/storage_schemas.py platform/server.py
git commit -m "feat: add S3 storage management routes and service"
```

---

## Chunk 3: BYOC Engine

### Task 8: BYOC Provisioner Service

**Files:**
- Create: `platform/byoc/__init__.py`
- Create: `platform/byoc/provisioner.py`
- Create: `platform/byoc/user_data.py`

- [ ] **Step 1: Create user-data script template**

Create `platform/byoc/__init__.py` (empty) and `platform/byoc/user_data.py`:
```python
"""EC2 user-data script generation for BYOC training."""


def generate_user_data(
    platform_url: str,
    job_id: str,
    job_token: str,
    whitematter_version: str,
    training_config: dict,
) -> str:
    """Generate bash user-data script for EC2 instance."""
    return f"""#!/bin/bash
set -e

# Log everything
exec > /var/log/whitematter-training.log 2>&1

echo "=== Whitematter BYOC Training ==="
echo "Job ID: {job_id}"
echo "Platform: {platform_url}"

# Install whitematter
pip install whitematter=={whitematter_version}

# Create training script
cat > /tmp/train.py << 'TRAIN_SCRIPT'
import json
import sys
import time
import urllib.request

PLATFORM_URL = "{platform_url}"
JOB_ID = "{job_id}"
JOB_TOKEN = "{job_token}"
CONFIG = json.loads('''{json.dumps(training_config)}''')

def api_call(path, data=None):
    url = f"{{PLATFORM_URL}}{{path}}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        url, data=body,
        headers={{"Authorization": f"Bearer {{JOB_TOKEN}}", "Content-Type": "application/json"}}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"API call failed: {{e}}")
        return None

def heartbeat():
    api_call(f"/training/{{JOB_ID}}/heartbeat")

def log_line(msg):
    api_call(f"/training/{{JOB_ID}}/log", {{"message": msg}})

def report_metrics(epoch, loss, accuracy):
    api_call(f"/training/{{JOB_ID}}/metrics", {{
        "epoch": epoch, "loss": loss, "accuracy": accuracy
    }})

def complete(metrics):
    api_call(f"/training/{{JOB_ID}}/complete", {{"metrics": metrics}})

def fail(error):
    api_call(f"/training/{{JOB_ID}}/fail", {{"error_message": error}})

try:
    import whitematter as wm
    heartbeat()
    log_line("whitematter loaded, starting training...")

    # TODO: Build model from CONFIG, download data from S3, train
    # This will be filled in when pybind11 bindings are extended

    log_line("Training complete")
    complete({{"status": "done"}})
except Exception as e:
    fail(str(e))
    sys.exit(1)
TRAIN_SCRIPT

python /tmp/train.py

# Self-terminate
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
"""
```

- [ ] **Step 2: Create provisioner service**

Create `platform/byoc/provisioner.py`:
```python
"""BYOC EC2 instance provisioner."""
import hashlib
import secrets
from datetime import datetime
import boto3
from sqlalchemy.orm import Session
from platform.db.auth_models import AWSCredential, ByocTrainingJob, ByocJobStatus
from platform.services.credential_service import CredentialService
from platform.byoc.user_data import generate_user_data

# AWS Deep Learning AMI (Ubuntu 22.04) — updated per-region
DEEP_LEARNING_AMIS = {
    "us-east-1": "ami-0a0c8eebcdd6dcbd0",
    "us-west-2": "ami-0a0c8eebcdd6dcbd0",
}
DEFAULT_AMI = "ami-0a0c8eebcdd6dcbd0"

WHITEMATTER_VERSION = "0.1.0"


class ByocProvisioner:
    def __init__(self, platform_url: str = "http://localhost:8080",
                 cred_service: CredentialService | None = None):
        self.platform_url = platform_url
        self.cred_service = cred_service or CredentialService()

    def _get_ec2_client(self, cred: AWSCredential):
        return boto3.client(
            "ec2",
            aws_access_key_id=self.cred_service.decrypt(cred.encrypted_access_key),
            aws_secret_access_key=self.cred_service.decrypt(cred.encrypted_secret_key),
            region_name=cred.default_region,
        )

    def launch_job(self, db: Session, cred: AWSCredential, job: ByocTrainingJob) -> str:
        """Launch a GPU EC2 instance for training. Returns instance ID."""
        ec2 = self._get_ec2_client(cred)

        # Generate one-time job token
        job_token = secrets.token_urlsafe(48)
        job.job_token_hash = hashlib.sha256(job_token.encode()).hexdigest()
        job.status = ByocJobStatus.LAUNCHING.value
        db.commit()

        # Generate user-data script
        user_data = generate_user_data(
            platform_url=self.platform_url,
            job_id=job.id,
            job_token=job_token,
            whitematter_version=WHITEMATTER_VERSION,
            training_config={
                "model_config": job.model_config,
                "dataset_config": job.dataset_config,
            },
        )

        ami = DEEP_LEARNING_AMIS.get(cred.default_region, DEFAULT_AMI)

        # Launch instance
        resp = ec2.run_instances(
            ImageId=ami,
            InstanceType=job.instance_type or cred.default_instance_type,
            MinCount=1,
            MaxCount=1,
            UserData=user_data,
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"whitematter-job-{job.id}"},
                    {"Key": "whitematter-job-id", "Value": job.id},
                ],
            }],
            InstanceInitiatedShutdownBehavior="terminate",
        )

        instance_id = resp["Instances"][0]["InstanceId"]
        job.instance_id = instance_id
        job.status = ByocJobStatus.RUNNING.value
        job.started_at = datetime.utcnow()
        job.region = cred.default_region
        db.commit()

        return instance_id

    def stop_job(self, db: Session, cred: AWSCredential, job: ByocTrainingJob):
        """Terminate the EC2 instance for a job."""
        if not job.instance_id:
            return
        ec2 = self._get_ec2_client(cred)
        ec2.terminate_instances(InstanceIds=[job.instance_id])
        job.status = ByocJobStatus.STOPPED.value
        job.completed_at = datetime.utcnow()
        db.commit()

    def check_instance_status(self, cred: AWSCredential, instance_id: str) -> str:
        """Check if an EC2 instance is still running."""
        ec2 = self._get_ec2_client(cred)
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]
        return state
```

- [ ] **Step 3: Commit**

```bash
git add platform/byoc/
git commit -m "feat: add BYOC provisioner and user-data script generation"
```

---

### Task 9: BYOC Training Routes + Callback Routes

**Files:**
- Create: `platform/routes/byoc_training.py`
- Create: `platform/schemas/byoc_schemas.py`
- Modify: `platform/server.py`

- [ ] **Step 1: Create BYOC schemas**

Create `platform/schemas/byoc_schemas.py`:
```python
"""BYOC training schemas."""
from pydantic import BaseModel


class LaunchRequest(BaseModel):
    model_config_data: dict  # renamed to avoid pydantic conflict
    dataset_config: dict
    instance_type: str = "g4dn.xlarge"
    region: str | None = None

class JobStatusResponse(BaseModel):
    id: str
    status: str
    instance_type: str
    instance_id: str | None = None
    region: str
    started_at: str | None = None
    completed_at: str | None = None
    metrics: dict | None = None
    error_message: str | None = None

class HeartbeatRequest(BaseModel):
    pass

class LogRequest(BaseModel):
    message: str

class MetricsRequest(BaseModel):
    epoch: int
    loss: float
    accuracy: float | None = None

class CompleteRequest(BaseModel):
    metrics: dict

class FailRequest(BaseModel):
    error_message: str
```

- [ ] **Step 2: Create BYOC training routes**

Create `platform/routes/byoc_training.py`:
```python
"""BYOC training routes (launch, status, stop) + callback routes."""
import hashlib
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from platform.db.database import get_db
from platform.db.auth_models import User, AWSCredential, ByocTrainingJob, ByocJobStatus
from platform.auth.dependencies import get_current_user
from platform.byoc.provisioner import ByocProvisioner
from platform.schemas.byoc_schemas import (
    LaunchRequest, JobStatusResponse, HeartbeatRequest,
    LogRequest, MetricsRequest, CompleteRequest, FailRequest,
)

router = APIRouter(tags=["training"])
provisioner = ByocProvisioner()

# Store logs in memory (could move to DB later)
_job_logs: dict[str, list[str]] = {}


# --- User-facing routes (require JWT auth) ---

@router.post("/training/launch", response_model=JobStatusResponse, status_code=201)
def launch_training(req: LaunchRequest,
                    user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    cred = db.query(AWSCredential).filter(AWSCredential.user_id == user.id).first()
    if not cred:
        raise HTTPException(status_code=400, detail="No AWS credentials configured")

    job = ByocTrainingJob(
        user_id=user.id,
        model_config=req.model_config_data,
        dataset_config=req.dataset_config,
        instance_type=req.instance_type,
        region=req.region or cred.default_region,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        provisioner.launch_job(db, cred, job)
    except Exception as e:
        job.status = ByocJobStatus.FAILED.value
        job.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to launch instance: {e}")

    return _job_to_response(job)


@router.get("/training/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str,
                   user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    job = db.query(ByocTrainingJob).filter(
        ByocTrainingJob.id == job_id, ByocTrainingJob.user_id == user.id
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


@router.get("/training/{job_id}/logs")
def get_job_logs(job_id: str,
                 user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    job = db.query(ByocTrainingJob).filter(
        ByocTrainingJob.id == job_id, ByocTrainingJob.user_id == user.id
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"logs": _job_logs.get(job_id, [])}


@router.post("/training/{job_id}/stop")
def stop_training(job_id: str,
                  user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    job = db.query(ByocTrainingJob).filter(
        ByocTrainingJob.id == job_id, ByocTrainingJob.user_id == user.id
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    cred = db.query(AWSCredential).filter(AWSCredential.user_id == user.id).first()
    if cred:
        provisioner.stop_job(db, cred, job)

    return _job_to_response(job)


# --- Callback routes (require job token auth) ---

def _verify_job_token(job_id: str, authorization: str, db: Session) -> ByocTrainingJob:
    """Verify one-time job token from GPU instance."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization")
    token = authorization[7:]
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    job = db.query(ByocTrainingJob).filter(ByocTrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.job_token_hash != token_hash:
        raise HTTPException(status_code=401, detail="Invalid job token")
    return job


@router.post("/training/{job_id}/heartbeat")
def heartbeat(job_id: str,
              authorization: str = Header(...),
              db: Session = Depends(get_db)):
    job = _verify_job_token(job_id, authorization, db)
    job.updated_at = datetime.utcnow()
    db.commit()
    return {"status": "ok"}


@router.post("/training/{job_id}/log")
def append_log(job_id: str, req: LogRequest,
               authorization: str = Header(...),
               db: Session = Depends(get_db)):
    _verify_job_token(job_id, authorization, db)
    if job_id not in _job_logs:
        _job_logs[job_id] = []
    _job_logs[job_id].append(req.message)
    return {"status": "ok"}


@router.post("/training/{job_id}/metrics")
def report_metrics(job_id: str, req: MetricsRequest,
                   authorization: str = Header(...),
                   db: Session = Depends(get_db)):
    job = _verify_job_token(job_id, authorization, db)
    metrics = job.metrics or {}
    if "epochs" not in metrics:
        metrics["epochs"] = []
    metrics["epochs"].append({
        "epoch": req.epoch, "loss": req.loss, "accuracy": req.accuracy
    })
    job.metrics = metrics
    db.commit()
    return {"status": "ok"}


@router.post("/training/{job_id}/complete")
def complete_job(job_id: str, req: CompleteRequest,
                 authorization: str = Header(...),
                 db: Session = Depends(get_db)):
    job = _verify_job_token(job_id, authorization, db)
    job.status = ByocJobStatus.COMPLETED.value
    job.completed_at = datetime.utcnow()
    job.metrics = req.metrics
    db.commit()
    return {"status": "ok"}


@router.post("/training/{job_id}/fail")
def fail_job(job_id: str, req: FailRequest,
             authorization: str = Header(...),
             db: Session = Depends(get_db)):
    job = _verify_job_token(job_id, authorization, db)
    job.status = ByocJobStatus.FAILED.value
    job.completed_at = datetime.utcnow()
    job.error_message = req.error_message
    db.commit()
    return {"status": "ok"}


def _job_to_response(job: ByocTrainingJob) -> JobStatusResponse:
    return JobStatusResponse(
        id=job.id,
        status=job.status,
        instance_type=job.instance_type,
        instance_id=job.instance_id,
        region=job.region,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        metrics=job.metrics,
        error_message=job.error_message,
    )
```

- [ ] **Step 3: Register in server.py**

Add `from routes.byoc_training import router as byoc_training_router` and `app.include_router(byoc_training_router)`.

- [ ] **Step 4: Commit**

```bash
git add platform/routes/byoc_training.py platform/schemas/byoc_schemas.py platform/server.py
git commit -m "feat: add BYOC training launch, status, stop, and callback routes"
```

---

## Chunk 4: Deployment Infrastructure

### Task 10: Production Docker + Compose

**Files:**
- Modify: `Dockerfile` (slim runtime stage)
- Create: `docker-compose.prod.yml`
- Create: `.env.example`

- [ ] **Step 1: Slim down Dockerfile runtime stage**

In `Dockerfile`, replace the runtime stage (Stage 4) `apt-get install` line to remove `build-essential g++ make libomp-dev`. Replace with:
```dockerfile
RUN apt-get update && apt-get install -y \
    libgomp1 \
    nginx \
    supervisor \
    sqlite3 \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*
```

- [ ] **Step 2: Create docker-compose.prod.yml**

```yaml
# Production deployment with PostgreSQL
services:
  web:
    build: .
    ports:
      - "80:5173"
    env_file: .env
    depends_on:
      - db
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s

  db:
    image: postgres:16-alpine
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: whitematter
      POSTGRES_USER: whitematter
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    restart: unless-stopped

volumes:
  pgdata:
```

- [ ] **Step 3: Create .env.example**

```
# Database
POSTGRES_PASSWORD=changeme
DATABASE_URL=postgresql://whitematter:changeme@db:5432/whitematter

# Auth
JWT_SECRET=generate-a-random-secret-min-32-chars

# BYOC
CREDENTIAL_ENCRYPTION_KEY=generate-a-fernet-key
PLATFORM_URL=http://your-domain-or-ip

# Optional
ANTHROPIC_API_KEY=sk-ant-...
WORKERS=4
```

- [ ] **Step 4: Commit**

```bash
git add Dockerfile docker-compose.prod.yml .env.example
git commit -m "feat: add production Docker compose with PostgreSQL, slim runtime"
```

---

### Task 11: AWS Deploy Script

**Files:**
- Create: `deploy/aws-setup.sh`

- [ ] **Step 1: Create deploy script**

Create `deploy/aws-setup.sh`:
```bash
#!/bin/bash
set -euo pipefail

echo "=== Whitematter Platform Setup ==="
echo "This script sets up whitematter on a fresh Ubuntu 22.04 EC2 instance."
echo ""

# Install Docker
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "Docker installed. You may need to log out and back in for group changes."
fi

# Install Docker Compose plugin
if ! docker compose version &> /dev/null; then
    echo "Installing Docker Compose plugin..."
    sudo apt-get update && sudo apt-get install -y docker-compose-plugin
fi

# Clone or update repo
if [ ! -d "/opt/whitematter" ]; then
    echo "Cloning whitematter..."
    sudo git clone https://github.com/hwang2409/whitematter.git /opt/whitematter
    sudo chown -R $USER:$USER /opt/whitematter
else
    echo "Updating whitematter..."
    cd /opt/whitematter && git pull
fi

cd /opt/whitematter

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env

    # Generate secrets
    JWT_SECRET=$(openssl rand -base64 32)
    POSTGRES_PASSWORD=$(openssl rand -base64 16)
    FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || openssl rand -base64 32)

    sed -i "s|generate-a-random-secret-min-32-chars|$JWT_SECRET|" .env
    sed -i "s|changeme|$POSTGRES_PASSWORD|g" .env
    sed -i "s|generate-a-fernet-key|$FERNET_KEY|" .env

    # Get public IP
    PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "localhost")
    sed -i "s|http://your-domain-or-ip|http://$PUBLIC_IP|" .env

    echo "Generated .env with random secrets."
    echo "Edit /opt/whitematter/.env to add your ANTHROPIC_API_KEY if needed."
fi

# Build and start
echo "Building and starting whitematter..."
docker compose -f docker-compose.prod.yml up -d --build

echo ""
echo "=== Setup Complete ==="
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "localhost")
echo "Platform: http://$PUBLIC_IP"
echo "API:      http://$PUBLIC_IP:8080/health"
echo ""
echo "Logs:     docker compose -f docker-compose.prod.yml logs -f"
echo "Stop:     docker compose -f docker-compose.prod.yml down"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x deploy/aws-setup.sh`

- [ ] **Step 3: Commit**

```bash
git add deploy/
git commit -m "feat: add AWS EC2 deployment bootstrap script"
```

---

## Chunk 5: Frontend Auth + Dashboard

### Task 12: Frontend Auth Service + Pages

**Files:**
- Create: `frontend/src/services/auth.ts`
- Create: `frontend/src/pages/LoginPage.tsx`
- Create: `frontend/src/pages/RegisterPage.tsx`
- Create: `frontend/src/context/AuthContext.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Create auth API service**

Create `frontend/src/services/auth.ts`:
```typescript
const API_BASE = "";

interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

interface UserResponse {
  id: string;
  email: string;
  oauth_provider: string | null;
  created_at: string;
}

export async function register(email: string, password: string): Promise<TokenResponse> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Registration failed");
  }
  return res.json();
}

export async function login(email: string, password: string): Promise<TokenResponse> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Login failed");
  }
  return res.json();
}

export async function getMe(token: string): Promise<UserResponse> {
  const res = await fetch(`${API_BASE}/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error("Not authenticated");
  return res.json();
}

export function getStoredToken(): string | null {
  return localStorage.getItem("access_token");
}

export function storeTokens(tokens: TokenResponse) {
  localStorage.setItem("access_token", tokens.access_token);
  localStorage.setItem("refresh_token", tokens.refresh_token);
}

export function clearTokens() {
  localStorage.removeItem("access_token");
  localStorage.removeItem("refresh_token");
}
```

- [ ] **Step 2: Create AuthContext**

Create `frontend/src/context/AuthContext.tsx`:
```tsx
import { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { getMe, getStoredToken, storeTokens, clearTokens } from "../services/auth";

interface User {
  id: string;
  email: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  loading: boolean;
  loginWithTokens: (tokens: { access_token: string; refresh_token: string; token_type: string }) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(getStoredToken());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      getMe(token)
        .then(setUser)
        .catch(() => { clearTokens(); setToken(null); setUser(null); })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, [token]);

  const loginWithTokens = (tokens: { access_token: string; refresh_token: string; token_type: string }) => {
    storeTokens(tokens);
    setToken(tokens.access_token);
  };

  const logout = () => {
    clearTokens();
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, loginWithTokens, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be inside AuthProvider");
  return ctx;
}
```

- [ ] **Step 3: Create LoginPage and RegisterPage**

Create `frontend/src/pages/LoginPage.tsx` and `frontend/src/pages/RegisterPage.tsx` — standard form components that call the auth service and use `useAuth().loginWithTokens()` on success.

- [ ] **Step 4: Update App.tsx to wrap with AuthProvider and show login if not authenticated**

Modify `frontend/src/App.tsx`:
- Wrap the app content with `<AuthProvider>`
- If `!user && !loading`, show LoginPage
- If `user`, show existing app content with a logout button

- [ ] **Step 5: Commit**

```bash
git add frontend/src/services/auth.ts frontend/src/context/AuthContext.tsx frontend/src/pages/
git commit -m "feat: add frontend auth (login, register, context)"
```

---

### Task 13: Frontend AWS Setup + S3 Manager Pages

**Files:**
- Create: `frontend/src/services/aws.ts`
- Create: `frontend/src/pages/AWSSetupPage.tsx`
- Create: `frontend/src/pages/S3ManagerPage.tsx`
- Create: `frontend/src/pages/DashboardPage.tsx`

- [ ] **Step 1: Create AWS API service**

Create `frontend/src/services/aws.ts` with functions for:
- `storeCredentials(token, data)` — POST /credentials/aws
- `getCredentials(token)` — GET /credentials/aws
- `updateCredentials(token, data)` — PUT /credentials/aws
- `deleteCredentials(token)` — DELETE /credentials/aws
- `listBuckets(token)` — GET /storage/buckets
- `createBucket(token, name)` — POST /storage/buckets
- `listObjects(token, bucket, prefix)` — GET /storage/buckets/{bucket}/objects
- `uploadFile(token, bucket, key, file)` — POST /storage/upload (multipart FormData)
- `deleteObject(token, bucket, key)` — DELETE /storage/objects
- `getPresignedUrl(token, bucket, key)` — POST /storage/presigned-url

All functions include `Authorization: Bearer ${token}` header.

- [ ] **Step 2: Create AWSSetupPage**

Create `frontend/src/pages/AWSSetupPage.tsx`:
- Form with access key, secret key, region dropdown, instance type dropdown
- Displays the required IAM policy JSON for user to copy
- Shows current credential status
- Save / Update / Delete buttons

- [ ] **Step 3: Create S3ManagerPage**

Create `frontend/src/pages/S3ManagerPage.tsx`:
- Bucket list sidebar
- Object browser with folder navigation (prefix-based)
- Drag-and-drop file upload
- Delete button per object
- Download via presigned URL
- Create bucket button

- [ ] **Step 4: Create DashboardPage**

Create `frontend/src/pages/DashboardPage.tsx`:
- Lists user's BYOC training jobs with status badges
- Shows model architectures
- Links to training monitor, S3 manager, AWS setup

- [ ] **Step 5: Add navigation between pages in App.tsx**

Update App.tsx to add tab/nav for: Dashboard, Data (S3), Train, Models, Settings (AWS Setup).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/services/aws.ts frontend/src/pages/
git commit -m "feat: add AWS setup, S3 manager, and dashboard pages"
```

---

## Chunk 6: PyPI Library Packaging

### Task 14: Clean Up setup.py + CI Wheel Build

**Files:**
- Modify: `platform/setup.py`
- Create: `.github/workflows/publish.yml`
- Create: `pyproject.toml`

- [ ] **Step 1: Update setup.py for PyPI**

Update `platform/setup.py`:
- Change `name` to `"whitematter"`
- Update `version` to `"0.1.0"`
- Change `description` to `"Lightweight neural network framework with GPU support"`
- Add `long_description` from README
- Add `author`, `url`, `classifiers`, `python_requires`

- [ ] **Step 2: Create pyproject.toml for cibuildwheel**

Create `pyproject.toml` at project root:
```toml
[build-system]
requires = ["setuptools>=68.0", "pybind11>=2.11.0"]
build-backend = "setuptools.build_meta"

[tool.cibuildwheel]
build = "cp311-* cp312-*"
skip = "*-win* *-musllinux*"
test-command = "python -c \"import whitematter; print('OK')\""

[tool.cibuildwheel.linux]
before-build = "yum install -y libomp-devel || apt-get install -y libomp-dev"

[tool.cibuildwheel.macos]
before-build = "brew install libomp"
```

- [ ] **Step 3: Create GitHub Actions workflow for publishing**

Create `.github/workflows/publish.yml`:
```yaml
name: Build and Publish Wheels
on:
  push:
    tags: ["v*"]
  workflow_dispatch:

jobs:
  build-wheels:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, macos-13]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install cibuildwheel
        run: pip install cibuildwheel
      - name: Build wheels
        run: cibuildwheel --output-dir dist
        working-directory: platform
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: platform/dist/*.whl

  publish:
    needs: build-wheels
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
      - uses: actions/download-artifact@v4
        with:
          path: dist
          merge-multiple: true
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

- [ ] **Step 4: Commit**

```bash
git add platform/setup.py pyproject.toml .github/
git commit -m "feat: add PyPI packaging with cibuildwheel CI"
```

---

## Summary

| Chunk | Tasks | What it delivers |
|-------|-------|-----------------|
| 1: Database & Auth | Tasks 1-4 | PostgreSQL, Alembic, User model, JWT auth routes |
| 2: AWS & S3 | Tasks 5-7 | Credential encryption, CRUD routes, S3 proxy |
| 3: BYOC Engine | Tasks 8-9 | EC2 provisioner, user-data scripts, callback routes |
| 4: Deployment | Tasks 10-11 | Slim Dockerfile, prod compose, deploy script |
| 5: Frontend | Tasks 12-13 | Auth pages, dashboard, AWS setup, S3 manager |
| 6: PyPI | Task 14 | setup.py cleanup, cibuildwheel, GitHub Actions |
