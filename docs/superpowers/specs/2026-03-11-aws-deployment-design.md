# Whitematter AWS Deployment + Library Design

## Overview

Deploy whitematter as a **public platform** and **Python library** with BYOC (Bring Your Own Cloud) training.

- **Platform website**: Model design UI, auth, training orchestration, monitoring — runs on cheap CPU EC2
- **BYOC engine**: Provisions GPU instances in the user's AWS account for training
- **Python library**: `pip install whitematter` for standalone local training

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Platform (CPU EC2 — t3.medium, ~$30/mo)         │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ React UI │  │ FastAPI   │  │ PostgreSQL   │  │
│  │ (nginx)  │──│ API       │──│ (Docker or   │  │
│  └──────────┘  └───────────┘  │  RDS later)  │  │
│                     │         └──────────────┘  │
│              ┌──────┴────────┐                   │
│              │  BYOC Engine  │                   │
│              │  (boto3)      │                   │
│              └──────┬────────┘                   │
└─────────────────────┼────────────────────────────┘
                      │ provisions via user's AWS keys
          ┌───────────▼────────────┐
          │  User's AWS Account    │
          │  ┌──────────────────┐  │
          │  │ GPU EC2 (g4dn)  │  │
          │  │ pip install      │  │
          │  │ whitematter      │  │
          │  │ → train → exit   │  │
          │  └──────────────────┘  │
          └────────────────────────┘

  +  pip install whitematter (standalone, local use)
```

## Components

### 1. Authentication

- Email/password signup + login
- OAuth: Google and GitHub (with state parameter + PKCE for CSRF protection)
- JWT access tokens + refresh tokens
- Password hashing with bcrypt
- FastAPI dependency for protected routes
- Rate limiting on auth endpoints

### 2. Platform API (extend existing FastAPI)

Existing routes stay. New routes added:

**Auth:**
- `POST /auth/register` — email/password signup
- `POST /auth/login` — email/password login
- `GET /auth/oauth/{provider}` — OAuth redirect (with state param)
- `POST /auth/oauth/{provider}/callback` — OAuth callback
- `POST /auth/refresh` — refresh JWT

**AWS credentials:**
- `POST /credentials/aws` — store encrypted AWS keys
- `GET /credentials/aws` — check if keys exist (never returns actual keys)
- `PUT /credentials/aws` — update/rotate keys
- `DELETE /credentials/aws` — remove stored keys

**S3 data management (uses user's AWS credentials):**
- `GET /storage/buckets` — list user's S3 buckets
- `POST /storage/buckets` — create a new S3 bucket
- `GET /storage/buckets/{bucket}/objects?prefix=` — list objects in bucket
- `POST /storage/upload` — upload file to user's S3 (multipart, streams through platform)
- `DELETE /storage/objects` — delete object from user's S3
- `POST /storage/presigned-url` — generate presigned download URL (browser downloads directly from S3)

**BYOC training:**
- `POST /training/launch` — launch BYOC training job
- `GET /training/{job_id}/status` — poll job status
- `GET /training/{job_id}/logs` — stream training logs
- `POST /training/{job_id}/stop` — terminate instance + job

**Callback routes (called by GPU instances, authenticated with one-time job token):**
- `POST /training/{job_id}/heartbeat` — instance health check
- `POST /training/{job_id}/log` — append log lines
- `POST /training/{job_id}/metrics` — report epoch metrics (loss, accuracy)
- `POST /training/{job_id}/complete` — signal completion with final results
- `POST /training/{job_id}/fail` — signal failure with error details

### 3. BYOC Engine (new module: `platform/byoc/`)

Responsible for managing GPU instances in user AWS accounts.

**Credential storage (MVP — access keys):**
- AWS access key + secret key per user
- Encrypted at rest with Fernet (symmetric encryption)
- Encryption key from environment variable, never hardcoded
- Keys never logged or returned via API
- **Required IAM policy**: Users must scope their keys to a minimal policy:
  - `ec2:RunInstances`, `ec2:TerminateInstances`, `ec2:DescribeInstances`
  - `ec2:CreateSecurityGroup`, `ec2:AuthorizeSecurityGroupIngress/Egress`
  - `ec2:CreateTags`, `ec2:DescribeTags`
  - `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`, `s3:ListBucket`, `s3:CreateBucket` (for managing training data in user's S3)
  - Platform UI displays this policy for users to copy during setup
- **Migration path**: IAM cross-account roles with external IDs (standard pattern used by Terraform Cloud, Datadog). User creates a role in their account with trust policy pointing to platform's AWS account, platform assumes via STS. No long-lived keys stored.

**Instance provisioning (boto3):**
- Default instance type: `g4dn.xlarge` (1x T4, 4 vCPU, 16GB RAM, ~$0.53/hr on-demand)
- User-configurable instance type
- Uses AWS Deep Learning AMI (NVIDIA drivers pre-installed)
- Security group: outbound HTTPS only (to platform API + PyPI)
- Instance tagged with job ID for tracking

**Job token authentication:**
- When launching a job, platform generates a cryptographically random one-time job token
- Token is passed to the EC2 instance via user-data (encrypted)
- All callback endpoints require this token as Bearer auth
- Token expires when job completes, fails, or times out

**EC2 user-data script:**
1. Install pinned whitematter version from PyPI (`pip install whitematter==X.Y.Z`)
2. Verify package checksum
3. Authenticate with platform API using job token
4. Pull training config from platform API
5. Pull dataset from user's S3 URI (specified in training config)
6. Run training, streaming logs/metrics back via callback endpoints
7. POST final results to platform API
8. Self-terminate instance

**Job lifecycle:**
- Launch → Running → Completed / Failed / Stopped
- Heartbeat every 60s from GPU instance; platform marks as failed if missed for 5 minutes
- Auto-terminate on completion, failure, or timeout
- Timeout configurable (default: 6 hours)

**Training data path:**
- Users upload datasets to their own S3 via the platform's S3 management UI
- Training config references an S3 URI (e.g., `s3://my-bucket/mnist/`)
- The GPU instance pulls data directly from the user's S3 using the same AWS credentials
- For built-in datasets (MNIST, CIFAR-10), whitematter downloads them automatically

### 4. Database (PostgreSQL)

Switch from SQLite to PostgreSQL for concurrent multi-user access.

**New tables** (added alongside existing tables):

```
users
  - id (UUID, PK)
  - email (unique)
  - password_hash (nullable — OAuth users don't have one)
  - oauth_provider (nullable)
  - oauth_id (nullable)
  - created_at

aws_credentials
  - id (UUID, PK)
  - user_id (FK → users, unique)
  - encrypted_access_key (bytes)
  - encrypted_secret_key (bytes)
  - default_region (string, default: us-east-1)
  - default_instance_type (string, default: g4dn.xlarge)
  - created_at, updated_at

byoc_training_jobs
  - id (UUID, PK)
  - user_id (FK → users)
  - model_config (JSON — architecture definition)
  - dataset_config (JSON — dataset S3 URI + preprocessing)
  - instance_type (string)
  - instance_id (string, nullable — set after launch)
  - region (string)
  - status (enum: pending, launching, running, completed, failed, stopped)
  - job_token_hash (string — hashed one-time token for callback auth)
  - started_at, completed_at
  - metrics (JSON — loss curves, accuracy, etc.)
  - error_message (text, nullable)

model_architectures
  - id (UUID, PK)
  - user_id (FK → users)
  - name (string)
  - config (JSON)
  - created_at, updated_at
```

**Existing tables preserved as-is:**
- `datasets` — dataset metadata (local platform features)
- `models` — trained model metadata
- `training_jobs` — local training job tracking (existing worker system)
- `training_history` — per-epoch metrics
- `blob_metadata` — blob storage

The `byoc_training_jobs` table is separate from the existing `training_jobs` table. Local training (via the existing worker system) and BYOC training (via user's AWS) are independent flows that share the same UI but use different backend tables and orchestration.

**Migration**: Alembic for schema migrations. Auto-generate initial migration from existing models, then add new tables. Migration files live in `platform/migrations/`.

### 5. Python Library (`pip install whitematter`)

Pybind11 wrapper around the C++ core. Existing `bindings/whitematter_py.cpp` + `platform/setup.py` already handle this.

**What's needed:**
- Clean up `setup.py` for PyPI publishing (name, version, description, classifiers)
- Build manylinux + macOS wheels via GitHub Actions (cibuildwheel)
- Platforms: Linux (x86_64, aarch64) + macOS (x86_64, arm64)
- Windows: not supported initially (document this)
- CUDA-optional: base package is CPU-only, `whitematter[cuda]` extra for CUDA
- Publish to PyPI
- Verify existing pybind11 bindings expose the full training API (Sequential, Linear, Adam, loss functions, backward, etc.) — extend if needed

**Library usage (standalone):**
```python
import whitematter as wm

model = wm.Sequential([
    wm.Linear(784, 128),
    wm.ReLU(),
    wm.Linear(128, 10)
])

optimizer = wm.Adam(model.parameters(), lr=0.001)
loss_fn = wm.CrossEntropyLoss()

for batch in dataloader:
    pred = model.forward(batch.x)
    loss = loss_fn(pred, batch.y)
    loss.backward()
    optimizer.step()
```

### 6. Frontend (extend existing React app)

New pages/components:

- **Auth pages**: Login, Register, OAuth buttons (Google, GitHub)
- **Dashboard**: User's training jobs, model architectures
- **AWS Setup**: Form to input access key + secret key, region picker, instance type selector, displays required IAM policy for user to copy
- **S3 Data Manager**: Browse buckets, create buckets, upload/download/delete files, drag-and-drop upload — all operating on user's own S3 via their stored AWS credentials
- **Training Launch**: Select model + pick dataset from S3 browser → configure → launch
- **Training Monitor**: Real-time status, log viewer, metrics charts (recharts already installed)

**CORS**: Lock down `allow_origins` from `["*"]` to the platform's actual domain.

### 7. Deployment Infrastructure

**Platform Docker (no GPU):**
- `Dockerfile` — slim, CPU-only build (no CUDA, no Metal)
- Runtime stage: only `libgomp1`, `nginx`, `supervisor`, `sqlite3`, `curl`, `libpq5` (PostgreSQL client)
- No `build-essential`, `g++`, or `make` in runtime stage
- Add PostgreSQL client library (`libpq-dev` in build stage, `libpq5` in runtime)
- Supervisor manages: nginx, FastAPI (uvicorn with 4 workers), background job poller

**HTTPS requirement**: HTTPS (Let's Encrypt + certbot) must be configured before enabling BYOC. The BYOC flow handles real AWS credentials and job tokens — these must not travel over plain HTTP. The platform can run on HTTP for development, but BYOC endpoints must reject non-HTTPS requests in production.

**docker-compose.prod.yml** (replaces existing docker-compose.yml for production):
```yaml
services:
  web:
    build: .
    ports:
      - "80:5173"
      - "443:443"    # When HTTPS is enabled
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

The existing `docker-compose.yml` (SQLite-based, single service) is kept for local development. `docker-compose.prod.yml` is for production deployment.

**`.env.example`:**
```
# Database
POSTGRES_PASSWORD=changeme
DATABASE_URL=postgresql://whitematter:changeme@db:5432/whitematter

# Auth
JWT_SECRET=generate-a-random-secret-min-32-chars

# BYOC
CREDENTIAL_ENCRYPTION_KEY=generate-a-fernet-key

# Optional
ANTHROPIC_API_KEY=sk-ant-...
WORKERS=4
```

**`deploy/aws-setup.sh`:**
- Bootstrap script for a fresh t3.medium EC2 instance (Ubuntu 22.04 AMI)
- Installs Docker + Docker Compose
- Clones repo or pulls image from ECR
- Copies `.env` from user input
- Starts platform with `docker-compose -f docker-compose.prod.yml up -d`
- Configures firewall (ports 80, 443 open)
- Prints access URL

## What the Platform Does NOT Do

- Store model weights or datasets on platform servers (all data lives in user's own S3)
- Run GPU compute (all GPU runs in user accounts)
- Manage billing or payments (users pay their own AWS bill)
- Host any file storage — platform proxies S3 operations using user's credentials

## Security Considerations

- AWS credentials encrypted at rest (Fernet), scoped to minimal IAM policy
- Encryption key from env var, never in code or DB
- JWT tokens with expiration + refresh flow
- HTTPS required before BYOC goes live (Let's Encrypt + nginx)
- BYOC instances use minimal security groups (outbound HTTPS only)
- One-time job tokens for GPU instance callback authentication
- Rate limiting on auth endpoints
- Input validation on all API endpoints
- CORS locked to platform domain (no wildcard)
- OAuth flows use state parameter + PKCE

## Migration Path

1. **SQLite → PostgreSQL**: Alembic migrations, update SQLAlchemy connection string
2. **HTTP → HTTPS**: Let's Encrypt + certbot (required before BYOC launch)
3. **Single EC2 → RDS**: Move PostgreSQL to RDS for managed backups
4. **Access keys → IAM roles**: Cross-account roles with external IDs as primary BYOC auth
5. **Spot instances**: Add spot instance support for BYOC (g4dn.xlarge ~$0.16/hr spot vs $0.53/hr on-demand) with interruption handling
