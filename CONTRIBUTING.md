# Contributing to Whitematter

Thank you for your interest in contributing. This document explains how to set up the development environment, run tests, and submit changes.

## Table of contents

- [Getting started for contributors](#getting-started-for-contributors)
- [Development setup](#development-setup)
- [Running tests](#running-tests)
- [Pull request process](#pull-request-process)
- [PR conventions](#pr-conventions)
- [Issue tracking](#issue-tracking)
- [API reference](#api-reference)

## Getting started for contributors

### Option A: Docker (fastest)

```bash
git clone https://github.com/hwang2409/whitematter.git
cd whitematter
docker compose up --build
```

The platform is now running at **http://localhost:5173** (frontend) and **http://localhost:8080** (API).

### Option B: Local dev

```bash
git clone https://github.com/hwang2409/whitematter.git
cd whitematter

# Backend
cd platform
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python server.py &
cd ..

# Frontend
cd frontend
npm install
npm run dev
```

### Option C: GitHub Codespaces

Click **Code > Codespaces > New codespace** on the repo page. The `.devcontainer/devcontainer.json` pre-installs all dependencies. Then run `make dev`.

## Development setup

### Prerequisites

- **C++**: C++17 compiler (g++, clang++)
- **Python**: 3.8+ (for bindings and platform)
- **Node.js**: For the frontend (LTS recommended)
- **macOS (optional)**: `brew install libomp` for OpenMP
- **CUDA (optional)**: For GPU builds on Linux; set `CUDA_PATH` if needed

### 1. Clone the repository

```bash
git clone https://github.com/hwang2409/whitematter.git
cd whitematter
```

### 2. Build the C++ core and run C++ tests

From the repository root:

```bash
# Build all targets (examples + test binary)
make

# Optional: GPU backends
# macOS Metal:   make METAL=1
# Linux CUDA:    make CUDA=1
```

### 3. Python (bindings / dev install)

From the repository root:

```bash
# Create and use a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package in editable mode
pip install -e .
```

To verify: `python -c "import whitematter as wm; print('OK')"`

### 4. Platform (FastAPI server + Python services)

The platform lives in `platform/` and has its own dependencies:

```bash
cd platform
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# Or: pip install -r requirements.txt  (if present)
cd ..
```

Start the backend (from repo root):

```bash
cd platform && python server.py
```

### 5. Frontend (React / Next.js)

```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173. Use the backend from step 4 for full platform functionality.

### Quick reference: local platform stack

| Component   | Command / location        |
|------------|----------------------------|
| C++ build  | `make` (repo root)         |
| C++ tests  | `make test` (repo root)    |
| Backend    | `cd platform && python server.py` |
| Frontend   | `cd frontend && npm run dev` |
| Python pkg | `pip install -e .` (repo root) |

## Running tests

### C++ unit tests (repo root)

```bash
make test              # Run all C++ tests
make test-tensor       # Tensor operations
make test-autograd     # Autograd
make test-layers       # Layers
make test-loss         # Loss functions
make test-optimizer    # Optimizers
```

### Platform Python tests

From the repository root:

```bash
cd platform
# With venv activated:
pytest
# Or: pytest -v
# Or: pytest path/to/test_file.py
```

### Frontend lint

```bash
cd frontend
npm run lint
```

Before opening a PR, run the C++ tests (`make test`) and, if you changed platform or frontend code, the corresponding tests/lint above.

## Pull request process

1. **Open an issue first** (or pick an existing one) so we can align on scope. Use the [issue templates](.github/ISSUE_TEMPLATE/) when possible (bug report, feature request).

2. **Fork and branch**  
   Create a branch from `main` (e.g. `fix/issue-123` or `feat/my-feature`).

3. **Implement and test**  
   - Keep changes focused.  
   - Add or update tests as needed.  
   - Run `make test` for C++ changes; run platform/frontend tests or lint if you touched those areas.

4. **Commit**  
   Use clear commit messages. Reference the issue number where relevant (e.g. `Fix loss scale in AMP (fixes #42)`).

5. **Push and open a PR**  
   - Target the `main` branch.  
   - Fill in the [pull request template](.github/PULL_REQUEST_TEMPLATE.md): describe what changed, how to test, and link any related issues.

6. **Review**  
   Address review feedback. Maintainers may request additional tests or small edits.

7. **Merge**  
   A maintainer will merge once the PR is approved and CI (if configured) passes.

We do not require a formal CLA for small fixes; by submitting a PR you agree that your contributions may be used under the project’s license (MIT).

## PR conventions

- **Branch naming**: `feat/description`, `fix/description`, or `docs/description`
- **Commit messages**: Use imperative mood (e.g. "Add LSTM layer" not "Added LSTM layer"). Reference issues with `fixes #N` or `closes #N`.
- **One concern per PR**: Keep PRs focused. A bug fix and a new feature should be separate PRs.
- **Tests required**: All C++ changes must pass `make test`. Platform changes need passing `pytest`. Frontend changes must pass `npm run lint`.
- **Changelog**: For user-facing changes, add an entry to `CHANGELOG.md` under the `[Unreleased]` section.

## API reference

The FastAPI backend auto-generates interactive API documentation:

- **Swagger UI**: [http://localhost:8080/docs](http://localhost:8080/docs) (when running locally)
- **ReDoc**: [http://localhost:8080/redoc](http://localhost:8080/redoc)

These are the authoritative references for all REST endpoints, request/response schemas, and authentication requirements.

## Issue tracking

We use **GitHub Issues** for bugs, features, and documentation. For structured reports:

- **Bug**: use [Bug report](.github/ISSUE_TEMPLATE/bug_report.md).
- **New feature or idea**: use [Feature request](.github/ISSUE_TEMPLATE/feature_request.md).

Include enough context (version, OS, steps to reproduce for bugs; use case and constraints for features) so maintainers can triage and act on issues quickly.

---

Thank you for contributing to Whitematter.
