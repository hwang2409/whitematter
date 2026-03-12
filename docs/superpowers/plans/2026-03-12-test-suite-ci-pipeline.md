# Test Suite & CI Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build comprehensive test suite and CI pipeline for production readiness across frontend and backend.

**Architecture:** Frontend uses Vitest + testing-library for unit tests, Playwright for E2E. Backend uses pytest with FastAPI TestClient. CI uses GitHub Actions with security scanning (pip-audit, npm audit, bandit), type checking (mypy, tsc), caching, and Docker-based staging deploy.

**Tech Stack:** Vitest, @testing-library/react, Playwright, pytest, pytest-asyncio, httpx, bandit, pip-audit, mypy, GitHub Actions, Docker Compose

---

## Chunk 1: Frontend Testing Infrastructure

### Task 1: Vitest + Testing Library Setup
**Files:**
- Create: `frontend/vitest.config.ts`
- Create: `frontend/src/__tests__/setup.ts`
- Modify: `frontend/package.json` (add test scripts)

- [x] Install vitest, @vitejs/plugin-react, jsdom, @testing-library/react, @testing-library/jest-dom, @testing-library/user-event
- [x] Create vitest.config.ts with jsdom environment, React plugin, path alias
- [x] Create setup.ts with jest-dom matchers, localStorage mock, window.location mock
- [x] Add scripts: test, test:watch, typecheck

### Task 2: Frontend Unit Tests
**Files:**
- Create: `frontend/src/__tests__/safeJson.test.ts`
- Create: `frontend/src/__tests__/auth-service.test.ts`
- Create: `frontend/src/__tests__/api-utils.test.ts`
- Create: `frontend/src/__tests__/AuthContext.test.tsx`
- Create: `frontend/src/__tests__/DashboardPage.test.tsx`

- [ ] Write safeJson tests (valid JSON, HTML response, empty, invalid)
- [ ] Write auth service tests (register, login, getMe, localStorage functions)
- [ ] Write API utility tests (ApiError, getErrorMessage, isRetryableError, timeout, API calls)
- [ ] Write AuthContext tests (provider rendering, useAuth hook, login/logout, token refresh)
- [ ] Write DashboardPage smoke test

### Task 3: Playwright E2E Setup
**Files:**
- Create: `frontend/playwright.config.ts`
- Create: `frontend/e2e/helpers.ts`
- Create: `frontend/e2e/auth.spec.ts`
- Create: `frontend/e2e/training-flow.spec.ts`

- [ ] Create Playwright config with chromium project, webServer for frontend + backend
- [ ] Create auth helpers (loginUser, registerUser)
- [ ] Write auth flow E2E (register → login → dashboard)
- [ ] Write training flow smoke E2E (navigate all pages after login)

---

## Chunk 2: Backend Test Coverage

### Task 4: Backend Test Infrastructure
**Files:**
- Modify: `platform/tests/conftest.py` (add TestClient, auth fixtures)

- [ ] Add FastAPI TestClient fixture
- [ ] Add auth_token fixture for authenticated requests
- [ ] Add mock_db fixture for database isolation

### Task 5: Route Tests
**Files:**
- Create: `platform/tests/test_routes_health.py`
- Create: `platform/tests/test_routes_auth.py`
- Create: `platform/tests/test_routes_training.py`
- Create: `platform/tests/test_routes_datasets.py`
- Create: `platform/tests/test_routes_predict.py`
- Create: `platform/tests/test_routes_design.py`
- Create: `platform/tests/test_routes_deploy.py`
- Create: `platform/tests/test_routes_credentials.py`

- [ ] Health route test (GET /health → 200)
- [ ] Auth route tests (register, login, me, refresh — success + error)
- [ ] Training route tests (start, status, cancel, custom — mock subprocess)
- [ ] Dataset route tests (upload, list, get, delete, import)
- [ ] Predict route tests (predict, custom predict, generate, info)
- [ ] Design route tests (suggest, validate, refine, preview-code, help)
- [ ] Deploy route tests (start, list, get, terminate)
- [ ] Credential route tests (save, update, delete)

### Task 6: Service & Integration Tests
**Files:**
- Create: `platform/tests/test_llm_service.py`
- Create: `platform/tests/test_codegen.py`
- Create: `platform/tests/test_ws_training.py`

- [ ] LLM service tests (mock/fallback, suggest, refine, help)
- [ ] Codegen tests (image arch, text arch, unknown layers, missing params)
- [ ] WebSocket training tests (connect, receive updates, disconnect)

---

## Chunk 3: CI Pipeline

### Task 7: Security Scanning
**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] Add pip-audit step to platform job
- [ ] Add bandit static analysis step
- [ ] Add npm audit step to frontend job

### Task 8: Caching
- [ ] Add actions/cache for ~/.cache/pip (keyed on requirements.txt)
- [ ] Verify npm cache via setup-node cache parameter

### Task 9: Type Checking
- [ ] Add mypy step for backend (routes/, services/)
- [ ] Add tsc --noEmit step for frontend

### Task 10: Staging Deploy
**Files:**
- Create: `docker-compose.staging.yml`
- Modify: `.github/workflows/ci.yml` (add staging-deploy job)

- [ ] Create staging docker-compose with PostgreSQL
- [ ] Add staging-deploy job (build + push to GHCR on merge to main)
