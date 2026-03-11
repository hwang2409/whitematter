# dongha.md — What I Added to This Project

This file records plans I implemented in the whitematter project. Add new entries when you implement a plan.

---

## 1. Zero-Copy Views for Reshape / Squeeze / Unsqueeze

**Summary:** Tensor storage was switched from raw pool pointers to `std::shared_ptr<float>` with a pool deleter. Reshape, squeeze, and unsqueeze were changed to O(1) views that share the data buffer instead of copying.

**From this chat:** Implementation followed the plan in order: (1) MemoryPool `acquire_shared`, (2) Tensor switch to shared_ptr storage, (3) view constructor and reshape as view, (4) squeeze and unsqueeze as views, (5) full test suite. One fix during implementation: in the from-vector constructor the parameter `data` shadowed the member `data()`; `std::memcpy` was updated to use `data_storage_.get()` instead of `data()`. `operator[]` and `at()` now use `data()`. Full test run: 184 tests passed; an earlier run had a transient bus error that did not reproduce.

### Changes

- **Memory pool (`core/memory_pool.h`, `core/memory_pool.cpp`)**
  - Added `#include <memory>` in the header.
  - Added `std::shared_ptr<float> acquire_shared(size_t n)` that returns a shared_ptr with a custom deleter calling `release(ptr, n)` so memory is returned to the pool when the last owner is destroyed. Deleter captures `n` for the correct size-class bucket.
  - Kept `acquire` and `release` for compatibility.

- **Tensor storage (`core/tensor.h`, `core/tensor.cpp`)**
  - Replaced `float* data_ptr_` / `float* grad_ptr_` with `std::shared_ptr<float> data_storage_` / `grad_storage_` (kept `data_size_` / `grad_size_`).
  - `data()` / `grad()` now return `data_storage_.get()` / `grad_storage_.get()` (or nullptr if empty); `grad_empty()` is `!grad_storage_`.
  - Default constructor: only sets `data_size_` / `grad_size_` to 0.
  - Destructor: no manual `release`; shared_ptr deleters return memory to the pool.
  - Shape-only and from-vector constructors: use `acquire_shared` for data and grad; on grad alloc failure, data shared_ptr goes out of scope and deleter runs.
  - From-vector constructor: `std::memcpy(data_storage_.get(), data.data(), ...)` to avoid parameter name clash with `data()`.
  - `operator[]` and `at()`: use `data()` instead of `data_ptr_`.
  - `backward()` lazy grad: use `acquire_shared(1)` and set `grad_storage_`, `grad_size_ = 1`.

- **View constructor**
  - Added private constructor:  
    `Tensor(std::shared_ptr<float> data_storage, size_t data_size, const std::vector<size_t>& shape, bool requires_grad)`.  
    View shares `data_storage_`; allocates its own `grad_storage_` (via `acquire_shared(data_size_)`) when `requires_grad` so backward can accumulate into the base without double-counting.

- **Reshape / squeeze / unsqueeze as views**
  - **reshape:** Computes `total = product(new_shape)`, asserts `total == size()`. Creates view with `std::make_shared<Tensor>(data_storage_, data_size_, new_shape, track)`. No `std::vector<float>`, no copy. If track and existing grad, copies grad into result. Sets `result->parents = {self_ptr}` and `result->grad_fn` (simd_add for grad accumulation).
  - **squeeze:** Builds `new_shape` (remove size-1 dim(s), or single dim if given). If `new_shape == shape`, returns `shared_from_this()`. Otherwise creates view same way as reshape; sets parents and grad_fn.
  - **unsqueeze:** Builds `new_shape` (insert 1 at dim). Creates view; sets parents and grad_fn when tracking.
  - **flatten:** Unchanged; it calls `reshape`, so it is a view automatically.
  - **permute:** Left as copy-based (no view semantics).

### Verification

- Full test suite: 184 tests passed (tensor 39, autograd 23, layers 74, loss 22, optimizer 26).

---

## 2. Thread-Local Pool Caches

**Summary:** The memory pool now uses a thread-local cache per worker; threads only lock the global mutex when refilling from or draining to the global pool in batches (and when the thread exits), which reduces lock contention under OpenMP.

**From this chat:** Implementation followed the Thread-Local Pool Caches plan: added `ThreadCache` (thread_local) with per–size-class buckets and a per-thread cap, batch refill/drain on `Impl`, routing of `acquire`/`release` through TLS, thread-exit drain in the cache destructor, then full test suite and this dongha update. A friend declaration was added so `memory_pool_detail::ThreadCache` can use the private `Impl` type.

### Changes

- **core/memory_pool.cpp**
  - Constants in anonymous namespace: `kMaxPerClassPerThread` (64), `kRefillBatch` (32).
  - `Impl`: added `acquire_batch(cls, count, out)` (under lock: fill `out` from global bucket or malloc) and `release_batch(from, cls)` (under lock: push all from `from` into global bucket, clear `from`); added `release_one(ptr, cls)` for thread-exit drain.
  - `memory_pool_detail::ThreadCache`: per-thread `unordered_map<size_t, vector<float*>>` buckets and `Impl* global_` (set on first use). `acquire(n, global)`: serve from local bucket if non-empty; else call `global->acquire_batch`, keep rest in local, return one. `release(ptr, original_n, global)`: push to local if under cap; else drain half of that size-class to global via `release_batch`, then push current ptr to local. Destructor: if `global_` non-null, release all cached buffers to global via `release_one`.
  - `thread_local ThreadCache t_thread_cache` in `memory_pool_detail`. `MemoryPool::acquire`/`release` call into `memory_pool_detail::t_thread_cache`; `acquire_shared` unchanged and thus uses the TLS path automatically.

- **core/memory_pool.h**
  - Forward declaration `namespace memory_pool_detail { struct ThreadCache; }` and `friend struct memory_pool_detail::ThreadCache;` so `ThreadCache` can use `MemoryPool::Impl*`.

### Verification

- Full test suite: 184 tests passed (tensor 39, autograd 23, layers 74, loss 22, optimizer 26).

---

## 3. Platform Refactor, Sequential unique_ptr, and Metal GPU Backend

**Summary:** (1) Platform: removed `server_v2.py`, confirmed modular `server.py` with `config.py`, `schemas.py`, `dependencies.py`, and `routes/*`; fixed codegen to skip `"transformer"` layer type in `_generate_layers` so text architecture tests pass. (2) Core: migrated `Sequential` from `std::vector<Module*>` to `std::vector<std::unique_ptr<Module>>` with move-only semantics and updated all call sites (onnx_export, layer.cpp train/eval/summary). (3) Device abstraction and Metal backend: added CPU/Metal device type, Tensor device field and `to(DeviceType)`, optional Metal matmul path when both inputs are on Metal, and `make METAL=1` on macOS.

### Changes

- **Platform**
  - `server_v2.py`: deleted (was 44K incomplete rewrite).
  - `server.py`: already refactored to ~70 lines (FastAPI app, CORS, routers, uvicorn entrypoint).
  - `platform/config.py`, `schemas.py`, `dependencies.py`, `routes/*`: present and used.
  - `platform/codegen/generator.py`: in `_generate_layers`, skip layer type `"transformer"` (composite; text models use TransformerLM template). Fixes `test_text_architecture_generation`.

- **Sequential unique_ptr (`core/layer.h`, `core/layer.cpp`, `core/onnx_export.cpp`)**
  - `layers` changed from `std::vector<Module*>` to `std::vector<std::unique_ptr<Module>>`.
  - Constructor takes `std::initializer_list<Module*>` and wraps each in `unique_ptr`; `add(Module*)` does the same.
  - Destructor defaulted; copy deleted; move defaulted.
  - `forward`, `parameters` unchanged (iteration works via `operator->`). `train`/`eval`/`summary` use `layer.get()` for `dynamic_cast`. `onnx_export.cpp`: iterate `model->layers` with `ptr.get()` for raw pointer.

- **Device and Metal**
  - **core/device.h**: `enum class DeviceType { CPU, METAL }`, `Device::cpu()`, `Device::metal()`, `Device::default_device()`, `Device::is_available()`, `metal_backend_available()`.
  - **core/device.cpp**: Device methods; `is_available()` for METAL calls `metal_backend_available()`.
  - **core/metal/metal_stub.cpp**: `metal_backend_available()` returns `false` (used when METAL=0 or non-Darwin).
  - **core/metal/metal_backend.h**: `MetalBackend::instance()`, `is_available()`, `matmul(A,B,C,M,N,K)`.
  - **core/metal/metal_backend.mm**: Objective-C++ Metal implementation (macOS only, `#ifdef __APPLE__`), `@available` checks, embedded matmul kernel source, copy A/B to GPU, dispatch, copy C back.
  - **core/metal/kernels.metal**: Metal kernels for matmul, elementwise add/mul/sub/div, relu/sigmoid/tanh (used as reference; backend currently embeds matmul source).
  - **Tensor**: `#include "device.h"`, `whitematter::DeviceType device` (default CPU), `TensorPtr to(DeviceType)` (copy + set device). In `matmul`, when both tensors have `device == METAL` and `metal_backend_available()`, dispatch to `MetalBackend::instance().matmul()` (copy-in/copy-out; result stays CPU).
  - **Makefile**: `device.cpp` and either `metal_stub.o` (default) or `metal_backend.o` (when `METAL=1` on Darwin). `METAL=1` adds `-DWHITEMATTER_METAL`, `-framework Metal -framework Foundation`. Default build remains Linux-friendly with no Metal deps.

### Verification

- Platform: `cd platform && python -m pytest tests/ -x` — 79 passed.
- C++: `make clean && make && make test` — 184 passed (default and `make METAL=1` on macOS).

---

## 4. Pull from main and CUDA GPU backend

**Summary:** (1) Merged and rebased on `origin/main`. (2) Added an optional CUDA backend so the project supports both Metal (macOS) and CUDA (cloud/Linux) for GPU compute: `DeviceType::CUDA`, stub/backend build via `make CUDA=1`, cuBLAS matmul and batched matmul, and tensor dispatch for matmul/bmm when both operands are on CUDA.

**From this chat:** Implementation followed the “Pull from main and add CUDA GPU backend” plan: device layer (CUDA type, `cuda()`, `cuda_backend_available()`), Makefile `CUDA=0/1`, `core/cuda/cuda_stub.cpp` and `core/cuda/cuda_backend.cu` (cuBLAS), tensor.cpp branches for CUDA matmul/bmm, README updated. Default build remains CPU-only; `make CUDA=1` requires nvcc and links cudart/cublas for cloud deployment.

### Changes

- **Sync with main**
  - Merged `origin/main` into `dongha`; later rebased and force-pushed. Build and 184 tests passed after merge.

- **Device (`core/device.h`, `core/device.cpp`)**
  - `enum class DeviceType` extended with `CUDA`. Added `Device::cuda()` and `bool cuda_backend_available()` (declared in device.h; implemented in cuda_stub or cuda_backend depending on build). `Device::is_available()` handles `DeviceType::CUDA` via `cuda_backend_available()`.

- **CUDA stub and backend**
  - **core/cuda/cuda_stub.cpp**: When CUDA is not built, defines `cuda_backend_available()` returning `false`. No CUDA headers; used for all non-CUDA builds.
  - **core/cuda/cuda_backend.h**: `CUDABackend::instance()`, `is_available()`, `matmul(A,B,C,M,N,K)`, `bmm(A,B,C,batch,M,K,N)`.
  - **core/cuda/cuda_backend.cu**: Singleton implementation. `is_available()` uses `cudaGetDeviceCount()`. `matmul` and `bmm` use cuBLAS (`cublasSgemm`, `cublasSgemmStridedBatched`) with host copy-in/copy-out; row-major result handled by transposing cuBLAS (column-major) output. Also defines `cuda_backend_available()` when CUDA=1.

- **Makefile**
  - `CUDA ?= 0`. When `CUDA=0`: add `cuda_stub.o` (built from `core/cuda/cuda_stub.cpp`). When `CUDA=1`: add `cuda_backend.o` (built with nvcc from `core/cuda/cuda_backend.cu`), `CXXFLAGS += -DWHITEMATTER_CUDA`, `LDFLAGS += -lcudart -lcublas`; optional `CUDA_PATH` for `-L$(CUDA_PATH)/lib64` and nvcc path. New rules for `cuda_stub.o` and `cuda_backend.o`.

- **Tensor (`core/tensor.cpp`, `core/tensor.h`)**
  - `#if defined(WHITEMATTER_CUDA)` includes `cuda/cuda_backend.h`. In `matmul`: when both operands have `device == CUDA` and `cuda_backend_available()`, call `CUDABackend::instance().matmul(...)` and attach same grad_fn pattern as Metal/CPU. In `bmm`: same for `CUDABackend::instance().bmm(...)` when both operands on CUDA. `tensor.h` comment for `to(DeviceType)` updated to “CPU, METAL, or CUDA”.

- **README.md**
  - Build options: documented “GPU backends” — `make METAL=1` (macOS), `make CUDA=1` (Linux/cloud, nvcc and toolkit required; `CUDA_PATH` optional). Infrastructure checklist: GPU support (CUDA/Metal) marked done.

### Verification

- `make clean && make && make test` (CUDA=0, default): 184 tests passed.
- Push to `origin dongha` after rebase on `origin/main`.

---

## 5. Repo cleanup: purge large/internal paths from history

**Summary:** Removed from git history (via `git filter-repo`) files and directories that should not be in a public repo: OS/editor cruft, Python artifacts, MNIST data, model binaries, personal dev log, and internal tooling. Updated `.gitignore` so they stay ignored; added `.gitkeep` in `data/`, `models/`, `platform/models/` so those dirs exist after clone.

**From this chat:** Ran `git filter-repo` with `--path ... --invert-paths` for: `.DS_Store`, `__pycache__/`, `whitematter.egg-info/`, `whitematter.cpython-311-darwin.so`, `mnist_cnn.onnx`, `data/`, `models/`, `platform/models/`, `dongha.md`, `.beads/`, `mayor/`, `deacon/`, `settings/`. Repo size dropped from ~69MB blobs to ~8MB. Re-added `origin` remote. `.gitignore` updated with explicit rules and exceptions for `data/.gitkeep`, `models/.gitkeep`, `platform/models/.gitkeep`. This file (dongha.md) is now gitignored and kept local only.

### Changes

- **History purge (git filter-repo)**
  - Purged paths: `.DS_Store`; `__pycache__/`; `whitematter.egg-info/`; `whitematter.cpython-311-darwin.so`; `mnist_cnn.onnx`; `data/` (53MB MNIST); `models/*.bin`, `models/*.json`; `platform/models/*.bin`, `platform/models/*.json`; `dongha.md`; `.beads/`; `mayor/`; `deacon/`; `settings/`.

- **.gitignore**
  - Added/expanded: `*.onnx`, `mnist_cnn.onnx`; `models/*.bin`, `models/*.json`, `!models/.gitkeep`; `platform/models/*.bin`, `platform/models/*.json`, `!platform/models/.gitkeep`; `dongha.md`; `.beads/`; `mayor/`; `deacon/`; `settings/`. Data: `data/*` with `!data/.gitkeep` so only `data/.gitkeep` is trackable.

- **Directory structure**
  - Added `data/.gitkeep`, `models/.gitkeep`, `platform/models/.gitkeep` (tracked) so scripts that expect these dirs still work after clone.

### Verification

- `.git` size ~6.4MB; blob size ~8MB. No purged paths remain in `git ls-files`. Remote re-added; force-push required for updated branches.

---

## 6. AWS Deployment Implementation Plan (2026-03-11)

**Summary:** Implemented the full AWS deployment plan from `docs/superpowers/plans/2026-03-11-aws-deployment-plan.md`: PostgreSQL + Alembic, User/auth models and JWT auth routes, credential encryption and AWS credential CRUD, S3 proxy and storage routes, BYOC provisioner and user-data scripts, BYOC training and callback routes, production Docker/compose and deploy script, frontend auth (login/register), AWS setup and S3 manager and dashboard pages, and PyPI packaging with cibuildwheel CI.

### Changes

- **Chunk 1 (Database & Auth):** `platform/requirements.txt` (psycopg2, alembic, passlib, jose, boto3, etc.); `platform/db/database.py` (DATABASE_URL env); Alembic init and migrations; `platform/db/auth_models.py` (User, AWSCredential, ByocTrainingJob, ModelArchitecture); `platform/services/auth_service.py` and `platform/auth/dependencies.py`; auth routes and schemas; server router registration.
- **Chunk 2 (AWS & S3):** `platform/services/credential_service.py`, `platform/services/s3_service.py`; credential and storage routes/schemas; credentials and storage routers in server.
- **Chunk 3 (BYOC):** `platform/byoc/user_data.py`, `platform/byoc/provisioner.py`; `platform/routes/byoc_training.py` and schemas; launch/status/stop and callback routes.
- **Chunk 4 (Deployment):** Dockerfile runtime slimmed (libgomp1, nginx, supervisor, libpq5); `docker-compose.prod.yml` and `.env.example`; `deploy/aws-setup.sh`.
- **Chunk 5 (Frontend):** `frontend/src/services/auth.ts`, `context/AuthContext.tsx`, Login/Register pages; `services/aws.ts`, AWSSetupPage, S3ManagerPage, DashboardPage; App.tsx nav (Dashboard, Data (S3), Train, Models, Predict, Settings). **Later:** Migrated frontend from Vite to Next.js (App Router, real routes).
- **Chunk 6 (PyPI):** `platform/setup.py` (long_description, classifiers, url); `platform/pyproject.toml` (cibuildwheel); `.github/workflows/publish.yml` (build wheels on tag, publish to PyPI).

### Verification

- Platform tests: 57 passed (auth, credential, dataset_manager). Branch: `feature/aws-deployment`.

---

## 8. Fix FastAPI schemas package import

**Summary:** Resolved a runtime import error where `ModelMetadata` and related Pydantic models could not be imported from `schemas` because the `platform/schemas` package had an empty `__init__.py`. Consolidated the schema definitions into the package so `from schemas import ...` works consistently across the platform.

### Changes

- **Platform schemas**
  - Populated `platform/schemas/__init__.py` with all Pydantic models previously only defined in `platform/schemas.py` (`LayerConfig`, optimizer/scheduler/augmentation configs, `TrainRequest`, `TrainStatus`, `ModelMetadata`, design/refine/custom-train and `GenerateRequest`), making `schemas` a proper package export.
  - Confirmed that `python3 server.py` now fails only on the expected `whitematter` editable install check (no longer on a `ModelMetadata` import error), so all `from schemas import ...` imports succeed.

---

## 7. Frontend: Vite → Next.js (App Router, Option B)

**Summary:** Migrated the frontend from Vite to Next.js 15 with the App Router and real routes (Option B). All “tabs” are now proper URLs; auth and API env var updated for Next.

### Changes

- **Next.js setup:** `next.config.mjs` (rewrites to backend on 8080), `package.json` (next 15, scripts: dev/build/start), `tsconfig.json` (paths `@/*` → `src/*`), `next-env.d.ts`.
- **App Router:** `src/app/layout.tsx` (root layout + Providers), `src/app/globals.css`, `src/app/providers.tsx` (AuthProvider client wrapper), `src/app/page.tsx` (redirect / → /dashboard or /login).
- **Auth routes:** `src/app/login/page.tsx`, `src/app/register/page.tsx` (wrap views; use `Link` and `router.replace` for post-login).
- **Authenticated area:** `src/app/(authenticated)/layout.tsx` (header, nav with `Link`, redirect when !user), and pages: `dashboard`, `data`, `train`, `models`, `predict`, `settings`.
- **Views:** Renamed `src/pages` → `src/views` so Next does not treat them as Pages Router routes. Login/Register/Dashboard/S3/AWSSetup use `@/` imports and `Link`/`useRouter`.
- **API/env:** `API_BASE` now uses `process.env.NEXT_PUBLIC_API_BASE` in `api.ts`, `services/auth.ts`, `services/aws.ts`. `getStoredToken`/`storeTokens`/`clearTokens` guarded for SSR (`typeof window === "undefined"`).
- **Client components:** Added `"use client"` to AuthContext, Providers, all view and tab components (ErrorBoundary, DatasetsTab, TrainTab, DesignHelper, ModelsTab, PredictTab, ConfirmDialog, Toast, TrainingChart, Login/Register/Dashboard/AWSSetup/S3Manager).
- **Removed:** `vite.config.ts`, `index.html`, `src/main.tsx`, `src/App.tsx`. ESLint: dropped `eslint-plugin-react-refresh`; `ignoreDuringBuilds: true` so existing lint does not block build.

### Follow-up

- **Sign up:** Auth switch links (Sign up / Sign in) given `className="auth-link"` and `.auth-link` in `AuthPages.css` so they are clearly clickable. Register/Login now show a clearer error when the API is unreachable: “Is the API server running? (See frontend README.)”
- **Run instructions:** `frontend/README.md` updated to state that `npm run dev` runs only the Next.js frontend; the backend must be started separately (`cd platform && python server.py`). Both must run for login/sign up to work.

### Verification

- `npm run build` succeeds. Routes: `/`, `/login`, `/register`, `/dashboard`, `/data`, `/train`, `/models`, `/predict`, `/settings`.

---

## 10. Dataset import (URL, Hugging Face) and S3-compatible storage (R2/B2)

**Summary:** Users can import datasets from a public HTTPS URL (ZIP or TXT), from Hugging Face Hub by dataset ID, and can connect Cloudflare R2 or Backblaze B2 (and other S3-compatible storage) in addition to AWS S3.

### Backend

- **Import from URL** (`POST /datasets/import/url`)
  - `services/url_fetcher.py`: Secure fetch (HTTPS only, 1 GB max, timeout). Returns content, content-type, suggested filename.
  - Route: fetch URL → if ZIP use legacy dataset_manager path; if TXT use dataset_service.upload_text. Name from request or inferred from URL/filename.
- **Import from Hugging Face** (`POST /datasets/import/huggingface`)
  - `services/huggingface_import.py`: Load via `datasets`; auto-detect image (image + label columns) or text. Image: build folder-per-class ZIP in memory → dataset_service.upload_zip. Text: concatenate text column → dataset_service.upload_text. Limits: 50k image rows, 100k text rows, 10M text chars.
  - Schemas: `schemas/import_schemas.py` (ImportFromUrlRequest, ImportFromHuggingFaceRequest).
- **S3-compatible storage**
  - `db/auth_models.py`: AWSCredential extended with optional `endpoint_url` and `provider` (aws|r2|b2|custom).
  - Migration `d58bb00add0e_add_s3_compatible_endpoint.py`: add columns to aws_credentials.
  - `services/s3_service.py`: All methods accept optional `endpoint_url`; boto3 client uses it when set (R2, B2, MinIO). create_bucket skips LocationConstraint when endpoint_url is set.
  - `routes/storage.py`: Pass `cred.endpoint_url` into every S3Service call.
  - `routes/credentials.py` + `schemas/credential_schemas.py`: Request/response include endpoint_url and provider.

### Frontend

- **Data page:** Route `/data` has two tabs: **Datasets** (import + upload) and **S3 Storage** (buckets/objects). `DataPage.tsx` wraps `DatasetsTab` and `S3ManagerPage`.
- **Datasets tab:** “Import from URL” (URL input, optional name, Import); “Import from Hugging Face” (dataset ID, optional name, split: train/validation/test, Import). Same loading/error/success and preview as upload. API: 120s timeout for both imports.
- **Settings (AWS):** Provider dropdown: AWS S3, Cloudflare R2, Backblaze B2, Other S3-compatible. When not AWS, show Endpoint URL field with placeholder. Credential payload sends endpoint_url and provider.
- **API:** `importDatasetFromUrl(url, name?)`, `importDatasetFromHuggingFace(datasetId, options?)`. `aws.CredentialData` and store/update pass endpoint_url and provider.

### Dependencies

- `platform/requirements.txt`: added `datasets>=2.18.0`, `huggingface_hub>=0.21.0`.

### Tests

- `platform/tests/test_url_fetcher.py`: Rejects HTTP, rejects invalid URL.

---

## 11. Minimal, friendly UI with Material UI (Apple-inspired)

**Summary:** The frontend was updated to use Material UI (MUI) with a minimal, friendly, Apple-inspired dark theme. Login, Register, home loading, authenticated layout (header + tabs), and Dashboard were rebuilt with MUI components for consistent typography, spacing, and accessibility.

### Changes

- **Dependencies:** Added `@mui/material`, `@emotion/react`, `@emotion/styled`, `@mui/icons-material`.
- **Theme (`frontend/src/theme.ts`):** New file. Dark palette, 12px border radius, system font stack, refined Button/TextField/Paper/AppBar/Card overrides; no heavy shadows, subtle borders.
- **Providers (`frontend/src/app/providers.tsx`):** Wrapped app with MUI `ThemeProvider` and `CssBaseline`.
- **Auth pages (`frontend/src/views/LoginPage.tsx`, `RegisterPage.tsx`):** Rebuilt with MUI `Paper`, `TextField` (with icon adornments), `Button`, `Alert`, `Link` (with Next.js `NextLink`); centered card layout, clear hierarchy. Removed dependency on `AuthPages.css` for layout (CSS file left in repo).
- **Authenticated layout (`frontend/src/app/(authenticated)/layout.tsx`):** Replaced custom header/nav with MUI `AppBar`, `Toolbar`, `Tabs`, `Tab` (with Next.js `Link` for routing), `Chip` for model count; loading state uses `CircularProgress`.
- **Dashboard (`frontend/src/views/DashboardPage.tsx`):** Rebuilt with MUI `Card`, `CardActionArea`, `CardContent`, `Stack`, icons (Dataset, School, Psychology, Settings), short sublabels per action.
- **Home page (`frontend/src/app/page.tsx`):** Loading state uses MUI `Box` and `CircularProgress`.

Existing views (Data, Train, Models, Predict, Settings) and `App.css` unchanged; they render inside the new MUI layout and paper content area.

### Verification

- `npm run build` in `frontend` completes successfully.

---

## 12. Viral Product Polish: Distribution and UX

**Summary:** Implemented the three-phase Viral Product Polish plan: (1) distribution and one-line boot — root `pyproject.toml`/`setup.py` for `pip install whitematter`, root `CMakeLists.txt` (FetchContent-friendly), `install.sh` and README updates; (2) viral UX — architecture graph with `@xyflow/react` and code sandbox preview (`POST /design/preview-code` + frontend viewer); (3) edit-and-sync — editable layer params in the Train tab so Refine, Preview code, and Train all use the edited architecture.

### Changes

- **Phase 1 — Friction layer**
  - **Root Python package:** `setup.py` and `pyproject.toml` at repo root build the C++ extension from `core/` and `bindings/`; Dockerfile python-builder stage now copies root `setup.py`, `pyproject.toml`, `core/`, `bindings/` and runs `pip wheel .`.
  - **CMake:** `CMakeLists.txt` at root builds static library `whitematter` from core + datasets, optional Metal/CUDA (OFF by default), OpenMP when found; install targets for FetchContent consumers.
  - **One-line boot:** `install.sh` checks Docker and Docker Compose, runs `docker compose up -d` from repo or clones repo then runs; README “Distribution” section documents pip, CMake, and curl \| bash one-liner.

- **Phase 2 — Viral UX**
  - **Architecture graph:** New `frontend/src/components/ArchitectureGraph.tsx` using `@xyflow/react`: one node per layer (label from type + key params), edges in sequence; integrated above the layer list in TrainTab with `.architecture-graph-section` styles.
  - **Code preview:** Backend `POST /design/preview-code` in `platform/routes/design.py` (body: `dataset_id`, `architecture`); resolves `dataset_config` from processed blobs or dataset metadata, calls `code_generator.generate` into a temp dir, returns `train_cpp`/`infer_cpp` strings. Frontend `api.previewGeneratedCode`, “Preview generated code” button and expandable read-only panels for `train.cpp`/`infer.cpp` in TrainTab. Schema `PreviewCodeRequest` in `platform/schemas/__init__.py`.

- **Phase 3 — Edit-and-sync**
  - **Editable layers:** TrainTab layer list replaced with editable rows: each layer shows type (read-only) and per-param inputs; `handleLayerParamChange` updates `architecture` state so Refine, Preview code, and Start Training use the edited architecture. Hint text: “Edits here are included when you Refine, Preview code, or Train.” CSS: `.layers-list-editable`, `.layer-item-editable`, `.layer-params-editable`, `.layer-param-field`.

### Verification

- Frontend `npm run build` succeeds. Platform `from schemas import PreviewCodeRequest` and `from routes.design import preview_code` import successfully.

---

*When you implement another plan, add a new numbered section above this line.*
