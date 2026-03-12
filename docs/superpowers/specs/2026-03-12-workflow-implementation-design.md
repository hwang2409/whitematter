# Whitematter Workflow Implementation Design

**Date:** 2026-03-12
**Scope:** SP1 (Sign Up), SP2 (Chat-Driven Training), SP3 (Post-Training Experience)
**Build order:** SP1 → SP2 → SP3 (each is a self-contained deliverable)

---

## SP1: Sign Up Polish

### Goal
Add Google OAuth button (env-var gated) to three surfaces and ensure account linking works.

### What exists
- Email/password auth (login + register pages)
- Backend `POST /auth/google` endpoint — accepts an `access_token` via `GoogleAuthRequest` and calls Google userinfo API
- Users get `plan: "free"` on registration
- `providers.tsx` exists for app-level providers

### Changes

**1. GoogleOAuthProvider wrapper**
- Add `@react-oauth/google` dependency to frontend
- Wrap app in `<GoogleOAuthProvider>` in `providers.tsx`
- Conditionally render only when `NEXT_PUBLIC_GOOGLE_CLIENT_ID` env var is set
- If env var is missing, the provider is omitted and no Google UI renders

**2. Google OAuth button on 3 surfaces**
- **Landing page hero** (`LandingPage.tsx`): Google button next to existing "Get Started" button. One click from landing → Google auth → chat.
- **Login page** (`LoginPage.tsx`): Google button above the email/password form
- **Register page** (`RegisterPage.tsx`): Google button above the email/password form
- Use `useGoogleLogin` hook with the **implicit grant flow** which returns an `access_token` — this matches what the backend expects (it calls Google userinfo API with the access token as Bearer)
- On success, stores JWT tokens and redirects to `/chat`

**3. Account linking**
- Verify backend `/auth/google` handles this case: user signs up with email/password, later signs in with Google using the same email
- Backend should find existing user by email and link Google ID rather than creating duplicate
- Fix if not working

**4. Env vars**
- Backend: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` (for future server-side token verification; not currently used by backend but reserved)
- Frontend: `NEXT_PUBLIC_GOOGLE_CLIENT_ID`
- Update `.env.example` with these entries (commented out, in Optional section)

### Files touched
- `frontend/package.json` (new dependency)
- `frontend/src/app/providers.tsx` (GoogleOAuthProvider)
- `frontend/src/views/LandingPage.tsx` (Google button in hero)
- `frontend/src/views/LoginPage.tsx` (Google button)
- `frontend/src/views/RegisterPage.tsx` (Google button)
- `.env.example` (new env vars)

---

## SP2: Chat-Driven Training Flow

### Goal
Complete the core product loop: welcome → Quick Start or describe problem → AI suggests architecture → upload data → train → live results in chat.

### What exists
- Chat UI (`ChatPage.tsx`) — **currently running on mock data** (`mockAssistantReply`). Not wired to real backend SSE streaming yet.
- Welcome/greeting message in `chat_service.py` (`GREETING_MESSAGE` constant)
- Quick Start chips (`QuickStartChips.tsx`) with 4 templates (CIFAR-10, Shakespeare, Sentiment, Custom) — **will be replaced with 2 chips for launch**
- Claude integration for architecture suggestions (`chat_service.py`, `prompts.py`)
- Backend SSE streaming in `chat_service.py` + `chat.py` (StreamingResponse)
- Training endpoints (`/train/custom`)
- Training job store with progress tracking
- WebSocket endpoint at `/ws/train/{job_id}` in `training.py` + frontend client `createTrainingWebSocket` in `api.ts` — **will be replaced by SSE**
- `TrainingProgress.tsx` — uses polling (2s interval), **will be refactored to SSE**
- Dataset upload endpoint (`/datasets/upload`)
- `TrainingChart.tsx` — fully functional Recharts component (loss/accuracy dual-axis), needs chat integration
- `frontend/src/lib/trainingErrors.ts` error parser
- Existing `ModelCard.tsx` — already has "Looks good, train it!" and "Make changes" buttons for architecture proposals. **Will be refactored/renamed to `InlineModelCard`** rather than creating a duplicate.
- `ChatMessage.tsx` — bubble renderer, needs modification for new message types
- Backend saves architecture messages with `message_type="architecture"` (chat_service.py)

### Prerequisite: Wire ChatPage to Backend

Before any training flow work, the frontend chat must be connected to the real backend:
- Replace `mockAssistantReply` with real API calls to `POST /chat/conversations` and `POST /chat/conversations/{id}/messages`
- Consume SSE streaming responses from the backend
- Load conversation history on page load
- This is a significant chunk of work and the foundation for everything in SP2

### A. Welcome + Quick Start

**Welcome message:**
- Content: "Hey! I help you build and train neural networks. Try a demo or describe what you want to build."
- Two Quick Start chips displayed directly below the welcome message (replaces existing 4 chips):
  - **"Quick Start (MNIST)"** — instant demo flow
  - **"I want to build something"** — opens the describe-your-problem flow
- CIFAR-10, Shakespeare, Sentiment chips removed for launch (can be added later)

**Quick Start MNIST flow:**
- Clicking "Quick Start (MNIST)" sends a message that triggers MNIST preset loading
- MNIST dataset (~12 MB processed .bin files) is **pre-bundled in R2/BlobStore** — no download wait, instant loading. Pre-bundle `train_images.bin`, `train_labels.bin`, `test_images.bin`, `test_labels.bin`, and `config.json`.
- **One-time setup:** Create `scripts/prebundle_mnist.py` which downloads MNIST, processes it into .bin format (using existing `process_mnist_idx` from `dependencies.py`), and stores the files locally under `presets/mnist/`. In production with R2, upload these files to the R2 bucket under `presets/mnist/`. In local dev without R2, serve from the local filesystem.
- `chat_service.py` copies the pre-bundled MNIST files into the user's dataset directory
- Skips the "describe your problem" flow — goes straight to architecture suggestion with a pre-configured MNIST CNN
- Shows `InlineModelCard` (refactored from existing `ModelCard.tsx`) with the suggested architecture and "Train it" button

### B. Architecture Suggestion → Train

**InlineModelCard (refactored from existing `ModelCard.tsx`):**
- Rename/refactor existing `ModelCard.tsx` which already has the right UI (architecture display + "Train it" / "Make changes" buttons)
- Rendered inline in chat when Claude suggests an architecture
- Shows: architecture name, layer summary, estimated param count, dataset info
- Two actions directly on the card:
  - **"Looks good, train it"** button — transitions conversation phase from architecture → training
  - **"Change something"** — text input for the user to request modifications
- Clicking "Train it" triggers the training flow (section D)

**How it works:**
- Claude's response includes structured architecture data (JSON) alongside the natural language explanation
- Frontend parses this and renders `InlineModelCard` instead of plain text
- Backend uses `message_type="architecture"` (existing convention) to signal this message type
- The card is interactive — not just a display component

### C. Dataset Upload in Chat

**Wire attach button:**
- Connect existing attach button in `ChatInput.tsx` to `/datasets/upload` endpoint
- Show upload progress inline in chat

**Drag-and-drop:**
- Add drop zone on the chat area
- Drop overlay shows file size limit per plan: "Drop your dataset here (max 200 MB on Free, 1 GB on Pro, 5 GB on Scale)"
- Client-side file size check BEFORE upload — reject immediately with clear message if over limit
- On successful upload, AI auto-detects format and confirms in chat

**Plan-based limits (new server-side enforcement):**
- Free: 200 MB max upload (enough for MNIST + CIFAR-10)
- Pro: 1 GB max upload
- Scale: 5 GB max upload
- Add server-side file size validation in `/datasets/upload` endpoint (currently has no size check)
- Client-side check for instant UX, server-side check for security

### D. "Train It" Trigger

**From InlineModelCard:**
- "Train it" button calls existing `/train/custom` endpoint
- Passes architecture config + dataset reference

**From chat text:**
- Claude detects "train it" intent and triggers training via the same code path

**CPU/GPU selector (Scale users only):**
- Free/Pro: CPU only, no selector shown
- Scale: Toggle chips on the train action — "CPU (instant)" / "GPU (faster, ~60s startup)"
- Default: GPU for Scale users, but they can switch to CPU for quick tests

**Edge case — concurrent training:**
- If a training job is already running in the conversation, disable the "Train it" button
- Show message: "Training already in progress"

### E. Live Training Results in Chat (SSE only)

**Transport: SSE (not WebSocket)**
- Use SSE for both chat streaming and training updates — single transport mechanism
- Remove existing WebSocket endpoint `/ws/train/{job_id}` from `training.py`
- Remove `createTrainingWebSocket` from `frontend/src/api.ts`
- Add new SSE endpoint: `GET /chat/conversations/{id}/training/stream` — streams epoch updates as SSE events
- Refactor `TrainingProgress.tsx` from polling (2s interval) to SSE consumption

**Frontend rendering:**
- Training progress renders as a live-updating chat message
- Shows: current epoch / total epochs, loss, accuracy
- `TrainingChart` component (already functional) renders inline — loss/accuracy line chart updates as epochs complete
- When training finishes, the message transitions to the completed model card (SP3)

**Edge case — page refresh during training:**
- On conversation load, check if there's an active training job
- If so, reconnect to SSE stream and resume displaying progress
- Show accumulated progress from job store (epochs already completed)

### F. Training Failure Handling

**Error display:**
- Errors render inline in chat with human-readable explanation
- Use existing `frontend/src/lib/trainingErrors.ts` parser to convert raw errors to friendly messages
- Examples: "Compilation error in layer 3", "Out of memory — model too large for dataset"

**AI recovery:**
- After showing error, Claude automatically offers to fix the architecture
- Example: "Looks like the model ran out of memory. Want me to make it smaller?"
- User can accept → Claude suggests modified architecture → new `InlineModelCard` → retrain

### Files touched (estimated)
- `frontend/src/views/ChatPage.tsx` (wire to backend, drag-drop zone, message type rendering)
- `frontend/src/components/ChatMessage.tsx` (render new message types based on `type` field)
- `frontend/src/components/ChatInput.tsx` (file upload wiring, drag-drop)
- `frontend/src/components/ModelCard.tsx` (refactor → InlineModelCard)
- `frontend/src/components/TrainingProgress.tsx` (polling → SSE refactor)
- `frontend/src/components/TrainingChart.tsx` (inline chat integration)
- `frontend/src/components/QuickStartChips.tsx` (replace 4 chips with 2: Quick Start MNIST + "I want to build something")
- `frontend/src/api.ts` (remove WebSocket client, add SSE helpers)
- `platform/services/chat_service.py` (Quick Start MNIST, structured architecture output)
- `platform/routes/chat.py` (SSE training stream endpoint)
- `platform/routes/training.py` (remove WebSocket endpoint)
- `platform/routes/datasets.py` (server-side size limits)

---

## SP3: Post-Training Experience

### Goal
When training completes, show a rich model card with "Try it" (predict in chat). Deploy deferred.

### What exists
- `ModelCard.tsx` component (will be `InlineModelCard` after SP2 refactor)
- `ShareCard.tsx` with html2canvas
- `POST /predict` endpoint — accepts `model_id` query param + file upload
- Model metadata storage (`ModelMetadata` schema)

### A. Completed Model Card in Chat

**CompletedModelCard component (new, extends InlineModelCard):**
- Appears inline in chat when training finishes successfully
- Shows: accuracy (e.g. 98.5%), parameter count (e.g. 207K), training time (e.g. 18s), architecture summary
- Three action buttons:
  - **"Try it"** — opens inline predict widget
  - **"Deploy as API"** — shows "Coming soon" badge / upgrade prompt for Free users
  - **"Share"** — opens pre-filled Twitter intent URL (`https://twitter.com/intent/tweet?text=...&url=https://whitematter.com`) with model stats. Separate "Save image" option uses existing `ShareCard.tsx` + html2canvas for PNG download.

### B. Predict in Chat

**Inline predict widget:**
- "Try it" expands an area on/below the card with a file drop zone
- User drops an image → calls `POST /predict` with `model_id` from conversation context → shows prediction result with confidence scores
- `model_id` is obtained from the conversation's `model_id` field (set when training completes)
- Results display inline — no page navigation
- Can predict multiple images sequentially

### C. Deploy as API (deferred)

- Button renders but shows "Coming soon" state
- Free users see upgrade prompt
- No infrastructure work — this ships later as a separate project

### Files touched (estimated)
- `frontend/src/components/CompletedModelCard.tsx` (new)
- `frontend/src/components/InlinePredictWidget.tsx` (new)
- `frontend/src/components/ChatMessage.tsx` (render CompletedModelCard for `training_complete` type)
- `frontend/src/views/ChatPage.tsx` (render completed card after training)

---

## Architecture Decisions

### SSE over WebSocket
Both chat streaming and training updates use SSE. Single transport, works through nginx, already proven in the codebase. Existing WebSocket infrastructure (`/ws/train/{job_id}`, `createTrainingWebSocket`) is removed.

### Structured chat messages
Chat messages carry a `message_type` field (matching existing backend convention) to distinguish rendering:
- `text` — normal markdown message
- `architecture` — renders `InlineModelCard` (matches existing `chat_service.py` convention)
- `training_progress` — renders live training chart
- `training_complete` — renders `CompletedModelCard`
- `training_error` — renders error with AI recovery offer
- `file_upload` — renders upload progress/confirmation

### Google OAuth is env-var gated
If `NEXT_PUBLIC_GOOGLE_CLIENT_ID` is not set, all Google UI is hidden. Email/password works standalone. This means local dev and contributors don't need Google credentials.

### Plan-based gating
- File upload limits enforced client-side (instant rejection) AND server-side (security)
- CPU/GPU toggle only shown to Scale users
- Deploy button shows "Coming soon" for all users

---

## Edge Cases

- **Concurrent training:** Disable "Train it" if a job is already running in the conversation
- **Page refresh during training:** Detect active job on load, reconnect to SSE, resume progress display
- **Account linking:** Google OAuth with existing email account links rather than duplicating
- **Dataset ownership:** Datasets uploaded in chat are scoped to the user's conversation. Verify backend filters by user context.
- **Claude API down/rate-limited:** If Claude returns 529 or times out, show friendly error in chat: "I'm having trouble thinking right now, try again in a moment" with a retry button. Never show raw API errors or blank screens.

---

## Out of Scope

- Public model card URLs / shareable links — deferred (Share button opens Twitter intent with whitematter.com link, but no per-model public pages yet)
- Deploy as API infrastructure — deferred, button shows "Coming soon"
- Settings page — remains stub
- Onboarding wizard/modal — chat welcome message serves this purpose
- Hugging Face dataset import UI
