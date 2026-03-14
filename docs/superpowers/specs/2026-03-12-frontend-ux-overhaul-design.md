# Frontend UX Overhaul — Design Spec

**Date**: 2026-03-12
**Branch**: `frontend-ux-overhaul`
**Approach**: Incremental — Group B → C → A, each committed independently.

---

## Group B: Onboarding & UX Workflow

### B1. First-Run Onboarding Wizard

**Trigger**: Dashboard detects `datasets.length === 0 && models.length === 0` after data loads.

**Component**: `components/OnboardingWizard.tsx` — replaces normal dashboard content (not a modal).

**Steps** (MUI `Stepper`):
1. **"Upload your first dataset"** — inline drag-drop zone + "Use sample dataset" button that calls `importDatasetFromHuggingFace("ylecun/mnist", { name: "MNIST Sample", split: "train" })`. Shows progress indicator. **Error handling**: if import fails or times out (120s limit), show an error alert with "Try again" button and a fallback "Upload manually" link to `/data`.
2. **"Design your architecture"** — CTA linking to `/architect?dataset={id}`. Brief explanation of the AI architect.
3. **"Train your model"** — links to `/train` with dataset pre-selected.
4. **"Make a prediction"** — links to `/predict`.

Each step has "Skip." Step 1 completion auto-advances. Steps 2-4 are informational CTAs. Wizard state persists in `localStorage` key `wm_onboarding_{userId}` (scoped per user). Resets if user has zero datasets AND zero models (handles dataset deletion case).

**Files**: New `components/OnboardingWizard.tsx`, modified `views/DashboardPage.tsx`.

### B2. Quick Start Button

**Location**: Dashboard — shown when no datasets exist OR as a Quick Action card when datasets exist but no models.

**Label**: "Train a digit classifier in 60 seconds"

**Behavior**:
1. If no MNIST dataset → imports via HuggingFace API (inline progress)
2. Once ready → calls `startCustomTraining(datasetId, architecture)` with full `Architecture` object
3. Navigates to `/train` where live chart appears via WebSocket

**Defaults**: `QUICK_START_CONFIG` constant in `lib/quickStart.ts` — a full `Architecture` object:
```ts
const QUICK_START_ARCHITECTURE: Architecture = {
  name: "MNIST Digit Classifier",
  description: "Simple CNN for handwritten digit recognition",
  data_type: "image",
  input_shape: [1, 28, 28],
  num_classes: 10,
  layers: [
    { type: "conv2d", params: { in_channels: 1, out_channels: 16, kernel_size: 3 } },
    { type: "relu", params: {} },
    { type: "maxpool2d", params: { kernel_size: 2 } },
    { type: "conv2d", params: { in_channels: 16, out_channels: 32, kernel_size: 3 } },
    { type: "relu", params: {} },
    { type: "maxpool2d", params: { kernel_size: 2 } },
    { type: "flatten", params: {} },
    { type: "linear", params: { in_features: 800, out_features: 10 } },
  ],
  training: {
    optimizer: { type: "adam", params: { lr: 0.001 } },
    scheduler: { type: "step_lr", params: { step_size: 5, gamma: 0.5 } },
    epochs: 10,
    batch_size: 32,
  },
};
```

**Files**: New `lib/quickStart.ts`, modified `views/DashboardPage.tsx`.

### B3. Dedicated `/architect` Route — AI Architect

> **Note**: Route is `/architect` (not `/design`) to avoid conflict with the existing Next.js rewrite rule at `/design/:path*` → backend API in `next.config.mjs` line 16.

**New route**: `app/(authenticated)/architect/page.tsx` → `views/ArchitectPage.tsx`

**Nav update**: Add "Design" between Data and Train in sidebar (`AutoAwesomeOutlined` icon, label "Design", href `/architect`). Move Predict icon to `BatchPredictionOutlined` (verified available in `@mui/icons-material` v7). Predict label stays "Predict".

**Layout** — two-column:
- **Left (60%)**: Dataset selector → `ArchitectureGraph` visualization → architecture summary chips + param counts → "Send to Training" button
- **Right (40%)**: `DesignHelper` chat. Enhanced with "Suggest Architecture" button calling `suggestArchitecture(datasetId, prompt)`. AI suggestions render in the graph in real-time. "Apply & Train" navigates to `/train?dataset={id}`.

**State sharing**: New `context/DesignContext.tsx` holding current `Architecture` object. Added to `Providers` tree in `app/providers.tsx`. Design page writes, Train page reads. SessionStorage key `wm_design_architecture` as fallback for cross-navigation (JSON-serialized `Architecture`). When Train page loads without DesignContext data, it falls back to existing behavior (user configures manually).

**Train page changes**: Remove DesignHelper sidebar toggle from `TrainTab`. Replace with "Open AI Architect →" link to `/architect`. Keep architecture display as read-only.

**Files**: New `views/ArchitectPage.tsx`, `app/(authenticated)/architect/page.tsx`, `context/DesignContext.tsx`. Modified `app/(authenticated)/layout.tsx` (nav), `app/providers.tsx` (add DesignProvider), `components/TrainTab.tsx` (remove helper sidebar), `app/(authenticated)/train/page.tsx` (simplify to thin wrapper).

### B4. Tooltips on Training Parameters

**Component**: `components/ParamTooltip.tsx` — `?` `IconButton` with MUI `Tooltip`.

**Content map** in `lib/paramTooltips.ts`:
| Parameter | Description | Range |
|-----------|-------------|-------|
| Optimizer | Controls how model weights are updated. Adam is a good default. | Adam, SGD, AdamW |
| Scheduler | Adjusts learning rate during training. | StepLR, CosineAnnealing, None |
| Batch size | Samples processed per weight update. Larger = faster, more memory. | 16–128, default 32 |
| Learning rate | Step size for weight updates. Too high = unstable, too low = slow. | 0.0001–0.01, default 0.001 |
| Augmentations | Random transforms to prevent overfitting. | RandomFlip, RandomRotation |

**Files**: New `components/ParamTooltip.tsx`, `lib/paramTooltips.ts`. Modified `components/TrainTab.tsx`.

### B5. AWS Optional Badge

**DataPage.tsx**: MUI `Alert` (severity="info") at top of S3 Storage tab content area (inside the `{tab === "storage" && ...}` branch, before `<S3ManagerPage />`): "S3 storage requires AWS or S3-compatible credentials. Configure in Settings. This is optional — core training and prediction work without it."

**ModelsTab.tsx**: `Typography` caption next to "Deploy to API" button: "Requires AWS credentials (optional)" with link to `/settings`.

**Files**: Modified `views/DataPage.tsx`, `components/ModelsTab.tsx`.

### B6. Error Message Parsing

**New utility**: `lib/trainingErrors.ts`

Pattern → user-friendly message map:
| Pattern | Message |
|---------|---------|
| `dimension mismatch` | Incompatible layer dimensions. Check output/input sizes between layers. |
| `out of memory` / `OOM` | Not enough memory. Try reducing batch size or simplifying architecture. |
| `CUDA error` | GPU error. Try restarting training or switching to CPU. |
| `invalid argument` | Invalid training parameter. Check architecture configuration. |
| `nan` / `loss is nan` | Training diverged (NaN loss). Try a lower learning rate. |

Falls back to raw message with collapsible "Show raw output" `<details>`.

**Files**: New `lib/trainingErrors.ts`. Modified `components/TrainTab.tsx`.

---

## Group C: Infra & Quality

### C1. CSP Headers (Token Storage)

**Decision**: Keep localStorage for JWT. Add strict CSP headers in `next.config.mjs` via `headers()`.

**Note**: `'unsafe-inline'` for `style-src` is required because MUI/Emotion injects inline `<style>` tags at runtime. A nonce-based approach (Emotion's `nonce` option + Next.js nonce support) is the proper long-term fix but out of scope for this pass. This is an intentional and acceptable trade-off.

```
Content-Security-Policy:
  default-src 'self';
  script-src 'self';
  style-src 'self' 'unsafe-inline' https://fonts.googleapis.com;
  font-src 'self' https://fonts.gstatic.com;
  connect-src 'self' ws: wss:;
  img-src 'self' data: https://img.shields.io

Referrer-Policy: strict-origin-when-cross-origin
X-Content-Type-Options: nosniff
```

**Production note**: If the backend runs on a different origin than the frontend in production, `connect-src` will need the backend origin added.

**Files**: Modified `next.config.mjs`.

### C2. Accessibility Pass

**Skip-nav**: Visually-hidden "Skip to main content" link as first focusable element in `app/(authenticated)/layout.tsx`. Target: `<Box component="main" id="main-content">`. The skip-nav link gets a visible `:focus` style (outline + background) to satisfy WCAG 2.4.1.

**ARIA labels**:
- Sidebar nav links: add `aria-label` (e.g., `aria-label="Dashboard"`)
- File upload drop zone in `DatasetsTab`: `role="button"`, `aria-label="Upload dataset file"`, `tabIndex={0}`, `onKeyDown` for Enter/Space
- Training chart container: `aria-label="Training progress chart"`
- Close buttons (`×`): `aria-label="Close"`

**`aria-live` regions**:
- Training progress display in `TrainTab`: `aria-live="polite"`
- Upload status text in `DatasetsTab`: `aria-live="polite"`

**Keyboard nav on file upload**: Add `tabIndex={0}`, `role="button"`, `onKeyDown` (Enter/Space → trigger file input click) to drop zone.

**Files**: Modified `app/(authenticated)/layout.tsx`, `components/DatasetsTab.tsx`, `components/TrainTab.tsx`, `components/TrainingChart.tsx`.

### C3. View Consolidation

**Direction**: All page logic in `views/`, route files are thin one-line wrappers.

**Specific refactors needed**:
- `train/page.tsx` (105 lines) → after B3, DesignHelper state moves to `ArchitectPage.tsx`. Remaining dataset/training state moves into `views/TrainPage.tsx` (rename from inline logic). Route file becomes thin wrapper.
- `predict/page.tsx` (24 lines) → model fetching/filtering logic moves to new `views/PredictPage.tsx`. Route file becomes thin wrapper.
- `models/page.tsx` (6 lines) — already a thin wrapper (passes empty callbacks to `ModelsTab`). No change needed.
- `dashboard/page.tsx`, `data/page.tsx`, `settings/page.tsx` — already thin wrappers. No change.

All route files end up as:
```tsx
"use client";
import XPage from "@/views/XPage";
export default function XRoute() { return <XPage />; }
```

**Files**: New `views/TrainPage.tsx`, `views/PredictPage.tsx`. Modified `app/(authenticated)/train/page.tsx`, `app/(authenticated)/predict/page.tsx`.

### C4. Remove Dead CSS

**Delete**:
- `views/AuthPages.css` — zero imports
- `views/DashboardPage.css` — zero imports
- `src/index.css` — zero imports (confirmed dead; `#root` selector is a Vite-era artifact, Next.js doesn't use `#root`)

**Keep**:
- `views/S3ManagerPage.css` — imported by `S3ManagerPage.tsx`
- `app/globals.css` — imported by layout

---

## Group A: Virality & First Impression

### A1. Landing Page

**Route**: `app/page.tsx` — client component. Uses `useAuth()` to check auth state: if authenticated, redirect to `/dashboard`; otherwise render `<LandingPage />`. Both the wrapper and `LandingPage` are client components (server component is not possible here since auth state lives in React context, and a child component inside a client component boundary runs on the client regardless).

**Component**: `views/LandingPage.tsx`

**Hero section**:
- "wm" logo (JetBrains Mono, matching sidebar branding)
- Headline: "Train neural networks from your browser."
- Subhead: "Design architectures with AI. Deploy with one click."
- CTA: "Get Started" → `/register`
- Secondary: "Already have an account? Sign in" → `/login`
- Demo GIF placeholder: 16:9 `Box` with border, rounded corners, "Demo GIF" placeholder text

**Feature highlights** — 3-column grid:
1. AI Architecture Designer — `AutoAwesomeOutlined` — "Describe what you want. Claude designs the neural network."
2. Live Training — `ShowChartOutlined` — "Watch your model train in real-time with live loss and accuracy charts."
3. One-Click Deploy — `RocketLaunchOutlined` — "Deploy trained models as API endpoints instantly."

**GitHub badge**: `<img>` shield badge below CTA. Placeholder org/repo for now.

**Footer**: "Built with whitematter" + GitHub link.

**Styling**: MUI `sx` props, no new CSS files.

**Files**: Modified `app/page.tsx`. New `views/LandingPage.tsx`.

### A2. OG + Social Tags

**Location**: `app/layout.tsx` `metadata` export.

```ts
export const metadata: Metadata = {
  title: "whitematter — Train neural networks from your browser",
  description: "Design architectures with AI, train models in real-time, and deploy with one click.",
  openGraph: {
    title: "whitematter",
    description: "Train neural networks from your browser. Design architectures with AI. Deploy with one click.",
    images: [{ url: "/og-image.png", width: 1200, height: 630 }],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "whitematter",
    description: "Train neural networks from your browser.",
    images: ["/og-image.png"],
  },
};
```

**OG image**: Generate and commit a real 1200x630 PNG — dark background (#0a0a0a), "wm" branding in accent color (#7EB8FF), tagline text. Created via a one-off Node script using canvas, committed as `public/og-image.png`.

**Files**: Modified `app/layout.tsx`. New `public/og-image.png`.

### A3. Share Results Button

**Location**: `ModelsTab.tsx` model detail panel, alongside existing action buttons.

**Behavior**: "Share" button:
1. Renders hidden DOM element: share card with model name, accuracy %, architecture chips, mini loss sparkline, "Built with whitematter" footer
2. `html2canvas` (lazy-loaded via `import('html2canvas')` to avoid main bundle bloat — ~180KB gzipped) converts to PNG
3. Downloads as `{model-name}-results.png`
4. Optional "Copy to clipboard" via `navigator.clipboard.write()`
5. **Fallback**: if html2canvas rendering fails, show toast "Failed to generate share image" and skip

**Share card design**: 600x400px, dark background (#0a0a0a), accent (#7EB8FF) accuracy number, architecture chips row, mini loss sparkline, "whitematter" branding.

**New dependency**: `html2canvas`

**Files**: New `components/ShareCard.tsx`. Modified `components/ModelsTab.tsx`. Updated `package.json`.

---

## New Files Summary

| File | Group | Purpose |
|------|-------|---------|
| `components/OnboardingWizard.tsx` | B1 | First-run wizard |
| `lib/quickStart.ts` | B2 | Quick start config constants |
| `views/ArchitectPage.tsx` | B3 | Dedicated AI architect page |
| `app/(authenticated)/architect/page.tsx` | B3 | Architect route wrapper |
| `context/DesignContext.tsx` | B3 | Shared architecture state |
| `components/ParamTooltip.tsx` | B4 | Tooltip component |
| `lib/paramTooltips.ts` | B4 | Tooltip content map |
| `lib/trainingErrors.ts` | B6 | Error pattern matching |
| `views/TrainPage.tsx` | C3 | Train page (extracted from route) |
| `views/PredictPage.tsx` | C3 | Predict page (extracted from route) |
| `views/LandingPage.tsx` | A1 | Public landing page |
| `public/og-image.png` | A2 | Social sharing image |
| `components/ShareCard.tsx` | A3 | Shareable results card |

## Modified Files Summary

| File | Groups | Changes |
|------|--------|---------|
| `views/DashboardPage.tsx` | B1, B2 | Onboarding wizard, quick start button |
| `app/(authenticated)/layout.tsx` | B3, C2 | Nav update (Design route), skip-nav, ARIA |
| `app/providers.tsx` | B3 | Add DesignContext provider |
| `components/TrainTab.tsx` | B3, B4, B6, C2 | Remove design sidebar, add tooltips, error parsing, aria-live |
| `app/(authenticated)/train/page.tsx` | B3, C3 | Simplify to thin wrapper |
| `app/(authenticated)/predict/page.tsx` | C3 | Simplify to thin wrapper |
| `views/DataPage.tsx` | B5 | AWS optional badge |
| `components/ModelsTab.tsx` | B5, A3 | AWS badge, share button |
| `components/DatasetsTab.tsx` | C2 | ARIA labels, keyboard nav on upload |
| `components/TrainingChart.tsx` | C2 | ARIA label |
| `app/layout.tsx` | A2 | OG + social meta tags |
| `app/page.tsx` | A1 | Landing page (replaces redirect) |
| `next.config.mjs` | C1 | CSP headers |

## Deleted Files

| File | Reason |
|------|--------|
| `views/AuthPages.css` | Zero imports — dead code |
| `views/DashboardPage.css` | Zero imports — dead code |
| `src/index.css` | Zero imports — dead code (Vite-era artifact) |

---

## Testing Strategy

**Existing tests that will need updates**:
- Any tests importing from `views/DashboardPage.tsx` (onboarding wizard changes the conditional rendering)
- Tests for `TrainTab` if they reference the DesignHelper sidebar toggle

**New tests to write** (vitest + testing-library):
- `OnboardingWizard` — renders stepper, step navigation, skip behavior
- `QuickStart` — button renders, triggers import flow
- `ParamTooltip` — renders tooltip content for each param key
- `trainingErrors` — unit tests for pattern matching (pure function, easy to test)
- `ArchitectPage` — renders two-column layout, dataset selector

**Manual QA checklist**:
- [ ] First login with zero data → onboarding wizard appears
- [ ] Quick start → MNIST import → training starts → chart appears
- [ ] `/architect` page → select dataset → chat with AI → architecture appears in graph
- [ ] "Send to Training" from architect → Train page has architecture pre-loaded
- [ ] Tooltips visible on all training params
- [ ] AWS badge visible on S3 tab and Deploy button
- [ ] Training failure → user-friendly error with "Show raw output"
- [ ] Skip-nav link visible on focus, navigates to main content
- [ ] File upload drop zone keyboard-accessible (Tab → Enter)
- [ ] Landing page renders for unauthenticated users, redirects for authenticated
- [ ] OG tags render correctly (test with `curl -s URL | grep og:`)
- [ ] Share button generates downloadable PNG
