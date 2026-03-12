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
1. **"Upload your first dataset"** — inline drag-drop zone + "Use sample dataset" button that calls `importDatasetFromHuggingFace("ylecun/mnist", { name: "MNIST Sample", split: "train" })`. Shows progress indicator.
2. **"Design your architecture"** — CTA linking to `/design?dataset={id}`. Brief explanation of the AI architect.
3. **"Train your model"** — links to `/train` with dataset pre-selected.
4. **"Make a prediction"** — links to `/predict`.

Each step has "Skip." Step 1 completion auto-advances. Steps 2-4 are informational CTAs. Wizard state persists in `localStorage` key `wm_onboarding_step`.

**Files**: New `components/OnboardingWizard.tsx`, modified `views/DashboardPage.tsx`.

### B2. Quick Start Button

**Location**: Dashboard — shown when no datasets exist OR as a Quick Action card when datasets exist but no models.

**Label**: "Train a digit classifier in 60 seconds"

**Behavior**:
1. If no MNIST dataset → imports via HuggingFace API (inline progress)
2. Once ready → calls `startCustomTraining` with hardcoded config
3. Navigates to `/train` where live chart appears via WebSocket

**Defaults**: `QUICK_START_CONFIG` constant in `lib/quickStart.ts`:
- Architecture: Conv2d(1,16,3)→ReLU→MaxPool(2)→Conv2d(16,32,3)→ReLU→MaxPool(2)→Flatten→Linear(32*5*5, 10)
- Training: Adam lr=0.001, batch_size=32, 10 epochs
- Scheduler: StepLR step_size=5, gamma=0.5

**Files**: New `lib/quickStart.ts`, modified `views/DashboardPage.tsx`.

### B3. Dedicated `/design` Route — AI Architect

**New route**: `app/(authenticated)/design/page.tsx` → `views/DesignPage.tsx`

**Nav update**: Add "Design" between Data and Train in sidebar (`AutoAwesomeOutlined` icon). Move Predict icon to `BatchPredictionOutlined`.

**Layout** — two-column:
- **Left (60%)**: Dataset selector → `ArchitectureGraph` visualization → architecture summary chips + param counts → "Send to Training" button
- **Right (40%)**: `DesignHelper` chat. Enhanced with "Suggest Architecture" button calling `suggestArchitecture(datasetId, prompt)`. AI suggestions render in the graph in real-time. "Apply & Train" navigates to `/train?dataset={id}`.

**State sharing**: New `context/DesignContext.tsx` holding current `Architecture` object. Design page writes, Train page reads. SessionStorage fallback for cross-navigation.

**Train page changes**: Remove DesignHelper sidebar toggle from `TrainTab`. Replace with "Open AI Architect →" link to `/design`. Keep architecture display as read-only.

**Files**: New `views/DesignPage.tsx`, `app/(authenticated)/design/page.tsx`, `context/DesignContext.tsx`. Modified `app/(authenticated)/layout.tsx` (nav), `components/TrainTab.tsx` (remove helper sidebar).

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

**DataPage.tsx**: MUI `Alert` (severity="info") at top of S3 Storage tab: "S3 storage requires AWS or S3-compatible credentials. Configure in Settings. This is optional — core training and prediction work without it."

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

**Decision**: Keep localStorage for JWT. Add strict CSP headers in `next.config.ts` via `headers()`:
- `Content-Security-Policy`: `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self' ws: wss:; img-src 'self' data: https://img.shields.io`
- `Referrer-Policy`: `strict-origin-when-cross-origin`
- `X-Content-Type-Options`: `nosniff`

**Files**: Modified `next.config.ts` (or `next.config.js` — check which exists).

### C2. Accessibility Pass

**Skip-nav**: Visually-hidden "Skip to main content" link as first focusable element in `app/(authenticated)/layout.tsx`. Target: `<Box component="main" id="main-content">`.

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

After B3, `train/page.tsx` DesignHelper state moves to `DesignPage.tsx`. All route files become:
```tsx
"use client";
import XPage from "@/views/XPage";
export default function XRoute() { return <XPage />; }
```

Verify all routes match this pattern after B3 changes.

### C4. Remove Dead CSS

**Delete**:
- `views/AuthPages.css` — zero imports
- `views/DashboardPage.css` — zero imports

**Keep**:
- `views/S3ManagerPage.css` — imported by `S3ManagerPage.tsx`
- `app/globals.css` — imported by layout

**Check**: `src/index.css` — verify if imported anywhere; delete if dead.

---

## Group A: Virality & First Impression

### A1. Landing Page

**Route**: `app/page.tsx` — client component wrapper. Auth check: if authenticated redirect to `/dashboard`, else render `<LandingPage />`. Landing page content itself is a server component imported by the wrapper.

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
2. `html2canvas` converts to PNG
3. Downloads as `{model-name}-results.png`
4. Optional "Copy to clipboard" via `navigator.clipboard.write()`

**Share card design**: 600x400px, dark background (#0a0a0a), accent (#7EB8FF) accuracy number, architecture chips row, mini loss sparkline, "whitematter" branding.

**New dependency**: `html2canvas`

**Files**: New `components/ShareCard.tsx`. Modified `components/ModelsTab.tsx`. Updated `package.json`.

---

## New Files Summary

| File | Group | Purpose |
|------|-------|---------|
| `components/OnboardingWizard.tsx` | B1 | First-run wizard |
| `lib/quickStart.ts` | B2 | Quick start config constants |
| `views/DesignPage.tsx` | B3 | Dedicated AI architect page |
| `app/(authenticated)/design/page.tsx` | B3 | Design route wrapper |
| `context/DesignContext.tsx` | B3 | Shared architecture state |
| `components/ParamTooltip.tsx` | B4 | Tooltip component |
| `lib/paramTooltips.ts` | B4 | Tooltip content map |
| `lib/trainingErrors.ts` | B6 | Error pattern matching |
| `views/LandingPage.tsx` | A1 | Public landing page |
| `public/og-image.png` | A2 | Social sharing image |
| `components/ShareCard.tsx` | A3 | Shareable results card |

## Modified Files Summary

| File | Groups | Changes |
|------|--------|---------|
| `views/DashboardPage.tsx` | B1, B2 | Onboarding wizard, quick start button |
| `app/(authenticated)/layout.tsx` | B3, C2 | Nav update (Design route), skip-nav, ARIA |
| `components/TrainTab.tsx` | B3, B4, B6, C2 | Remove design sidebar, add tooltips, error parsing, aria-live |
| `views/DataPage.tsx` | B5 | AWS optional badge |
| `components/ModelsTab.tsx` | B5, A3 | AWS badge, share button |
| `components/DatasetsTab.tsx` | C2 | ARIA labels, keyboard nav on upload |
| `components/TrainingChart.tsx` | C2 | ARIA label |
| `app/layout.tsx` | A2 | OG + social meta tags |
| `app/page.tsx` | A1 | Landing page (replaces redirect) |
| `next.config.ts` | C1 | CSP headers |

## Deleted Files

| File | Reason |
|------|--------|
| `views/AuthPages.css` | Zero imports — dead code |
| `views/DashboardPage.css` | Zero imports — dead code |
| `src/index.css` | Verify; delete if no imports |
