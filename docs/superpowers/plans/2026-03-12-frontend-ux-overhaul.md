# Frontend UX Overhaul Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overhaul the whitematter frontend for better onboarding, UX workflow, accessibility, security headers, and virality — 12 items across 3 groups.

**Architecture:** Next.js 15 App Router + React 19 + MUI 7. All page logic lives in `views/`, route files are thin wrappers. Shared state via React context. No backend changes.

**Tech Stack:** Next.js 15, React 19, MUI 7, TypeScript, recharts, html2canvas (new), vitest + testing-library

**Spec:** `docs/superpowers/specs/2026-03-12-frontend-ux-overhaul-design.md`

---

## Chunk 1: Foundation — Utilities, Context, and Small Components (B2, B4, B6)

These are leaf-node files with no dependencies on other new code. Build them first so everything downstream can import them.

### Task 1: Quick Start Config (B2)

**Files:**
- Create: `frontend/src/lib/quickStart.ts`
- Test: `frontend/src/__tests__/quickStart.test.ts`

- [ ] **Step 1: Write test for quick start architecture shape**

```ts
// frontend/src/__tests__/quickStart.test.ts
import { describe, it, expect } from "vitest";
import { QUICK_START_ARCHITECTURE, QUICK_START_DATASET_HF_ID } from "@/lib/quickStart";

describe("QUICK_START_ARCHITECTURE", () => {
  it("has required Architecture fields", () => {
    expect(QUICK_START_ARCHITECTURE.name).toBe("MNIST Digit Classifier");
    expect(QUICK_START_ARCHITECTURE.data_type).toBe("image");
    expect(QUICK_START_ARCHITECTURE.input_shape).toEqual([1, 28, 28]);
    expect(QUICK_START_ARCHITECTURE.num_classes).toBe(10);
    expect(QUICK_START_ARCHITECTURE.layers.length).toBeGreaterThan(0);
    expect(QUICK_START_ARCHITECTURE.training.optimizer.type).toBe("adam");
    expect(QUICK_START_ARCHITECTURE.training.epochs).toBe(10);
    expect(QUICK_START_ARCHITECTURE.training.batch_size).toBe(32);
  });

  it("exports HuggingFace dataset ID", () => {
    expect(QUICK_START_DATASET_HF_ID).toBe("ylecun/mnist");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/__tests__/quickStart.test.ts`
Expected: FAIL — module not found

- [ ] **Step 3: Implement quickStart.ts**

```ts
// frontend/src/lib/quickStart.ts
import type { Architecture } from "@/api";

export const QUICK_START_DATASET_HF_ID = "ylecun/mnist";
export const QUICK_START_DATASET_NAME = "MNIST Sample";

export const QUICK_START_ARCHITECTURE: Architecture = {
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

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/__tests__/quickStart.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/quickStart.ts frontend/src/__tests__/quickStart.test.ts
git commit -m "feat(B2): add quick start architecture config"
```

---

### Task 2: Training Error Parser (B6)

**Files:**
- Create: `frontend/src/lib/trainingErrors.ts`
- Test: `frontend/src/__tests__/trainingErrors.test.ts`

- [ ] **Step 1: Write tests for error pattern matching**

```ts
// frontend/src/__tests__/trainingErrors.test.ts
import { describe, it, expect } from "vitest";
import { parseTrainingError } from "@/lib/trainingErrors";

describe("parseTrainingError", () => {
  it("parses dimension mismatch", () => {
    const result = parseTrainingError("Error: dimension mismatch at layer 3");
    expect(result.friendly).toContain("incompatible layer dimensions");
    expect(result.raw).toContain("dimension mismatch");
  });

  it("parses out of memory", () => {
    const result = parseTrainingError("CUDA out of memory");
    expect(result.friendly).toContain("memory");
  });

  it("parses OOM", () => {
    const result = parseTrainingError("OOM when allocating tensor");
    expect(result.friendly).toContain("memory");
  });

  it("parses CUDA error", () => {
    const result = parseTrainingError("CUDA error: device-side assert");
    expect(result.friendly).toContain("GPU error");
  });

  it("parses NaN loss", () => {
    const result = parseTrainingError("loss is nan at epoch 5");
    expect(result.friendly).toContain("diverged");
  });

  it("parses invalid argument", () => {
    const result = parseTrainingError("invalid argument for kernel_size");
    expect(result.friendly).toContain("Invalid training parameter");
  });

  it("falls back to raw message for unknown errors", () => {
    const result = parseTrainingError("something unexpected happened");
    expect(result.friendly).toBe("something unexpected happened");
    expect(result.raw).toBe("something unexpected happened");
  });

  it("is case-insensitive", () => {
    const result = parseTrainingError("DIMENSION MISMATCH error");
    expect(result.friendly).toContain("incompatible layer dimensions");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/__tests__/trainingErrors.test.ts`
Expected: FAIL — module not found

- [ ] **Step 3: Implement trainingErrors.ts**

```ts
// frontend/src/lib/trainingErrors.ts

interface ParsedError {
  friendly: string;
  raw: string;
}

const ERROR_PATTERNS: { pattern: RegExp; message: string }[] = [
  {
    pattern: /dimension mismatch/i,
    message: "Your architecture has incompatible layer dimensions. Check that output sizes match input sizes between layers.",
  },
  {
    pattern: /out of memory|oom/i,
    message: "Not enough memory for this configuration. Try reducing batch size or simplifying the architecture.",
  },
  {
    pattern: /cuda error/i,
    message: "GPU error occurred. Try restarting training or switching to CPU.",
  },
  {
    pattern: /loss is nan|nan/i,
    message: "Training diverged (loss became NaN). Try a lower learning rate.",
  },
  {
    pattern: /invalid argument/i,
    message: "Invalid training parameter. Check your architecture configuration.",
  },
];

export function parseTrainingError(raw: string): ParsedError {
  for (const { pattern, message } of ERROR_PATTERNS) {
    if (pattern.test(raw)) {
      return { friendly: message, raw };
    }
  }
  return { friendly: raw, raw };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/__tests__/trainingErrors.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/trainingErrors.ts frontend/src/__tests__/trainingErrors.test.ts
git commit -m "feat(B6): add training error message parser"
```

---

### Task 3: Param Tooltips Data + Component (B4)

**Files:**
- Create: `frontend/src/lib/paramTooltips.ts`
- Create: `frontend/src/components/ParamTooltip.tsx`
- Test: `frontend/src/__tests__/paramTooltips.test.ts`

- [ ] **Step 1: Write test for tooltip data and component**

```ts
// frontend/src/__tests__/paramTooltips.test.ts
import { describe, it, expect } from "vitest";
import { PARAM_TOOLTIPS } from "@/lib/paramTooltips";

describe("PARAM_TOOLTIPS", () => {
  it("has entries for all training params", () => {
    const keys = ["optimizer", "scheduler", "batch_size", "learning_rate", "augmentations"];
    for (const key of keys) {
      expect(PARAM_TOOLTIPS[key]).toBeDefined();
      expect(PARAM_TOOLTIPS[key].description).toBeTruthy();
      expect(PARAM_TOOLTIPS[key].range).toBeTruthy();
    }
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/__tests__/paramTooltips.test.ts`
Expected: FAIL — module not found

- [ ] **Step 3: Implement paramTooltips.ts**

```ts
// frontend/src/lib/paramTooltips.ts

export interface ParamTooltipData {
  description: string;
  range: string;
}

export const PARAM_TOOLTIPS: Record<string, ParamTooltipData> = {
  optimizer: {
    description: "Controls how model weights are updated during training. Adam is a good default for most tasks.",
    range: "Adam, SGD, AdamW",
  },
  scheduler: {
    description: "Adjusts the learning rate during training to improve convergence.",
    range: "StepLR, CosineAnnealing, None",
  },
  batch_size: {
    description: "Number of samples processed before updating weights. Larger batches train faster but use more memory.",
    range: "16–128, default 32",
  },
  learning_rate: {
    description: "Step size for weight updates. Too high causes instability, too low causes slow training.",
    range: "0.0001–0.01, default 0.001",
  },
  augmentations: {
    description: "Random transforms applied to training data to prevent overfitting and improve generalization.",
    range: "RandomFlip, RandomRotation for images",
  },
};
```

- [ ] **Step 4: Implement ParamTooltip.tsx component**

```tsx
// frontend/src/components/ParamTooltip.tsx
"use client";
import IconButton from "@mui/material/IconButton";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import HelpOutlineOutlined from "@mui/icons-material/HelpOutlineOutlined";
import { PARAM_TOOLTIPS } from "@/lib/paramTooltips";

interface Props {
  paramKey: string;
}

export default function ParamTooltip({ paramKey }: Props) {
  const data = PARAM_TOOLTIPS[paramKey];
  if (!data) return null;

  return (
    <Tooltip
      title={
        <Box sx={{ p: 0.5 }}>
          <Typography variant="body2" sx={{ mb: 0.5 }}>
            {data.description}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Recommended: {data.range}
          </Typography>
        </Box>
      }
      arrow
      placement="top"
    >
      <IconButton
        size="small"
        aria-label={`Info about ${paramKey.replace(/_/g, " ")}`}
        sx={{ ml: 0.5, p: 0.25, color: "text.secondary" }}
      >
        <HelpOutlineOutlined sx={{ fontSize: 16 }} />
      </IconButton>
    </Tooltip>
  );
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/__tests__/paramTooltips.test.ts`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/paramTooltips.ts frontend/src/components/ParamTooltip.tsx frontend/src/__tests__/paramTooltips.test.ts
git commit -m "feat(B4): add param tooltips data and component"
```

---

### Task 4: Design Context (B3 prerequisite)

**Files:**
- Create: `frontend/src/context/DesignContext.tsx`
- Modify: `frontend/src/app/providers.tsx`

- [ ] **Step 1: Create DesignContext**

```tsx
// frontend/src/context/DesignContext.tsx
"use client";
import { createContext, useContext, useState, useCallback, ReactNode } from "react";
import type { Architecture } from "@/api";

const SESSION_KEY = "wm_design_architecture";

interface DesignContextType {
  architecture: Architecture | null;
  setArchitecture: (arch: Architecture | null) => void;
  clearArchitecture: () => void;
}

const DesignContext = createContext<DesignContextType | null>(null);

export function DesignProvider({ children }: { children: ReactNode }) {
  const [architecture, setArchitectureState] = useState<Architecture | null>(() => {
    if (typeof window === "undefined") return null;
    try {
      const stored = sessionStorage.getItem(SESSION_KEY);
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });

  const setArchitecture = useCallback((arch: Architecture | null) => {
    setArchitectureState(arch);
    if (typeof window !== "undefined") {
      if (arch) {
        sessionStorage.setItem(SESSION_KEY, JSON.stringify(arch));
      } else {
        sessionStorage.removeItem(SESSION_KEY);
      }
    }
  }, []);

  const clearArchitecture = useCallback(() => {
    setArchitectureState(null);
    if (typeof window !== "undefined") {
      sessionStorage.removeItem(SESSION_KEY);
    }
  }, []);

  return (
    <DesignContext.Provider value={{ architecture, setArchitecture, clearArchitecture }}>
      {children}
    </DesignContext.Provider>
  );
}

export function useDesign() {
  const ctx = useContext(DesignContext);
  if (!ctx) throw new Error("useDesign must be inside DesignProvider");
  return ctx;
}
```

- [ ] **Step 2: Add DesignProvider to providers.tsx**

In `frontend/src/app/providers.tsx`, import `DesignProvider` and wrap it inside `AuthProvider`:

```tsx
// Add import at top:
import { DesignProvider } from "@/context/DesignContext";

// Wrap children inside AuthProvider:
<AuthProvider>
  <DesignProvider>{children}</DesignProvider>
</AuthProvider>
```

- [ ] **Step 3: Verify build compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/context/DesignContext.tsx frontend/src/app/providers.tsx
git commit -m "feat(B3): add DesignContext for architecture state sharing"
```

---

## Chunk 2: Nav + Sidebar Update, AWS Badge, Tooltips Integration (B3 nav, B5, B4 integration)

### Task 5: Update Sidebar Nav — Add Design, Fix Icons (B3)

**Files:**
- Modify: `frontend/src/app/(authenticated)/layout.tsx`

- [ ] **Step 1: Update nav array and imports**

In `frontend/src/app/(authenticated)/layout.tsx`:

Add import at top:
```tsx
import BatchPredictionOutlined from "@mui/icons-material/BatchPredictionOutlined";
import ArchitectureOutlined from "@mui/icons-material/ArchitectureOutlined";
```

Replace the `NAV` array (lines 23-30) with:
```tsx
const NAV: { href: string; label: string; icon: React.ReactNode }[] = [
  { href: "/dashboard", label: "Dashboard", icon: <DashboardOutlined fontSize="small" /> },
  { href: "/data", label: "Data", icon: <DatasetOutlined fontSize="small" /> },
  { href: "/architect", label: "Design", icon: <AutoAwesomeOutlined fontSize="small" /> },
  { href: "/train", label: "Train", icon: <PlayArrowOutlined fontSize="small" /> },
  { href: "/models", label: "Models", icon: <CategoryOutlined fontSize="small" /> },
  { href: "/predict", label: "Predict", icon: <BatchPredictionOutlined fontSize="small" /> },
  { href: "/settings", label: "Settings", icon: <SettingsOutlined fontSize="small" /> },
];
```

Remove unused `ArchitectureOutlined` import if not needed. Keep `AutoAwesomeOutlined` (now used for Design).

- [ ] **Step 2: Add aria-labels to nav links**

In the nav link rendering (around line 112-158), add `aria-label={label}` to each `<Box component={Link}>`:

```tsx
<Box
  component={Link}
  href={href}
  aria-label={label}
  sx={{...}}
>
```

- [ ] **Step 3: Add skip-nav link and main content ID**

At the very top of the return JSX (inside `<ErrorBoundary>`, before the flex container), add:

```tsx
<Box
  component="a"
  href="#main-content"
  sx={{
    position: "absolute",
    left: "-9999px",
    top: "auto",
    width: "1px",
    height: "1px",
    overflow: "hidden",
    "&:focus": {
      position: "fixed",
      top: 8,
      left: 8,
      width: "auto",
      height: "auto",
      overflow: "visible",
      zIndex: 9999,
      bgcolor: "primary.main",
      color: "primary.contrastText",
      px: 2,
      py: 1,
      borderRadius: 1,
      fontSize: "0.875rem",
      fontWeight: 600,
      textDecoration: "none",
      outline: "2px solid",
      outlineColor: "primary.main",
      outlineOffset: 2,
    },
  }}
>
  Skip to main content
</Box>
```

On the `<Box component="main">` (around line 190), add `id="main-content"`:

```tsx
<Box
  component="main"
  id="main-content"
  sx={{...}}
>
```

- [ ] **Step 4: Verify build compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/\(authenticated\)/layout.tsx
git commit -m "feat(B3,C2): update sidebar nav with Design route, add skip-nav and aria-labels"
```

---

### Task 6: AWS Optional Badges (B5)

**Files:**
- Modify: `frontend/src/views/DataPage.tsx`
- Modify: `frontend/src/components/ModelsTab.tsx`

- [ ] **Step 1: Add AWS badge to DataPage S3 tab**

In `frontend/src/views/DataPage.tsx`, add import:
```tsx
import Alert from "@mui/material/Alert";
import Link from "next/link";
```

Replace the `{tab === "storage" && <S3ManagerPage />}` block (around line 38) with:
```tsx
{tab === "storage" && (
  <>
    <Alert
      severity="info"
      sx={{ mb: 2 }}
    >
      S3 storage requires AWS or S3-compatible credentials.{" "}
      <Link href="/settings" style={{ color: "inherit", fontWeight: 600 }}>
        Configure in Settings
      </Link>
      . This is optional — core training and prediction work without it.
    </Alert>
    <S3ManagerPage />
  </>
)}
```

- [ ] **Step 2: Add AWS badge to ModelsTab deploy section**

In `frontend/src/components/ModelsTab.tsx`, find the "Deploy to API" button (around line 547-549). Add a caption below it:

```tsx
<Button variant="contained" onClick={() => setDeployModalOpen(true)}>
  Deploy to API
</Button>
<Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
  Requires AWS credentials (optional).{" "}
  <Link href="/settings" style={{ color: "inherit", textDecoration: "underline" }}>
    Settings
  </Link>
</Typography>
```

Make sure `Link` from `next/link` is imported (it already is at line 3).

- [ ] **Step 3: Verify build compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/views/DataPage.tsx frontend/src/components/ModelsTab.tsx
git commit -m "feat(B5): add AWS optional badges on S3 and deploy sections"
```

---

### Task 7: Integrate Tooltips into TrainTab (B4)

**Files:**
- Modify: `frontend/src/components/TrainTab.tsx`

- [ ] **Step 1: Add ParamTooltip imports to TrainTab**

At the top of `frontend/src/components/TrainTab.tsx`, add:
```tsx
import ParamTooltip from "./ParamTooltip";
```

- [ ] **Step 2: Add tooltips next to parameter labels**

Find each `<InputLabel>` or `<Typography>` for: Optimizer, Scheduler, Batch Size, Learning Rate, and Augmentations sections. Add `<ParamTooltip paramKey="..." />` inline next to each label.

For example, next to the Optimizer `<InputLabel>`:
```tsx
<Box sx={{ display: "flex", alignItems: "center" }}>
  <InputLabel>Optimizer</InputLabel>
  <ParamTooltip paramKey="optimizer" />
</Box>
```

Repeat for:
- `scheduler` — next to Scheduler label
- `batch_size` — next to Batch Size label
- `learning_rate` — next to Learning Rate label
- `augmentations` — next to Augmentations label

- [ ] **Step 3: Verify build compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/TrainTab.tsx
git commit -m "feat(B4): integrate param tooltips into training form"
```

---

### Task 8: Integrate Error Parser into TrainTab (B6)

**Files:**
- Modify: `frontend/src/components/TrainTab.tsx`

- [ ] **Step 1: Import error parser**

At the top of `TrainTab.tsx`, add:
```tsx
import { parseTrainingError } from "@/lib/trainingErrors";
```

- [ ] **Step 2: Replace raw error display with parsed errors**

Find where training failure messages are displayed (search for status === "failed" or error display). Replace the raw error text rendering with:

```tsx
{(() => {
  const parsed = parseTrainingError(errorMessage);
  return (
    <Box>
      <Typography color="error.main" variant="body2">
        {parsed.friendly}
      </Typography>
      {parsed.friendly !== parsed.raw && (
        <details style={{ marginTop: 8 }}>
          <summary style={{ cursor: "pointer", fontSize: "0.75rem", color: "inherit", opacity: 0.6 }}>
            Show raw output
          </summary>
          <Box
            component="pre"
            sx={{
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: "0.75rem",
              p: 1,
              mt: 0.5,
              bgcolor: "action.hover",
              borderRadius: 1,
              overflow: "auto",
              whiteSpace: "pre-wrap",
            }}
          >
            {parsed.raw}
          </Box>
        </details>
      )}
    </Box>
  );
})()}
```

- [ ] **Step 3: Add aria-live to training progress section**

Find the training progress display area (epoch counter, loss, accuracy). Wrap it with:
```tsx
<Box aria-live="polite" role="status">
  {/* existing progress content */}
</Box>
```

- [ ] **Step 4: Verify build compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/TrainTab.tsx
git commit -m "feat(B6,C2): integrate error parser and aria-live in TrainTab"
```

---

## Chunk 3: Architect Page + Train Page Refactor (B3, C3)

### Task 9: Create ArchitectPage (B3)

**Files:**
- Create: `frontend/src/views/ArchitectPage.tsx`
- Create: `frontend/src/app/(authenticated)/architect/page.tsx`

- [ ] **Step 1: Create the Architect route wrapper**

```tsx
// frontend/src/app/(authenticated)/architect/page.tsx
"use client";
import ArchitectPage from "@/views/ArchitectPage";

export default function ArchitectRoute() {
  return <ArchitectPage />;
}
```

- [ ] **Step 2: Create ArchitectPage.tsx**

```tsx
// frontend/src/views/ArchitectPage.tsx
"use client";
import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { useRouter } from "next/navigation";
import * as api from "@/api";
import { useDesign } from "@/context/DesignContext";
import DesignHelper from "@/components/DesignHelper";
import ArchitectureGraph from "@/components/ArchitectureGraph";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Paper from "@mui/material/Paper";
import Chip from "@mui/material/Chip";
import Alert from "@mui/material/Alert";

export default function ArchitectPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { architecture, setArchitecture } = useDesign();

  const [datasets, setDatasets] = useState<api.CustomDataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>(
    searchParams.get("dataset") || ""
  );
  const [loading, setLoading] = useState(true);
  const [suggesting, setSuggesting] = useState(false);
  const [error, setError] = useState("");
  const [chatMessages, setChatMessages] = useState<
    { role: "user" | "assistant"; content: string }[]
  >([]);

  useEffect(() => {
    api
      .getCustomDatasets()
      .then((ds) => {
        setDatasets(ds.filter((d) => d.status === "ready"));
      })
      .catch(() => setError("Failed to load datasets"))
      .finally(() => setLoading(false));
  }, []);

  const selectedDataset = datasets.find((d) => d.id === selectedDatasetId);

  async function handleSuggest() {
    if (!selectedDatasetId) return;
    setSuggesting(true);
    setError("");
    try {
      const result = await api.suggestArchitecture(
        selectedDatasetId,
        "Suggest a good architecture for this dataset"
      );
      setArchitecture(result.architecture);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to suggest architecture");
    } finally {
      setSuggesting(false);
    }
  }

  function handleSendToTraining() {
    if (!architecture || !selectedDatasetId) return;
    router.push(`/train?dataset=${selectedDatasetId}`);
  }

  return (
    <Box>
      <Typography variant="h2" sx={{ mb: 0.5 }}>
        AI Architecture Designer
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Describe what you want to build. Claude designs the neural network.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError("")}>
          {error}
        </Alert>
      )}

      <Box sx={{ display: "flex", gap: 3, alignItems: "flex-start" }}>
        {/* Left column: dataset + architecture */}
        <Box sx={{ flex: "0 0 60%", minWidth: 0 }}>
          <FormControl fullWidth size="small" sx={{ mb: 2 }}>
            <InputLabel>Dataset</InputLabel>
            <Select
              value={selectedDatasetId}
              label="Dataset"
              onChange={(e) => setSelectedDatasetId(e.target.value)}
              disabled={loading}
            >
              {datasets.map((d) => (
                <MenuItem key={d.id} value={d.id}>
                  {d.name} ({d.data_type}, {d.num_classes} classes)
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {!selectedDatasetId && (
            <Typography color="text.secondary" sx={{ py: 4, textAlign: "center" }}>
              Select a dataset to start designing an architecture.
            </Typography>
          )}

          {selectedDatasetId && !architecture && (
            <Paper
              variant="outlined"
              sx={{ p: 3, textAlign: "center", borderColor: "divider" }}
            >
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Use the chat on the right to describe your model, or let AI suggest one.
              </Typography>
              <Button
                variant="contained"
                onClick={handleSuggest}
                disabled={suggesting}
              >
                {suggesting ? "Generating..." : "Suggest Architecture"}
              </Button>
            </Paper>
          )}

          {architecture && (
            <Box>
              <Paper variant="outlined" sx={{ p: 2, mb: 2, borderColor: "divider" }}>
                <Typography variant="h3" sx={{ mb: 1 }}>
                  {architecture.name}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
                  {architecture.description}
                </Typography>
                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mb: 1.5 }}>
                  {architecture.layers.map((layer, i) => (
                    <Box key={i} sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                      <Chip
                        size="small"
                        label={layer.type}
                        sx={{
                          fontFamily: '"JetBrains Mono", monospace',
                          fontSize: "0.6875rem",
                        }}
                      />
                      {i < architecture.layers.length - 1 && (
                        <Typography component="span" color="text.secondary" sx={{ fontSize: "0.75rem" }}>
                          →
                        </Typography>
                      )}
                    </Box>
                  ))}
                </Box>
                <ArchitectureGraph architecture={architecture} />
              </Paper>

              <Box sx={{ display: "flex", gap: 1 }}>
                <Button
                  variant="contained"
                  onClick={handleSendToTraining}
                  disabled={!selectedDatasetId}
                >
                  Send to Training
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => setArchitecture(null)}
                >
                  Clear
                </Button>
              </Box>
            </Box>
          )}
        </Box>

        {/* Right column: chat */}
        <Box sx={{ flex: "0 0 40%", minWidth: 0, position: "sticky", top: 24 }}>
          <Paper variant="outlined" sx={{ borderColor: "divider", overflow: "hidden" }}>
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                p: 1.25,
                borderBottom: "1px solid",
                borderColor: "divider",
              }}
            >
              <Typography variant="subtitle1" fontWeight={600}>
                AI Design Assistant
              </Typography>
            </Box>
            <DesignHelper
              datasetType={selectedDataset?.data_type}
              currentArchitecture={architecture}
              messages={chatMessages}
              onMessagesChange={setChatMessages}
            />
          </Paper>
        </Box>
      </Box>
    </Box>
  );
}
```

- [ ] **Step 3: Verify build compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors (may have warnings about ArchitectureGraph props — check and adapt to actual component signature)

- [ ] **Step 4: Commit**

```bash
git add frontend/src/views/ArchitectPage.tsx frontend/src/app/\(authenticated\)/architect/page.tsx
git commit -m "feat(B3): add dedicated /architect page for AI architecture designer"
```

---

### Task 10: Refactor Train Page — Remove Design Sidebar, Simplify Route (B3, C3)

**Files:**
- Create: `frontend/src/views/TrainPage.tsx`
- Modify: `frontend/src/app/(authenticated)/train/page.tsx`
- Modify: `frontend/src/components/TrainTab.tsx`

- [ ] **Step 1: Remove helper props from TrainTab interface**

In `frontend/src/components/TrainTab.tsx`, remove the helper-related props from the `Props` interface (lines 33-35):
```tsx
// REMOVE these three props:
helperOpen?: boolean;
onHelperToggle?: (open: boolean) => void;
onHelperContextChange?: (context: { datasetType?: string; architecture?: api.Architecture | null }) => void;
```

And from the destructured params (lines 43-45). Remove any code that references `helperOpen`, `onHelperToggle`, or `onHelperContextChange`.

Add a link to the architect page. Find where the design helper toggle button was (search for `onHelperToggle` usage) and replace with:
```tsx
<Button
  variant="outlined"
  component={Link}
  href="/architect"
  sx={{ textDecoration: "none" }}
>
  Open AI Architect →
</Button>
```

Add `import Link from "next/link";` at the top if not already imported.

- [ ] **Step 2: Create views/TrainPage.tsx**

Extract the dataset-loading logic from the current `train/page.tsx` into a proper view:

```tsx
// frontend/src/views/TrainPage.tsx
"use client";
import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import TrainTab from "@/components/TrainTab";
import { getCustomDatasets } from "@/api";
import type { CustomDataset } from "@/api";
import Box from "@mui/material/Box";

export default function TrainPage() {
  const searchParams = useSearchParams();
  const [datasets, setDatasets] = useState<CustomDataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(
    searchParams.get("dataset") || null
  );

  const loadData = async () => {
    try {
      const datasetsData = await getCustomDatasets();
      setDatasets(datasetsData);
    } catch (err) {
      console.error("Failed to load data:", err);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const readyDatasets = datasets.filter((d) => d.status === "ready");

  return (
    <Box sx={{ width: "100%", maxWidth: "100%", p: 0 }}>
      <TrainTab
        datasets={readyDatasets}
        selectedDataset={selectedDataset}
        onDatasetChange={setSelectedDataset}
        onTrainingComplete={loadData}
      />
    </Box>
  );
}
```

- [ ] **Step 3: Simplify train/page.tsx to thin wrapper**

```tsx
// frontend/src/app/(authenticated)/train/page.tsx
"use client";
import TrainPage from "@/views/TrainPage";

export default function TrainRoute() {
  return <TrainPage />;
}
```

- [ ] **Step 4: Verify build compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/views/TrainPage.tsx frontend/src/app/\(authenticated\)/train/page.tsx frontend/src/components/TrainTab.tsx
git commit -m "feat(B3,C3): extract TrainPage view, remove design sidebar from TrainTab"
```

---

### Task 11: Extract PredictPage View (C3)

**Files:**
- Create: `frontend/src/views/PredictPage.tsx`
- Modify: `frontend/src/app/(authenticated)/predict/page.tsx`

- [ ] **Step 1: Create views/PredictPage.tsx**

Move logic from `predict/page.tsx`:

```tsx
// frontend/src/views/PredictPage.tsx
"use client";
import { useState, useEffect } from "react";
import PredictTab from "@/components/PredictTab";
import { getModels } from "@/api";
import type { Model } from "@/api";

export default function PredictPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  useEffect(() => {
    getModels()
      .then((data) => setModels(data.filter((m) => m.status === "completed")))
      .catch(() => {});
  }, []);

  return (
    <PredictTab
      models={models}
      selectedModel={selectedModel}
      onModelChange={setSelectedModel}
    />
  );
}
```

- [ ] **Step 2: Simplify predict/page.tsx**

```tsx
// frontend/src/app/(authenticated)/predict/page.tsx
"use client";
import PredictPage from "@/views/PredictPage";

export default function PredictRoute() {
  return <PredictPage />;
}
```

- [ ] **Step 3: Verify build compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/views/PredictPage.tsx frontend/src/app/\(authenticated\)/predict/page.tsx
git commit -m "refactor(C3): extract PredictPage view, thin route wrapper"
```

---

## Chunk 4: Onboarding Wizard + Quick Start (B1, B2)

### Task 12: OnboardingWizard Component (B1)

**Files:**
- Create: `frontend/src/components/OnboardingWizard.tsx`

- [ ] **Step 1: Create OnboardingWizard**

```tsx
// frontend/src/components/OnboardingWizard.tsx
"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import NextLink from "next/link";
import * as api from "@/api";
import {
  QUICK_START_DATASET_HF_ID,
  QUICK_START_DATASET_NAME,
} from "@/lib/quickStart";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Stepper from "@mui/material/Stepper";
import Step from "@mui/material/Step";
import StepLabel from "@mui/material/StepLabel";
import Alert from "@mui/material/Alert";
import CircularProgress from "@mui/material/CircularProgress";
import Link from "@mui/material/Link";

const STEPS = [
  "Upload your first dataset",
  "Design your architecture",
  "Train your model",
  "Make a prediction",
];

interface Props {
  userId: string;
  onDatasetImported: (dataset: api.CustomDataset) => void;
}

function getStorageKey(userId: string) {
  return `wm_onboarding_${userId}`;
}

function getSavedStep(userId: string): number {
  if (typeof window === "undefined") return 0;
  const saved = localStorage.getItem(getStorageKey(userId));
  return saved ? parseInt(saved, 10) : 0;
}

function saveStep(userId: string, step: number) {
  if (typeof window === "undefined") return;
  localStorage.setItem(getStorageKey(userId), String(step));
}

export default function OnboardingWizard({ userId, onDatasetImported }: Props) {
  const router = useRouter();
  const [activeStep, setActiveStep] = useState(getSavedStep(userId));
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState("");
  const [importedDataset, setImportedDataset] = useState<api.CustomDataset | null>(null);

  function goToStep(step: number) {
    setActiveStep(step);
    saveStep(userId, step);
  }

  async function handleImportSample() {
    setImporting(true);
    setError("");
    try {
      const dataset = await api.importDatasetFromHuggingFace(QUICK_START_DATASET_HF_ID, {
        name: QUICK_START_DATASET_NAME,
        split: "train",
      });
      setImportedDataset(dataset);
      onDatasetImported(dataset);
      goToStep(1);
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Failed to import sample dataset. Please try again."
      );
    } finally {
      setImporting(false);
    }
  }

  return (
    <Box>
      <Typography variant="h2" sx={{ mb: 0.5 }}>
        Welcome to whitematter
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Let&apos;s get you up and running in a few steps.
      </Typography>

      <Stepper activeStep={activeStep} sx={{ mb: 3 }}>
        {STEPS.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError("")}>
          {error}
          <Box sx={{ mt: 1 }}>
            <Button size="small" variant="outlined" onClick={handleImportSample} sx={{ mr: 1 }}>
              Try again
            </Button>
            <Link component={NextLink} href="/data" color="inherit" sx={{ fontSize: "0.875rem" }}>
              Upload manually
            </Link>
          </Box>
        </Alert>
      )}

      {activeStep === 0 && (
        <Box sx={{ textAlign: "center", py: 3 }}>
          <Typography variant="body1" sx={{ mb: 2 }}>
            Start by uploading a dataset, or use our sample MNIST dataset to get started instantly.
          </Typography>
          <Box sx={{ display: "flex", gap: 2, justifyContent: "center", flexWrap: "wrap" }}>
            <Button
              variant="contained"
              onClick={handleImportSample}
              disabled={importing}
              startIcon={importing ? <CircularProgress size={16} /> : undefined}
            >
              {importing ? "Importing MNIST..." : "Use sample dataset"}
            </Button>
            <Button variant="outlined" component={NextLink} href="/data">
              Upload your own
            </Button>
          </Box>
          <Button
            size="small"
            sx={{ mt: 2, color: "text.secondary" }}
            onClick={() => goToStep(1)}
          >
            Skip
          </Button>
        </Box>
      )}

      {activeStep === 1 && (
        <Box sx={{ textAlign: "center", py: 3 }}>
          <Typography variant="body1" sx={{ mb: 1 }}>
            Use the AI-powered architecture designer to create a neural network for your data.
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Describe what you want to build and Claude will design the architecture.
          </Typography>
          <Button
            variant="contained"
            component={NextLink}
            href={importedDataset ? `/architect?dataset=${importedDataset.id}` : "/architect"}
          >
            Open AI Architect
          </Button>
          <br />
          <Button size="small" sx={{ mt: 2, color: "text.secondary" }} onClick={() => goToStep(2)}>
            Skip
          </Button>
        </Box>
      )}

      {activeStep === 2 && (
        <Box sx={{ textAlign: "center", py: 3 }}>
          <Typography variant="body1" sx={{ mb: 1 }}>
            Configure training parameters and watch your model train in real-time.
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            You&apos;ll see a live chart with loss and accuracy as training progresses.
          </Typography>
          <Button variant="contained" component={NextLink} href="/train">
            Start Training
          </Button>
          <br />
          <Button size="small" sx={{ mt: 2, color: "text.secondary" }} onClick={() => goToStep(3)}>
            Skip
          </Button>
        </Box>
      )}

      {activeStep === 3 && (
        <Box sx={{ textAlign: "center", py: 3 }}>
          <Typography variant="body1" sx={{ mb: 1 }}>
            Test your trained model by uploading an image for prediction.
          </Typography>
          <Button variant="contained" component={NextLink} href="/predict">
            Try Prediction
          </Button>
          <br />
          <Button
            size="small"
            sx={{ mt: 2, color: "text.secondary" }}
            onClick={() => goToStep(4)}
          >
            Finish
          </Button>
        </Box>
      )}
    </Box>
  );
}
```

- [ ] **Step 2: Verify build compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/OnboardingWizard.tsx
git commit -m "feat(B1): add OnboardingWizard component"
```

---

### Task 13: Integrate Onboarding + Quick Start into Dashboard (B1, B2)

**Files:**
- Modify: `frontend/src/views/DashboardPage.tsx`

- [ ] **Step 1: Add imports**

At top of `frontend/src/views/DashboardPage.tsx`, add:
```tsx
import OnboardingWizard from "@/components/OnboardingWizard";
import { QUICK_START_ARCHITECTURE, QUICK_START_DATASET_HF_ID, QUICK_START_DATASET_NAME } from "@/lib/quickStart";
import * as api from "@/api";
import CircularProgress from "@mui/material/CircularProgress";
import RocketLaunchOutlined from "@mui/icons-material/RocketLaunchOutlined";
```

- [ ] **Step 2: Add quick start state and handler**

Inside the `DashboardPage` component, after existing state declarations, add:
```tsx
const [quickStarting, setQuickStarting] = useState(false);
const [quickStartError, setQuickStartError] = useState("");

async function handleQuickStart() {
  setQuickStarting(true);
  setQuickStartError("");
  try {
    // Check if MNIST dataset already exists
    let mnistDataset = datasets.find(
      (d) => d.name === QUICK_START_DATASET_NAME || d.name.toLowerCase().includes("mnist")
    );
    if (!mnistDataset) {
      mnistDataset = await api.importDatasetFromHuggingFace(QUICK_START_DATASET_HF_ID, {
        name: QUICK_START_DATASET_NAME,
        split: "train",
      });
    }
    // Start training
    const job = await api.startCustomTraining(mnistDataset.id, QUICK_START_ARCHITECTURE);
    // Navigate to train page
    router.push("/train");
  } catch (e) {
    setQuickStartError(e instanceof Error ? e.message : "Quick start failed");
  } finally {
    setQuickStarting(false);
  }
}
```

Add `const router = useRouter();` and `import { useRouter } from "next/navigation";` if not already present.

- [ ] **Step 3: Add onboarding wizard conditional render**

In the JSX return, after the welcome message and before the stat cards, add:

```tsx
{!loading && datasets.length === 0 && models.length === 0 && (
  <OnboardingWizard
    userId={user?.id || "anonymous"}
    onDatasetImported={(dataset) => {
      setDatasets((prev) => [dataset, ...prev]);
    }}
  />
)}
```

Only show the normal dashboard content when not in onboarding state. Wrap the existing stat cards, recent activity, and quick actions in:

```tsx
{(loading || datasets.length > 0 || models.length > 0) && (
  <>
    {/* existing stat cards, activity, quick actions */}
  </>
)}
```

- [ ] **Step 4: Add Quick Start card to Quick Actions**

In the Quick Actions section, add a new card before the existing ones:

```tsx
{models.length === 0 && (
  <Card
    variant="outlined"
    sx={{
      flex: "1 1 240px",
      maxWidth: 320,
      borderColor: "primary.main",
      borderRadius: 1,
      borderWidth: 2,
    }}
  >
    <CardActionArea onClick={handleQuickStart} disabled={quickStarting} sx={{ display: "block" }}>
      <CardContent sx={{ py: 2, "&:last-child": { pb: 2 } }}>
        <RocketLaunchOutlined
          sx={{ color: "primary.main", fontSize: 28, mb: 0.5 }}
        />
        <Typography variant="subtitle2" fontWeight={600} color="primary.main">
          {quickStarting ? "Starting..." : "Train a digit classifier in 60 seconds"}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          One click → MNIST + CNN → live training
        </Typography>
      </CardContent>
    </CardActionArea>
  </Card>
)}
```

Import `RocketLaunchOutlined` and `CardContent` if needed.

- [ ] **Step 5: Verify build compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add frontend/src/views/DashboardPage.tsx
git commit -m "feat(B1,B2): integrate onboarding wizard and quick start into dashboard"
```

---

## Chunk 5: Accessibility, CSP, Dead CSS Cleanup (C1, C2, C4)

### Task 14: Accessibility — DatasetsTab + TrainingChart (C2)

**Files:**
- Modify: `frontend/src/components/DatasetsTab.tsx`
- Modify: `frontend/src/components/TrainingChart.tsx`

- [ ] **Step 1: Add keyboard nav and ARIA to DatasetsTab upload zone**

In `frontend/src/components/DatasetsTab.tsx`, find the upload drop zone `<Box>` (around line 399-445). Add these attributes:

```tsx
<Box
  onDrop={handleDrop}
  onDragOver={handleDragOver}
  onClick={() => !file && fileInputRef.current?.click()}
  onKeyDown={(e: React.KeyboardEvent) => {
    if ((e.key === "Enter" || e.key === " ") && !file) {
      e.preventDefault();
      fileInputRef.current?.click();
    }
  }}
  tabIndex={0}
  role="button"
  aria-label="Upload dataset file"
  sx={{...existing styles...}}
>
```

Add `aria-live="polite"` to the upload status area. Find where `uploadedDataset` success message is shown and wrap it:

```tsx
<Box aria-live="polite">
  {uploadedDataset && (
    // ...existing upload success content
  )}
</Box>
```

- [ ] **Step 2: Add ARIA label to TrainingChart**

In `frontend/src/components/TrainingChart.tsx`, add `aria-label` to the outer `<Box>`:

```tsx
<Box
  aria-label="Training progress chart"
  role="img"
  sx={{...existing styles...}}
>
```

- [ ] **Step 3: Verify build compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/DatasetsTab.tsx frontend/src/components/TrainingChart.tsx
git commit -m "feat(C2): add keyboard nav, ARIA labels, and aria-live regions"
```

---

### Task 15: CSP Headers (C1)

**Files:**
- Modify: `frontend/next.config.mjs`

- [ ] **Step 1: Add security headers**

In `frontend/next.config.mjs`, add an `async headers()` function alongside the existing `rewrites()`:

```js
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  eslint: { ignoreDuringBuilds: true },
  typescript: { ignoreBuildErrors: false },
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          {
            key: "Content-Security-Policy",
            value: [
              "default-src 'self'",
              "script-src 'self'",
              "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
              "font-src 'self' https://fonts.gstatic.com",
              "connect-src 'self' ws: wss:",
              "img-src 'self' data: https://img.shields.io",
            ].join("; "),
          },
          {
            key: "Referrer-Policy",
            value: "strict-origin-when-cross-origin",
          },
          {
            key: "X-Content-Type-Options",
            value: "nosniff",
          },
        ],
      },
    ];
  },
  async rewrites() {
    // ...existing rewrites unchanged
  },
};

export default nextConfig;
```

- [ ] **Step 2: Verify dev server starts**

Run: `cd frontend && npx next dev` (start and check no CSP errors in console, then stop)

- [ ] **Step 3: Commit**

```bash
git add frontend/next.config.mjs
git commit -m "feat(C1): add CSP, Referrer-Policy, and X-Content-Type-Options headers"
```

---

### Task 16: Delete Dead CSS Files (C4)

**Files:**
- Delete: `frontend/src/views/AuthPages.css`
- Delete: `frontend/src/views/DashboardPage.css`
- Delete: `frontend/src/index.css`

- [ ] **Step 1: Delete the files**

```bash
rm frontend/src/views/AuthPages.css frontend/src/views/DashboardPage.css frontend/src/index.css
```

- [ ] **Step 2: Verify build compiles (no broken imports)**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add -u frontend/src/views/AuthPages.css frontend/src/views/DashboardPage.css frontend/src/index.css
git commit -m "chore(C4): remove dead CSS files (AuthPages, DashboardPage, index)"
```

---

## Chunk 6: Landing Page, OG Tags, Share Button (A1, A2, A3)

### Task 17: Landing Page (A1)

**Files:**
- Create: `frontend/src/views/LandingPage.tsx`
- Modify: `frontend/src/app/page.tsx`

- [ ] **Step 1: Create LandingPage.tsx**

```tsx
// frontend/src/views/LandingPage.tsx
"use client";
import NextLink from "next/link";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Container from "@mui/material/Container";
import AutoAwesomeOutlined from "@mui/icons-material/AutoAwesomeOutlined";
import ShowChartOutlined from "@mui/icons-material/ShowChartOutlined";
import RocketLaunchOutlined from "@mui/icons-material/RocketLaunchOutlined";
import Link from "@mui/material/Link";

const FEATURES = [
  {
    icon: <AutoAwesomeOutlined sx={{ fontSize: 32, color: "primary.main" }} />,
    title: "AI Architecture Designer",
    description: "Describe what you want to build. Claude designs the neural network.",
  },
  {
    icon: <ShowChartOutlined sx={{ fontSize: 32, color: "primary.main" }} />,
    title: "Live Training",
    description: "Watch your model train in real-time with live loss and accuracy charts.",
  },
  {
    icon: <RocketLaunchOutlined sx={{ fontSize: 32, color: "primary.main" }} />,
    title: "One-Click Deploy",
    description: "Deploy trained models as API endpoints instantly.",
  },
];

export default function LandingPage() {
  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "background.default" }}>
      {/* Hero */}
      <Container maxWidth="md" sx={{ pt: { xs: 8, md: 14 }, pb: 8, textAlign: "center" }}>
        <Typography
          component="span"
          sx={{
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: "1.5rem",
            fontWeight: 700,
            color: "primary.main",
            letterSpacing: "0.02em",
            display: "block",
            mb: 3,
          }}
        >
          wm
        </Typography>

        <Typography
          variant="h1"
          sx={{
            fontSize: { xs: "2rem", md: "3rem" },
            fontWeight: 700,
            letterSpacing: "-0.03em",
            mb: 2,
            color: "text.primary",
          }}
        >
          Train neural networks from your browser.
        </Typography>

        <Typography
          variant="h2"
          sx={{
            fontSize: { xs: "1.1rem", md: "1.35rem" },
            fontWeight: 400,
            color: "text.secondary",
            mb: 4,
            maxWidth: 560,
            mx: "auto",
          }}
        >
          Design architectures with AI. Deploy with one click.
        </Typography>

        <Box sx={{ display: "flex", gap: 2, justifyContent: "center", flexWrap: "wrap", mb: 3 }}>
          <Button
            variant="contained"
            size="large"
            component={NextLink}
            href="/register"
            sx={{ px: 4, py: 1.5, fontSize: "1rem" }}
          >
            Get Started
          </Button>
          <Button
            variant="outlined"
            size="large"
            component={NextLink}
            href="/login"
            sx={{ px: 4, py: 1.5, fontSize: "1rem" }}
          >
            Sign in
          </Button>
        </Box>

        {/* GitHub badge placeholder */}
        <Box sx={{ mb: 4 }}>
          <img
            src="https://img.shields.io/github/stars/user/whitematter?style=social"
            alt="GitHub stars"
            style={{ height: 20 }}
          />
        </Box>

        {/* Demo GIF placeholder */}
        <Box
          sx={{
            width: "100%",
            maxWidth: 720,
            mx: "auto",
            aspectRatio: "16/9",
            border: "1px solid",
            borderColor: "divider",
            borderRadius: 2,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            bgcolor: "background.paper",
          }}
        >
          <Typography color="text.secondary" variant="body2">
            Demo GIF
          </Typography>
        </Box>
      </Container>

      {/* Features */}
      <Container maxWidth="md" sx={{ pb: 10 }}>
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: { xs: "1fr", md: "repeat(3, 1fr)" },
            gap: 4,
          }}
        >
          {FEATURES.map((feature) => (
            <Box key={feature.title} sx={{ textAlign: "center" }}>
              {feature.icon}
              <Typography variant="h3" sx={{ mt: 1.5, mb: 1 }}>
                {feature.title}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {feature.description}
              </Typography>
            </Box>
          ))}
        </Box>
      </Container>

      {/* Footer */}
      <Box
        component="footer"
        sx={{
          py: 3,
          textAlign: "center",
          borderTop: "1px solid",
          borderColor: "divider",
        }}
      >
        <Typography variant="body2" color="text.secondary">
          Built with whitematter ·{" "}
          <Link
            href="https://github.com/user/whitematter"
            target="_blank"
            rel="noopener noreferrer"
            color="primary"
          >
            GitHub
          </Link>
        </Typography>
      </Box>
    </Box>
  );
}
```

- [ ] **Step 2: Update app/page.tsx to show landing page or redirect**

```tsx
// frontend/src/app/page.tsx
"use client";
import { useAuth } from "@/context/AuthContext";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";
import LandingPage from "@/views/LandingPage";

export default function HomePage() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (loading) return;
    if (user) {
      router.replace("/dashboard");
    }
  }, [user, loading, router]);

  if (loading) {
    return (
      <Box
        sx={{
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          bgcolor: "background.default",
        }}
      >
        <CircularProgress size={32} sx={{ color: "text.secondary" }} />
      </Box>
    );
  }

  if (user) {
    // Will redirect via useEffect
    return null;
  }

  return <LandingPage />;
}
```

- [ ] **Step 3: Verify build compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/views/LandingPage.tsx frontend/src/app/page.tsx
git commit -m "feat(A1): add public landing page with hero, features, and GitHub badge"
```

---

### Task 18: OG + Social Meta Tags (A2)

**Files:**
- Modify: `frontend/src/app/layout.tsx`

- [ ] **Step 1: Update metadata in layout.tsx**

Replace the existing `metadata` export in `frontend/src/app/layout.tsx`:

```tsx
export const metadata: Metadata = {
  title: "whitematter — Train neural networks from your browser",
  description:
    "Design architectures with AI, train models in real-time, and deploy with one click.",
  openGraph: {
    title: "whitematter",
    description:
      "Train neural networks from your browser. Design architectures with AI. Deploy with one click.",
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

- [ ] **Step 2: Generate OG image placeholder**

Create a simple OG image using a Node script:

```bash
cd frontend && node -e "
const { createCanvas } = require('canvas');
// If canvas not available, create a simple SVG-based approach
" 2>/dev/null || echo "canvas not available"
```

If `canvas` is not available, create a minimal SVG and convert, or create the file as a simple placeholder. Simplest approach: create a basic HTML file and screenshot, or just create a solid color PNG.

As a fallback, create a minimal valid PNG placeholder:
```bash
# Create a simple 1200x630 placeholder PNG using ImageMagick if available, or just note it for manual creation
convert -size 1200x630 xc:"#0a0a0a" -fill "#7EB8FF" -pointsize 72 -gravity center -annotate 0 "wm\nwhitematter" frontend/public/og-image.png 2>/dev/null || echo "TODO: create public/og-image.png manually (1200x630, dark bg, wm branding)"
```

If neither tool is available, create a minimal placeholder and note it needs replacing.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/layout.tsx
git add frontend/public/og-image.png 2>/dev/null  # if generated
git commit -m "feat(A2): add OG and Twitter Card meta tags"
```

---

### Task 19: Share Results Button (A3)

**Files:**
- Create: `frontend/src/components/ShareCard.tsx`
- Modify: `frontend/src/components/ModelsTab.tsx`

- [ ] **Step 1: Install html2canvas**

```bash
cd frontend && npm install html2canvas
```

- [ ] **Step 2: Create ShareCard component**

```tsx
// frontend/src/components/ShareCard.tsx
"use client";
import { useRef, useState } from "react";
import type { Model } from "@/api";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import ShareOutlined from "@mui/icons-material/ShareOutlined";
import Toast, { useToast } from "./Toast";

interface Props {
  model: Model;
}

export default function ShareCard({ model }: Props) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [generating, setGenerating] = useState(false);
  const toast = useToast();

  async function handleShare() {
    if (!cardRef.current) return;
    setGenerating(true);
    try {
      const html2canvas = (await import("html2canvas")).default;
      const canvas = await html2canvas(cardRef.current, {
        backgroundColor: "#0a0a0a",
        scale: 2,
      });
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, "image/png")
      );
      if (!blob) throw new Error("Failed to generate image");

      // Download
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${model.name.replace(/\s+/g, "-")}-results.png`;
      a.click();
      URL.revokeObjectURL(url);

      // Try clipboard
      try {
        await navigator.clipboard.write([
          new ClipboardItem({ "image/png": blob }),
        ]);
        toast.success("Image downloaded and copied to clipboard!");
      } catch {
        toast.success("Image downloaded!");
      }
    } catch {
      toast.error("Failed to generate share image");
    } finally {
      setGenerating(false);
    }
  }

  const lastHistory = model.training_history?.[model.training_history.length - 1];
  const archParts = model.architecture
    .replace(/_/g, " ")
    .split(/[\s,-]+/)
    .filter(Boolean);

  return (
    <>
      {/* Hidden render target */}
      <Box
        ref={cardRef}
        sx={{
          position: "absolute",
          left: "-9999px",
          width: 600,
          height: 400,
          bgcolor: "#0a0a0a",
          p: 4,
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          border: "1px solid rgba(126,184,255,0.2)",
          borderRadius: 2,
        }}
      >
        <Box>
          <Typography
            sx={{
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: "0.875rem",
              color: "#7EB8FF",
              mb: 1,
            }}
          >
            wm
          </Typography>
          <Typography sx={{ fontSize: "1.5rem", fontWeight: 700, color: "#fff", mb: 0.5 }}>
            {model.name.replace(/^custom_/, "").replace(/_/g, " ")}
          </Typography>
          <Typography
            sx={{
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: "3rem",
              fontWeight: 700,
              color: "#7EB8FF",
              lineHeight: 1,
              mb: 2,
            }}
          >
            {model.best_accuracy.toFixed(1)}%
          </Typography>
        </Box>
        <Box>
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mb: 2 }}>
            {archParts.slice(0, 8).map((part, i) => (
              <Chip
                key={i}
                size="small"
                label={part}
                sx={{
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: "0.625rem",
                  bgcolor: "rgba(126,184,255,0.15)",
                  color: "#7EB8FF",
                  border: "1px solid rgba(126,184,255,0.3)",
                }}
              />
            ))}
          </Box>
          <Box sx={{ display: "flex", gap: 2, fontSize: "0.75rem", color: "rgba(255,255,255,0.5)" }}>
            <span>{model.epochs_trained} epochs</span>
            {lastHistory && <span>Loss: {lastHistory.loss.toFixed(4)}</span>}
            <span>{model.dataset.startsWith("custom:") ? "Custom dataset" : model.dataset}</span>
          </Box>
        </Box>
        <Typography
          sx={{
            fontSize: "0.6875rem",
            color: "rgba(255,255,255,0.3)",
            fontFamily: '"JetBrains Mono", monospace',
          }}
        >
          Built with whitematter
        </Typography>
      </Box>

      {/* Visible button */}
      <Button
        variant="outlined"
        size="small"
        startIcon={<ShareOutlined />}
        onClick={handleShare}
        disabled={generating}
      >
        {generating ? "Generating..." : "Share"}
      </Button>

      <Toast toasts={toast.toasts} onDismiss={toast.dismissToast} />
    </>
  );
}
```

- [ ] **Step 3: Add ShareCard to ModelsTab**

In `frontend/src/components/ModelsTab.tsx`, add import:
```tsx
import ShareCard from "./ShareCard";
```

Find the action buttons area for completed models (around line 554-567). Add `<ShareCard model={selectedModel} />` in the button row:

```tsx
{selectedModel.status === "completed" && (
  <>
    <Button variant="outlined" component={Link} href="/predict" sx={{ textDecoration: "none" }}>
      Predict
    </Button>
    <ShareCard model={selectedModel} />
    <Button variant="outlined" disabled sx={{ color: "text.secondary" }}>
      Export ONNX
    </Button>
    {/* ...rest of buttons */}
  </>
)}
```

- [ ] **Step 4: Verify build compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/ShareCard.tsx frontend/src/components/ModelsTab.tsx frontend/package.json frontend/package-lock.json
git commit -m "feat(A3): add share results button with downloadable card image"
```

---

## Final Verification

### Task 20: Full Build + Type Check

- [ ] **Step 1: Run type check**

```bash
cd frontend && npx tsc --noEmit
```

Expected: No errors

- [ ] **Step 2: Run tests**

```bash
cd frontend && npx vitest run
```

Expected: All tests pass (including new tests from Tasks 1-3)

- [ ] **Step 3: Run build**

```bash
cd frontend && npm run build
```

Expected: Build succeeds

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A && git commit -m "fix: address build/type issues from UX overhaul"
```
