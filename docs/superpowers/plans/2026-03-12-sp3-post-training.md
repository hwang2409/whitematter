# SP3: Post-Training Experience — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When training completes, show a rich model card with accuracy/params/time, "Try it" (predict in chat), "Share" (Twitter intent), and "Deploy as API" (coming soon).

**Architecture:** `CompletedModelCard` component renders inline in chat when `message_type === "training_complete"`. Extends the styling of `InlineModelCard` (from SP2). `InlinePredictWidget` is a sub-component that expands on the card for drag-and-drop image prediction via existing `POST /predict` endpoint.

**Tech Stack:** React, MUI, html2canvas (existing), FastAPI

**Spec:** `docs/superpowers/specs/2026-03-12-workflow-implementation-design.md` (SP3 section)

**Depends on:** SP2 must be complete (ChatPage wired to backend, message types rendering).

---

## Chunk 1: CompletedModelCard and Share

### Task 1: Create CompletedModelCard component

**Files:**
- Create: `frontend/src/components/CompletedModelCard.tsx`

- [ ] **Step 1: Create the component**

```tsx
"use client";
import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import Divider from "@mui/material/Divider";
import InlinePredictWidget from "./InlinePredictWidget";

interface CompletedModelCardProps {
  modelId: string;
  accuracy: number;
  params: string;
  trainingTime: string;
  architecture: string;
  datasetName: string;
}

export default function CompletedModelCard({
  modelId,
  accuracy,
  params,
  trainingTime,
  architecture,
  datasetName,
}: CompletedModelCardProps) {
  const [showPredict, setShowPredict] = useState(false);

  const accuracyPct = (accuracy * 100).toFixed(1);

  function handleShare() {
    const text = encodeURIComponent(
      `Just trained a model with ${accuracyPct}% accuracy (${params} params) in ${trainingTime} on @whitematter`
    );
    const url = encodeURIComponent("https://whitematter.com");
    window.open(
      `https://twitter.com/intent/tweet?text=${text}&url=${url}`,
      "_blank",
      "width=550,height=420"
    );
  }

  function handleSaveImage() {
    // Use existing ShareCard/html2canvas logic
    // Import dynamically to avoid loading html2canvas eagerly
    import("html2canvas").then((html2canvas) => {
      const el = document.getElementById(`model-card-${modelId}`);
      if (!el) return;
      html2canvas.default(el).then((canvas) => {
        const link = document.createElement("a");
        link.download = `whitematter-${modelId}.png`;
        link.href = canvas.toDataURL();
        link.click();
      });
    });
  }

  return (
    <Box
      id={`model-card-${modelId}`}
      sx={{
        border: 1,
        borderColor: "divider",
        borderRadius: 2,
        p: 2,
        my: 1,
        bgcolor: "background.paper",
      }}
    >
      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
        Training Complete
      </Typography>

      <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", mb: 1.5 }}>
        <Chip label={`${accuracyPct}% accuracy`} color="success" size="small" />
        <Chip label={params} size="small" variant="outlined" />
        <Chip label={trainingTime} size="small" variant="outlined" />
      </Box>

      <Typography variant="body2" color="text.secondary" gutterBottom>
        {architecture} on {datasetName}
      </Typography>

      <Divider sx={{ my: 1.5 }} />

      <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
        <Button
          size="small"
          variant="contained"
          onClick={() => setShowPredict(!showPredict)}
        >
          {showPredict ? "Close" : "Try it"}
        </Button>
        <Button size="small" variant="outlined" disabled>
          Deploy as API (coming soon)
        </Button>
        <Button size="small" variant="outlined" onClick={handleShare}>
          Share
        </Button>
        <Button size="small" variant="text" onClick={handleSaveImage}>
          Save image
        </Button>
      </Box>

      {showPredict && (
        <Box sx={{ mt: 2 }}>
          <InlinePredictWidget modelId={modelId} />
        </Box>
      )}
    </Box>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/CompletedModelCard.tsx
git commit -m "feat(sp3): create CompletedModelCard with share and save image"
```

---

### Task 2: Create InlinePredictWidget component

**Files:**
- Create: `frontend/src/components/InlinePredictWidget.tsx`

- [ ] **Step 1: Create the component**

```tsx
"use client";
import { useState, useCallback } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import CircularProgress from "@mui/material/CircularProgress";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

interface Prediction {
  label: string;
  confidence: number;
}

interface InlinePredictWidgetProps {
  modelId: string;
}

export default function InlinePredictWidget({ modelId }: InlinePredictWidgetProps) {
  const [predictions, setPredictions] = useState<Prediction[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleFile = useCallback(async (file: File) => {
    setLoading(true);
    setError("");
    setPredictions(null);
    setPreviewUrl(URL.createObjectURL(file));

    const token = localStorage.getItem("access_token");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/predict?model_id=${modelId}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();
      setPredictions(data.predictions || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }, [modelId]);

  return (
    <Box>
      {/* Drop zone */}
      <Box
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const file = e.dataTransfer.files[0];
          if (file) handleFile(file);
        }}
        onClick={() => {
          const input = document.createElement("input");
          input.type = "file";
          input.accept = "image/*";
          input.onchange = (e) => {
            const file = (e.target as HTMLInputElement).files?.[0];
            if (file) handleFile(file);
          };
          input.click();
        }}
        sx={{
          border: "2px dashed",
          borderColor: dragOver ? "primary.main" : "divider",
          borderRadius: 1,
          p: 2,
          textAlign: "center",
          cursor: "pointer",
          bgcolor: dragOver ? "action.hover" : "transparent",
          transition: "all 0.2s",
        }}
      >
        <Typography variant="body2" color="text.secondary">
          Drop an image here or click to upload
        </Typography>
      </Box>

      {/* Preview + Results */}
      {(previewUrl || loading || predictions || error) && (
        <Box sx={{ mt: 1.5, display: "flex", gap: 2, alignItems: "flex-start" }}>
          {previewUrl && (
            <Box
              component="img"
              src={previewUrl}
              sx={{ width: 80, height: 80, objectFit: "cover", borderRadius: 1 }}
            />
          )}
          <Box sx={{ flex: 1 }}>
            {loading && <CircularProgress size={20} />}
            {error && <Typography color="error" variant="body2">{error}</Typography>}
            {predictions && predictions.map((p, i) => (
              <Box key={i} sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
                <Typography variant="body2">{p.label}</Typography>
                <Typography variant="body2" fontWeight="bold">
                  {(p.confidence * 100).toFixed(1)}%
                </Typography>
              </Box>
            ))}
          </Box>
        </Box>
      )}
    </Box>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/InlinePredictWidget.tsx
git commit -m "feat(sp3): create InlinePredictWidget for predict-in-chat"
```

---

### Task 3: Wire CompletedModelCard into ChatMessage

**Files:**
- Modify: `frontend/src/components/ChatMessage.tsx`

- [ ] **Step 1: Import and render CompletedModelCard for `training_complete` type**

In `ChatMessage.tsx`, find the `training_complete` case (around line 72). Replace the placeholder with:

```tsx
import CompletedModelCard from "./CompletedModelCard";

// In the training_complete conditional:
if (message.type === "training_complete") {
  const meta = message.metadata || {};
  return (
    <CompletedModelCard
      modelId={meta.model_id as string}
      accuracy={meta.accuracy as number}
      params={meta.params as string}
      trainingTime={meta.training_time as string}
      architecture={meta.architecture as string}
      datasetName={meta.dataset_name as string}
    />
  );
}
```

- [ ] **Step 2: Verify ChatMessage correctly renders all message types**

Check that all types are handled: `text`, `architecture`, `training_progress`, `training_complete`, `training_error`, `file_upload`, `prediction`.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ChatMessage.tsx
git commit -m "feat(sp3): wire CompletedModelCard into chat message rendering"
```

---

### Task 4: Emit `training_complete` message from backend

**Files:**
- Modify: `platform/services/chat_service.py`

- [ ] **Step 1: When training completes, save a `training_complete` message**

In the training completion handler (or in the SSE stream endpoint when status becomes "completed"), create a message:

```python
# After detecting training completion:
model_metadata = load_model_metadata(job["model_id"])
complete_msg = ConversationMessage(
    conversation_id=conversation_id,
    role="assistant",
    content="Training complete!",
    message_type="training_complete",
    metadata={
        "model_id": job["model_id"],
        "accuracy": job.get("accuracy", 0),
        "params": f"{model_metadata.parameters // 1000}K" if model_metadata else "unknown",
        "training_time": f"{job.get('elapsed_seconds', 0):.0f}s",
        "architecture": model_metadata.architecture if model_metadata else "",
        "dataset_name": job.get("dataset_name", ""),
    },
)
db.add(complete_msg)
conversation.phase = ConversationPhase.COMPLETED.value
conversation.model_id = job["model_id"]
db.commit()
```

- [ ] **Step 2: Commit**

```bash
git add platform/services/chat_service.py
git commit -m "feat(sp3): emit training_complete message with model stats"
```

---

## SP3 Complete

After all 4 tasks:
- `CompletedModelCard` renders inline in chat after training
- Shows accuracy, params, training time
- "Try it" opens inline predict widget (drop image → see prediction)
- "Share" opens Twitter intent with model stats
- "Save image" downloads PNG via html2canvas
- "Deploy as API" shows "coming soon" (disabled)
