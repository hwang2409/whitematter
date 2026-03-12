# SP2: Chat-Driven Training Flow — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the core product loop: welcome → Quick Start or describe problem → AI suggests architecture → upload data → train → live results in chat.

**Architecture:** Frontend ChatPage wired to real backend SSE streaming. Chat messages carry `message_type` field for polymorphic rendering (text, architecture, training_progress, etc). Training updates use SSE (WebSocket removed). MNIST pre-bundled for instant Quick Start.

**Tech Stack:** React, Next.js, MUI, Recharts, FastAPI, SQLAlchemy, Anthropic SDK, SSE

**Spec:** `docs/superpowers/specs/2026-03-12-workflow-implementation-design.md` (SP2 section)

---

## Chunk 1: Wire ChatPage to Backend

### Task 1: Add SSE helper to api.ts

**Files:**
- Modify: `frontend/src/api.ts`

The file already has `sendChatMessage` (line 794) that parses SSE. Review it and verify it:
1. Creates a fetch request to the correct endpoint
2. Parses `data: ` lines from the stream
3. Handles `[DONE]` signal
4. Returns an abort handle

- [ ] **Step 1: Review existing `sendChatMessage` in `api.ts`**

Read `frontend/src/api.ts` lines 794-872. Verify the function signature matches:
```typescript
sendChatMessage(
  conversationId: string,
  content: string,
  onChunk: (chunk: string) => void,
  onDone: (fullMessage: ChatMessage) => void,
  onError: (error: Error) => void,
): { abort: () => void }
```

- [ ] **Step 2: Add `createConversation` and `getConversation` API functions if missing**

Check if these exist. If not, add them near the chat functions:

```typescript
export async function createConversation(token: string): Promise<Conversation> {
  const res = await fetch(`${API_BASE}/chat/conversations`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error("Failed to create conversation");
  return res.json();
}

export async function getConversation(token: string, id: string): Promise<{ conversation: Conversation; messages: ChatMessage[] }> {
  const res = await fetch(`${API_BASE}/chat/conversations/${id}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error("Failed to load conversation");
  return res.json();
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api.ts
git commit -m "feat(sp2): add createConversation and getConversation API helpers"
```

---

### Task 2: Wire ChatPage to real backend

**Files:**
- Modify: `frontend/src/views/ChatPage.tsx`

This is the most critical task. Replace mock data with real API calls.

- [ ] **Step 1: Replace mock greeting with backend conversation creation**

Current state: `ChatPage` has a hardcoded `GREETING_MESSAGE` (line 17) and `mockAssistantReply` (line 26). Replace with:

1. On mount, create a new conversation via `POST /chat/conversations` (or load existing one if `conversationId` prop is set)
2. The backend returns the conversation with the greeting message already attached
3. Load messages from the response and set them as initial state

Remove `GREETING_MESSAGE` constant and `mockAssistantReply` function.

```tsx
// Replace the useEffect/initialization with:
const { token } = useAuth();

useEffect(() => {
  async function init() {
    if (!token) return;
    setLoading(true);
    try {
      if (conversationId) {
        const data = await getConversation(token, conversationId);
        setMessages(data.messages);
        setPhase(data.conversation.phase as ConversationPhase);
      } else {
        const conv = await createConversation(token);
        // Backend returns conversation with greeting message
        const data = await getConversation(token, conv.id);
        setMessages(data.messages);
        setConversationId(conv.id);
      }
    } catch (err) {
      console.error("Failed to load conversation:", err);
    } finally {
      setLoading(false);
    }
  }
  init();
}, [token, conversationId]);
```

- [ ] **Step 2: Replace handleSend with real SSE streaming**

Replace the current `handleSend` function (lines 75-110) which uses `mockAssistantReply`:

```tsx
async function handleSend(text: string) {
  if (!text.trim() || streaming || !token || !currentConversationId) return;

  // Add user message to UI immediately
  const userMsg: ChatMessage = { role: "user", type: "text", content: text };
  setMessages((prev) => [...prev, userMsg]);

  // Add placeholder assistant message
  const assistantMsg: ChatMessage = { role: "assistant", type: "text", content: "" };
  setMessages((prev) => [...prev, assistantMsg]);
  setStreaming(true);

  sendChatMessage(
    currentConversationId,
    text,
    token,
    // onChunk: append to last message
    (chunk: string) => {
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        updated[updated.length - 1] = { ...last, content: last.content + chunk };
        return updated;
      });
    },
    // onDone: replace last message with full response
    (fullMessage: ChatMessage) => {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = fullMessage;
        return updated;
      });
      setStreaming(false);
      // Update phase if returned in message metadata
      if (fullMessage.metadata?.phase) {
        setPhase(fullMessage.metadata.phase as ConversationPhase);
      }
    },
    // onError
    (error: Error) => {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          type: "text",
          content: "I'm having trouble thinking right now, try again in a moment.",
          metadata: { error: true },
        };
        return updated;
      });
      setStreaming(false);
    },
  );
}
```

Note: The `sendChatMessage` function in `api.ts` needs to accept a `token` parameter for auth. Check and add if missing.

- [ ] **Step 3: Verify chat sends messages and streams responses**

```bash
cd /Users/gimdongha/Desktop/Projects/whitematter
source .venv/bin/activate
make dev
```

Open `http://localhost:3000/chat`. Register/login, verify:
1. Greeting message loads from backend
2. Typing a message sends it and streams a response
3. If `ANTHROPIC_API_KEY` is not set, mock LLM should still return something

- [ ] **Step 4: Commit**

```bash
git add frontend/src/views/ChatPage.tsx frontend/src/api.ts
git commit -m "feat(sp2): wire ChatPage to real backend SSE streaming"
```

---

### Task 3: Update ChatMessage rendering for message types

**Files:**
- Modify: `frontend/src/components/ChatMessage.tsx`

- [ ] **Step 1: Review current message type handling**

Read `ChatMessage.tsx`. It already has conditionals for `architecture`, `training_progress`, `training_complete`, `file_upload`, `prediction` types (lines 20-138). Verify these render the right components. The `architecture` case should render `ModelCard` — confirm it passes the right props from `message.metadata`.

- [ ] **Step 2: Add Claude API error handling**

In the default text rendering case, check for `message.metadata?.error` and render a retry button:

```tsx
{message.metadata?.error && (
  <Button
    size="small"
    onClick={() => {/* retry logic via onRetry prop */}}
    sx={{ mt: 1 }}
  >
    Try again
  </Button>
)}
```

Add `onRetry?: () => void` to `ChatMessageBubbleProps`.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ChatMessage.tsx
git commit -m "feat(sp2): add error state and retry button to chat messages"
```

---

## Chunk 2: Quick Start and Welcome

### Task 4: Update QuickStartChips to 2 chips

**Files:**
- Modify: `frontend/src/components/QuickStartChips.tsx`

- [ ] **Step 1: Replace 4 chips with 2**

Replace the `QUICK_STARTS` constant (lines 7-12):

```tsx
const QUICK_STARTS = [
  {
    label: "Quick Start (MNIST)",
    message: "I want to try the Quick Start with MNIST",
  },
  {
    label: "I want to build something",
    message: "I want to build a custom neural network",
  },
];
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/QuickStartChips.tsx
git commit -m "feat(sp2): replace 4 quick start chips with 2 (MNIST + custom)"
```

---

### Task 5: Update welcome message in backend

**Files:**
- Modify: `platform/services/chat_service.py` (line 27-35)

- [ ] **Step 1: Update GREETING_MESSAGE**

Replace the current multi-line greeting with:

```python
GREETING_MESSAGE = (
    "Hey! I help you build and train neural networks. "
    "Try a demo or describe what you want to build."
)
```

- [ ] **Step 2: Commit**

```bash
git add platform/services/chat_service.py
git commit -m "feat(sp2): update welcome message to be concise"
```

---

### Task 6: Create MNIST prebundle script

**Files:**
- Create: `scripts/prebundle_mnist.py`

- [ ] **Step 1: Create the script**

This script downloads MNIST, processes it using the existing `process_mnist_idx` function, and saves the .bin files to `presets/mnist/`.

```python
#!/usr/bin/env python3
"""
One-time setup: Download MNIST, process into .bin format, save to presets/mnist/.
In production, upload the resulting files to R2 under presets/mnist/.
"""
import sys
import struct
import gzip
import urllib.request
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PRESETS_DIR = PROJECT_ROOT / "presets" / "mnist"
RAW_DIR = PRESETS_DIR / "raw"

MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

def download_mnist():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MNIST_URLS.items():
        dest = RAW_DIR / name.replace(".gz", "")
        if dest.exists():
            print(f"  Already exists: {dest.name}")
            continue
        print(f"  Downloading {name}...")
        data = urllib.request.urlopen(url).read()
        with open(dest, "wb") as f:
            f.write(gzip.decompress(data))

def save_tensor(path, data):
    TENSOR_MAGIC = 0x54454E53
    with open(path, "wb") as f:
        f.write(struct.pack("I", TENSOR_MAGIC))
        f.write(struct.pack("I", len(data.shape)))
        for dim in data.shape:
            f.write(struct.pack("Q", dim))
        data = np.ascontiguousarray(data, dtype=np.float32)
        f.write(data.tobytes())

def read_idx_images(filepath):
    with open(filepath, "rb") as f:
        struct.unpack(">I", f.read(4))  # magic
        num = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, 1, rows, cols).astype(np.float32) / 255.0, rows, cols

def read_idx_labels(filepath):
    with open(filepath, "rb") as f:
        struct.unpack(">I", f.read(4))  # magic
        struct.unpack(">I", f.read(4))  # num
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32)

def main():
    print("Step 1: Downloading MNIST...")
    download_mnist()

    print("Step 2: Processing...")
    train_images, rows, cols = read_idx_images(RAW_DIR / "train-images-idx3-ubyte")
    train_labels = read_idx_labels(RAW_DIR / "train-labels-idx1-ubyte")
    test_images, _, _ = read_idx_images(RAW_DIR / "t10k-images-idx3-ubyte")
    test_labels = read_idx_labels(RAW_DIR / "t10k-labels-idx1-ubyte")

    mean = [float(train_images.mean())]
    std_val = [float(max(train_images.std(), 1e-7))]
    train_images = (train_images - mean[0]) / std_val[0]
    test_images = (test_images - mean[0]) / std_val[0]

    print("Step 3: Saving .bin files...")
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    save_tensor(PRESETS_DIR / "train_images.bin", train_images)
    save_tensor(PRESETS_DIR / "train_labels.bin", train_labels)
    save_tensor(PRESETS_DIR / "test_images.bin", test_images)
    save_tensor(PRESETS_DIR / "test_labels.bin", test_labels)

    import json
    config = {
        "target_size": [rows, cols],
        "channels": 1,
        "mean": mean,
        "std": std_val,
        "num_classes": 10,
        "class_names": [str(i) for i in range(10)],
        "train_samples": len(train_images),
        "test_samples": len(test_images),
        "input_shape": [1, rows, cols],
    }
    with open(PRESETS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Cleanup raw files
    import shutil
    shutil.rmtree(RAW_DIR)

    total_mb = sum(f.stat().st_size for f in PRESETS_DIR.glob("*.bin")) / 1024 / 1024
    print(f"Done! Saved to {PRESETS_DIR} ({total_mb:.1f} MB)")
    print("For production: upload presets/mnist/ contents to R2 bucket under presets/mnist/")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```bash
cd /Users/gimdongha/Desktop/Projects/whitematter
python3 scripts/prebundle_mnist.py
```

Expected: Creates `presets/mnist/` with ~12 MB of .bin files + config.json.

- [ ] **Step 3: Add presets/ to .gitignore (binary data, don't commit)**

```bash
echo "presets/" >> .gitignore
```

- [ ] **Step 4: Commit**

```bash
git add scripts/prebundle_mnist.py .gitignore
git commit -m "feat(sp2): add MNIST prebundle script for Quick Start"
```

---

### Task 7: Backend Quick Start MNIST handler in chat_service

**Files:**
- Modify: `platform/services/chat_service.py`

- [ ] **Step 1: Add MNIST preset loading to chat_service**

When the user sends a message containing "Quick Start" or "MNIST", the chat service should:
1. Copy pre-bundled MNIST files to the user's dataset directory
2. Skip to architecture suggestion phase
3. Return a pre-configured MNIST CNN architecture

Add a method to `ChatService`:

```python
import shutil

PRESETS_DIR = Path(__file__).resolve().parent.parent.parent / "presets"

def _handle_quick_start_mnist(self, db, conversation, user):
    """Handle Quick Start MNIST flow: load preset, suggest architecture."""
    # Copy MNIST preset to user's dataset
    mnist_src = PRESETS_DIR / "mnist"
    if not mnist_src.exists():
        raise ValueError("MNIST preset not found. Run: python scripts/prebundle_mnist.py")

    dataset = dataset_service.create_dataset("mnist-quickstart")
    dest = Path(dataset["path"])
    dest.mkdir(parents=True, exist_ok=True)
    for f in mnist_src.glob("*"):
        shutil.copy2(f, dest / f.name)

    # Update conversation phase
    conversation.phase = ConversationPhase.ARCHITECTURE.value
    conversation.dataset_id = dataset["id"]
    db.commit()

    # Return pre-configured MNIST architecture
    architecture = {
        "name": "MNIST CNN",
        "description": "A simple CNN for handwritten digit classification",
        "layers": "Conv2d(1,16,3) → ReLU → MaxPool → Conv2d(16,32,3) → ReLU → MaxPool → FC(32*5*5, 128) → ReLU → FC(128, 10)",
        "trainingConfig": "SGD lr=0.01, batch_size=64, epochs=5",
        "params": "~207K",
        "dataset": "MNIST (60K train, 10K test)",
    }
    return architecture
```

- [ ] **Step 2: Integrate into `process_message`**

In the `process_message` method, detect Quick Start intent early:

```python
async def process_message(self, db, conversation_id, user, content):
    # ... existing conversation lookup ...

    # Quick Start detection
    if "quick start" in content.lower() or "mnist" in content.lower() and conversation.phase == ConversationPhase.GREETING.value:
        architecture = self._handle_quick_start_mnist(db, conversation, user)
        # Yield architecture message as SSE
        msg = ConversationMessage(
            conversation_id=conversation.id,
            role="assistant",
            content=f"Great! I've loaded MNIST for you. Here's a simple CNN that works well for digit classification:",
            message_type="architecture",
            metadata=architecture,
        )
        db.add(msg)
        db.commit()
        yield f"data: {json.dumps({'type': 'architecture', 'content': msg.content, 'metadata': architecture})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # ... rest of existing process_message ...
```

- [ ] **Step 3: Commit**

```bash
git add platform/services/chat_service.py
git commit -m "feat(sp2): add Quick Start MNIST handler to chat service"
```

---

## Chunk 3: Dataset Upload in Chat

### Task 8: Wire attach button in ChatInput to file upload

**Files:**
- Modify: `frontend/src/components/ChatInput.tsx`
- Modify: `frontend/src/views/ChatPage.tsx`

- [ ] **Step 1: Add file input and upload handler to ChatInput**

Add `onFileUpload` callback prop and wire the existing attach button:

```tsx
interface ChatInputProps {
  onSend: (text: string) => void;
  onFileUpload?: (file: File) => void;
  disabled?: boolean;
  placeholder?: string;
  maxUploadMB?: number;
}
```

Add a hidden file input and wire the attach button (around line 48-59):

```tsx
const fileInputRef = useRef<HTMLInputElement>(null);

// Replace the attach IconButton onClick:
<IconButton onClick={() => fileInputRef.current?.click()} disabled={disabled}>
  <AttachFileIcon />
</IconButton>
<input
  ref={fileInputRef}
  type="file"
  accept=".zip"
  hidden
  onChange={(e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (maxUploadMB && file.size > maxUploadMB * 1024 * 1024) {
      alert(`File too large. Maximum upload size is ${maxUploadMB} MB.`);
      return;
    }
    onFileUpload?.(file);
    e.target.value = "";
  }}
/>
```

- [ ] **Step 2: Add file upload handler to ChatPage**

In `ChatPage.tsx`, add a handler that uploads to `/datasets/upload`:

```tsx
async function handleFileUpload(file: File) {
  if (!token || !currentConversationId) return;

  // Add upload message to chat
  const uploadMsg: ChatMessage = {
    role: "user",
    type: "file_upload",
    content: `Uploading ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)...`,
    metadata: { fileName: file.name, status: "uploading" },
  };
  setMessages((prev) => [...prev, uploadMsg]);

  const formData = new FormData();
  formData.append("file", file);
  formData.append("name", file.name.replace(".zip", ""));

  try {
    const res = await fetch(`${API_BASE}/datasets/upload`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: formData,
    });
    if (!res.ok) throw new Error("Upload failed");
    const result = await res.json();

    // Update message to show success
    setMessages((prev) => {
      const updated = [...prev];
      updated[updated.length - 1] = {
        ...updated[updated.length - 1],
        content: `Uploaded ${file.name} successfully!`,
        metadata: { ...result, status: "complete" },
      };
      return updated;
    });

    // Send a message to the AI about the uploaded dataset
    handleSend(`I've uploaded a dataset: ${file.name}`);
  } catch (err) {
    setMessages((prev) => {
      const updated = [...prev];
      updated[updated.length - 1] = {
        ...updated[updated.length - 1],
        content: `Failed to upload ${file.name}. Please try again.`,
        metadata: { status: "error" },
      };
      return updated;
    });
  }
}
```

Pass it to ChatInput:
```tsx
<ChatInput
  onSend={handleSend}
  onFileUpload={handleFileUpload}
  maxUploadMB={uploadLimitMB}
  disabled={streaming}
/>
```

Where `uploadLimitMB` comes from the user's plan (200 for free, 1000 for pro, 5000 for scale).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ChatInput.tsx frontend/src/views/ChatPage.tsx
git commit -m "feat(sp2): wire attach button to dataset upload"
```

---

### Task 9: Add drag-and-drop to ChatPage

**Files:**
- Modify: `frontend/src/views/ChatPage.tsx`

- [ ] **Step 1: Add drag-and-drop zone**

Wrap the chat message area with drag handlers:

```tsx
const [dragOver, setDragOver] = useState(false);

// On the outer chat container:
<Box
  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
  onDragLeave={() => setDragOver(false)}
  onDrop={(e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith(".zip")) {
      handleFileUpload(file);
    }
  }}
  sx={{ position: "relative", flex: 1, display: "flex", flexDirection: "column" }}
>
  {/* Drag overlay */}
  {dragOver && (
    <Box sx={{
      position: "absolute", inset: 0, zIndex: 10,
      bgcolor: "rgba(0,0,0,0.5)", display: "flex",
      alignItems: "center", justifyContent: "center",
      borderRadius: 2, border: "2px dashed",
      borderColor: "primary.main",
    }}>
      <Typography variant="h6" color="white">
        Drop your dataset here (max {uploadLimitMB} MB)
      </Typography>
    </Box>
  )}
  {/* existing chat content */}
</Box>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/views/ChatPage.tsx
git commit -m "feat(sp2): add drag-and-drop dataset upload to chat"
```

---

### Task 10: Add server-side upload size limits

**Files:**
- Modify: `platform/routes/datasets.py` (around line 30)

- [ ] **Step 1: Add plan-based file size validation**

After the ZIP extension check (line 39), add:

```python
from services.billing_service import BillingService

UPLOAD_LIMITS = {
    "free": 200 * 1024 * 1024,   # 200 MB
    "pro": 1000 * 1024 * 1024,   # 1 GB
    "scale": 5000 * 1024 * 1024, # 5 GB
}

# Inside upload_dataset, after ZIP check:
file_content = await file.read()
max_size = UPLOAD_LIMITS.get(user.plan, UPLOAD_LIMITS["free"])
if len(file_content) > max_size:
    max_mb = max_size // (1024 * 1024)
    raise HTTPException(
        status_code=413,
        detail=f"File too large. Your plan allows up to {max_mb} MB. Upgrade for more storage."
    )
```

Note: The current code reads `file_content = await file.read()` already (around line 50). Move the size check right after that read, before processing.

- [ ] **Step 2: Commit**

```bash
git add platform/routes/datasets.py
git commit -m "feat(sp2): add plan-based server-side upload size limits"
```

---

## Chunk 4: Training Flow

### Task 11: Remove WebSocket infrastructure

**Files:**
- Modify: `platform/routes/training.py` (remove `/ws/train/{job_id}` endpoint)
- Modify: `frontend/src/api.ts` (remove `createTrainingWebSocket`)

- [ ] **Step 1: Remove WebSocket endpoint from backend**

Read `platform/routes/training.py` and find the WebSocket endpoint (around line 320). Remove the entire function and its import of `WebSocket` from fastapi.

- [ ] **Step 2: Remove WebSocket client from frontend**

Remove `createTrainingWebSocket` function (lines 260-337) from `frontend/src/api.ts`. Remove any imports it uses that are now unused.

- [ ] **Step 3: Commit**

```bash
git add platform/routes/training.py frontend/src/api.ts
git commit -m "refactor(sp2): remove WebSocket training infrastructure, replaced by SSE"
```

---

### Task 12: Add SSE training stream endpoint

**Files:**
- Modify: `platform/routes/chat.py`

- [ ] **Step 1: Add SSE endpoint for training updates**

Add a new endpoint that streams training progress as SSE events:

```python
@router.get("/chat/conversations/{conversation_id}/training/stream")
async def stream_training_progress(
    conversation_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Stream training progress as SSE events."""
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id,
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    async def event_stream():
        while True:
            status = training_jobs.get_by_conversation(conversation_id)
            if status:
                yield f"data: {json.dumps(status)}\n\n"
                if status.get("status") in ("completed", "failed", "cancelled"):
                    break
            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

- [ ] **Step 2: Commit**

```bash
git add platform/routes/chat.py
git commit -m "feat(sp2): add SSE endpoint for training progress streaming"
```

---

### Task 13: Refactor TrainingProgress to use SSE

**Files:**
- Modify: `frontend/src/components/TrainingProgress.tsx`

- [ ] **Step 1: Replace polling with SSE consumption**

Replace the `useEffect` polling (lines 30-64) with EventSource:

```tsx
useEffect(() => {
  const token = localStorage.getItem("access_token");
  const eventSource = new EventSource(
    `${API_BASE}/chat/conversations/${conversationId}/training/stream`,
    // Note: EventSource doesn't support auth headers natively.
    // Use fetch-based SSE or pass token as query param.
  );

  // Alternative: use fetch-based SSE
  const controller = new AbortController();

  async function connectSSE() {
    try {
      const res = await fetch(
        `${API_BASE}/chat/conversations/${conversationId}/training/stream`,
        {
          headers: { Authorization: `Bearer ${token}` },
          signal: controller.signal,
        }
      );
      const reader = res.body?.getReader();
      const decoder = new TextDecoder();

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value);
        const lines = text.split("\n");
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = JSON.parse(line.slice(6));
            setStatus(data);
            setHistory((prev) => [...prev, {
              epoch: data.epoch,
              loss: data.loss,
              accuracy: data.accuracy,
            }]);
            if (["completed", "failed", "cancelled"].includes(data.status)) {
              onComplete?.(data);
              return;
            }
          }
        }
      }
    } catch (err) {
      if (!controller.signal.aborted) console.error("SSE error:", err);
    }
  }

  connectSSE();
  return () => controller.abort();
}, [conversationId, jobId]);
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/TrainingProgress.tsx
git commit -m "refactor(sp2): replace polling with SSE in TrainingProgress"
```

---

### Task 14: Add CPU/GPU toggle for Scale users

**Files:**
- Modify: `frontend/src/components/ModelCard.tsx`

- [ ] **Step 1: Add compute selector to ModelCard**

Add a `userPlan` prop. If `userPlan === "scale"`, show CPU/GPU toggle chips above the "Train it" button:

```tsx
interface ModelCardProps {
  // ... existing props ...
  userPlan?: string;
  onApprove: (compute?: "cpu" | "gpu") => void;
}

// Inside the component, before the "Train it" button:
const [compute, setCompute] = useState<"cpu" | "gpu">("gpu");

{userPlan === "scale" && (
  <Box sx={{ display: "flex", gap: 1, mb: 1 }}>
    <Chip
      label="CPU (instant)"
      onClick={() => setCompute("cpu")}
      color={compute === "cpu" ? "primary" : "default"}
      variant={compute === "cpu" ? "filled" : "outlined"}
    />
    <Chip
      label="GPU (faster, ~60s startup)"
      onClick={() => setCompute("gpu")}
      color={compute === "gpu" ? "primary" : "default"}
      variant={compute === "gpu" ? "filled" : "outlined"}
    />
  </Box>
)}

// Update button onClick:
<Button onClick={() => onApprove(userPlan === "scale" ? compute : "cpu")}>
  Looks good, train it!
</Button>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/ModelCard.tsx
git commit -m "feat(sp2): add CPU/GPU compute toggle for Scale users"
```

---

## Chunk 5: Training Error Handling

### Task 15: Handle training failures in chat

**Files:**
- Modify: `frontend/src/components/ChatMessage.tsx`

- [ ] **Step 1: Implement `training_error` message type rendering**

In `ChatMessage.tsx`, the `training_error` case currently doesn't exist or is a placeholder. Add:

```tsx
if (message.type === "training_error") {
  const errorMsg = message.metadata?.friendlyMessage || message.content;
  return (
    <Box sx={{ p: 2, bgcolor: "error.main", color: "white", borderRadius: 2, my: 1 }}>
      <Typography variant="subtitle2">Training Failed</Typography>
      <Typography variant="body2" sx={{ mt: 0.5 }}>{errorMsg}</Typography>
      {message.metadata?.suggestion && (
        <Typography variant="body2" sx={{ mt: 1, fontStyle: "italic" }}>
          {message.metadata.suggestion}
        </Typography>
      )}
    </Box>
  );
}
```

- [ ] **Step 2: Wire error parsing from trainingErrors.ts**

In the `TrainingProgress` `onComplete` callback (called from ChatPage), when status is "failed":

```tsx
import { parseTrainingError } from "@/lib/trainingErrors";

// In onComplete handler:
if (status.status === "failed") {
  const parsed = parseTrainingError(status.message || "Unknown error");
  const errorMsg: ChatMessage = {
    role: "assistant",
    type: "training_error",
    content: parsed.message,
    metadata: {
      friendlyMessage: parsed.message,
      suggestion: parsed.suggestion,
    },
  };
  setMessages((prev) => [...prev, errorMsg]);

  // Follow up with AI recovery suggestion
  handleSend(`Training failed with error: ${parsed.message}. Can you suggest a fix?`);
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ChatMessage.tsx frontend/src/views/ChatPage.tsx
git commit -m "feat(sp2): handle training failures with friendly errors and AI recovery"
```

---

## SP2 Complete

After all 15 tasks:
- ChatPage wired to real backend (no more mock data)
- SSE streaming for both chat and training
- Quick Start MNIST flow works end-to-end
- Dataset upload via attach button and drag-and-drop
- Plan-based upload limits (client + server)
- CPU/GPU toggle for Scale users
- Training errors handled with AI recovery suggestions
- WebSocket infrastructure removed
