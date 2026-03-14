# UI Redesign — "Living Workspace" Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform WhiteMatter's frontend from a generic chatbot UI into a distinctive, recognizable product with a violet+amber color system, breakout training cards, and polished chat interactions.

**Architecture:** Three-phase rollout keeping the existing layout intact (sidebar + chat + top bar). Phase 1 swaps the color system and polishes chat messages. Phase 2 builds the signature TrainingCard component system. Phase 3 adds motion and accessibility polish. All changes stay within MUI's theming system.

**Tech Stack:** Next.js 15, React 19, MUI v7, Recharts, TypeScript, Vitest + Testing Library

**Spec:** `docs/superpowers/specs/2026-03-13-ui-redesign-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `frontend/src/components/training/TrainingCard.tsx` | Breakout training card — orchestrates ProgressRing, StageIndicator, SparklineChart, expand/collapse |
| `frontend/src/components/training/ProgressRing.tsx` | SVG circular progress indicator with violet/amber stroke |
| `frontend/src/components/training/StageIndicator.tsx` | Horizontal dot + label stage progression display |
| `frontend/src/components/training/SparklineChart.tsx` | Minimal inline Recharts loss chart (no axes, no grid) |
| `frontend/src/__tests__/TrainingCard.test.tsx` | Tests for TrainingCard rendering, states, expand/collapse |
| `frontend/src/__tests__/ProgressRing.test.tsx` | Tests for ProgressRing rendering and value display |
| `frontend/src/__tests__/StageIndicator.test.tsx` | Tests for stage dot states at each pipeline point |
| `frontend/src/__tests__/SparklineChart.test.tsx` | Tests for SparklineChart rendering |
| `frontend/src/__tests__/ChatMessage.test.tsx` | Tests for message type rendering and visual distinction |
| `frontend/src/__tests__/theme.test.ts` | Tests for color tokens and themeTokens export |

### Modified Files

| File | What Changes |
|------|-------------|
| `frontend/src/theme.ts` | New violet/amber color tokens, completion palette, updated themeTokens export, JetBrains Mono → DM Mono |
| `frontend/src/components/ChatMessage.tsx` | AI/user visual distinction, W monogram avatar, code block accent, TrainingCard integration |
| `frontend/src/components/ChatInput.tsx` | Focus ring → violet |
| `frontend/src/components/QuickStartChips.tsx` | Violet hover glow and border |
| `frontend/src/components/TrainingProgress.tsx` | Refactored to use TrainingCard |
| `frontend/src/components/TrainingChart.tsx` | Violet/amber palette, JetBrains Mono → DM Mono |
| `frontend/src/components/ShareCard.tsx` | JetBrains Mono → DM Mono, color update |
| `frontend/src/components/CompletedModelCard.tsx` | Remove gradient, update to violet/amber palette |
| `frontend/src/components/InlinePredictWidget.tsx` | Drop zone border → violet on drag-over |
| `frontend/src/views/ChatPage.tsx` | Empty state redesign, typing indicator, isLatestActive prop logic |

---

## Chunk 1: Phase 1 — Color System + Chat Polish

### Task 1: Update Color Tokens in theme.ts

**Files:**
- Modify: `frontend/src/theme.ts:5-7` (color constants)
- Modify: `frontend/src/theme.ts:9-33` (dark/light palettes)
- Modify: `frontend/src/theme.ts:37-50` (palette config in getTheme)
- Modify: `frontend/src/theme.ts:221-231` (themeTokens export)
- Test: `frontend/src/__tests__/theme.test.ts`

- [ ] **Step 1: Write failing test for new color tokens**

```typescript
// frontend/src/__tests__/theme.test.ts
import { describe, it, expect } from 'vitest';
import { getTheme, themeTokens } from '../theme';

describe('theme color tokens', () => {
  it('dark theme uses violet as primary', () => {
    const theme = getTheme('dark');
    expect(theme.palette.primary.main).toBe('#8B5CF6');
    expect(theme.palette.primary.light).toBe('#A78BFA');
    expect(theme.palette.primary.dark).toBe('#7C3AED');
  });

  it('light theme uses darker violet for contrast', () => {
    const theme = getTheme('light');
    expect(theme.palette.primary.main).toBe('#7C3AED');
    expect(theme.palette.primary.light).toBe('#8B5CF6');
    expect(theme.palette.primary.dark).toBe('#6D28D9');
  });

  it('success stays green (not amber)', () => {
    const darkTheme = getTheme('dark');
    expect(darkTheme.palette.success.main).toBe('#22c55e');
  });

  it('themeTokens exports correct accent and completion values', () => {
    expect(themeTokens.accent).toBe('#8B5CF6');
    expect(themeTokens.accentLight).toBe('#A78BFA');
    expect(themeTokens.accentMuted).toBe('rgba(139, 92, 246, 0.15)');
    expect(themeTokens.completion).toBe('#F59E0B');
    expect(themeTokens.completionLight).toBe('#FBBF24');
    expect(themeTokens.completionMuted).toBe('rgba(245, 158, 11, 0.12)');
    expect(themeTokens.fontMono).toBe("'DM Mono', monospace");
  });

  it('dark card background is warm-tinted', () => {
    const theme = getTheme('dark');
    // Access custom token via theme — implementation will store in palette or custom
    expect(themeTokens.card).toBe('#1C1A18');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/__tests__/theme.test.ts`
Expected: FAIL — current accent is `#78716C`, themeTokens lacks completion/fontMono

- [ ] **Step 3: Update color constants and palettes**

Replace lines 5-7 of `theme.ts`:

```typescript
const ACCENT = "#8B5CF6";
const ACCENT_LIGHT = "rgba(139, 92, 246, 0.08)";
const ACCENT_MUTED = "rgba(139, 92, 246, 0.35)";
```

Update dark palette object (lines 9-20) — change `card` value:

```typescript
card: "#1C1A18",
```

Update light palette object (lines 22-33):

```typescript
card: "#F3F2F0",
```

Update `getTheme` palette section to use darker violet values for light mode:

```typescript
primary: {
  main: colors.mode === 'dark' ? '#8B5CF6' : '#7C3AED',
  light: colors.mode === 'dark' ? '#A78BFA' : '#8B5CF6',
  dark: colors.mode === 'dark' ? '#7C3AED' : '#6D28D9',
},
```

Note: Keep `success.main` as `#22c55e` — do not change.

- [ ] **Step 4: Update themeTokens export**

Replace lines 221-231 of `theme.ts`:

```typescript
export const themeTokens = {
  accent: '#8B5CF6',
  accentLight: '#A78BFA',
  accentMuted: 'rgba(139, 92, 246, 0.15)',
  completion: '#F59E0B',
  completionLight: '#FBBF24',
  completionMuted: 'rgba(245, 158, 11, 0.12)',
  bg: '#141311',
  surface: '#1E1D1B',
  card: '#1C1A18',
  text: '#F2F1EE',
  textSecondary: '#9C9A95',
  border: 'rgba(255,255,255,0.07)',
  borderHover: 'rgba(255,255,255,0.14)',
  textMuted: '#6B6963',
  error: '#ef4444',
  fontMono: "'DM Mono', monospace",
};
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/__tests__/theme.test.ts`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/src/theme.ts frontend/src/__tests__/theme.test.ts
git commit -m "feat: update color system to violet/amber palette"
```

---

### Task 2: Migrate JetBrains Mono to DM Mono

**Files:**
- Modify: `frontend/src/components/TrainingChart.tsx:67,79,92,107`
- Modify: `frontend/src/components/ShareCard.tsx:84,104,123,153`
- Modify: `frontend/src/components/ErrorBoundary.tsx:87`
- Modify: `frontend/src/components/ModelsTab.tsx` (10 occurrences)
- Modify: `frontend/src/app/globals.css:1` (font import)

- [ ] **Step 1: Replace JetBrains Mono in TrainingChart.tsx**

Find all instances of `'"JetBrains Mono", monospace'` in TrainingChart.tsx and replace with `"'DM Mono', monospace"`.

Four occurrences at lines 67, 79, 92, 107.

- [ ] **Step 2: Replace JetBrains Mono in ShareCard.tsx**

Find all instances of `'"JetBrains Mono", monospace'` in ShareCard.tsx and replace with `"'DM Mono', monospace"`.

Four occurrences at lines 84, 104, 123, 153.

- [ ] **Step 3: Replace JetBrains Mono in ErrorBoundary.tsx**

Replace `'"JetBrains Mono", monospace'` with `"'DM Mono', monospace"` at line 87.

- [ ] **Step 4: Replace JetBrains Mono in ModelsTab.tsx**

Replace all instances of `'"JetBrains Mono", monospace'` with `"'DM Mono', monospace"`. Ten occurrences across the file.

- [ ] **Step 5: Update globals.css font import**

In `frontend/src/app/globals.css`, update or remove the `@import url` for JetBrains Mono. If DM Mono is not already imported via `globals.css` or `<link>` in the layout, add the DM Mono import. If JetBrains Mono is no longer used anywhere, remove its import.

- [ ] **Step 6: Search for any remaining JetBrains Mono references**

Run: `cd frontend && grep -r "JetBrains" src/`
Expected: No results

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/TrainingChart.tsx frontend/src/components/ShareCard.tsx frontend/src/components/ErrorBoundary.tsx frontend/src/components/ModelsTab.tsx frontend/src/app/globals.css
git commit -m "refactor: migrate JetBrains Mono to DM Mono"
```

---

### Task 3: Chat Message Visual Distinction — AI vs User

**Files:**
- Modify: `frontend/src/components/ChatMessage.tsx:19-41` (avatar), `220-252` (user msg), `255-320` (AI msg)
- Test: `frontend/src/__tests__/ChatMessage.test.tsx`

- [ ] **Step 1: Write failing test for message visual distinction**

```typescript
// frontend/src/__tests__/ChatMessage.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import ChatMessageBubble from '../components/ChatMessage';

const theme = getTheme('dark');

const wrap = (ui: React.ReactElement) =>
  render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

describe('ChatMessageBubble', () => {
  it('renders AI avatar with W monogram', () => {
    wrap(
      <ChatMessageBubble
        message={{ id: 'test-1', role: 'assistant', content: 'Hello', type: 'text', createdAt: new Date().toISOString() }}
      />
    );
    expect(screen.getByText('W')).toBeTruthy();
  });

  it('does not render avatar for user messages', () => {
    wrap(
      <ChatMessageBubble
        message={{ id: 'test-2', role: 'user', content: 'Hi', type: 'text', createdAt: new Date().toISOString() }}
      />
    );
    expect(screen.queryByText('W')).toBeNull();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/__tests__/ChatMessage.test.tsx`
Expected: FAIL — current avatar shows "wm" not "W"

- [ ] **Step 3: Update AI Avatar to W monogram**

In `ChatMessage.tsx`, update the AiAvatar component (lines 19-41):

```typescript
const AiAvatar = () => (
  <Box
    sx={{
      width: 28,
      height: 28,
      borderRadius: '50%',
      bgcolor: 'primary.main',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      flexShrink: 0,
    }}
  >
    <Typography
      sx={{
        fontFamily: "'DM Serif Display', Georgia, serif",
        fontSize: '0.8rem',
        fontWeight: 400,
        color: '#FFFFFF',
        lineHeight: 1,
      }}
    >
      W
    </Typography>
  </Box>
);
```

- [ ] **Step 4: Update user message styling**

In `ChatMessage.tsx`, update user message rendering (around lines 220-252):

- Keep right-alignment
- Keep background: `rgba(255,255,255,0.06)` for dark, `#F2F1EE` for light (unchanged)
- Ensure border-radius stays `18px 18px 4px 18px`
- Remove any avatar from user messages if present

- [ ] **Step 5: Update AI message styling — remove background**

In `ChatMessage.tsx`, update assistant message rendering (around lines 255-320):

- Set background to `'transparent'` (no background, sits on page bg)
- Ensure AI avatar (W monogram) displays to the left

- [ ] **Step 6: Update code block styling with violet left-border**

In `ChatMessage.tsx`, update code block styling (around lines 276-290):

```typescript
// For code blocks (pre > code)
'& pre': {
  backgroundColor: '#161514',
  border: '1px solid rgba(255,255,255,0.07)',
  borderLeft: '4px solid',
  borderLeftColor: 'primary.main',
  borderRadius: '8px',
  padding: '12px 16px',
  overflow: 'auto',
  fontFamily: "'DM Mono', monospace",
  fontSize: '0.8125rem',
},
```

- [ ] **Step 7: Run tests**

Run: `cd frontend && npx vitest run src/__tests__/ChatMessage.test.tsx`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/ChatMessage.tsx frontend/src/__tests__/ChatMessage.test.tsx
git commit -m "feat: add AI/user message visual distinction with W monogram"
```

---

### Task 4: ChatInput Focus Ring Update

**Files:**
- Modify: `frontend/src/components/ChatInput.tsx:84-90`

- [ ] **Step 1: Update focus ring to violet**

In `ChatInput.tsx`, update the `&:focus-within` styles (lines 84-90):

```typescript
'&:focus-within': {
  borderColor: 'primary.main',
  boxShadow: '0 0 0 2px rgba(139, 92, 246, 0.15)',
},
```

Remove the dark/light mode conditional — violet works on both backgrounds.

- [ ] **Step 2: Verify visually**

Run: `cd frontend && npm run dev`
Navigate to a chat, click the input field, confirm violet focus ring appears.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ChatInput.tsx
git commit -m "feat: update ChatInput focus ring to violet"
```

---

### Task 5: QuickStart Chips Hover Styling

**Files:**
- Modify: `frontend/src/components/QuickStartChips.tsx:46-70`

- [ ] **Step 1: Update chip hover and selection styles**

In `QuickStartChips.tsx`, update chip sx props (lines 46-70):

```typescript
sx={{
  borderRadius: '9999px',
  border: '1px solid',
  borderColor: 'divider',
  bgcolor: 'transparent',
  color: 'text.secondary',
  px: 2,
  py: 1,
  cursor: 'pointer',
  transition: 'all 0.15s ease-out',
  '&:hover': {
    borderColor: 'primary.main',
    boxShadow: '0 0 12px rgba(139, 92, 246, 0.1)',
    color: 'text.primary',
  },
  ...(isSelected && {
    bgcolor: 'rgba(139, 92, 246, 0.08)',
    borderColor: 'primary.main',
    color: 'text.primary',
  }),
}}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/QuickStartChips.tsx
git commit -m "feat: add violet hover glow to QuickStart chips"
```

---

### Task 6: Empty State Redesign

**Files:**
- Modify: `frontend/src/views/ChatPage.tsx:422-466`

- [ ] **Step 1: Redesign empty state layout**

In `ChatPage.tsx`, update the empty state section (lines 422-466):

```typescript
{/* Empty state — shown when phase is greeting and no messages */}
{phase === 'greeting' && messages.length <= 1 && (
  <Box
    sx={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      flex: 1,
      gap: 3,
      px: 2,
    }}
  >
    {/* Watermark logomark */}
    <Typography
      sx={{
        fontFamily: "'DM Serif Display', Georgia, serif",
        fontSize: '6rem',
        fontWeight: 400,
        color: 'text.primary',
        opacity: 0.06,
        lineHeight: 1,
        userSelect: 'none',
      }}
    >
      W
    </Typography>

    {/* Heading */}
    <Typography
      variant="h1"
      sx={{
        fontFamily: "'DM Serif Display', Georgia, serif",
        fontSize: '1.5rem',
        fontWeight: 400,
        color: 'text.secondary',
        letterSpacing: '-0.02em',
        mt: -2,
      }}
    >
      What would you like to train?
    </Typography>

    {/* Chat input */}
    <Box sx={{ width: '100%', maxWidth: 600 }}>
      <ChatInput
        onSend={handleSend}
        onFileUpload={handleFileUpload}
        disabled={sending}
        placeholder="Describe what you want to build..."
      />
    </Box>

    {/* Quick start chips */}
    <QuickStartChips onSelect={handleSend} />
  </Box>
)}
```

- [ ] **Step 2: Verify visually**

Run: `cd frontend && npm run dev`
Navigate to `/chat` (new conversation). Confirm:
- Large faded "W" watermark centered
- "What would you like to train?" heading below
- Chat input below that
- Chips at the bottom

- [ ] **Step 3: Commit**

```bash
git add frontend/src/views/ChatPage.tsx
git commit -m "feat: redesign empty chat state with logomark and heading"
```

---

### Task 7: Typing Indicator

**Files:**
- Modify: `frontend/src/views/ChatPage.tsx:485-504`

- [ ] **Step 1: Replace streaming indicator with violet bar**

In `ChatPage.tsx`, update the streaming/thinking indicator (lines 485-504):

```typescript
{streaming && (
  <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.5, px: 2, py: 1 }}>
    {/* AI Avatar */}
    <Box
      sx={{
        width: 28,
        height: 28,
        borderRadius: '50%',
        bgcolor: 'primary.main',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        animation: 'pulse 1.5s ease-in-out infinite',
        '@keyframes pulse': {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.5 },
        },
      }}
    >
      <Typography
        sx={{
          fontFamily: "'DM Serif Display', Georgia, serif",
          fontSize: '0.8rem',
          color: '#FFFFFF',
          lineHeight: 1,
        }}
      >
        W
      </Typography>
    </Box>
    {/* Sliding bar indicator */}
    <Box
      sx={{
        width: 40,
        height: 3,
        borderRadius: 2,
        bgcolor: 'primary.main',
        mt: '12px',
        animation: 'slideBar 1.2s ease-in-out infinite',
        '@keyframes slideBar': {
          '0%': { transform: 'translateX(0)', opacity: 0.4 },
          '50%': { transform: 'translateX(12px)', opacity: 1 },
          '100%': { transform: 'translateX(0)', opacity: 0.4 },
        },
      }}
    />
  </Box>
)}
```

- [ ] **Step 2: Verify visually**

Run: `cd frontend && npm run dev`
Send a message in chat. Confirm:
- Pulsing violet W avatar appears
- Small violet bar slides next to it while AI is thinking

- [ ] **Step 3: Commit**

```bash
git add frontend/src/views/ChatPage.tsx
git commit -m "feat: add violet typing indicator with pulsing avatar"
```

---

### Task 8: Update CompletedModelCard — Remove Gradient

**Files:**
- Modify: `frontend/src/components/CompletedModelCard.tsx:76-86` (gradient bar)

- [ ] **Step 1: Replace gradient accent bar with solid violet**

In `CompletedModelCard.tsx`, update the accent bar (lines 76-86):

```typescript
{/* Accent bar — solid violet, no gradient */}
<Box
  sx={{
    height: 3,
    bgcolor: 'primary.main',
    borderRadius: '16px 16px 0 0',
  }}
/>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/CompletedModelCard.tsx
git commit -m "fix: replace gradient accent bar with solid violet"
```

---

### Task 9: Update InlinePredictWidget Drag-over Color

**Files:**
- Modify: `frontend/src/components/InlinePredictWidget.tsx`

- [ ] **Step 1: Update drag-over styling**

Find the drag-over background color `rgba(120,113,108,0.04)` and replace with `rgba(139,92,246,0.04)` (violet tint).

The `borderColor` on drag-over should already use `primary.main` which is now violet.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/InlinePredictWidget.tsx
git commit -m "fix: update drag-over color to match violet palette"
```

---

### Task 10: Update TrainingChart Palette

**Files:**
- Modify: `frontend/src/components/TrainingChart.tsx`

- [ ] **Step 1: Update chart colors to violet/amber**

In TrainingChart.tsx, replace the hardcoded color constants (around lines 39-40):

```typescript
// Before:
const accentLight = "rgba(126, 184, 255, 0.1)";
const accentLighter = "rgba(126, 184, 255, 0.6)";

// After:
const accentLight = themeTokens.accentMuted;  // violet at 0.15 opacity
const accentLighter = themeTokens.completion;  // #F59E0B amber for accuracy line
```

Also update any remaining direct color references:
- Loss line stroke: Should use `themeTokens.accent` (not hardcoded `#78716C`)
- Accuracy line stroke: Use `themeTokens.completion` (#F59E0B)
- Area fill: Use `themeTokens.accentMuted` for violet fill
- Grid color: Keep as-is (`rgba(255,255,255,0.06)`)

- [ ] **Step 2: Verify visually**

Run: `cd frontend && npm run dev`
Start a training session or view a completed model. Confirm chart uses violet for loss, amber for accuracy.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/TrainingChart.tsx
git commit -m "feat: update TrainingChart to violet/amber palette"
```

---

### Task 11: Phase 1 Integration Test

- [ ] **Step 1: Run all existing tests**

Run: `cd frontend && npx vitest run`
Expected: All tests pass (existing + new theme test + new ChatMessage test)

- [ ] **Step 2: Run build to catch type errors**

Run: `cd frontend && npm run build`
Expected: Build succeeds with no errors

- [ ] **Step 3: Run lint**

Run: `cd frontend && npm run lint`
Expected: No new lint errors

- [ ] **Step 4: Visual QA checklist**

Run: `cd frontend && npm run dev`

Verify in browser:
- [ ] Sidebar accent colors are now violet
- [ ] Chat input focus ring is violet
- [ ] QuickStart chips glow violet on hover
- [ ] Empty state shows W watermark + heading + chips
- [ ] AI messages show W monogram in violet circle
- [ ] User messages have no avatar, right-aligned
- [ ] Code blocks have violet left border
- [ ] Typing indicator shows pulsing W + sliding violet bar
- [ ] CompletedModelCard has solid violet bar (no gradient)
- [ ] InlinePredictWidget drag-over shows violet tint background
- [ ] Toggle light/dark mode — both look correct
- [ ] All monospace text uses DM Mono (no JetBrains Mono)

- [ ] **Step 5: Commit any fixes from QA**

```bash
git add -A
git commit -m "fix: phase 1 QA fixes"
```

---

## Chunk 2: Phase 2 — Training Cards

### Task 12: Build ProgressRing Component

**Files:**
- Create: `frontend/src/components/training/ProgressRing.tsx`
- Test: `frontend/src/__tests__/ProgressRing.test.tsx`

- [ ] **Step 1: Write failing test**

```typescript
// frontend/src/__tests__/ProgressRing.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import ProgressRing from '../components/training/ProgressRing';

const theme = getTheme('dark');
const wrap = (ui: React.ReactElement) =>
  render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

describe('ProgressRing', () => {
  it('renders percentage text', () => {
    wrap(<ProgressRing value={72} />);
    expect(screen.getByText('72%')).toBeTruthy();
  });

  it('renders 0% when value is 0', () => {
    wrap(<ProgressRing value={0} />);
    expect(screen.getByText('0%')).toBeTruthy();
  });

  it('renders completed state with amber color', () => {
    wrap(<ProgressRing value={100} completed />);
    expect(screen.getByText('100%')).toBeTruthy();
  });

  it('has progressbar role with correct aria attributes', () => {
    wrap(<ProgressRing value={72} />);
    const ring = screen.getByRole('progressbar');
    expect(ring.getAttribute('aria-valuenow')).toBe('72');
    expect(ring.getAttribute('aria-valuemin')).toBe('0');
    expect(ring.getAttribute('aria-valuemax')).toBe('100');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/__tests__/ProgressRing.test.tsx`
Expected: FAIL — component doesn't exist

- [ ] **Step 3: Create ProgressRing component**

```bash
mkdir -p frontend/src/components/training
```

```typescript
// frontend/src/components/training/ProgressRing.tsx
'use client';

import { Box, Typography } from '@mui/material';
import { themeTokens } from '../../theme';

interface ProgressRingProps {
  value: number;
  size?: number;
  strokeWidth?: number;
  completed?: boolean;
}

export default function ProgressRing({
  value,
  size = 64,
  strokeWidth = 4,
  completed = false,
}: ProgressRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;
  const color = completed ? themeTokens.completion : themeTokens.accent;

  return (
    <Box
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={100}
      sx={{
        position: 'relative',
        width: size,
        height: size,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.07)"
          strokeWidth={strokeWidth}
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 0.5s ease-out, stroke 0.5s ease-out' }}
        />
      </svg>
      <Typography
        sx={{
          position: 'absolute',
          fontFamily: "'DM Mono', monospace",
          fontWeight: 700,
          fontSize: size * 0.22,
          color: 'text.primary',
        }}
      >
        {value}%
      </Typography>
    </Box>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/__tests__/ProgressRing.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/training/ProgressRing.tsx frontend/src/__tests__/ProgressRing.test.tsx
git commit -m "feat: add ProgressRing SVG component"
```

---

### Task 13: Build StageIndicator Component

**Files:**
- Create: `frontend/src/components/training/StageIndicator.tsx`
- Test: `frontend/src/__tests__/StageIndicator.test.tsx`

- [ ] **Step 1: Write failing test**

```typescript
// frontend/src/__tests__/StageIndicator.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import StageIndicator from '../components/training/StageIndicator';

const theme = getTheme('dark');
const wrap = (ui: React.ReactElement) =>
  render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

type Stage = 'preparing' | 'training' | 'evaluating' | 'complete' | 'ready';

describe('StageIndicator', () => {
  it('renders all 5 stage labels', () => {
    wrap(<StageIndicator currentStage="preparing" />);
    expect(screen.getByText('Preparing')).toBeTruthy();
    expect(screen.getByText('Training')).toBeTruthy();
    expect(screen.getByText('Evaluating')).toBeTruthy();
    expect(screen.getByText('Complete')).toBeTruthy();
    expect(screen.getByText('Ready')).toBeTruthy();
  });

  it('marks completed stages with checkmarks', () => {
    wrap(<StageIndicator currentStage="evaluating" />);
    // Preparing and Training should have check icons
    const checks = screen.getAllByTestId('stage-check');
    expect(checks.length).toBe(2);
  });

  it('has aria-label describing current stage', () => {
    wrap(<StageIndicator currentStage="training" />);
    const indicator = screen.getByRole('status');
    expect(indicator.getAttribute('aria-label')).toContain('Training');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/__tests__/StageIndicator.test.tsx`
Expected: FAIL — component doesn't exist

- [ ] **Step 3: Create StageIndicator component**

```typescript
// frontend/src/components/training/StageIndicator.tsx
'use client';

import { Box, Typography } from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import { themeTokens } from '../../theme';

export type TrainingStage = 'preparing' | 'training' | 'evaluating' | 'complete' | 'ready';

const STAGES: { key: TrainingStage; label: string }[] = [
  { key: 'preparing', label: 'Preparing' },
  { key: 'training', label: 'Training' },
  { key: 'evaluating', label: 'Evaluating' },
  { key: 'complete', label: 'Complete' },
  { key: 'ready', label: 'Ready' },
];

const STAGE_INDEX: Record<TrainingStage, number> = {
  preparing: 0,
  training: 1,
  evaluating: 2,
  complete: 3,
  ready: 4,
};

interface StageIndicatorProps {
  currentStage: TrainingStage;
}

export default function StageIndicator({ currentStage }: StageIndicatorProps) {
  const currentIdx = STAGE_INDEX[currentStage];

  return (
    <Box
      role="status"
      aria-label={`Current stage: ${STAGES[currentIdx].label}`}
      sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}
    >
      {STAGES.map((stage, idx) => {
        const isCompleted = idx < currentIdx;
        const isActive = idx === currentIdx;
        const isAmber = idx >= 3 && isCompleted;
        const isAmberActive = idx >= 3 && isActive && currentIdx >= 3;

        const dotColor = isCompleted || isActive
          ? (isAmber || isAmberActive ? themeTokens.completion : themeTokens.accent)
          : themeTokens.textMuted;

        return (
          <Box
            key={stage.key}
            sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.5, minWidth: 56 }}
          >
            {/* Dot */}
            <Box
              sx={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                bgcolor: isCompleted || isActive ? dotColor : 'transparent',
                border: isCompleted || isActive ? 'none' : `2px solid ${themeTokens.textMuted}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                ...(isActive && !isCompleted && {
                  animation: 'stagePulse 1.5s ease-in-out infinite',
                  '@keyframes stagePulse': {
                    '0%, 100%': { opacity: 1 },
                    '50%': { opacity: 0.5 },
                  },
                }),
              }}
            >
              {isCompleted && (
                <CheckIcon
                  data-testid="stage-check"
                  sx={{ fontSize: 8, color: '#FFFFFF' }}
                />
              )}
            </Box>
            {/* Label */}
            <Typography
              sx={{
                fontSize: '0.625rem',
                fontFamily: "'DM Mono', monospace",
                color: isActive ? 'text.primary' : themeTokens.textMuted,
                fontWeight: isActive ? 600 : 400,
              }}
            >
              {stage.label}
            </Typography>
          </Box>
        );
      })}
    </Box>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/__tests__/StageIndicator.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/training/StageIndicator.tsx frontend/src/__tests__/StageIndicator.test.tsx
git commit -m "feat: add StageIndicator component with dot progression"
```

---

### Task 14: Build SparklineChart Component

**Files:**
- Create: `frontend/src/components/training/SparklineChart.tsx`
- Test: `frontend/src/__tests__/SparklineChart.test.tsx`

- [ ] **Step 1: Write failing test**

```typescript
// frontend/src/__tests__/SparklineChart.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import SparklineChart from '../components/training/SparklineChart';

const theme = getTheme('dark');
const wrap = (ui: React.ReactElement) =>
  render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

describe('SparklineChart', () => {
  it('renders without crashing', () => {
    const data = [
      { epoch: 1, loss: 2.3 },
      { epoch: 2, loss: 1.8 },
      { epoch: 3, loss: 1.2 },
    ];
    const { container } = wrap(<SparklineChart data={data} />);
    expect(container.querySelector('.recharts-line')).toBeTruthy();
  });

  it('renders nothing when data is empty', () => {
    const { container } = wrap(<SparklineChart data={[]} />);
    expect(container.querySelector('.recharts-line')).toBeNull();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/__tests__/SparklineChart.test.tsx`
Expected: FAIL — component doesn't exist

- [ ] **Step 3: Create SparklineChart component**

```typescript
// frontend/src/components/training/SparklineChart.tsx
'use client';

import { Box } from '@mui/material';
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import { themeTokens } from '../../theme';

interface SparklineDataPoint {
  epoch: number;
  loss: number;
}

interface SparklineChartProps {
  data: SparklineDataPoint[];
  height?: number;
}

export default function SparklineChart({ data, height = 48 }: SparklineChartProps) {
  if (data.length === 0) return null;

  return (
    <Box sx={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <Line
            type="monotone"
            dataKey="loss"
            stroke={themeTokens.accent}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/__tests__/SparklineChart.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/training/SparklineChart.tsx frontend/src/__tests__/SparklineChart.test.tsx
git commit -m "feat: add SparklineChart inline loss chart"
```

---

### Task 15: Build TrainingCard Component

**Files:**
- Create: `frontend/src/components/training/TrainingCard.tsx`
- Test: `frontend/src/__tests__/TrainingCard.test.tsx`

- [ ] **Step 1: Write failing test**

```typescript
// frontend/src/__tests__/TrainingCard.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import TrainingCard from '../components/training/TrainingCard';

const theme = getTheme('dark');
const wrap = (ui: React.ReactElement) =>
  render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

describe('TrainingCard', () => {
  const defaultProps = {
    modelName: 'MNIST Classifier',
    stage: 'training' as const,
    epoch: 12,
    totalEpochs: 50,
    loss: 0.12,
    accuracy: 94.2,
    lossHistory: [
      { epoch: 1, loss: 2.3 },
      { epoch: 5, loss: 0.8 },
      { epoch: 12, loss: 0.12 },
    ],
    isLatestActive: true,
  };

  it('renders model name', () => {
    wrap(<TrainingCard {...defaultProps} />);
    expect(screen.getByText('MNIST Classifier')).toBeTruthy();
  });

  it('renders progress percentage', () => {
    wrap(<TrainingCard {...defaultProps} />);
    expect(screen.getByText('24%')).toBeTruthy(); // 12/50 = 24%
  });

  it('renders epoch count', () => {
    wrap(<TrainingCard {...defaultProps} />);
    expect(screen.getByText(/12\/50/)).toBeTruthy();
  });

  it('renders loss and accuracy stats', () => {
    wrap(<TrainingCard {...defaultProps} />);
    expect(screen.getByText(/0\.12/)).toBeTruthy();
    expect(screen.getByText(/94\.2/)).toBeTruthy();
  });

  it('has aria-live for screen readers', () => {
    const { container } = wrap(<TrainingCard {...defaultProps} />);
    const liveRegion = container.querySelector('[aria-live="polite"]');
    expect(liveRegion).toBeTruthy();
  });

  it('shows expand button', () => {
    wrap(<TrainingCard {...defaultProps} />);
    expect(screen.getByText('Expand Details')).toBeTruthy();
  });

  it('expands on click to show chart', () => {
    wrap(<TrainingCard {...defaultProps} />);
    fireEvent.click(screen.getByText('Expand Details'));
    expect(screen.getByText('Collapse')).toBeTruthy();
  });

  it('renders completed state', () => {
    wrap(
      <TrainingCard
        {...defaultProps}
        stage="complete"
        epoch={50}
        accuracy={97.5}
      />
    );
    expect(screen.getByText(/97\.5/)).toBeTruthy();
  });

  it('renders error state', () => {
    wrap(
      <TrainingCard
        {...defaultProps}
        stage="training"
        error="Out of memory"
      />
    );
    expect(screen.getByText(/Out of memory/)).toBeTruthy();
  });

  it('hides progress ring in error state', () => {
    wrap(
      <TrainingCard
        {...defaultProps}
        error="Out of memory"
      />
    );
    expect(screen.queryByRole('progressbar')).toBeNull();
  });

  it('shows cancel button when onCancel provided and active', () => {
    const onCancel = vi.fn();
    wrap(<TrainingCard {...defaultProps} onCancel={onCancel} />);
    expect(screen.getByText('Cancel')).toBeTruthy();
  });

  it('hides cancel button in completed state', () => {
    const onCancel = vi.fn();
    wrap(
      <TrainingCard
        {...defaultProps}
        stage="complete"
        epoch={50}
        onCancel={onCancel}
      />
    );
    expect(screen.queryByText('Cancel')).toBeNull();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/__tests__/TrainingCard.test.tsx`
Expected: FAIL — component doesn't exist

- [ ] **Step 3: Create TrainingCard component**

```typescript
// frontend/src/components/training/TrainingCard.tsx
'use client';

import { useState } from 'react';
import { Box, Typography, Button, Collapse } from '@mui/material';
import { themeTokens } from '../../theme';
import ProgressRing from './ProgressRing';
import StageIndicator, { TrainingStage } from './StageIndicator';
import SparklineChart from './SparklineChart';
import TrainingChart from '../TrainingChart';
import InlinePredictWidget from '../InlinePredictWidget';

interface TrainingCardProps {
  modelName: string;
  stage: TrainingStage;
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  lossHistory: { epoch: number; loss: number; accuracy?: number }[];
  isLatestActive: boolean;
  modelId?: string;
  error?: string;
  architecture?: string[];
  datasetName?: string;
  learningRate?: number;
  batchSize?: number;
  optimizer?: string;
  onCancel?: () => void;
  onRetry?: () => void;
}

export default function TrainingCard({
  modelName,
  stage,
  epoch,
  totalEpochs,
  loss,
  accuracy,
  lossHistory,
  isLatestActive,
  modelId,
  error,
  architecture,
  datasetName,
  learningRate,
  batchSize,
  optimizer,
  onCancel,
  onRetry,
}: TrainingCardProps) {
  const [expanded, setExpanded] = useState(false);

  const percentage = totalEpochs > 0 ? Math.round((epoch / totalEpochs) * 100) : 0;
  const isCompleted = stage === 'complete' || stage === 'ready';
  const isError = !!error;
  const isActive = !isCompleted && !isError;

  const borderColor = isError
    ? themeTokens.error
    : isCompleted
      ? themeTokens.completion
      : themeTokens.accent;

  const glowColor = isError
    ? 'rgba(239, 68, 68, 0.12)'
    : isCompleted
      ? themeTokens.completionMuted
      : themeTokens.accentMuted;

  return (
    <Box
      aria-live="polite"
      sx={{
        width: '90%',
        maxWidth: 640,
        mx: 'auto',
        bgcolor: themeTokens.card,
        border: `1px solid ${borderColor}`,
        borderRadius: '16px',
        p: 2.5,
        position: 'relative',
        overflow: 'hidden',
        boxShadow: `0 0 16px ${glowColor}`,
        // Pulsing glow via pseudo-element (GPU-compositable)
        ...(isActive && isLatestActive && {
          '&::after': {
            content: '""',
            position: 'absolute',
            inset: -1,
            borderRadius: '16px',
            boxShadow: `0 0 20px ${themeTokens.accentMuted}`,
            animation: 'glowPulse 2s ease-in-out infinite',
            pointerEvents: 'none',
            '@keyframes glowPulse': {
              '0%, 100%': { opacity: 1 },
              '50%': { opacity: 0.3 },
            },
          },
        }),
      }}
    >
      {/* Header */}
      <Typography
        sx={{
          fontFamily: "'DM Serif Display', Georgia, serif",
          fontSize: '1rem',
          fontWeight: 400,
          color: 'text.primary',
          mb: 1.5,
        }}
      >
        {isError ? 'Training Failed' : isCompleted ? 'Training Complete' : 'Training'}: {modelName}
      </Typography>

      {/* Stage Indicator */}
      <Box sx={{ mb: 2 }}>
        <StageIndicator currentStage={stage} />
      </Box>

      {/* Error state */}
      {isError && (
        <Box sx={{ mb: 2 }}>
          <Typography sx={{ color: themeTokens.error, fontSize: '0.875rem', mb: 1 }}>
            {error}
          </Typography>
          {onRetry && (
            <Button size="small" variant="outlined" onClick={onRetry} sx={{ borderColor: themeTokens.error, color: themeTokens.error }}>
              Retry
            </Button>
          )}
        </Box>
      )}

      {/* Progress section — hidden on error */}
      {!isError && (
        <>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2.5, mb: 2 }}>
            {/* Progress Ring */}
            <ProgressRing value={isCompleted ? 100 : percentage} completed={isCompleted} />

            {/* Stats */}
            <Box sx={{ flex: 1 }}>
              <Typography sx={{ fontFamily: "'DM Mono', monospace", fontSize: '0.875rem', color: 'text.primary' }}>
                Epoch {epoch}/{totalEpochs}
              </Typography>
              <Typography sx={{ fontFamily: "'DM Mono', monospace", fontSize: '0.8125rem', color: 'text.secondary' }}>
                Loss: {loss.toFixed(4)}
              </Typography>
              <Typography sx={{ fontFamily: "'DM Mono', monospace", fontSize: '0.8125rem', color: 'text.secondary' }}>
                Accuracy: {accuracy.toFixed(1)}%
              </Typography>
            </Box>
          </Box>

          {/* Sparkline */}
          {lossHistory.length > 1 && (
            <Box sx={{ mb: 2 }}>
              <SparklineChart data={lossHistory} />
            </Box>
          )}
        </>
      )}

      {/* Actions */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Button
          size="small"
          onClick={() => setExpanded(!expanded)}
          sx={{ color: 'text.secondary', textTransform: 'none', fontSize: '0.8125rem' }}
        >
          {expanded ? 'Collapse' : 'Expand Details'}
        </Button>
        {isActive && onCancel && (
          <Button
            size="small"
            onClick={onCancel}
            sx={{ color: themeTokens.textMuted, textTransform: 'none', fontSize: '0.8125rem' }}
          >
            Cancel
          </Button>
        )}
      </Box>

      {/* Expanded details — uses MUI Collapse for animated height transition */}
      <Collapse in={expanded} timeout={200} easing="ease-out">
        <Box
          sx={{
            mt: 2,
            pt: 2,
            borderTop: `1px solid ${themeTokens.border}`,
          }}
        >
          {/* Full chart */}
          {lossHistory.length > 0 && (
            <Box sx={{ mb: 2, height: 200 }}>
              <TrainingChart
                data={lossHistory.map((d) => ({
                  epoch: d.epoch,
                  loss: d.loss,
                  accuracy: d.accuracy ?? 0,
                }))}
              />
            </Box>
          )}

          {/* Hyperparameters */}
          {(learningRate || batchSize || optimizer) && (
            <Box sx={{ mb: 2 }}>
              <Typography sx={{ fontSize: '0.75rem', color: themeTokens.textMuted, mb: 0.5, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                Hyperparameters
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                {optimizer && (
                  <Typography sx={{ fontFamily: "'DM Mono', monospace", fontSize: '0.8125rem', color: 'text.secondary' }}>
                    Optimizer: {optimizer}
                  </Typography>
                )}
                {learningRate && (
                  <Typography sx={{ fontFamily: "'DM Mono', monospace", fontSize: '0.8125rem', color: 'text.secondary' }}>
                    LR: {learningRate}
                  </Typography>
                )}
                {batchSize && (
                  <Typography sx={{ fontFamily: "'DM Mono', monospace", fontSize: '0.8125rem', color: 'text.secondary' }}>
                    Batch: {batchSize}
                  </Typography>
                )}
              </Box>
            </Box>
          )}

          {/* Inline predict widget for completed models */}
          {isCompleted && modelId && (
            <Box sx={{ mt: 2 }}>
              <InlinePredictWidget modelId={modelId} />
            </Box>
          )}
        </Box>
      </Collapse>
    </Box>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/__tests__/TrainingCard.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/training/TrainingCard.tsx frontend/src/__tests__/TrainingCard.test.tsx
git commit -m "feat: add TrainingCard breakout component"
```

---

### Task 16: Integrate TrainingCard into ChatMessage

**Files:**
- Modify: `frontend/src/components/ChatMessage.tsx:47-167`
- Modify: `frontend/src/views/ChatPage.tsx` (isLatestActive prop logic)

- [ ] **Step 1: Update ChatMessage imports**

Add to ChatMessage.tsx imports:

```typescript
import TrainingCard from './training/TrainingCard';
```

- [ ] **Step 2: Update training_progress layout wrapper**

The `training_progress` case keeps using `TrainingProgress` (which manages SSE and will render TrainingCard internally after Task 17). Only update the layout wrapper to center the card and remove the AI avatar:

```typescript
case 'training_progress':
  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', py: 1 }}>
      <TrainingProgress
        conversationId={message.metadata?.conversation_id as string}
        jobId={message.metadata?.job_id as string}
        onComplete={message.metadata?.onComplete as (() => void) | undefined}
        isLatestActive={message.metadata?.isLatestActive as boolean ?? false}
      />
    </Box>
  );
```

Note: `TrainingProgress` will be updated in Task 17 to render TrainingCard internally and accept `isLatestActive` as a prop.

- [ ] **Step 3: Replace training_complete rendering**

Update the `training_complete` case (around lines 148-167) to render TrainingCard in completed state. Note: use snake_case metadata field names matching the backend:

```typescript
case 'training_complete':
  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', py: 1 }}>
      <TrainingCard
        modelName={message.metadata?.model_name as string ?? 'Model'}
        stage="complete"
        epoch={message.metadata?.total_epochs as number ?? 0}
        totalEpochs={message.metadata?.total_epochs as number ?? 0}
        loss={message.metadata?.loss as number ?? 0}
        accuracy={message.metadata?.accuracy as number ?? 0}
        lossHistory={message.metadata?.loss_history as { epoch: number; loss: number; accuracy?: number }[] ?? []}
        isLatestActive={false}
        modelId={message.metadata?.model_id as string}
        optimizer={message.metadata?.optimizer as string}
        learningRate={message.metadata?.learning_rate as number}
        batchSize={message.metadata?.batch_size as number}
      />
    </Box>
  );
```

- [ ] **Step 4: Replace training_error rendering**

Update the `training_error` case (around lines 107-145) to render TrainingCard in error state:

```typescript
case 'training_error':
  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', py: 1 }}>
      <TrainingCard
        modelName={message.metadata?.model_name as string ?? 'Model'}
        stage={message.metadata?.stage as TrainingStage ?? 'training'}
        epoch={message.metadata?.epoch as number ?? 0}
        totalEpochs={message.metadata?.total_epochs as number ?? 0}
        loss={message.metadata?.loss as number ?? 0}
        accuracy={message.metadata?.accuracy as number ?? 0}
        lossHistory={message.metadata?.loss_history as { epoch: number; loss: number; accuracy?: number }[] ?? []}
        isLatestActive={false}
        error={message.metadata?.error as string ?? message.content}
      />
    </Box>
  );
```

- [ ] **Step 5: Add isLatestActive logic to ChatPage**

In `ChatPage.tsx`, when mapping messages to determine `isLatestActive`, find the last message with `type === 'training_progress'` and set `isLatestActive: true` only for that one:

```typescript
// In ChatPage.tsx, in the message mapping section
const lastActiveTrainingIdx = messages.reduceRight(
  (found, msg, idx) =>
    found === -1 && msg.type === 'training_progress' ? idx : found,
  -1
);

// Then when rendering messages, pass isLatestActive in metadata:
messages.map((msg, idx) => ({
  ...msg,
  metadata: {
    ...msg.metadata,
    isLatestActive: idx === lastActiveTrainingIdx,
  },
}))
```

- [ ] **Step 6: Run tests**

Run: `cd frontend && npx vitest run`
Expected: All tests pass

- [ ] **Step 7: Run build**

Run: `cd frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/ChatMessage.tsx frontend/src/views/ChatPage.tsx
git commit -m "feat: integrate TrainingCard into ChatMessage flow"
```

---

### Task 17: Refactor TrainingProgress to Wire TrainingCard

**Files:**
- Modify: `frontend/src/components/TrainingProgress.tsx`

- [ ] **Step 1: Update TrainingProgress to emit TrainingCard-compatible data**

The existing `TrainingProgress.tsx` manages the SSE connection and tracks epoch history. Instead of rendering its own UI, it should now pass data up to the parent (ChatPage) which will feed it into the TrainingCard via message metadata.

Keep TrainingProgress as the SSE data manager, but update its rendered output to use TrainingCard internally. Add `isLatestActive` as a new prop passed through from ChatMessage.

Update the TrainingProgress props interface to accept `isLatestActive`:

```typescript
interface TrainingProgressProps {
  conversationId: string;
  jobId: string;
  onComplete?: () => void;
  isLatestActive?: boolean;  // NEW — controls pulsing glow
}
```

Replace the entire render section with:

```typescript
import TrainingCard from './training/TrainingCard';

return (
  <TrainingCard
    modelName={status?.model_name ?? 'Model'}
    stage={mapStatusToStage(status?.status)}
    epoch={status?.epoch ?? 0}
    totalEpochs={status?.total_epochs ?? 0}
    loss={status?.loss ?? 0}
    accuracy={status?.accuracy ?? 0}
    lossHistory={epochHistory.map((e) => ({ epoch: e.epoch, loss: e.loss, accuracy: e.accuracy }))}
    isLatestActive={isLatestActive ?? false}
    onCancel={onCancel}
  />
);
```

Add a helper function to map backend status strings to TrainingStage:

```typescript
import { TrainingStage } from './training/StageIndicator';

function mapStatusToStage(status?: string): TrainingStage {
  switch (status) {
    case 'preparing': return 'preparing';
    case 'training': return 'training';
    case 'evaluating': return 'evaluating';
    case 'completed': return 'complete';
    case 'ready': return 'ready';
    default: return 'preparing';
  }
}
```

- [ ] **Step 2: Run tests and build**

Run: `cd frontend && npx vitest run && npm run build`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/TrainingProgress.tsx
git commit -m "refactor: TrainingProgress now renders TrainingCard"
```

---

### Task 18: Phase 2 Integration Test

- [ ] **Step 1: Run full test suite**

Run: `cd frontend && npx vitest run`
Expected: All tests pass

- [ ] **Step 2: Run build**

Run: `cd frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 3: Visual QA checklist**

Run: `cd frontend && npm run dev`

Verify in browser:
- [ ] Training card appears inline in chat during active training
- [ ] Card has violet border and pulsing glow
- [ ] Progress ring shows correct percentage
- [ ] Stage dots progress correctly
- [ ] Sparkline chart renders loss history
- [ ] "Expand Details" reveals full chart and hyperparameters
- [ ] Completed training shows amber border/ring
- [ ] Error state shows red border with error message
- [ ] Only the most recent active card pulses
- [ ] Card is wider than regular messages (~90% width)

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: phase 2 QA fixes"
```

---

## Chunk 3: Phase 3 — Polish + Accessibility

### Task 19: Message Appear Animations

**Files:**
- Modify: `frontend/src/components/ChatMessage.tsx`

- [ ] **Step 1: Add fade-in + slide animation to message wrapper**

In ChatMessage.tsx, wrap the outermost Box of each message type with an animation:

```typescript
// Add to the outermost Box sx of each message type render:
animation: 'messageAppear 0.15s ease-out',
'@keyframes messageAppear': {
  from: { opacity: 0, transform: 'translateY(8px)' },
  to: { opacity: 1, transform: 'translateY(0)' },
},
```

- [ ] **Step 2: Run tests to verify no regressions**

Run: `cd frontend && npx vitest run`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ChatMessage.tsx
git commit -m "feat: add message appear animation"
```

---

### Task 20: Reduced Motion Support

**Files:**
- Modify: `frontend/src/components/ChatMessage.tsx`
- Modify: `frontend/src/components/training/TrainingCard.tsx`
- Modify: `frontend/src/components/training/StageIndicator.tsx`
- Modify: `frontend/src/components/training/ProgressRing.tsx`
- Modify: `frontend/src/components/QuickStartChips.tsx`
- Modify: `frontend/src/views/ChatPage.tsx`

- [ ] **Step 1: Add prefers-reduced-motion media query to all animated components**

In every component that has CSS animations or transitions, add the reduced-motion override. Full list of animations from the spec:

```typescript
'@media (prefers-reduced-motion: reduce)': {
  animation: 'none',
  transition: 'none',
},
```

Components and specific animations to cover:

| Component | Animation | Override needed |
|-----------|-----------|----------------|
| **ChatMessage.tsx** | `messageAppear` keyframe | `animation: 'none'` |
| **TrainingCard.tsx** | `glowPulse` pseudo-element | `animation: 'none'` on `&::after` |
| **TrainingCard.tsx** | Collapse expand/collapse | MUI Collapse already respects reduced-motion |
| **TrainingCard.tsx** | Border color transition (violet→amber on complete) | `transition: 'none'` on border/boxShadow |
| **StageIndicator.tsx** | `stagePulse` keyframe (active dot) | `animation: 'none'` |
| **StageIndicator.tsx** | Dot fill transition (opacity+scale 0.3s) | `transition: 'none'` |
| **ProgressRing.tsx** | `stroke-dashoffset` transition | `transition: 'none'` on SVG circle style |
| **QuickStartChips.tsx** | Hover transition (`all 0.15s ease-out`) | `transition: 'none'` |
| **ChatPage.tsx** | Typing indicator `pulse` keyframe | `animation: 'none'` |
| **ChatPage.tsx** | Typing indicator `slideBar` keyframe | `animation: 'none'` |

Note: Page transition fades are deferred — Next.js App Router does not natively support route transition animations without additional libraries. This is not worth the complexity for Phase 3.

- [ ] **Step 2: Run tests**

Run: `cd frontend && npx vitest run`
Expected: All tests pass (animations don't affect test assertions)

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ChatMessage.tsx frontend/src/components/training/TrainingCard.tsx frontend/src/components/training/StageIndicator.tsx frontend/src/components/training/ProgressRing.tsx frontend/src/components/QuickStartChips.tsx frontend/src/views/ChatPage.tsx
git commit -m "a11y: respect prefers-reduced-motion across all animations"
```

---

### Task 21: Focus State Updates

**Files:**
- Modify: `frontend/src/theme.ts` (MUI component overrides)

- [ ] **Step 1: Update global focus-visible styles in theme**

In `theme.ts`, add a global `MuiButtonBase` override for focus-visible (covers Button, IconButton, Tab, MenuItem, Chip):

```typescript
MuiButtonBase: {
  styleOverrides: {
    root: {
      '&:focus-visible': {
        outline: '2px solid #8B5CF6',
        outlineOffset: '2px',
      },
    },
  },
},
```

Also add `MuiOutlinedInput` override (text inputs do NOT extend MuiButtonBase):

```typescript
MuiOutlinedInput: {
  styleOverrides: {
    root: {
      '&.Mui-focused': {
        '& .MuiOutlinedInput-notchedOutline': {
          borderColor: '#8B5CF6',
          borderWidth: '2px',
        },
      },
      '&:focus-visible': {
        outline: '2px solid #8B5CF6',
        outlineOffset: '2px',
      },
    },
  },
},
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/theme.ts
git commit -m "a11y: add violet focus-visible outlines to all interactive elements"
```

---

### Task 22: Models Tab Empty State

**Files:**
- Modify: `frontend/src/components/ModelsTab.tsx`

- [ ] **Step 1: Add empty state to ModelsTab**

When the models list is empty, show a centered empty state similar to the chat empty state:

```typescript
import { themeTokens } from '../theme';

{models.length === 0 && (
  <Box
    sx={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      py: 8,
      gap: 2,
    }}
  >
    <Typography
      sx={{
        fontFamily: "'DM Serif Display', Georgia, serif",
        fontSize: '4rem',
        color: 'text.primary',
        opacity: 0.06,
        lineHeight: 1,
        userSelect: 'none',
      }}
    >
      W
    </Typography>
    <Typography
      sx={{
        fontFamily: "'DM Serif Display', Georgia, serif",
        fontSize: '1.25rem',
        color: 'text.secondary',
      }}
    >
      No models yet
    </Typography>
    <Typography
      sx={{
        fontSize: '0.875rem',
        color: themeTokens.textMuted,
      }}
    >
      Start a conversation to train your first model.
    </Typography>
  </Box>
)}
```

- [ ] **Step 2: Run tests to verify no regressions**

Run: `cd frontend && npx vitest run`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ModelsTab.tsx
git commit -m "feat: add empty state to ModelsTab"
```

---

### Task 23: Final Build and QA

- [ ] **Step 1: Run full test suite**

Run: `cd frontend && npx vitest run`
Expected: All tests pass

- [ ] **Step 2: Run build**

Run: `cd frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 3: Run lint**

Run: `cd frontend && npm run lint`
Expected: No lint errors

- [ ] **Step 4: Full visual QA**

Run: `cd frontend && npm run dev`

Test dark mode:
- [ ] All violet accents render correctly
- [ ] Training cards pulse, complete to amber, show errors in red
- [ ] Message animations are smooth
- [ ] Empty states show W watermark
- [ ] Focus rings are visible on tab navigation (Tab through entire UI)
- [ ] Code blocks have violet left border
- [ ] All monospace text is DM Mono

Test light mode:
- [ ] Violet accents are darker for contrast
- [ ] Card backgrounds adapt
- [ ] Text remains readable
- [ ] Focus rings visible on light background

Test reduced motion:
- [ ] Enable `prefers-reduced-motion: reduce` in browser dev tools
- [ ] All animations are disabled (no pulsing, no slides)
- [ ] State changes still visible through color/text

Test accessibility:
- [ ] Training card `aria-live="polite"` announces state changes to screen reader
- [ ] Progress ring has `role="progressbar"` with correct `aria-valuenow`/`aria-valuemin`/`aria-valuemax`
- [ ] Stage indicator has descriptive `aria-label` (e.g., "Current stage: Training")
- [ ] All state changes are communicated via text, not just color

- [ ] **Step 5: Commit any final fixes**

```bash
git add frontend/src/components/ frontend/src/views/ frontend/src/theme.ts
git commit -m "fix: final QA polish"
```
