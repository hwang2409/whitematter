# WhiteMatter UI Redesign — "Living Workspace"

## Overview

Redesign the WhiteMatter frontend from a generic chatbot UI into a distinctive, recognizable product experience. The chat-first workflow stays intact, but training cards "break out" of the conversation as rich, live-updating visual elements. A new color system (electric violet + warm amber on a warm dark base) replaces the current bland taupe palette, creating instant brand recognition.

**Target audience:** Non-technical users who want to train ML models without writing code.

**Design influences:** Stripe (bold color confidence, premium feel) + Notion/Arc (warm, approachable, slightly playful).

**Constraints:**
- No gradients anywhere — solid colors only
- No emojis in UI copy
- Preserve the existing chat-first navigational flow
- Keep the existing layout structure (sidebar + chat + top bar)

---

## 1. Color System & Visual Identity

### Signature Colors

Replace the current taupe accent (`#78716C`) with a dual-accent system:

| Token | Value | Usage |
|-------|-------|-------|
| `accent.main` | `#8B5CF6` (electric violet) | Active states, CTAs, training card borders, focus rings, AI avatar |
| `accent.light` | `#A78BFA` | Hover states, lighter UI accents |
| `accent.dark` | `#7C3AED` | Pressed states |
| `success` / completion | `#F59E0B` (warm amber) | Completion indicators, success states, progress fills |
| `success.light` | `#FBBF24` | Hover on success elements |

### Background & Surface (unchanged)

| Token | Value |
|-------|-------|
| `bg` | `#141311` |
| `surface` | `#1E1D1B` |
| `card` | `#1C1A18` (slight warm tint adjustment) |
| `border` | `rgba(255,255,255,0.07)` |
| `borderHover` | `rgba(255,255,255,0.14)` |

### Text (unchanged)

| Token | Value |
|-------|-------|
| `text.primary` | `#F2F1EE` |
| `text.secondary` | `#9C9A95` |
| `text.muted` | `#6B6963` |

### System Colors

| Token | Value |
|-------|-------|
| `error` | `#ef4444` (unchanged) |
| `warning` | `#eab308` (unchanged) |
| `success` | `#22c55e` → `#F59E0B` (amber replaces green for success/completion) |

### Rationale

The violet-to-amber pairing is uncommon in ML tooling (most use blue or green). Violet reads as "intelligent/creative," amber as "warm/approachable" — matching the Stripe + Notion influence. Both pass WCAG AA contrast on the dark background (violet: 5.2:1, amber: 4.6:1 — amber used only for large indicators, never body text).

---

## 2. Training Cards — The Signature Element

Training cards are the visual centerpiece. They break out of the chat stream — wider than regular messages, with their own visual language.

### Card Anatomy

```
┌─────────────────────────────────────────────┐
│  Training: MNIST Classifier                 │  Header with model name
│  ┌───────────────────────────────────────┐  │
│  │  ●●●○○  Preparing > Training > ...    │  │  Stage indicator (dots + labels)
│  └───────────────────────────────────────┘  │
│                                             │
│  ╭─────────╮  Epoch 12/50                   │
│  │   72%   │  Loss: 0.34 > 0.12             │  Progress ring + live stats
│  ╰─────────╯  Accuracy: 94.2%              │
│                                             │
│  ┌───────────────────────────────────────┐  │
│  │  sparkline loss chart                  │  │  Inline mini chart
│  └───────────────────────────────────────┘  │
│                                             │
│  [ Expand Details ]           [ Cancel ]    │  Actions
└─────────────────────────────────────────────┘
```

### Visual Treatment

- **Width:** ~90% of content area (vs ~70% for regular chat messages)
- **Border:** 1px solid violet (`#8B5CF6`) during active training
- **Glow:** `box-shadow: 0 0 20px rgba(139, 92, 246, 0.15)` — pulses on 2s ease-in-out cycle during active training
- **Completion:** Border transitions to amber (`#F59E0B`), glow settles to steady `box-shadow: 0 0 16px rgba(245, 158, 11, 0.12)`
- **Border radius:** 16px
- **Background:** `#1C1A18` (card surface)

### Stage Progression

Five stages: `Preparing` > `Training` > `Evaluating` > `Complete` > `Ready`

- Each stage is a dot that fills with solid violet as it completes
- Active stage dot pulses (opacity 0.5 to 1.0, 1.5s cycle)
- Completed stage dots show a small checkmark icon
- Stages after completion fill with amber

### Progress Ring

- SVG circle, 64px diameter
- Stroke color: violet during training, amber on completion
- Smooth incremental fill animation (CSS transition on stroke-dashoffset)
- Center text: percentage in DM Mono, bold

### Sparkline Chart

- Minimal line chart (no axes, no grid) showing loss over recent epochs
- Line color: violet
- Dimensions: full card width, ~48px tall
- Uses Recharts (already in the project), restyled

### Expanded State

Clicking "Expand Details" animates the card taller (0.2s ease-out) to reveal:

- Full loss/accuracy chart (Recharts, restyled with violet/amber)
- Epoch-by-epoch stats table
- Hyperparameter summary (learning rate, batch size, optimizer)
- "Try a prediction" inline widget (uses existing InlinePredictWidget)

### Error / Cancelled States

- Border shifts to error red (`#ef4444`)
- Glow becomes `box-shadow: 0 0 16px rgba(239, 68, 68, 0.12)`
- Clear error message written for non-technical users ("What went wrong?")
- Retry action button

### Accessibility

- `aria-live="polite"` on the training card container for screen reader updates
- `role="progressbar"` with `aria-valuenow`, `aria-valuemin`, `aria-valuemax` on the progress ring
- Stage indicators have `aria-label` describing current stage
- All state changes communicated via text, not just color

---

## 3. Chat Message Redesign

### User Messages

- **Alignment:** Right-aligned
- **Background:** Warm surface (`#1C1A18`)
- **Border radius:** 16px top-left, 16px top-right, 4px bottom-right, 16px bottom-left (chat bubble shape)
- **No avatar** — position communicates ownership

### AI Messages

- **Alignment:** Left-aligned
- **Background:** None (sits on page background for breathing room)
- **Avatar:** Small WhiteMatter "W" monogram in violet, displayed to the left of the message
- **Thinking state:** Avatar gets a subtle violet pulse (opacity 0.5 to 1.0, 1.5s cycle)

### Typing Indicator

- Replaces the three-dot pattern
- Solid violet animated bar that slides left-to-right (no gradient)
- Small, subtle — ~40px wide, 3px tall, below the AI avatar
- Animation: `translateX` slide, 1.2s ease-in-out, infinite

### Code/Config Blocks

- Dark inset background: `#161514`
- 1px border: `rgba(255,255,255,0.07)`
- Left border accent: 4px solid violet (`#8B5CF6`)
- Font: DM Mono (already in theme)
- Border radius: 8px

### QuickStart Chips

- **Layout stays horizontal** (existing pattern preserved)
- **Styling updates:**
  - Border: 1px solid `rgba(255,255,255,0.07)`, transitions to violet on hover
  - Hover: Subtle violet glow (`box-shadow: 0 0 12px rgba(139, 92, 246, 0.1)`)
  - Padding: Slightly more generous (8px 16px)
  - Font: DM Mono (already set in theme)
  - No emojis — clean text labels only

### Empty State (New Chat)

- **Center of screen:** WhiteMatter logomark, large, at watermark opacity (~0.06)
- **Below logomark:** "What would you like to train?" in DM Serif Display, `text.secondary` color
- **Below heading:** Existing chips row, restyled with new palette
- **Feel:** A landing moment, not an empty room

---

## 4. Motion & Animation

All animations respect `prefers-reduced-motion` — when enabled, all animations are disabled and state changes are instant.

| Element | Animation | Duration | Easing |
|---------|-----------|----------|--------|
| Training card glow | Pulse (box-shadow opacity) | 2s | ease-in-out |
| Stage dot fill | Opacity + scale | 0.3s | ease-out |
| Stage dot active | Pulse (opacity 0.5-1.0) | 1.5s | ease-in-out |
| Progress ring | stroke-dashoffset transition | 0.5s | ease-out |
| Card expand/collapse | Height + opacity | 0.2s | ease-out |
| Message appear | Fade-in + translateY(8px to 0) | 0.15s | ease-out |
| Page transitions | Opacity fade | 0.15s | ease-out |
| Hover states | All properties | 0.15s | ease-out (already in theme) |
| Typing indicator bar | translateX slide | 1.2s | ease-in-out, infinite |
| AI avatar thinking | Opacity pulse (0.5-1.0) | 1.5s | ease-in-out |
| Training → complete | Border color + box-shadow color | 0.5s | ease-out |

---

## 5. Accessibility

### Contrast

| Element | Foreground | Background | Ratio | Level |
|---------|------------|------------|-------|-------|
| Body text | `#F2F1EE` | `#141311` | 15.8:1 | AAA |
| Secondary text | `#9C9A95` | `#141311` | 7.1:1 | AAA |
| Violet accent | `#8B5CF6` | `#141311` | 5.2:1 | AA |
| Amber (large only) | `#F59E0B` | `#141311` | 4.6:1 | AA Large |

### Focus States

- All interactive elements: 2px solid violet outline with 2px offset
- Visible on keyboard navigation only (`:focus-visible`)
- High contrast against dark background

### Screen Readers

- Training cards use `aria-live="polite"` for status updates
- Progress rings use `role="progressbar"` with value attributes
- Stage indicators have descriptive `aria-label`
- All state changes communicated via text, not just color/animation

### Motion Sensitivity

- `@media (prefers-reduced-motion: reduce)` disables all animations
- State changes remain visible through color and text changes

---

## 6. Component Impact Map

### New Components

| Component | Description |
|-----------|-------------|
| `TrainingCard` | The breakout training card with progress ring, stages, sparkline, expand/collapse |
| `ProgressRing` | SVG circular progress indicator |
| `StageIndicator` | Horizontal dot + label stage progression |
| `SparklineChart` | Minimal inline Recharts line chart |

### Modified Components

| Component | Changes |
|-----------|---------|
| `ChatMessage.tsx` | New visual distinction (AI vs user), bubble shapes, code block styling |
| `ChatInput.tsx` | Updated focus ring to violet |
| `QuickStartChips.tsx` | Hover glow, padding, border updates |
| `TrainingProgress.tsx` | Refactored to use new TrainingCard system |
| `TrainingChart.tsx` | Restyled with violet/amber palette |
| `ShareCard.tsx` | Updated to new color system |
| `theme.ts` | New color tokens, updated component overrides |

### Unchanged Components

- Sidebar layout and navigation
- Top bar and profile menu
- ModelsTab / ModelCard (existing styling, just color token swap)
- Auth pages (login/register)
- Settings page
- ErrorBoundary, Toast, ConfirmDialog

---

## 7. Implementation Phases

### Phase 1: Color System + Chat Polish (2-3 weeks)

- Update `theme.ts` with new violet/amber color tokens
- Swap all accent color references throughout components
- Chat message visual distinction (AI left/no-bg, user right/surface-bg)
- AI avatar (W monogram) with thinking pulse
- Typing indicator (violet bar)
- Code block left-border accent
- Chip hover states
- Empty state (new chat) with logomark + heading + chips
- Focus ring updates (violet, focus-visible)

### Phase 2: Training Cards (3-4 weeks)

- Build `TrainingCard` component with:
  - Progress ring (SVG)
  - Stage indicator (dots + labels)
  - Sparkline chart (Recharts)
  - Live stats display
  - Expand/collapse interaction
- Pulsing glow animation during active training
- Completion state (amber border + glow)
- Error/cancelled states
- Wire up to existing WebSocket training updates
- Accessibility: aria-live, progressbar role, screen reader labels

### Phase 3: Polish + Accessibility (2-3 weeks)

- Message appear animations (fade + slide)
- Page transition fades
- `prefers-reduced-motion` support across all animations
- Accessibility audit (contrast, focus, screen readers)
- Empty states for models tab
- Final QA pass across light/dark themes

### Total timeline: 7-10 weeks

---

## 8. Constraints & Trade-offs

- **No layout changes:** Sidebar, top bar, routing all stay as-is. Risk is minimized by keeping the structural shell intact.
- **No gradients:** All color usage is solid. Simplifies implementation and keeps the design clean.
- **MUI component system:** All new components built within MUI's theming system. No CSS escape hatches unless absolutely necessary.
- **Mobile:** Training cards need responsive handling — on small screens, they span full width and the sparkline collapses. Progress ring and stats remain visible.
- **Performance:** Pulsing glow animations use `box-shadow` which can trigger repaints. Limit to one visible training card animating at a time; completed cards use static shadows.
- **Light theme:** All new color tokens need light-theme counterparts. Violet works on both; amber needs a darker variant for light backgrounds.

---

## 9. Success Criteria

- Users can identify WhiteMatter from a screenshot (violet + amber on dark, training cards)
- Non-technical users understand training progress without reading documentation
- Training card states (active, complete, error) are clear without relying solely on color
- All interactive elements have visible focus states
- Animations respect `prefers-reduced-motion`
- No regressions in existing chat workflow or navigation
