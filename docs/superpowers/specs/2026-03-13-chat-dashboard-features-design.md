# Chat Dashboard Core Features — Design Spec

## Context

The WhiteMatter chat dashboard has a basic sidebar, chat interface, models view, and placeholder settings page. This spec covers the core UX features needed to make the dashboard functional: collapsible sidebar, chat history management, models table view, and a full settings page.

## 1. Collapsible Sidebar

### Behavior
- **Expanded:** 260px (current), shows logo, "New Chat" button, search bar, nav icons, conversation list, user area
- **Collapsed:** ~60px icon rail, shows only nav icons (Chat/Models/Settings) stacked vertically, `+` icon button for new chat, user avatar at bottom
- **Toggle:** Chevron button at bottom of sidebar (above user area). `ChevronLeft` when expanded, `ChevronRight` when collapsed
- **Animation:** `transition: width 0.2s ease` on sidebar; main content uses `flex: 1` to fill automatically
- **Persistence:** `localStorage("whitematter-sidebar")` via new `SidebarContext`

### New Files
- `frontend/src/context/SidebarContext.tsx` — `collapsed` boolean + `toggleSidebar()` function

### Files to Modify
- `frontend/src/app/(authenticated)/layout.tsx` — sidebar width conditional, collapsed/expanded rendering
- `frontend/src/app/providers.tsx` — wrap with `SidebarProvider`

## 2. Chat History Management

### Three-Dot Menu
- `MoreHoriz` icon appears on hover, right-aligned on each chat item in sidebar
- MUI `Menu` anchored to the icon with options:
  - **Star / Unstar** — toggles `is_starred` on conversation
  - **Rename** — swaps title to inline `TextField`, Enter saves, Escape cancels
  - **Delete** — opens existing `ConfirmDialog`, then calls `DELETE /chat/conversations/:id`

### Search Bar
- `TextField` with `SearchOutlined` icon, placed between "New Chat" button and nav icons row
- Client-side filtering of conversation list by title (no backend search endpoint needed)
- Only visible when sidebar is expanded and on `/chat` routes

### Starred Chats
- Starred conversations sort to top within each date group
- Small `StarOutlined` icon shown next to starred chat titles
- `is_starred` field returned from `GET /chat/conversations`

### Backend Changes
- Add `is_starred = Column(Boolean, default=False)` to `Conversation` model in `platform/db/chat_models.py`
- Add `PATCH /chat/conversations/:id` endpoint — accepts `{ title?, is_starred? }` partial update
- Add `DELETE /chat/conversations/:id` endpoint (does not exist yet). Must cascade-delete associated `ConversationMessage` rows.
- Alembic migration for `is_starred` column
- Update `GET /chat/conversations` to include `is_starred` in response

### Edge Cases
- **Deleting active conversation:** If user deletes the conversation they're currently viewing (`/chat/:id`), redirect to `/chat` after deletion
- **Rename empty title:** If user clears the rename field and presses Enter, revert to previous title (don't save empty)

### Frontend API Changes
- Add `updateConversation(id, { title?, is_starred? })` to `frontend/src/api.ts`
- Add `deleteConversation(id)` to `frontend/src/api.ts`
- Add `isStarred` to `Conversation` type (camelCase to match existing `createdAt`/`updatedAt` convention)

## 3. Models — Table/List View with Inline Expansion

### Layout Change
Replace current side-by-side card+detail panel with a full-width sortable data table.

### Table Columns
| Column | Sortable | Content |
|--------|----------|---------|
| Name | Yes | `formatModelName(model.name)` |
| Architecture | No | Architecture string |
| Status | Yes | Colored `Chip` (existing `getStatusColor`) |
| Accuracy | Yes | `best_accuracy` formatted |
| Loss | Yes | Final loss from training history |
| Created | Yes | `created_at` formatted |

### Inline Expansion
- Click row to expand/collapse detail section below it
- Only one row expanded at a time
- Expanded content includes all existing detail panel content:
  - Training history table
  - API endpoint + cURL example (for image models)
  - Text generation UI (for text models)
  - Deploy button + deploy modal
  - Action buttons: Predict, Share, Export ONNX, Resume Training, Delete

### Files to Modify
- `frontend/src/components/ModelsTab.tsx` — refactor layout from card grid to table with expandable rows
- All existing functionality (deploy modal, text generation, delete confirm, share card) is preserved

## 4. Settings Page — 4 Tabs

### Layout
Vertical tab navigation on the left (~200px), content panel on the right. Tab state managed with `useState`, optionally synced to URL hash.

### General Tab
- **Theme toggle:** Light/dark switch wired to existing `ThemeContext` (via `useThemeMode()` which returns `{ mode, setMode }`)
- **Default preferences:** Placeholder section for future model training defaults (e.g., default epochs, learning rate). For v1, just show the theme toggle — defaults can be added when training configuration is exposed in the UI.

### Account Tab
- **Change password:** Current password + new password + confirm fields
  - Backend: `POST /auth/change-password` with `{ current_password, new_password }`
  - **OAuth users:** If user signed up via Google OAuth (no `password_hash`), hide the change password section and show "Signed in with Google" instead
- **Export data:** Button that triggers download of all user data as JSON
  - Backend: `GET /auth/export` returns JSON containing: conversations (with messages), models (with training history), datasets (metadata only, not raw files), and AWS credentials (masked)

### Billing Tab
- **Current plan card:** Shows Free/Pro/Scale with feature highlights
- **Usage stats:** Models created count, datasets count — derived from existing DB queries (no training hours tracking exists yet)
  - Backend: `GET /billing/usage` (new endpoint, returns `{ models_count, datasets_count, conversations_count }`)
- **Upgrade/downgrade:** Buttons that create Stripe Checkout sessions or open Stripe Customer Portal
  - Backend: Uses existing `POST /billing/checkout` and `POST /billing/portal` endpoints
  - Plan details from existing `GET /billing/status` endpoint

### Connect Tab
- **AWS Credentials management** using existing `frontend/src/services/aws.ts`
  - Functions available: `storeCredentials`, `getCredentials`, `updateCredentials`, `deleteCredentials`
- Form: Access Key ID, Secret Access Key, Region, S3 Endpoint (optional)
- Keys displayed masked (last 4 chars only)
- Connection test button to validate credentials
- Add/edit/delete flow

### New Files
- `frontend/src/components/settings/GeneralTab.tsx`
- `frontend/src/components/settings/AccountTab.tsx`
- `frontend/src/components/settings/BillingTab.tsx`
- `frontend/src/components/settings/ConnectTab.tsx`

### Files to Modify
- `frontend/src/app/(authenticated)/settings/page.tsx` — replace placeholder with tab layout + tab panels
- `platform/routes/auth.py` — add `change-password` and `export` endpoints
- `platform/schemas/auth_schemas.py` — add request/response schemas

### Backend Endpoints (New)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/auth/change-password` | Change password |
| GET | `/auth/export` | Export all user data as JSON |
| GET | `/billing/usage` | Usage stats for current user |
| POST | `/billing/checkout` | Create Stripe Checkout session (already exists) |
| POST | `/billing/portal` | Open Stripe Customer Portal (already exists) |

## 5. Database Changes Summary

### Conversations table
- Add `is_starred` column: `Boolean, default=False`

### Migration
- Single Alembic migration: `add_conversation_is_starred`

## 6. State Management Summary

| State | Location | Persistence |
|-------|----------|-------------|
| Sidebar collapsed | `SidebarContext` | `localStorage("whitematter-sidebar")` |
| Theme mode | `ThemeContext` (existing) | `localStorage("whitematter-theme")` |
| Search query | `layout.tsx` local state | None (ephemeral) |
| Rename editing | `layout.tsx` local state | None |
| Models sort column/dir | `ModelsTab.tsx` local state | None |
| Models expanded row | `ModelsTab.tsx` local state | None |
| Settings active tab | `settings/page.tsx` local state | URL hash (optional) |
| Default preferences | `GeneralTab` | Deferred to future iteration |

## 7. Implementation Order

1. **Sidebar collapse** — `SidebarContext` + layout changes (no backend)
2. **Chat history** — DB migration, backend endpoints, then frontend menu/search/star
3. **Models table** — frontend-only refactor of `ModelsTab.tsx`
4. **Settings: General** — frontend-only (theme toggle + localStorage prefs)
5. **Settings: Account** — backend endpoints + frontend form
6. **Settings: Connect** — frontend-only (uses existing AWS service)
7. **Settings: Billing** — backend Stripe integration + frontend (most complex, do last)
