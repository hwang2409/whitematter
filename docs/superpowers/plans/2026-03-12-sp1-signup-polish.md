# SP1: Sign Up Polish — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add env-var-gated Google OAuth button to landing, login, and register pages with account linking.

**Architecture:** Frontend uses `@react-oauth/google` with `useGoogleLogin` (implicit grant flow) to get an `access_token`, which is sent to the existing `POST /auth/google` backend endpoint. The `GoogleOAuthProvider` wrapper in `providers.tsx` is conditionally rendered based on `NEXT_PUBLIC_GOOGLE_CLIENT_ID`. Backend account linking is verified and fixed if needed.

**Tech Stack:** React, `@react-oauth/google`, Next.js, FastAPI, SQLAlchemy

**Spec:** `docs/superpowers/specs/2026-03-12-workflow-implementation-design.md` (SP1 section)

---

## Chunk 1: Google OAuth Integration

### Task 1: Install dependency and update env config

**Files:**
- Modify: `frontend/package.json`
- Modify: `.env.example`

- [ ] **Step 1: Install `@react-oauth/google`**

```bash
cd frontend && npm install @react-oauth/google
```

- [ ] **Step 2: Add Google env vars to `.env.example`**

Add under a new `# ── Google OAuth (optional)` section, commented out:

```
# ── Google OAuth (optional) ───────────────────────────────
# GOOGLE_CLIENT_ID=
# GOOGLE_CLIENT_SECRET=
# NEXT_PUBLIC_GOOGLE_CLIENT_ID=
```

- [ ] **Step 3: Commit**

```bash
git add frontend/package.json frontend/package-lock.json .env.example
git commit -m "feat(sp1): add @react-oauth/google dependency and env vars"
```

---

### Task 2: Add GoogleOAuthProvider to providers.tsx

**Files:**
- Modify: `frontend/src/app/providers.tsx` (lines 1-31)

- [ ] **Step 1: Add conditional GoogleOAuthProvider wrapper**

The current `Providers` component (lines 21-31) wraps children with `ThemeContextProvider → ThemedApp → AuthProvider → DesignProvider`. Add `GoogleOAuthProvider` as the outermost wrapper, conditionally rendered.

```tsx
// Add import at top
import { GoogleOAuthProvider } from "@react-oauth/google";

// Add wrapper component before Providers
function GoogleOAuthWrapper({ children }: { children: React.ReactNode }) {
  const clientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;
  if (!clientId) return <>{children}</>;
  return <GoogleOAuthProvider clientId={clientId}>{children}</GoogleOAuthProvider>;
}

// In Providers, wrap everything with GoogleOAuthWrapper:
export default function Providers({ children }: { children: React.ReactNode }) {
  return (
    <GoogleOAuthWrapper>
      <ThemeContextProvider>
        <ThemedApp>
          <AuthProvider>
            <DesignProvider>{children}</DesignProvider>
          </AuthProvider>
        </ThemedApp>
      </ThemeContextProvider>
    </GoogleOAuthWrapper>
  );
}
```

- [ ] **Step 2: Verify app still renders without env var set**

```bash
cd frontend && npm run dev
```

Expected: App loads normally, no errors. No Google button visible yet.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/providers.tsx
git commit -m "feat(sp1): add conditional GoogleOAuthProvider wrapper"
```

---

### Task 3: Create reusable GoogleSignInButton component

**Files:**
- Create: `frontend/src/components/GoogleSignInButton.tsx`

- [ ] **Step 1: Create the component**

This component encapsulates the Google login flow. It uses `useGoogleLogin` with the implicit grant flow (returns `access_token`), sends it to `POST /auth/google`, and calls `loginWithTokens` on success.

```tsx
"use client";
import { useGoogleLogin } from "@react-oauth/google";
import { useAuth } from "@/context/AuthContext";
import { useRouter } from "next/navigation";
import Button from "@mui/material/Button";
import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

interface GoogleSignInButtonProps {
  label?: string;
  fullWidth?: boolean;
}

export default function GoogleSignInButton({
  label = "Continue with Google",
  fullWidth = true,
}: GoogleSignInButtonProps) {
  const { loginWithTokens } = useAuth();
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const googleLogin = useGoogleLogin({
    onSuccess: async (tokenResponse) => {
      setLoading(true);
      setError("");
      try {
        const res = await fetch(`${API_BASE}/auth/google`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ access_token: tokenResponse.access_token }),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.detail || "Google sign-in failed");
        }
        const tokens = await res.json();
        loginWithTokens(tokens);
        router.replace("/chat");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Google sign-in failed");
      } finally {
        setLoading(false);
      }
    },
    onError: () => setError("Google sign-in was cancelled"),
  });

  // Don't render if Google OAuth is not configured
  if (!process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID) return null;

  return (
    <>
      <Button
        onClick={() => googleLogin()}
        disabled={loading}
        fullWidth={fullWidth}
        variant="outlined"
        size="large"
        sx={{
          textTransform: "none",
          borderColor: "divider",
          color: "text.primary",
          py: 1.2,
        }}
      >
        {loading ? "Signing in..." : label}
      </Button>
      {error && (
        <p style={{ color: "red", fontSize: "0.85rem", marginTop: 4 }}>
          {error}
        </p>
      )}
    </>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/GoogleSignInButton.tsx
git commit -m "feat(sp1): create reusable GoogleSignInButton component"
```

---

### Task 4: Add Google button to LoginPage

**Files:**
- Modify: `frontend/src/views/LoginPage.tsx`

- [ ] **Step 1: Add GoogleSignInButton above the form**

Add import at top:
```tsx
import GoogleSignInButton from "@/components/GoogleSignInButton";
```

Add the button and divider above the email field. Insert after the heading/title and before the email `TextField`. Look for the `<TextField label="Email"` line and insert before it:

```tsx
<GoogleSignInButton />
<Box sx={{ display: "flex", alignItems: "center", my: 2 }}>
  <Box sx={{ flex: 1, borderBottom: 1, borderColor: "divider" }} />
  <Box sx={{ px: 2, color: "text.secondary", fontSize: "0.85rem" }}>or</Box>
  <Box sx={{ flex: 1, borderBottom: 1, borderColor: "divider" }} />
</Box>
```

- [ ] **Step 2: Verify login page renders**

Open `http://localhost:3000/login`. Without `NEXT_PUBLIC_GOOGLE_CLIENT_ID` set, no Google button should appear. With it set, button should show above the "or" divider.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/views/LoginPage.tsx
git commit -m "feat(sp1): add Google sign-in button to login page"
```

---

### Task 5: Add Google button to RegisterPage

**Files:**
- Modify: `frontend/src/views/RegisterPage.tsx`

- [ ] **Step 1: Add GoogleSignInButton above the form**

Same pattern as LoginPage. Add import:
```tsx
import GoogleSignInButton from "@/components/GoogleSignInButton";
```

Insert before the email `TextField`:
```tsx
<GoogleSignInButton label="Sign up with Google" />
<Box sx={{ display: "flex", alignItems: "center", my: 2 }}>
  <Box sx={{ flex: 1, borderBottom: 1, borderColor: "divider" }} />
  <Box sx={{ px: 2, color: "text.secondary", fontSize: "0.85rem" }}>or</Box>
  <Box sx={{ flex: 1, borderBottom: 1, borderColor: "divider" }} />
</Box>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/views/RegisterPage.tsx
git commit -m "feat(sp1): add Google sign-in button to register page"
```

---

### Task 6: Add Google button to LandingPage hero

**Files:**
- Modify: `frontend/src/views/LandingPage.tsx`

- [ ] **Step 1: Add GoogleSignInButton next to "Get Started" button**

Add import:
```tsx
import GoogleSignInButton from "@/components/GoogleSignInButton";
```

Find the "Get Started" button (around line 89-97). Wrap it and the Google button in a flex container:

```tsx
<Box sx={{ display: "flex", gap: 2, flexDirection: { xs: "column", sm: "row" }, alignItems: "center" }}>
  {/* Existing Get Started button stays */}
  <Button component={NextLink} href="/register" variant="contained" size="large">
    Get Started
  </Button>
  <GoogleSignInButton label="Sign in with Google" fullWidth={false} />
</Box>
```

- [ ] **Step 2: Verify landing page renders**

Open `http://localhost:3000`. "Get Started" button should still work. Google button appears next to it only when env var is set.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/views/LandingPage.tsx
git commit -m "feat(sp1): add Google sign-in button to landing page hero"
```

---

### Task 7: Verify and fix backend account linking

**Files:**
- Modify (if needed): `platform/routes/auth.py` (lines 81-135)

- [ ] **Step 1: Review the `/auth/google` endpoint for account linking**

Read `platform/routes/auth.py` lines 81-135. Check if the endpoint:
1. Fetches Google user info (email, google_id)
2. Checks for existing user by `google_id` first
3. If not found by google_id, checks for existing user by `email`
4. If found by email (email/password user), links the google_id to that user instead of creating a new one

The current code (line 106) queries `User.google_id == google_id`. If no match, it creates a new user (line 116). It does NOT check for existing email, meaning a user who registered with email/password and then tries Google OAuth with the same email will get a duplicate account error.

- [ ] **Step 2: Fix account linking**

After the google_id lookup fails (around line 113), add an email lookup before creating a new user:

```python
# After: user = db.query(User).filter(User.google_id == google_id).first()
if not user:
    # Check if a user with this email already exists (account linking)
    user = db.query(User).filter(User.email == email).first()
    if user:
        # Link Google ID to existing account
        user.google_id = google_id
        if picture:
            user.avatar_url = picture
        db.commit()
        db.refresh(user)
```

Only create a new user if neither google_id nor email match.

- [ ] **Step 3: Test account linking manually**

1. Register with email/password: `POST /auth/register` with `email=test@example.com`
2. Sign in with Google using same email: `POST /auth/google`
3. Verify: same user ID, google_id now populated, no duplicate

- [ ] **Step 4: Commit**

```bash
git add platform/routes/auth.py
git commit -m "fix(sp1): add account linking for Google OAuth with existing email users"
```

---

## SP1 Complete

After all 7 tasks:
- Google OAuth button appears on landing, login, register pages (only when `NEXT_PUBLIC_GOOGLE_CLIENT_ID` is set)
- Backend handles account linking for existing email users
- `.env.example` updated with Google env vars
- No changes needed for local dev without Google credentials
