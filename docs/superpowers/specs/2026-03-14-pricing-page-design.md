# Pricing Page Design Spec

## Overview

Add a dedicated `/pricing` page inside the authenticated route group where logged-in users can view and upgrade to paid plans. Update the Billing settings tab to show current plan status with links to Stripe portal and the pricing page. Add `interval` support to the backend checkout endpoint.

## Decisions

- **Route**: `/pricing` inside `(authenticated)` — requires login, includes sidebar
- **Layout**: Side-by-side pricing cards with monthly/annual toggle
- **Annual discount**: 20% off (rounded down to clean numbers)
- **Checkout**: Extends existing `POST /billing/checkout` Stripe endpoint with `interval` param

## Pricing Tiers

Prices are rounded down from exact 20% discount for clean display numbers.

### Pro — $29/mo or $23/mo annually ($276/yr)

- Unlimited training runs
- Unlimited models
- 20 GB storage
- 1 deploy endpoint

### Scale — $59/mo or $47/mo annually ($564/yr)

- Everything in Pro
- 100 GB storage
- GPU access ($0.99/hr)
- 5 deploy endpoints
- "Recommended" badge

## Page Structure

### Header

- Title: "Choose your plan"
- Subtitle with billing toggle

### Billing Toggle

- MUI `ToggleButtonGroup` pill-style: Monthly | Annual
- "Save 20%" chip next to Annual in violet
- Default: Monthly

### Pricing Cards

Two `Paper` cards side by side (stack vertically on mobile `xs`).

Each card contains:
- Plan name (h3)
- Price with `/mo` suffix
- "Billed $X/year" note when annual is selected
- Feature list with checkmark icons
- CTA button

**Pro card**: Standard styling. CTA: "Upgrade to Pro" (violet contained button).

**Scale card**: Subtle violet border highlight. "Recommended" badge. CTA: "Upgrade to Scale" (violet contained button).

**Current plan handling**: If user is already on a plan, that card's CTA shows "Current plan" as a disabled outlined button.

**Downgrade handling**: If user is on Scale and views Pro, no downgrade button — managed via Stripe portal from the Billing settings tab.

### Loading & Error States

- **Loading**: Show a centered `CircularProgress` while fetching user plan status
- **Checkout loading**: CTA button shows a spinner and disables after click while creating checkout session
- **Checkout error**: Show a snackbar/alert with "Something went wrong. Please try again." if `POST /billing/checkout` fails

### Checkout Flow

1. User clicks upgrade CTA
2. CTA shows loading spinner, disables
3. Frontend calls `POST /billing/checkout` with `{ plan, interval }` (interval: "month" or "year")
4. Backend creates Stripe checkout session, returns URL
5. Frontend redirects to Stripe Checkout
6. Stripe redirects back to `/pricing?billing=success` or `/pricing?billing=cancelled`

## Billing Tab Update

The `BillingTab` in Settings (`/settings`, tab index 1) changes from placeholder to:

- Current plan name and status display
- "Manage subscription" button → calls `POST /billing/portal` to open Stripe customer portal
- "Change plan" link → navigates to `/pricing`

## Navigation Changes

- "Upgrade plan" menu item in profile dropdown (layout.tsx ~line 722): change from `router.push("/settings")` to `router.push("/pricing")`. Do NOT change the adjacent "Settings" menu item.

## Backend Changes

The current `POST /billing/checkout` only accepts `plan: str`. It needs to also accept `interval`.

### Changes to `platform/routes/billing.py`

- Add `interval: str = "month"` to the checkout request schema (default "month" for backwards compatibility)
- Pass interval through to `create_checkout_session`

### Changes to `platform/services/billing_service.py`

- Add annual Stripe price IDs to the `PLANS` dict (e.g., `stripe_annual_price_id` alongside existing `stripe_price_id`)
- Update `create_checkout_session` to select the correct price ID based on interval
- Update success/cancel redirect URLs to `/pricing?billing=success` and `/pricing?billing=cancelled`

## Files to Create/Modify

### New Files

- `frontend/src/app/(authenticated)/pricing/page.tsx` — route wrapper
- `frontend/src/views/PricingPage.tsx` — pricing page component

### Modified Files

- `frontend/src/components/settings/BillingTab.tsx` — plan summary with Stripe portal link
- `frontend/src/app/(authenticated)/layout.tsx` — update "Upgrade plan" navigation target
- `platform/routes/billing.py` — add interval param to checkout endpoint
- `platform/services/billing_service.py` — add annual price IDs, interval-based price selection, update redirect URLs

## Theme Consistency

- Font: DM Sans (body), DM Serif Display (headings)
- Accent: `#8B5CF6` (violet) for CTAs, highlights, badges
- Cards: `Paper` with `borderRadius: 16px`
- Dark/light mode: fully theme-aware using MUI `sx` callbacks
- Focus styles: violet outline (`2px solid #8B5CF6`)
