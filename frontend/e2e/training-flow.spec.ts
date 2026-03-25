import { test, expect } from "@playwright/test";

/**
 * Smoke tests for page rendering.
 *
 * These verify that unauthenticated routes render correctly.
 * Authenticated route tests require a running backend and are
 * skipped in CI when the backend is unavailable.
 */
test.describe("Page smoke tests", () => {
  test("login page loads", async ({ page }) => {
    await page.goto("/login");
    await expect(page.getByText("Welcome back")).toBeVisible();
  });

  test("register page loads", async ({ page }) => {
    await page.goto("/register");
    await expect(page.getByText("Create your account")).toBeVisible();
  });

  test("root redirects to chat then login", async ({ page }) => {
    await page.goto("/");
    // Unauthenticated: / -> /chat -> /login
    await page.waitForURL("**/login");
    await expect(page.getByText("Welcome back")).toBeVisible();
  });

  test("settings redirects to login when unauthenticated", async ({ page }) => {
    await page.goto("/settings");
    await page.waitForURL("**/login");
    await expect(page.getByText("Welcome back")).toBeVisible();
  });

  test("models redirects to login when unauthenticated", async ({ page }) => {
    await page.goto("/models");
    await page.waitForURL("**/login");
    await expect(page.getByText("Welcome back")).toBeVisible();
  });
});
