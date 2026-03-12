import { test, expect } from "@playwright/test";
import { registerUser, uniqueTestEmail } from "./helpers";

const TEST_PASSWORD = "TestPassword123!";

/**
 * Navigation smoke tests for authenticated pages.
 *
 * These verify that each main view renders correctly after authentication.
 * Full training/prediction flows require a running C++ backend and real
 * datasets, so the tests here focus on page structure and navigation.
 */
test.describe("Authenticated page navigation", () => {
  let testEmail: string;

  test.beforeEach(async ({ page }) => {
    testEmail = uniqueTestEmail();
    await registerUser(page, testEmail, TEST_PASSWORD);
  });

  test("dashboard page loads with expected sections", async ({ page }) => {
    await expect(page).toHaveURL(/\/dashboard/);
    await expect(page.getByText("Dashboard")).toBeVisible();
    await expect(page.getByText(/welcome back/i)).toBeVisible();
    await expect(page.getByText("Recent Activity")).toBeVisible();
    await expect(page.getByText("Quick Actions")).toBeVisible();
  });

  test("data page loads and shows datasets tab", async ({ page }) => {
    await page.goto("/data");
    await expect(page.getByText("Data")).toBeVisible();
    await expect(page.getByRole("tab", { name: /datasets/i })).toBeVisible();
    await expect(page.getByRole("tab", { name: /s3 storage/i })).toBeVisible();

    // The Datasets sub-heading should be visible on the default tab
    await expect(page.getByRole("heading", { name: "Datasets" })).toBeVisible();
  });

  test("train page loads with configuration form", async ({ page }) => {
    await page.goto("/train");
    await expect(page.getByText("Train a Model")).toBeVisible();
  });

  test("models page loads", async ({ page }) => {
    await page.goto("/models");
    await expect(page.getByText("Trained Models")).toBeVisible();
  });

  test("predict page loads", async ({ page }) => {
    await page.goto("/predict");
    await expect(page.getByText("Make Predictions")).toBeVisible();
  });

  test("settings page loads", async ({ page }) => {
    await page.goto("/settings");
    await expect(page.getByText("Settings")).toBeVisible();
  });

  test("sidebar navigation works across all pages", async ({ page }) => {
    // Dashboard is already loaded from beforeEach

    // Navigate via sidebar tooltips/links — sidebar uses icon links with
    // tooltip labels. The <a> elements have href attributes we can target.
    await page.locator('nav a[href="/data"]').click();
    await expect(page).toHaveURL(/\/data/);
    await expect(page.getByRole("heading", { name: "Data" })).toBeVisible();

    await page.locator('nav a[href="/train"]').click();
    await expect(page).toHaveURL(/\/train/);
    await expect(page.getByText("Train a Model")).toBeVisible();

    await page.locator('nav a[href="/models"]').click();
    await expect(page).toHaveURL(/\/models/);
    await expect(page.getByText("Trained Models")).toBeVisible();

    await page.locator('nav a[href="/predict"]').click();
    await expect(page).toHaveURL(/\/predict/);
    await expect(page.getByText("Make Predictions")).toBeVisible();

    await page.locator('nav a[href="/settings"]').click();
    await expect(page).toHaveURL(/\/settings/);
    await expect(page.getByText("Settings")).toBeVisible();

    // Navigate back to dashboard
    await page.locator('nav a[href="/dashboard"]').click();
    await expect(page).toHaveURL(/\/dashboard/);
    await expect(page.getByText("Dashboard")).toBeVisible();
  });

  test("user email is displayed in the top bar", async ({ page }) => {
    await expect(page.getByText(testEmail)).toBeVisible();
  });
});
