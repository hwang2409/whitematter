import { test, expect } from "@playwright/test";
import { registerUser, loginUser, logoutUser, uniqueTestEmail } from "./helpers";

const TEST_PASSWORD = "TestPassword123!";

test.describe("Authentication flow", () => {
  let testEmail: string;

  test.beforeAll(() => {
    testEmail = uniqueTestEmail();
  });

  test("register a new account and land on dashboard", async ({ page }) => {
    await page.goto("/register");

    // Verify the registration page renders
    await expect(page.getByText("Create your account")).toBeVisible();

    // Fill in the registration form
    await page.getByLabel(/email/i).fill(testEmail);
    await page.getByLabel(/password/i).fill(TEST_PASSWORD);

    // Submit
    await page.getByRole("button", { name: /sign up/i }).click();

    // Should redirect to dashboard
    await page.waitForURL("**/dashboard");
    await expect(page.getByText("Dashboard")).toBeVisible();
    await expect(page.getByText(/welcome back/i)).toBeVisible();
  });

  test("log out and return to login page", async ({ page }) => {
    // First register and land on dashboard
    await registerUser(page, uniqueTestEmail(), TEST_PASSWORD);

    // Click the "Log out" button in the top bar
    await logoutUser(page);

    // Should be back on the login page
    await expect(page.getByText("Sign in to your account")).toBeVisible();
  });

  test("log in with existing credentials", async ({ page }) => {
    // Register a user first so we have valid credentials
    const email = uniqueTestEmail();
    await registerUser(page, email, TEST_PASSWORD);
    await logoutUser(page);

    // Now log in with the same credentials
    await page.goto("/login");
    await expect(page.getByText("Sign in to your account")).toBeVisible();

    await page.getByLabel(/email/i).fill(email);
    await page.getByLabel(/password/i).fill(TEST_PASSWORD);
    await page.getByRole("button", { name: /sign in/i }).click();

    // Should redirect to dashboard
    await page.waitForURL("**/dashboard");
    await expect(page.getByText("Dashboard")).toBeVisible();
    await expect(page.getByText(email)).toBeVisible();
  });

  test("unauthenticated user is redirected to login", async ({ page }) => {
    // Clear any stored tokens
    await page.goto("/login");
    await page.evaluate(() => {
      localStorage.removeItem("access_token");
      localStorage.removeItem("refresh_token");
    });

    // Try to access a protected page
    await page.goto("/dashboard");
    await page.waitForURL("**/login");
    await expect(page.getByText("Sign in to your account")).toBeVisible();
  });

  test("navigate between login and register pages", async ({ page }) => {
    await page.goto("/login");
    await expect(page.getByText("Sign in to your account")).toBeVisible();

    // Click the "Sign up" link to go to register
    await page.getByRole("link", { name: /sign up/i }).click();
    await page.waitForURL("**/register");
    await expect(page.getByText("Create your account")).toBeVisible();

    // Click the "Sign in" link to go back to login
    await page.getByRole("link", { name: /sign in/i }).click();
    await page.waitForURL("**/login");
    await expect(page.getByText("Sign in to your account")).toBeVisible();
  });
});
