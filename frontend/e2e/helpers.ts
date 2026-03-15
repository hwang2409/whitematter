import { Page } from "@playwright/test";

/**
 * Register a new user account and wait for redirect to chat.
 */
export async function registerUser(
  page: Page,
  email: string,
  password: string,
) {
  await page.goto("/register");
  await page.getByLabel(/email/i).fill(email);
  await page.getByLabel(/password/i).fill(password);
  await page.getByRole("button", { name: /sign up/i }).click();
  await page.waitForURL("**/chat");
}

/**
 * Log in with existing credentials and wait for redirect to chat.
 */
export async function loginUser(
  page: Page,
  email: string,
  password: string,
) {
  await page.goto("/login");
  await page.getByLabel(/email/i).fill(email);
  await page.getByLabel(/password/i).fill(password);
  await page.getByRole("button", { name: /sign in/i }).click();
  await page.waitForURL("**/chat");
}

/**
 * Log out from the authenticated layout.
 */
export async function logoutUser(page: Page) {
  await page.getByRole("button", { name: /log out/i }).click();
  await page.waitForURL("**/login");
}

/**
 * Generate a unique test email to avoid collisions between test runs.
 */
export function uniqueTestEmail(): string {
  const timestamp = Date.now();
  const rand = Math.random().toString(36).slice(2, 8);
  return `e2e-${timestamp}-${rand}@test.local`;
}
