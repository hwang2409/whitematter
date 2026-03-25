import { test, expect } from "@playwright/test";

test.describe("Authentication flow", () => {
  test("login page renders", async ({ page }) => {
    await page.goto("/login");
    await expect(page.getByText("Welcome back")).toBeVisible();
  });

  test("register page renders", async ({ page }) => {
    await page.goto("/register");
    await expect(page.getByText("Create your account")).toBeVisible();
  });

  test("unauthenticated user is redirected to login", async ({ page }) => {
    await page.goto("/login");
    await page.evaluate(() => {
      localStorage.removeItem("access_token");
      localStorage.removeItem("refresh_token");
    });

    await page.goto("/chat");
    await page.waitForURL("**/login");
    await expect(page.getByText("Welcome back")).toBeVisible();
  });

  test("navigate between login and register pages", async ({ page }) => {
    await page.goto("/login");
    await expect(page.getByText("Welcome back")).toBeVisible();

    await page.getByRole("link", { name: /sign up/i }).click();
    await page.waitForURL("**/register");
    await expect(page.getByText("Create your account")).toBeVisible();

    await page.getByRole("link", { name: /sign in/i }).click();
    await page.waitForURL("**/login");
    await expect(page.getByText("Welcome back")).toBeVisible();
  });
});
