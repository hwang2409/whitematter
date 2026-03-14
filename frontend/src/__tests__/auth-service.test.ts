import { describe, it, expect, vi, beforeEach } from "vitest";
import { register, login, getMe, getStoredToken, storeTokens, clearTokens } from "@/services/auth";

// Mock safeParseJson - used internally by auth service
vi.mock("@/lib/safeJson", () => ({
  safeParseJson: async (res: Response) => {
    const text = await res.text();
    const trimmed = text.trim();
    if (!trimmed) return {};
    return JSON.parse(text);
  },
}));

const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

const mockTokens = {
  access_token: "test-access-token",
  refresh_token: "test-refresh-token",
  token_type: "bearer",
};

const mockUser = {
  id: "user-123",
  email: "test@example.com",
  oauth_provider: null,
  created_at: "2025-01-01T00:00:00Z",
};

beforeEach(() => {
  mockFetch.mockReset();
  localStorage.clear();
});

describe("register", () => {
  it("sends POST to /auth/register and returns tokens", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify(mockTokens), { status: 200 })
    );

    const result = await register("test@example.com", "password123");

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/auth/register"),
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: "test@example.com", password: "password123" }),
      })
    );
    expect(result).toEqual(mockTokens);
  });

  it("throws with server detail on error response", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "Email already exists" }), { status: 409 })
    );

    await expect(register("test@example.com", "password123")).rejects.toThrow(
      "Email already exists"
    );
  });

  it("throws generic message when no detail in error response", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response("{}", { status: 500 })
    );

    await expect(register("test@example.com", "password123")).rejects.toThrow(
      "Registration failed"
    );
  });
});

describe("login", () => {
  it("sends POST to /auth/login and returns tokens", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify(mockTokens), { status: 200 })
    );

    const result = await login("test@example.com", "password123");

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/auth/login"),
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: "test@example.com", password: "password123" }),
      })
    );
    expect(result).toEqual(mockTokens);
  });

  it("throws with server detail on error response", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "Invalid credentials" }), { status: 401 })
    );

    await expect(login("test@example.com", "wrong")).rejects.toThrow(
      "Invalid credentials"
    );
  });

  it("throws generic message when no detail in error response", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response("{}", { status: 500 })
    );

    await expect(login("test@example.com", "password123")).rejects.toThrow(
      "Login failed"
    );
  });
});

describe("getMe", () => {
  it("sends GET to /auth/me with Bearer token and returns user", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify(mockUser), { status: 200 })
    );

    const result = await getMe("my-token");

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/auth/me"),
      expect.objectContaining({
        headers: { Authorization: "Bearer my-token" },
      })
    );
    expect(result).toEqual(mockUser);
  });

  it("throws on non-ok response", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response("", { status: 401 })
    );

    await expect(getMe("bad-token")).rejects.toThrow("Not authenticated");
  });
});

describe("getStoredToken", () => {
  it("returns null when no token stored", () => {
    expect(getStoredToken()).toBeNull();
  });

  it("returns stored access_token", () => {
    localStorage.setItem("access_token", "stored-token");
    expect(getStoredToken()).toBe("stored-token");
  });
});

describe("storeTokens", () => {
  it("stores access_token and refresh_token in localStorage", () => {
    storeTokens(mockTokens);
    expect(localStorage.getItem("access_token")).toBe("test-access-token");
    expect(localStorage.getItem("refresh_token")).toBe("test-refresh-token");
  });
});

describe("clearTokens", () => {
  it("removes access_token and refresh_token from localStorage", () => {
    localStorage.setItem("access_token", "tok1");
    localStorage.setItem("refresh_token", "tok2");

    clearTokens();

    expect(localStorage.getItem("access_token")).toBeNull();
    expect(localStorage.getItem("refresh_token")).toBeNull();
  });
});
