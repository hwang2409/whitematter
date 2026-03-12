import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { ReactNode } from "react";
import { AuthProvider, useAuth } from "@/context/AuthContext";

// Mock the auth service
vi.mock("@/services/auth", () => ({
  getMe: vi.fn(),
  getStoredToken: vi.fn(),
  storeTokens: vi.fn(),
  clearTokens: vi.fn(),
}));

import { getMe, getStoredToken, storeTokens, clearTokens } from "@/services/auth";

const mockGetMe = vi.mocked(getMe);
const mockGetStoredToken = vi.mocked(getStoredToken);
const mockStoreTokens = vi.mocked(storeTokens);
const mockClearTokens = vi.mocked(clearTokens);

function wrapper({ children }: { children: ReactNode }) {
  return <AuthProvider>{children}</AuthProvider>;
}

beforeEach(() => {
  vi.clearAllMocks();
  mockGetStoredToken.mockReturnValue(null);
});

describe("AuthProvider", () => {
  it("renders children", () => {
    mockGetStoredToken.mockReturnValue(null);
    const { result } = renderHook(() => useAuth(), { wrapper });
    expect(result.current).toBeDefined();
  });

  it("sets loading=false when no token is stored", async () => {
    mockGetStoredToken.mockReturnValue(null);

    const { result } = renderHook(() => useAuth(), { wrapper });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
    expect(result.current.user).toBeNull();
    expect(result.current.token).toBeNull();
  });

  it("calls getMe and sets user when token exists", async () => {
    mockGetStoredToken.mockReturnValue("stored-token");
    mockGetMe.mockResolvedValueOnce({
      id: "user-1",
      email: "test@example.com",
      oauth_provider: null,
      created_at: "2025-01-01T00:00:00Z",
    });

    const { result } = renderHook(() => useAuth(), { wrapper });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
    expect(mockGetMe).toHaveBeenCalledWith("stored-token");
    expect(result.current.user).toEqual({ id: "user-1", email: "test@example.com" });
    expect(result.current.token).toBe("stored-token");
  });

  it("clears tokens when getMe fails", async () => {
    mockGetStoredToken.mockReturnValue("bad-token");
    mockGetMe.mockRejectedValueOnce(new Error("Not authenticated"));

    const { result } = renderHook(() => useAuth(), { wrapper });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
    expect(mockClearTokens).toHaveBeenCalled();
    expect(result.current.user).toBeNull();
    expect(result.current.token).toBeNull();
  });
});

describe("loginWithTokens", () => {
  it("stores tokens and updates token state", async () => {
    mockGetStoredToken.mockReturnValue(null);

    const { result } = renderHook(() => useAuth(), { wrapper });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Now simulate loginWithTokens - this will update the token state,
    // triggering the useEffect to call getMe
    mockGetMe.mockResolvedValueOnce({
      id: "user-2",
      email: "new@example.com",
      oauth_provider: null,
      created_at: "2025-01-01T00:00:00Z",
    });

    act(() => {
      result.current.loginWithTokens({
        access_token: "new-access",
        refresh_token: "new-refresh",
        token_type: "bearer",
      });
    });

    expect(mockStoreTokens).toHaveBeenCalledWith({
      access_token: "new-access",
      refresh_token: "new-refresh",
      token_type: "bearer",
    });

    await waitFor(() => {
      expect(result.current.token).toBe("new-access");
    });
  });
});

describe("logout", () => {
  it("clears tokens and user state", async () => {
    mockGetStoredToken.mockReturnValue("stored-token");
    mockGetMe.mockResolvedValueOnce({
      id: "user-1",
      email: "test@example.com",
      oauth_provider: null,
      created_at: "2025-01-01T00:00:00Z",
    });

    const { result } = renderHook(() => useAuth(), { wrapper });

    await waitFor(() => {
      expect(result.current.user).not.toBeNull();
    });

    act(() => {
      result.current.logout();
    });

    expect(mockClearTokens).toHaveBeenCalled();
    expect(result.current.user).toBeNull();
    expect(result.current.token).toBeNull();
  });
});

describe("useAuth", () => {
  it("throws when used outside AuthProvider", () => {
    // Suppress the console.error from React for this expected error
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => {
      renderHook(() => useAuth());
    }).toThrow("useAuth must be inside AuthProvider");
    spy.mockRestore();
  });
});
