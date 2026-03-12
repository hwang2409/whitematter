import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { ThemeProvider, createTheme } from "@mui/material/styles";

// Mock next/link before importing the component
vi.mock("next/link", () => ({
  default: ({ children, href, ...props }: { children: React.ReactNode; href: string; [key: string]: unknown }) => (
    <a href={href} {...props}>{children}</a>
  ),
}));

// Mock the auth context
vi.mock("@/context/AuthContext", () => ({
  useAuth: vi.fn(),
}));

// Mock the API module
vi.mock("@/api", () => ({
  getCustomDatasets: vi.fn(),
  getModels: vi.fn(),
}));

import { useAuth } from "@/context/AuthContext";
import { getCustomDatasets, getModels } from "@/api";
import DashboardPage from "@/views/DashboardPage";

const mockUseAuth = vi.mocked(useAuth);
const mockGetCustomDatasets = vi.mocked(getCustomDatasets);
const mockGetModels = vi.mocked(getModels);

const theme = createTheme();

function renderWithProviders(ui: React.ReactElement) {
  return render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);
}

beforeEach(() => {
  vi.clearAllMocks();
  mockUseAuth.mockReturnValue({
    user: { id: "user-1", email: "test@example.com" },
    token: "test-token",
    loading: false,
    loginWithTokens: vi.fn(),
    logout: vi.fn(),
  });
  // Default: API calls that never resolve (to test loading state)
  mockGetCustomDatasets.mockReturnValue(new Promise(() => {}));
  mockGetModels.mockReturnValue(new Promise(() => {}));
});

describe("DashboardPage", () => {
  it("renders without crashing", () => {
    renderWithProviders(<DashboardPage />);
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
  });

  it("shows the user email in the welcome message", () => {
    renderWithProviders(<DashboardPage />);
    expect(screen.getByText(/Welcome back, test@example.com/)).toBeInTheDocument();
  });

  it("shows loading state initially", () => {
    renderWithProviders(<DashboardPage />);
    // During loading, stat cards show em dash
    const dashes = screen.getAllByText("\u2014");
    expect(dashes.length).toBeGreaterThanOrEqual(3);
    // Activity section shows Loading text
    expect(screen.getByText("Loading\u2026")).toBeInTheDocument();
  });

  it("renders quick action cards", () => {
    renderWithProviders(<DashboardPage />);
    expect(screen.getByText("Upload Dataset")).toBeInTheDocument();
    expect(screen.getByText("New Training Run")).toBeInTheDocument();
    expect(screen.getByText("Try Prediction")).toBeInTheDocument();
  });

  it("renders section headings", () => {
    renderWithProviders(<DashboardPage />);
    expect(screen.getByText("Recent Activity")).toBeInTheDocument();
    expect(screen.getByText("Quick Actions")).toBeInTheDocument();
  });
});
