"use client";

import { createTheme } from "@mui/material/styles";

const ACCENT = "#7EB8FF";
const ACCENT_LIGHT = "rgba(126, 184, 255, 0.15)";
const ACCENT_MUTED = "rgba(126, 184, 255, 0.5)";

const dark = {
  bg: "#0a0a0a",
  surface: "#1a1a1a",
  border: "rgba(255,255,255,0.08)",
  textPrimary: "#ffffff",
  textSecondary: "#a3a3a3",
  secondary: "rgba(255,255,255,0.6)",
  inputBg: "rgba(255,255,255,0.04)",
  chipBg: "rgba(255,255,255,0.04)",
};

const light = {
  bg: "#fafafa",
  surface: "#ffffff",
  border: "rgba(0,0,0,0.12)",
  textPrimary: "rgba(0,0,0,0.87)",
  textSecondary: "rgba(0,0,0,0.6)",
  secondary: "rgba(0,0,0,0.6)",
  inputBg: "rgba(0,0,0,0.04)",
  chipBg: "rgba(0,0,0,0.06)",
};

export type ThemeMode = "light" | "dark";

export function getTheme(mode: ThemeMode) {
  const colors = mode === "dark" ? dark : light;
  return createTheme({
    palette: {
      mode,
      primary: {
        main: ACCENT,
        light: "#9ecaff",
        dark: "#5a9ae8",
        contrastText: "#0a0a0a",
      },
      secondary: {
        main: colors.secondary,
        light: mode === "dark" ? "rgba(255,255,255,0.8)" : "rgba(0,0,0,0.8)",
        dark: mode === "dark" ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)",
      },
      background: {
        default: colors.bg,
        paper: colors.surface,
      },
      text: {
        primary: colors.textPrimary,
        secondary: colors.textSecondary,
        disabled: mode === "dark" ? "#525252" : "rgba(0,0,0,0.38)",
      },
      divider: colors.border,
      error: { main: "#ef4444" },
      success: { main: "#22c55e" },
      warning: { main: "#eab308" },
    },
    shape: { borderRadius: 8 },
    typography: {
      fontFamily: [
        "-apple-system",
        "BlinkMacSystemFont",
        '"Segoe UI"',
        "Roboto",
        "sans-serif",
      ].join(","),
      h1: { fontSize: "1.5rem", fontWeight: 600, letterSpacing: "-0.02em" },
      h2: { fontSize: "1.25rem", fontWeight: 600, letterSpacing: "-0.02em" },
      h3: { fontSize: "1rem", fontWeight: 600 },
      body1: { fontSize: "0.9375rem", lineHeight: 1.5 },
      body2: { fontSize: "0.875rem", lineHeight: 1.5 },
      button: { textTransform: "none", fontWeight: 500 },
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            WebkitFontSmoothing: "antialiased",
            MozOsxFontSmoothing: "grayscale",
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            padding: "10px 20px",
            fontSize: "0.9375rem",
            transition: "all 0.2s ease-out",
          },
          contained: { boxShadow: "none", "&:hover": { boxShadow: "none" } },
          outlined: {
            borderColor: colors.border,
            "&:hover": {
              borderColor: ACCENT_MUTED,
              backgroundColor: ACCENT_LIGHT,
            },
          },
        },
      },
      MuiTextField: {
        defaultProps: { variant: "outlined", size: "medium" },
        styleOverrides: {
          root: {
            "& .MuiOutlinedInput-root": {
              borderRadius: 8,
              backgroundColor: colors.inputBg,
              "&:hover .MuiOutlinedInput-notchedOutline": {
                borderColor: mode === "dark" ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.23)",
              },
              "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                borderColor: ACCENT,
                borderWidth: 1,
              },
              "&.Mui-focused": { backgroundColor: "rgba(126,184,255,0.06)" },
              "& fieldset": { borderColor: colors.border },
            },
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: "none",
            borderRadius: 8,
            border: `1px solid ${colors.border}`,
            backgroundColor: colors.surface,
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            border: `1px solid ${colors.border}`,
            backgroundColor: colors.surface,
            transition: "border-color 0.2s ease-out, box-shadow 0.2s ease-out",
            "&:hover": { borderColor: ACCENT_MUTED },
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: 6,
            border: `1px solid ${colors.border}`,
            backgroundColor: colors.chipBg,
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: "0.75rem",
            "&.MuiChip-filled": {
              backgroundColor: ACCENT_LIGHT,
              borderColor: ACCENT_MUTED,
              color: ACCENT,
            },
          },
        },
      },
      MuiTable: {
        styleOverrides: {
          root: {
            "& .MuiTableCell-head": {
              color: colors.textSecondary,
              fontSize: "0.625rem",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              borderBottom: `1px solid ${colors.border}`,
            },
            "& .MuiTableCell-body": {
              borderBottom: `1px solid ${colors.border}`,
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: "0.8125rem",
            },
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundImage: "none",
            borderBottom: `1px solid ${colors.border}`,
          },
        },
      },
      MuiTab: {
        styleOverrides: {
          root: { textTransform: "none", fontWeight: 500, fontSize: "0.875rem" },
        },
      },
      MuiLink: {
        styleOverrides: {
          root: {
            color: ACCENT,
            textDecoration: "none",
            "&:hover": { textDecoration: "underline" },
          },
        },
      },
      MuiAlert: { styleOverrides: { root: { borderRadius: 8 } } },
    },
  });
}

// Backwards compatibility: default export dark theme (will be overridden by ThemeProvider with context)
export const theme = getTheme("dark");

export const themeTokens = {
  accent: ACCENT,
  accentLight: ACCENT_LIGHT,
  accentMuted: ACCENT_MUTED,
  // Mode-agnostic; for mode-specific use theme.palette in sx
  bg: dark.bg,
  surface: dark.surface,
  border: dark.border,
};
