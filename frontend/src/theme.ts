"use client";

import { createTheme } from "@mui/material/styles";

const ACCENT = "#78716C";
const ACCENT_LIGHT = "rgba(120, 113, 108, 0.08)";
const ACCENT_MUTED = "rgba(120, 113, 108, 0.35)";

const dark = {
  bg: "#141311",
  surface: "#1E1D1B",
  card: "#1A1917",
  border: "rgba(255,255,255,0.07)",
  borderHover: "rgba(255,255,255,0.14)",
  textPrimary: "#F2F1EE",
  textSecondary: "#9C9A95",
  textMuted: "#6B6963",
  inputBg: "rgba(255,255,255,0.04)",
  chipBg: "rgba(255,255,255,0.05)",
};

const light = {
  bg: "#F9F8F6",
  surface: "#FFFFFF",
  card: "#F2F1EE",
  border: "rgba(0,0,0,0.06)",
  borderHover: "rgba(0,0,0,0.12)",
  textPrimary: "#1A1A1A",
  textSecondary: "#888580",
  textMuted: "#B5B3AF",
  inputBg: "rgba(0,0,0,0.02)",
  chipBg: "rgba(0,0,0,0.03)",
};

export type ThemeMode = "light" | "dark";

export function getTheme(mode: ThemeMode) {
  const colors = mode === "dark" ? dark : light;
  return createTheme({
    palette: {
      mode,
      primary: {
        main: ACCENT,
        light: "#A8A29E",
        dark: "#57534E",
        contrastText: "#FFFFFF",
      },
      secondary: {
        main: colors.textSecondary,
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
        disabled: colors.textMuted,
      },
      divider: colors.border,
      error: { main: "#ef4444" },
      success: { main: "#22c55e" },
      warning: { main: "#eab308" },
    },
    shape: { borderRadius: 10 },
    typography: {
      fontFamily: "'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif",
      h1: {
        fontFamily: "'DM Serif Display', Georgia, serif",
        fontSize: "2.25rem",
        fontWeight: 400,
        letterSpacing: "-0.02em",
      },
      h2: {
        fontFamily: "'DM Serif Display', Georgia, serif",
        fontSize: "1.5rem",
        fontWeight: 400,
        letterSpacing: "-0.02em",
      },
      h3: { fontSize: "1rem", fontWeight: 600 },
      body1: { fontSize: "0.9375rem", lineHeight: 1.6 },
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
            borderRadius: 10,
            padding: "10px 24px",
            fontSize: "0.875rem",
            transition: "all 0.15s ease-out",
          },
          contained: {
            boxShadow: "none",
            "&:hover": { boxShadow: "none" },
          },
          outlined: {
            borderColor: colors.border,
            "&:hover": {
              borderColor: colors.borderHover,
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
              borderRadius: 10,
              backgroundColor: colors.inputBg,
              "&:hover .MuiOutlinedInput-notchedOutline": {
                borderColor: colors.borderHover,
              },
              "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                borderColor: ACCENT,
                borderWidth: 1,
              },
              "&.Mui-focused": { backgroundColor: ACCENT_LIGHT },
              "& fieldset": { borderColor: colors.border },
            },
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: "none",
            borderRadius: 16,
            border: `1px solid ${colors.border}`,
            backgroundColor: colors.surface,
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 16,
            border: `1px solid ${colors.border}`,
            backgroundColor: colors.surface,
            transition: "border-color 0.15s ease-out",
            "&:hover": { borderColor: colors.borderHover },
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: 6,
            border: `1px solid ${colors.border}`,
            backgroundColor: colors.chipBg,
            fontFamily: "'DM Mono', monospace",
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
              fontFamily: "'DM Mono', monospace",
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
      MuiAlert: { styleOverrides: { root: { borderRadius: 10 } } },
    },
  });
}

export const theme = getTheme("dark");

export const themeTokens = {
  accent: ACCENT,
  accentLight: ACCENT_LIGHT,
  accentMuted: ACCENT_MUTED,
  bg: dark.bg,
  surface: dark.surface,
  card: dark.card,
  border: dark.border,
  borderHover: dark.borderHover,
  textMuted: dark.textMuted,
};
