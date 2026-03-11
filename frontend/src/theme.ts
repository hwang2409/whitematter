"use client";

import { createTheme } from "@mui/material/styles";

// Accent: cool blue-white (neural pathways / whitematter). Single accent for all interactive elements and data viz.
const ACCENT = "#7EB8FF";
const ACCENT_LIGHT = "rgba(126, 184, 255, 0.15)";
const ACCENT_MUTED = "rgba(126, 184, 255, 0.5)";
const BG = "#0a0a0a";
const SURFACE = "#1a1a1a";
const BORDER = "rgba(255,255,255,0.08)";

export const theme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: ACCENT,
      light: "#9ecaff",
      dark: "#5a9ae8",
      contrastText: "#0a0a0a",
    },
    secondary: {
      main: "rgba(255,255,255,0.6)",
      light: "rgba(255,255,255,0.8)",
      dark: "rgba(255,255,255,0.4)",
    },
    background: {
      default: BG,
      paper: SURFACE,
    },
    text: {
      primary: "#ffffff",
      secondary: "#a3a3a3",
      disabled: "#525252",
    },
    divider: BORDER,
    error: { main: "#ef4444" },
    success: { main: "#22c55e" },
    warning: { main: "#eab308" },
  },
  shape: {
    borderRadius: 8,
  },
  typography: {
    fontFamily: [
      "-apple-system",
      "BlinkMacSystemFont",
      '"Segoe UI"',
      "Roboto",
      "sans-serif",
    ].join(","),
    h1: {
      fontSize: "1.5rem",
      fontWeight: 600,
      letterSpacing: "-0.02em",
    },
    h2: {
      fontSize: "1.25rem",
      fontWeight: 600,
      letterSpacing: "-0.02em",
    },
    h3: {
      fontSize: "1rem",
      fontWeight: 600,
    },
    body1: {
      fontSize: "0.9375rem",
      lineHeight: 1.5,
    },
    body2: {
      fontSize: "0.875rem",
      lineHeight: 1.5,
      color: "rgba(250,250,250,0.64)",
    },
    button: {
      textTransform: "none",
      fontWeight: 500,
    },
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
        contained: {
          boxShadow: "none",
          "&:hover": {
            boxShadow: "none",
          },
        },
        outlined: {
          borderColor: BORDER,
          "&:hover": {
            borderColor: ACCENT_MUTED,
            backgroundColor: ACCENT_LIGHT,
          },
        },
      },
    },
    MuiTextField: {
      defaultProps: {
        variant: "outlined",
        size: "medium",
      },
      styleOverrides: {
        root: {
          "& .MuiOutlinedInput-root": {
            borderRadius: 8,
            backgroundColor: "rgba(255,255,255,0.04)",
            fontFamily: 'inherit',
            "&:hover .MuiOutlinedInput-notchedOutline": {
              borderColor: "rgba(255,255,255,0.2)",
            },
            "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
              borderColor: ACCENT,
              borderWidth: 1,
            },
            "&.Mui-focused": {
              backgroundColor: "rgba(126,184,255,0.06)",
            },
            "& fieldset": {
              borderColor: BORDER,
            },
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
          borderRadius: 8,
          border: `1px solid ${BORDER}`,
          backgroundColor: SURFACE,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          border: `1px solid ${BORDER}`,
          backgroundColor: SURFACE,
          transition: "border-color 0.2s ease-out, box-shadow 0.2s ease-out",
          "&:hover": {
            borderColor: ACCENT_MUTED,
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 6,
          border: `1px solid ${BORDER}`,
          backgroundColor: "rgba(255,255,255,0.04)",
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
            color: "rgba(255,255,255,0.5)",
            fontSize: "0.625rem",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            borderBottom: `1px solid ${BORDER}`,
          },
          "& .MuiTableCell-body": {
            borderBottom: `1px solid ${BORDER}`,
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
          borderBottom: `1px solid ${BORDER}`,
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: "none",
          fontWeight: 500,
          fontSize: "0.875rem",
        },
      },
    },
    MuiLink: {
      styleOverrides: {
        root: {
          color: ACCENT,
          textDecoration: "none",
          "&:hover": {
            textDecoration: "underline",
          },
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
  },
});

// Export tokens for use in sx and custom components
export const themeTokens = {
  accent: ACCENT,
  accentLight: ACCENT_LIGHT,
  accentMuted: ACCENT_MUTED,
  bg: BG,
  surface: SURFACE,
  border: BORDER,
};
