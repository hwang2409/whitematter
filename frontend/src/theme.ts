"use client";

import { createTheme } from "@mui/material/styles";

// Minimal, friendly, Apple-inspired dark theme: clean typography, generous spacing, subtle radius
export const theme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#ffffff",
      light: "#f5f5f5",
      dark: "#e0e0e0",
    },
    background: {
      default: "#0a0a0a",
      paper: "#141414",
    },
    text: {
      primary: "#fafafa",
      secondary: "#a3a3a3",
      disabled: "#525252",
    },
    divider: "rgba(255,255,255,0.08)",
    error: { main: "#ef4444" },
    success: { main: "#22c55e" },
    warning: { main: "#eab308" },
  },
  shape: {
    borderRadius: 12,
  },
  typography: {
    fontFamily: [
      "-apple-system",
      "BlinkMacSystemFont",
      '"SF Pro Text"',
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
          borderRadius: 10,
          padding: "10px 20px",
          fontSize: "0.9375rem",
        },
        contained: {
          boxShadow: "none",
          "&:hover": {
            boxShadow: "none",
          },
        },
        outlined: {
          borderWidth: 1,
          "&:hover": {
            borderWidth: 1,
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
            borderRadius: 10,
            backgroundColor: "rgba(255,255,255,0.04)",
            "&:hover .MuiOutlinedInput-notchedOutline": {
              borderColor: "rgba(255,255,255,0.2)",
            },
            "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
              borderColor: "rgba(255,255,255,0.5)",
              borderWidth: 1,
            },
            "&.Mui-focused": {
              backgroundColor: "rgba(255,255,255,0.06)",
            },
            "& fieldset": {
              borderColor: "rgba(255,255,255,0.12)",
            },
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
          borderRadius: 12,
          border: "1px solid rgba(255,255,255,0.08)",
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
          borderBottom: "1px solid rgba(255,255,255,0.08)",
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
          borderRadius: 10,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          border: "1px solid rgba(255,255,255,0.08)",
        },
      },
    },
  },
});
