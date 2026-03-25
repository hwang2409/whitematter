"use client";

import { createTheme } from "@mui/material/styles";

const ACCENT = "#F97316";
const ACCENT_HOVER = "#EA580C";
const ACCENT_SUBTLE = "rgba(249, 115, 22, 0.08)";

const dark = {
  bg: "#09090B",
  surface: "#18181B",
  surfaceHover: "#27272A",
  card: "#18181B",
  border: "#27272A",
  borderHover: "#3F3F46",
  textPrimary: "#FAFAFA",
  textSecondary: "#A1A1AA",
  textMuted: "#52525B",
  inputBg: "rgba(255, 255, 255, 0.03)",
  chipBg: "rgba(255, 255, 255, 0.04)",
};

const light = {
  bg: "#FAFAF9",
  surface: "#FFFFFF",
  surfaceHover: "#F4F4F5",
  card: "#FFFFFF",
  border: "#E4E4E7",
  borderHover: "#D4D4D8",
  textPrimary: "#09090B",
  textSecondary: "#71717A",
  textMuted: "#A1A1AA",
  inputBg: "rgba(0, 0, 0, 0.02)",
  chipBg: "rgba(0, 0, 0, 0.03)",
};

export type ThemeMode = "light" | "dark";

export function getTheme(mode: ThemeMode) {
  const c = mode === "dark" ? dark : light;
  const accentMain = mode === "dark" ? ACCENT : ACCENT_HOVER;

  return createTheme({
    palette: {
      mode,
      primary: {
        main: accentMain,
        light: ACCENT,
        dark: ACCENT_HOVER,
        contrastText: "#FFFFFF",
      },
      secondary: {
        main: c.textSecondary,
      },
      background: {
        default: c.bg,
        paper: c.surface,
      },
      text: {
        primary: c.textPrimary,
        secondary: c.textSecondary,
        disabled: c.textMuted,
      },
      divider: c.border,
      error: { main: "#EF4444" },
      success: { main: "#22C55E" },
      warning: { main: "#EAB308" },
    },

    shape: { borderRadius: 8 },

    typography: {
      fontFamily: "'Outfit', -apple-system, BlinkMacSystemFont, sans-serif",
      h1: {
        fontFamily: "'Instrument Serif', Georgia, serif",
        fontSize: "2.5rem",
        fontWeight: 400,
        letterSpacing: "-0.02em",
        lineHeight: 1.15,
      },
      h2: {
        fontFamily: "'Instrument Serif', Georgia, serif",
        fontSize: "1.75rem",
        fontWeight: 400,
        letterSpacing: "-0.01em",
        lineHeight: 1.2,
      },
      h3: {
        fontFamily: "'Outfit', sans-serif",
        fontSize: "1rem",
        fontWeight: 600,
        letterSpacing: "-0.01em",
      },
      body1: {
        fontSize: "0.9375rem",
        lineHeight: 1.6,
        letterSpacing: "-0.006em",
      },
      body2: {
        fontSize: "0.875rem",
        lineHeight: 1.5,
        letterSpacing: "-0.006em",
      },
      button: {
        textTransform: "none",
        fontWeight: 500,
        letterSpacing: "-0.006em",
      },
      overline: {
        fontFamily: "'Outfit', sans-serif",
        fontSize: "0.6875rem",
        fontWeight: 600,
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        lineHeight: 1.5,
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
            padding: "8px 20px",
            fontSize: "0.8125rem",
            fontWeight: 500,
            transition: "all 0.15s ease-out",
          },
          contained: {
            boxShadow: "none",
            "&:hover": {
              boxShadow: "none",
            },
          },
          containedPrimary: {
            backgroundColor: accentMain,
            color: "#FFFFFF",
            "&:hover": {
              backgroundColor: ACCENT_HOVER,
            },
          },
          outlined: {
            borderColor: c.border,
            color: c.textSecondary,
            "&:hover": {
              borderColor: c.borderHover,
              backgroundColor: mode === "dark" ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)",
              color: c.textPrimary,
            },
          },
          text: {
            color: c.textSecondary,
            "&:hover": {
              backgroundColor: mode === "dark" ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.04)",
              color: c.textPrimary,
            },
          },
          sizeSmall: {
            padding: "6px 14px",
            fontSize: "0.75rem",
          },
        },
      },

      MuiTextField: {
        defaultProps: { variant: "outlined", size: "medium" },
        styleOverrides: {
          root: {
            "& .MuiOutlinedInput-root": {
              borderRadius: 8,
              backgroundColor: c.inputBg,
              fontSize: "0.875rem",
              fontFamily: "'Outfit', sans-serif",
              "&:hover .MuiOutlinedInput-notchedOutline": {
                borderColor: c.borderHover,
              },
              "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                borderColor: accentMain,
                borderWidth: 1,
              },
              "& fieldset": {
                borderColor: c.border,
              },
            },
            "& .MuiInputLabel-root": {
              fontSize: "0.8125rem",
              fontFamily: "'Outfit', sans-serif",
            },
          },
        },
      },

      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: "none",
            borderRadius: 12,
            border: `1px solid ${c.border}`,
            backgroundColor: c.surface,
          },
        },
      },

      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            border: `1px solid ${c.border}`,
            backgroundColor: c.surface,
            transition: "border-color 0.15s ease-out, box-shadow 0.15s ease-out",
            "&:hover": {
              borderColor: c.borderHover,
            },
          },
        },
      },

      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: 6,
            border: `1px solid ${c.border}`,
            backgroundColor: c.chipBg,
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "0.6875rem",
            fontWeight: 500,
            "&.MuiChip-filled": {
              backgroundColor: ACCENT_SUBTLE,
              borderColor: "rgba(249, 115, 22, 0.2)",
              color: accentMain,
            },
          },
        },
      },

      MuiTable: {
        styleOverrides: {
          root: {
            "& .MuiTableCell-head": {
              color: c.textMuted,
              fontSize: "0.625rem",
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              fontFamily: "'Outfit', sans-serif",
              borderBottom: `1px solid ${c.border}`,
              padding: "8px 12px",
            },
            "& .MuiTableCell-body": {
              borderBottom: `1px solid ${c.border}`,
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: "0.8125rem",
              padding: "10px 12px",
            },
          },
        },
      },

      MuiDialog: {
        styleOverrides: {
          paper: {
            borderRadius: 12,
            border: `1px solid ${c.border}`,
            backgroundColor: c.surface,
            backgroundImage: "none",
          },
        },
      },

      MuiMenu: {
        styleOverrides: {
          paper: {
            borderRadius: 10,
            border: `1px solid ${c.border}`,
            backgroundColor: c.surface,
            boxShadow: mode === "dark"
              ? "0 8px 32px rgba(0, 0, 0, 0.5)"
              : "0 8px 32px rgba(0, 0, 0, 0.1)",
          },
        },
      },

      MuiMenuItem: {
        styleOverrides: {
          root: {
            fontSize: "0.8125rem",
            fontFamily: "'Outfit', sans-serif",
            borderRadius: 6,
            margin: "2px 4px",
            padding: "8px 12px",
            "&:hover": {
              backgroundColor: mode === "dark" ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.04)",
            },
          },
        },
      },

      MuiTab: {
        styleOverrides: {
          root: {
            textTransform: "none",
            fontWeight: 500,
            fontSize: "0.8125rem",
            fontFamily: "'Outfit', sans-serif",
          },
        },
      },

      MuiLink: {
        styleOverrides: {
          root: {
            color: accentMain,
            textDecoration: "none",
            "&:hover": { textDecoration: "underline" },
          },
        },
      },

      MuiAlert: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            fontSize: "0.8125rem",
          },
        },
      },

      MuiCircularProgress: {
        styleOverrides: {
          root: {
            color: accentMain,
          },
        },
      },

      MuiSlider: {
        styleOverrides: {
          root: {
            color: accentMain,
          },
        },
      },

      MuiTooltip: {
        styleOverrides: {
          tooltip: {
            backgroundColor: mode === "dark" ? "#27272A" : "#18181B",
            color: "#FAFAFA",
            fontSize: "0.75rem",
            fontFamily: "'Outfit', sans-serif",
            borderRadius: 6,
            padding: "4px 10px",
          },
          arrow: {
            color: mode === "dark" ? "#27272A" : "#18181B",
          },
        },
      },
    },
  });
}

export const theme = getTheme("dark");

export const themeTokens = {
  accent: ACCENT,
  accentHover: ACCENT_HOVER,
  accentSubtle: ACCENT_SUBTLE,
  bg: dark.bg,
  surface: dark.surface,
  card: dark.card,
  border: dark.border,
  borderHover: dark.borderHover,
  textMuted: dark.textMuted,
};
