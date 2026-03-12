"use client";
import { useMemo } from "react";
import { AuthProvider } from "@/context/AuthContext";
import { DesignProvider } from "@/context/DesignContext";
import { ThemeContextProvider, useThemeMode } from "@/context/ThemeContext";
import { ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { getTheme } from "@/theme";

function ThemedApp({ children }: { children: React.ReactNode }) {
  const { mode } = useThemeMode();
  const theme = useMemo(() => getTheme(mode), [mode]);
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {children}
    </ThemeProvider>
  );
}

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeContextProvider>
      <ThemedApp>
        <AuthProvider>
          <DesignProvider>{children}</DesignProvider>
        </AuthProvider>
      </ThemedApp>
    </ThemeContextProvider>
  );
}
