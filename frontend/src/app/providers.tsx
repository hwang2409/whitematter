"use client";
import { useMemo } from "react";
import { AuthProvider } from "@/context/AuthContext";
import { DesignProvider } from "@/context/DesignContext";
import { ThemeContextProvider, useThemeMode } from "@/context/ThemeContext";
import { ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { getTheme } from "@/theme";
import { GoogleOAuthProvider } from "@react-oauth/google";

function GoogleOAuthWrapper({ children }: { children: React.ReactNode }) {
  const clientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;
  if (!clientId) return <>{children}</>;
  return <GoogleOAuthProvider clientId={clientId}>{children}</GoogleOAuthProvider>;
}

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
    <GoogleOAuthWrapper>
      <ThemeContextProvider>
        <ThemedApp>
          <AuthProvider>
            <DesignProvider>{children}</DesignProvider>
          </AuthProvider>
        </ThemedApp>
      </ThemeContextProvider>
    </GoogleOAuthWrapper>
  );
}
