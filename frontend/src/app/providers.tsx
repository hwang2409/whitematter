"use client";
import { AuthProvider } from "@/context/AuthContext";
import { ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { theme } from "@/theme";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AuthProvider>{children}</AuthProvider>
    </ThemeProvider>
  );
}
