"use client";

import { createContext, useContext, useState, useMemo, useEffect } from "react";
import type { ThemeMode } from "@/theme";

const STORAGE_KEY = "whitematter-theme";

type ThemeContextValue = {
  mode: ThemeMode;
  setMode: (mode: ThemeMode) => void;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeContextProvider({ children }: { children: React.ReactNode }) {
  const [mode, setModeState] = useState<ThemeMode>("dark");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY) as ThemeMode | null;
      if (stored === "light" || stored === "dark") setModeState(stored);
    } catch {}
    setMounted(true);
  }, []);

  const setMode = useMemo(
    () =>
      (next: ThemeMode) => {
        setModeState(next);
        try {
          localStorage.setItem(STORAGE_KEY, next);
        } catch {}
      },
    []
  );

  const value = useMemo(() => ({ mode, setMode }), [mode, setMode]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useThemeMode() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useThemeMode must be used within ThemeContextProvider");
  return ctx;
}
