"use client";
import {
  createContext,
  useContext,
  useState,
  useCallback,
  ReactNode,
} from "react";
import type { Architecture } from "@/api";

const SESSION_KEY = "wm_design_architecture";

interface DesignContextType {
  architecture: Architecture | null;
  setArchitecture: (arch: Architecture | null) => void;
  clearArchitecture: () => void;
}

const DesignContext = createContext<DesignContextType | null>(null);

export function DesignProvider({ children }: { children: ReactNode }) {
  const [architecture, setArchitectureState] = useState<Architecture | null>(
    () => {
      if (typeof window === "undefined") return null;
      try {
        const stored = sessionStorage.getItem(SESSION_KEY);
        return stored ? JSON.parse(stored) : null;
      } catch {
        return null;
      }
    }
  );

  const setArchitecture = useCallback((arch: Architecture | null) => {
    setArchitectureState(arch);
    if (typeof window !== "undefined") {
      if (arch) {
        sessionStorage.setItem(SESSION_KEY, JSON.stringify(arch));
      } else {
        sessionStorage.removeItem(SESSION_KEY);
      }
    }
  }, []);

  const clearArchitecture = useCallback(() => {
    setArchitectureState(null);
    if (typeof window !== "undefined") {
      sessionStorage.removeItem(SESSION_KEY);
    }
  }, []);

  return (
    <DesignContext.Provider
      value={{ architecture, setArchitecture, clearArchitecture }}
    >
      {children}
    </DesignContext.Provider>
  );
}

export function useDesign() {
  const ctx = useContext(DesignContext);
  if (!ctx) throw new Error("useDesign must be inside DesignProvider");
  return ctx;
}
