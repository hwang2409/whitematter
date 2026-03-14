"use client";
import { createContext, useContext, useState, useMemo, useEffect } from "react";

const STORAGE_KEY = "whitematter-sidebar";

type SidebarContextValue = {
  collapsed: boolean;
  toggleSidebar: () => void;
};

const SidebarContext = createContext<SidebarContextValue | null>(null);

export function SidebarProvider({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored === "true") setCollapsed(true);
      else if (stored === "false") setCollapsed(false);
    } catch {}
  }, []);

  const toggleSidebar = useMemo(
    () => () => {
      setCollapsed((prev) => {
        const next = !prev;
        try {
          localStorage.setItem(STORAGE_KEY, String(next));
        } catch {}
        return next;
      });
    },
    [],
  );

  const value = useMemo(
    () => ({ collapsed, toggleSidebar }),
    [collapsed, toggleSidebar],
  );

  return (
    <SidebarContext.Provider value={value}>{children}</SidebarContext.Provider>
  );
}

export function useSidebar() {
  const ctx = useContext(SidebarContext);
  if (!ctx)
    throw new Error("useSidebar must be used within a SidebarProvider");
  return ctx;
}
