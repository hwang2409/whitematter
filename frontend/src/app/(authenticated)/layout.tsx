"use client";
import { useAuth } from "@/context/AuthContext";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { useEffect, useState } from "react";
import { getModels } from "@/api";
import type { Model } from "@/api";
import ErrorBoundary from "@/components/ErrorBoundary";

const NAV = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/data", label: "Data (S3)" },
  { href: "/train", label: "Train" },
  { href: "/models", label: "Models" },
  { href: "/predict", label: "Predict" },
  { href: "/settings", label: "Settings" },
] as const;

export default function AuthenticatedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { user, loading, logout } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const [models, setModels] = useState<Model[]>([]);

  useEffect(() => {
    if (!user) return;
    getModels()
      .then(setModels)
      .catch(() => {});
  }, [user]);

  useEffect(() => {
    if (loading) return;
    if (!user) {
      router.replace("/login");
    }
  }, [user, loading, router]);

  if (loading || !user) {
    return (
      <div className="app" style={{ alignItems: "center", justifyContent: "center", minHeight: "100vh" }}>
        <p>Loading…</p>
      </div>
    );
  }

  const completedModels = models.filter((m) => m.status === "completed");

  return (
    <ErrorBoundary>
      <div className="app">
        <header className="header">
          <h1>whitematter</h1>
          <div className="header-actions">
            <span className="header-email">{user.email}</span>
            <button type="button" className="header-logout" onClick={logout}>
              Log out
            </button>
          </div>
        </header>

        <nav className="workflow-nav">
          {NAV.map(({ href, label }) => {
            const isActive = pathname === href || (href !== "/dashboard" && pathname?.startsWith(href));
            const isComplete =
              (href === "/train" && completedModels.length > 0) ||
              (href === "/models" && completedModels.length > 0);
            return (
              <Link
                key={href}
                href={href}
                className={`workflow-step ${isActive ? "active" : ""} ${isComplete ? "complete" : ""}`}
              >
                {(href === "/train" || href === "/models") && (
                  <span className="step-num">
                    {isComplete ? "✓" : ""}
                  </span>
                )}
                <span className="step-label">{label}</span>
                {href === "/models" && completedModels.length > 0 && (
                  <span className="step-badge">{completedModels.length}</span>
                )}
              </Link>
            );
          })}
        </nav>

        <div className="app-body">
          <main className="content">{children}</main>
        </div>
      </div>
    </ErrorBoundary>
  );
}
