"use client";
import { useAuth } from "@/context/AuthContext";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { useEffect, useState } from "react";
import { getModels } from "@/api";
import type { Model } from "@/api";
import ErrorBoundary from "@/components/ErrorBoundary";
import Box from "@mui/material/Box";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";

const NAV: { href: string; label: string }[] = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/data", label: "Data" },
  { href: "/train", label: "Train" },
  { href: "/models", label: "Models" },
  { href: "/predict", label: "Predict" },
  { href: "/settings", label: "Settings" },
];

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
      <Box
        sx={{
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          bgcolor: "background.default",
        }}
      >
        <CircularProgress size={32} sx={{ color: "text.secondary" }} />
      </Box>
    );
  }

  const completedModels = models.filter((m) => m.status === "completed");
  const activeIndex = NAV.findIndex(
    (item) => pathname === item.href || (item.href !== "/dashboard" && pathname?.startsWith(item.href))
  );
  const currentTab = activeIndex >= 0 ? activeIndex : 0;

  return (
    <ErrorBoundary>
      <Box sx={{ minHeight: "100vh", bgcolor: "background.default" }}>
        <AppBar position="static" color="default" elevation={0}>
          <Toolbar
            variant="dense"
            sx={{
              minHeight: { xs: 56, sm: 64 },
              px: { xs: 2, sm: 3 },
              justifyContent: "space-between",
            }}
          >
            <Typography
              variant="h1"
              component="span"
              sx={{ fontSize: "1.25rem", fontWeight: 600 }}
            >
              whitematter
            </Typography>
            <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <Typography variant="body2" color="text.secondary" noWrap sx={{ maxWidth: 180 }}>
                {user.email}
              </Typography>
              <Button
                variant="outlined"
                size="small"
                onClick={logout}
                sx={{ textTransform: "none" }}
              >
                Log out
              </Button>
            </Box>
          </Toolbar>
          <Tabs
            value={currentTab}
            variant="scrollable"
            scrollButtons="auto"
            allowScrollButtonsMobile
            sx={{
              minHeight: 48,
              px: { xs: 1, sm: 2 },
              "& .MuiTab-root": { minHeight: 48, textTransform: "none" },
              "& .MuiTabs-indicator": { display: "none" },
              "& .Mui-selected": { color: "text.primary", fontWeight: 600 },
            }}
          >
            {NAV.map(({ href, label }) => {
              const isActive =
                pathname === href || (href !== "/dashboard" && pathname?.startsWith(href));
              return (
                <Tab
                  key={href}
                  label={
                    <Box component="span" sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                      {label}
                      {href === "/models" && completedModels.length > 0 && (
                        <Chip
                          label={completedModels.length}
                          size="small"
                          sx={{
                            height: 20,
                            fontSize: "0.75rem",
                            bgcolor: "rgba(255,255,255,0.12)",
                            color: "text.secondary",
                          }}
                        />
                      )}
                    </Box>
                  }
                  component={Link}
                  href={href}
                  sx={{
                    color: isActive ? "text.primary" : "text.secondary",
                  }}
                />
              );
            })}
          </Tabs>
        </AppBar>

        <Box
          component="main"
          sx={{
            maxWidth: 1100,
            margin: "0 auto",
            p: { xs: 2, sm: 3 },
          }}
        >
          <Box
            sx={{
              bgcolor: "background.paper",
              borderRadius: 2,
              border: "1px solid",
              borderColor: "divider",
              p: { xs: 2, sm: 3 },
            }}
          >
            {children}
          </Box>
        </Box>
      </Box>
    </ErrorBoundary>
  );
}
