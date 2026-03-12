"use client";
import { useAuth } from "@/context/AuthContext";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { useEffect, useState } from "react";
import { getModels } from "@/api";
import type { Model } from "@/api";
import ErrorBoundary from "@/components/ErrorBoundary";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Tooltip from "@mui/material/Tooltip";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import DashboardOutlined from "@mui/icons-material/DashboardOutlined";
import DatasetOutlined from "@mui/icons-material/DatasetOutlined";
import PlayArrowOutlined from "@mui/icons-material/PlayArrowOutlined";
import CategoryOutlined from "@mui/icons-material/CategoryOutlined";
import AutoAwesomeOutlined from "@mui/icons-material/AutoAwesomeOutlined";
import BatchPredictionOutlined from "@mui/icons-material/BatchPredictionOutlined";
import SettingsOutlined from "@mui/icons-material/SettingsOutlined";
import { themeTokens } from "@/theme";

const NAV: { href: string; label: string; icon: React.ReactNode }[] = [
  { href: "/dashboard", label: "Dashboard", icon: <DashboardOutlined fontSize="small" /> },
  { href: "/data", label: "Data", icon: <DatasetOutlined fontSize="small" /> },
  { href: "/architect", label: "Design", icon: <AutoAwesomeOutlined fontSize="small" /> },
  { href: "/train", label: "Train", icon: <PlayArrowOutlined fontSize="small" /> },
  { href: "/models", label: "Models", icon: <CategoryOutlined fontSize="small" /> },
  { href: "/predict", label: "Predict", icon: <BatchPredictionOutlined fontSize="small" /> },
  { href: "/settings", label: "Settings", icon: <SettingsOutlined fontSize="small" /> },
];

const SIDEBAR_WIDTH = 60;

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
        <CircularProgress size={32} sx={{ color: "primary.main" }} />
      </Box>
    );
  }

  const completedModels = models.filter((m) => m.status === "completed");

  return (
    <ErrorBoundary>
      <Box
        component="a"
        href="#main-content"
        sx={{
          position: "absolute",
          left: "-9999px",
          top: "auto",
          width: "1px",
          height: "1px",
          overflow: "hidden",
          "&:focus": {
            position: "fixed",
            top: 8,
            left: 8,
            width: "auto",
            height: "auto",
            overflow: "visible",
            zIndex: 9999,
            bgcolor: "primary.main",
            color: "primary.contrastText",
            px: 2,
            py: 1,
            borderRadius: 1,
            fontSize: "0.875rem",
            fontWeight: 600,
            textDecoration: "none",
            outline: "2px solid",
            outlineColor: "primary.main",
            outlineOffset: 2,
          },
        }}
      >
        Skip to main content
      </Box>
      <Box sx={{ display: "flex", minHeight: "100vh", bgcolor: "background.default" }}>
        {/* Vertical sidebar */}
        <Box
          component="nav"
          sx={{
            width: SIDEBAR_WIDTH,
            flexShrink: 0,
            borderRight: "1px solid",
            borderColor: "divider",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            py: 1.5,
            bgcolor: "background.paper",
          }}
        >
          <Typography
            component="span"
            sx={{
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: "0.75rem",
              fontWeight: 700,
              color: "primary.main",
              letterSpacing: "0.02em",
              mb: 2,
            }}
          >
            wm
          </Typography>
          {NAV.map(({ href, label, icon }) => {
            const isActive =
              pathname === href ||
              (href !== "/dashboard" && pathname?.startsWith(href));
            return (
              <Tooltip key={href} title={label} placement="right" arrow>
                <Box
                  component={Link}
                  href={href}
                  aria-label={label}
                  sx={{
                    width: 40,
                    height: 40,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    borderRadius: 1,
                    color: isActive ? "primary.main" : "text.secondary",
                    borderLeft: "3px solid",
                    borderLeftColor: isActive ? "primary.main" : "transparent",
                    ml: isActive ? "-3px" : 0,
                    "&:hover": {
                      color: "primary.main",
                      bgcolor: "action.hover",
                    },
                    textDecoration: "none",
                    transition: "color 0.2s ease-out, background-color 0.2s ease-out",
                  }}
                >
                  {href === "/models" && completedModels.length > 0 ? (
                    <Box sx={{ position: "relative", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      {icon}
                      <Chip
                        label={completedModels.length}
                        size="small"
                        sx={{
                          position: "absolute",
                          top: -4,
                          right: -8,
                          height: 16,
                          fontSize: "0.625rem",
                          bgcolor: themeTokens.accentLight,
                          color: "primary.main",
                          "& .MuiChip-label": { px: 0.5 },
                        }}
                      />
                    </Box>
                  ) : (
                    icon
                  )}
                </Box>
              </Tooltip>
            );
          })}
        </Box>

        <Box sx={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
          {/* Top bar: user + logout */}
          <Box
            sx={{
              height: 56,
              flexShrink: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "flex-end",
              px: 2,
              borderBottom: "1px solid",
              borderColor: "divider",
              bgcolor: "background.default",
            }}
          >
            <Typography variant="body2" color="text.secondary" noWrap sx={{ maxWidth: 180, mr: 2 }}>
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

          <Box
            component="main"
            id="main-content"
            sx={{
              flex: 1,
              maxWidth: 1100,
              margin: "0 auto",
              width: "100%",
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
      </Box>
    </ErrorBoundary>
  );
}
