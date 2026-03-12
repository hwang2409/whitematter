"use client";
import { useAuth } from "@/context/AuthContext";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { useEffect } from "react";
import ErrorBoundary from "@/components/ErrorBoundary";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Tooltip from "@mui/material/Tooltip";
import CircularProgress from "@mui/material/CircularProgress";
import ChatOutlined from "@mui/icons-material/ChatOutlined";
import CategoryOutlined from "@mui/icons-material/CategoryOutlined";
import SettingsOutlined from "@mui/icons-material/SettingsOutlined";

const NAV: { href: string; label: string; icon: React.ReactNode }[] = [
  { href: "/chat", label: "Chat", icon: <ChatOutlined fontSize="small" /> },
  { href: "/models", label: "Models", icon: <CategoryOutlined fontSize="small" /> },
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

  useEffect(() => {
    if (loading) return;
    if (!user) {
      router.replace("/login");
      return;
    }
    // Redirect old routes to /chat
    if (
      pathname === "/dashboard" ||
      pathname === "/data" ||
      pathname === "/train" ||
      pathname === "/predict" ||
      pathname === "/architect"
    ) {
      router.replace("/chat");
    }
  }, [user, loading, router, pathname]);

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

  return (
    <ErrorBoundary>
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
              (href !== "/chat" && pathname?.startsWith(href)) ||
              (href === "/chat" && pathname?.startsWith("/chat"));
            return (
              <Tooltip key={href} title={label} placement="right" arrow>
                <Box
                  component={Link}
                  href={href}
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
                  {icon}
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
            sx={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              minHeight: 0,
            }}
          >
            {children}
          </Box>
        </Box>
      </Box>
    </ErrorBoundary>
  );
}
