"use client";
import { useAuth } from "@/context/AuthContext";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { useEffect, useState } from "react";
import ErrorBoundary from "@/components/ErrorBoundary";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Tooltip from "@mui/material/Tooltip";
import CircularProgress from "@mui/material/CircularProgress";
import ChatOutlined from "@mui/icons-material/ChatOutlined";
import CategoryOutlined from "@mui/icons-material/CategoryOutlined";
import SettingsOutlined from "@mui/icons-material/SettingsOutlined";
import AddOutlined from "@mui/icons-material/AddOutlined";
import { getConversations } from "@/api";
import type { Conversation } from "@/api";

const NAV: { href: string; label: string; icon: React.ReactNode }[] = [
  { href: "/chat", label: "Chat", icon: <ChatOutlined fontSize="small" /> },
  { href: "/models", label: "Models", icon: <CategoryOutlined fontSize="small" /> },
  { href: "/settings", label: "Settings", icon: <SettingsOutlined fontSize="small" /> },
];

const SIDEBAR_WIDTH = 260;

function formatRelativeDate(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export default function AuthenticatedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { user, loading, logout } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const [conversations, setConversations] = useState<Conversation[]>([]);

  useEffect(() => {
    if (loading) return;
    if (!user) {
      router.replace("/login");
      return;
    }
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

  // Fetch conversation list
  useEffect(() => {
    if (!user) return;
    getConversations()
      .then((convs) => setConversations(convs ?? []))
      .catch(() => {});
  }, [user, pathname]);

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

  const isOnChat = pathname?.startsWith("/chat");

  // Group conversations by date
  const grouped: { label: string; items: Conversation[] }[] = [];
  const buckets: Record<string, Conversation[]> = {};
  for (const c of conversations) {
    const label = formatRelativeDate(c.updatedAt || c.createdAt);
    if (!buckets[label]) buckets[label] = [];
    buckets[label].push(c);
  }
  for (const [label, items] of Object.entries(buckets)) {
    grouped.push({ label, items });
  }

  const userInitial = user.email?.[0]?.toUpperCase() ?? "U";

  return (
    <ErrorBoundary>
      <Box sx={{ display: "flex", minHeight: "100vh", bgcolor: "background.default" }}>
        {/* Sidebar */}
        <Box
          component="nav"
          sx={{
            width: SIDEBAR_WIDTH,
            flexShrink: 0,
            borderRight: "1px solid",
            borderColor: "divider",
            display: "flex",
            flexDirection: "column",
            bgcolor: "background.paper",
            overflow: "hidden",
          }}
        >
          {/* Logo */}
          <Box sx={{ px: 2.5, pt: 2, pb: 1.5 }}>
            <Typography
              sx={{
                fontFamily: "'DM Sans', sans-serif",
                fontSize: "1rem",
                fontWeight: 600,
                letterSpacing: "-0.02em",
                color: "text.primary",
              }}
            >
              whitematter
            </Typography>
          </Box>

          {/* New chat button */}
          <Box sx={{ px: 1.5, mb: 2 }}>
            <Button
              component={Link}
              href="/chat"
              startIcon={<AddOutlined sx={{ fontSize: "16px !important" }} />}
              fullWidth
              sx={{
                justifyContent: "flex-start",
                px: 1.5,
                py: 1,
                borderRadius: "10px",
                border: "1px solid",
                borderColor: "divider",
                color: "text.secondary",
                fontSize: "0.875rem",
                fontWeight: 400,
                textTransform: "none",
                "&:hover": {
                  borderColor: "action.hover",
                  bgcolor: "background.default",
                  color: "text.primary",
                },
              }}
            >
              New chat
            </Button>
          </Box>

          {/* Nav icons row */}
          <Box sx={{ display: "flex", gap: 0.5, px: 1.5, mb: 2 }}>
            {NAV.map(({ href, label, icon }) => {
              const isActive =
                pathname === href ||
                (href !== "/chat" && pathname?.startsWith(href)) ||
                (href === "/chat" && pathname?.startsWith("/chat"));
              return (
                <Tooltip key={href} title={label} placement="bottom" arrow>
                  <Box
                    component={Link}
                    href={href}
                    sx={{
                      width: 36,
                      height: 36,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      borderRadius: "8px",
                      color: isActive ? "primary.main" : "text.secondary",
                      bgcolor: isActive ? "rgba(108,92,231,0.08)" : "transparent",
                      "&:hover": {
                        color: "primary.main",
                        bgcolor: "rgba(108,92,231,0.06)",
                      },
                      textDecoration: "none",
                      transition: "all 0.15s ease-out",
                    }}
                  >
                    {icon}
                  </Box>
                </Tooltip>
              );
            })}
          </Box>

          {/* Conversation list */}
          {isOnChat && (
            <Box sx={{ flex: 1, overflowY: "auto", px: 1.5 }}>
              {grouped.map(({ label, items }) => (
                <Box key={label} sx={{ mb: 1.5 }}>
                  <Typography
                    sx={{
                      fontSize: "0.6875rem",
                      fontWeight: 600,
                      letterSpacing: "0.06em",
                      textTransform: "uppercase",
                      color: "text.disabled",
                      px: 1,
                      mb: 0.5,
                    }}
                  >
                    {label}
                  </Typography>
                  {items.map((conv) => {
                    const convPath = `/chat/${conv.id}`;
                    const isActive = pathname === convPath;
                    return (
                      <Box
                        key={conv.id}
                        component={Link}
                        href={convPath}
                        sx={{
                          display: "block",
                          px: 1.5,
                          py: 0.75,
                          borderRadius: "8px",
                          fontSize: "0.875rem",
                          color: isActive ? "text.primary" : "text.secondary",
                          bgcolor: isActive ? "background.default" : "transparent",
                          fontWeight: isActive ? 500 : 400,
                          cursor: "pointer",
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          textDecoration: "none",
                          mb: "2px",
                          transition: "all 0.12s",
                          "&:hover": {
                            bgcolor: "background.default",
                            color: "text.primary",
                          },
                        }}
                      >
                        {conv.title || "New conversation"}
                      </Box>
                    );
                  })}
                </Box>
              ))}
            </Box>
          )}

          <Box sx={{ flex: isOnChat ? 0 : 1 }} />

          {/* User area */}
          <Box
            sx={{
              px: 1.5,
              py: 1.5,
              borderTop: "1px solid",
              borderColor: "divider",
              display: "flex",
              alignItems: "center",
              gap: 1,
            }}
          >
            <Box
              sx={{
                width: 32,
                height: 32,
                borderRadius: "50%",
                bgcolor: "background.default",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "0.8125rem",
                fontWeight: 600,
                color: "text.secondary",
                flexShrink: 0,
              }}
            >
              {userInitial}
            </Box>
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography
                sx={{
                  fontSize: "0.8125rem",
                  color: "text.secondary",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {user.email}
              </Typography>
              <Typography
                sx={{
                  fontSize: "0.6875rem",
                  color: "primary.main",
                  fontWeight: 500,
                }}
              >
                {user.plan ? `${user.plan.charAt(0).toUpperCase()}${user.plan.slice(1)} plan` : "Free plan"}
              </Typography>
            </Box>
            <Button
              size="small"
              onClick={logout}
              sx={{
                minWidth: 0,
                px: 1,
                py: 0.5,
                fontSize: "0.75rem",
                color: "text.secondary",
                "&:hover": { color: "text.primary" },
              }}
            >
              Log out
            </Button>
          </Box>
        </Box>

        {/* Main content */}
        <Box sx={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
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
