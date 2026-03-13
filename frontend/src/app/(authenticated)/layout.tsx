"use client";
import { useAuth } from "@/context/AuthContext";
import { useSidebar } from "@/context/SidebarContext";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { useEffect, useState, useRef } from "react";
import ErrorBoundary from "@/components/ErrorBoundary";
import ConfirmDialog from "@/components/ConfirmDialog";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Tooltip from "@mui/material/Tooltip";
import IconButton from "@mui/material/IconButton";
import CircularProgress from "@mui/material/CircularProgress";
import TextField from "@mui/material/TextField";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import InputAdornment from "@mui/material/InputAdornment";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import ChatOutlined from "@mui/icons-material/ChatOutlined";
import CategoryOutlined from "@mui/icons-material/CategoryOutlined";
import SettingsOutlined from "@mui/icons-material/SettingsOutlined";
import AddOutlined from "@mui/icons-material/AddOutlined";
import ChevronLeft from "@mui/icons-material/ChevronLeft";
import ChevronRight from "@mui/icons-material/ChevronRight";
import SearchOutlined from "@mui/icons-material/SearchOutlined";
import MoreHoriz from "@mui/icons-material/MoreHoriz";
import StarOutlined from "@mui/icons-material/StarOutlined";
import StarBorderOutlined from "@mui/icons-material/StarBorderOutlined";
import EditOutlined from "@mui/icons-material/EditOutlined";
import DeleteOutlined from "@mui/icons-material/DeleteOutlined";
import { getConversations, updateConversation, deleteConversation } from "@/api";
import type { Conversation } from "@/api";

const NAV: { href: string; label: string; icon: React.ReactNode }[] = [
  { href: "/chat", label: "Chat", icon: <ChatOutlined fontSize="small" /> },
  { href: "/models", label: "Models", icon: <CategoryOutlined fontSize="small" /> },
  { href: "/settings", label: "Settings", icon: <SettingsOutlined fontSize="small" /> },
];

const SIDEBAR_WIDTH_EXPANDED = 260;
const SIDEBAR_WIDTH_COLLAPSED = 60;

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
  const { collapsed, toggleSidebar } = useSidebar();
  const router = useRouter();
  const pathname = usePathname();
  const [conversations, setConversations] = useState<Conversation[]>([]);

  // Chat history management state
  const [searchQuery, setSearchQuery] = useState("");
  const [menuAnchorEl, setMenuAnchorEl] = useState<HTMLElement | null>(null);
  const [menuConvId, setMenuConvId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);

  const sidebarWidth = collapsed ? SIDEBAR_WIDTH_COLLAPSED : SIDEBAR_WIDTH_EXPANDED;

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

  // Focus rename input when it appears
  useEffect(() => {
    if (renamingId && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [renamingId]);

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

  // Filter conversations by search query
  const filtered = searchQuery
    ? conversations.filter((c) =>
        (c.title || "New conversation").toLowerCase().includes(searchQuery.toLowerCase())
      )
    : conversations;

  // Group conversations by date, starred first within each group
  const grouped: { label: string; items: Conversation[] }[] = [];
  const buckets: Record<string, Conversation[]> = {};
  for (const c of filtered) {
    const label = formatRelativeDate(c.updatedAt || c.createdAt);
    if (!buckets[label]) buckets[label] = [];
    buckets[label].push(c);
  }
  for (const [label, items] of Object.entries(buckets)) {
    items.sort((a, b) => {
      if (a.isStarred && !b.isStarred) return -1;
      if (!a.isStarred && b.isStarred) return 1;
      return 0;
    });
    grouped.push({ label, items });
  }

  const userInitial = user.email?.[0]?.toUpperCase() ?? "U";

  // Handlers
  function handleMenuOpen(e: React.MouseEvent<HTMLElement>, convId: string) {
    e.preventDefault();
    e.stopPropagation();
    setMenuAnchorEl(e.currentTarget);
    setMenuConvId(convId);
  }

  function handleMenuClose() {
    setMenuAnchorEl(null);
    setMenuConvId(null);
  }

  async function handleToggleStar() {
    if (!menuConvId) return;
    const conv = conversations.find((c) => c.id === menuConvId);
    if (!conv) return;
    const newStarred = !conv.isStarred;
    // Optimistic update
    setConversations((prev) =>
      prev.map((c) => (c.id === menuConvId ? { ...c, isStarred: newStarred } : c))
    );
    handleMenuClose();
    try {
      await updateConversation(menuConvId, { isStarred: newStarred });
    } catch {
      // Revert on error
      setConversations((prev) =>
        prev.map((c) => (c.id === conv.id ? { ...c, isStarred: conv.isStarred } : c))
      );
    }
  }

  function handleStartRename() {
    if (!menuConvId) return;
    const conv = conversations.find((c) => c.id === menuConvId);
    setRenameValue(conv?.title || "");
    setRenamingId(menuConvId);
    handleMenuClose();
  }

  async function handleRenameSubmit(convId: string) {
    const trimmed = renameValue.trim();
    if (!trimmed) {
      // Don't save empty — revert
      setRenamingId(null);
      return;
    }
    const original = conversations.find((c) => c.id === convId);
    // Optimistic update
    setConversations((prev) =>
      prev.map((c) => (c.id === convId ? { ...c, title: trimmed } : c))
    );
    setRenamingId(null);
    try {
      await updateConversation(convId, { title: trimmed });
    } catch {
      // Revert
      if (original) {
        setConversations((prev) =>
          prev.map((c) => (c.id === convId ? { ...c, title: original.title } : c))
        );
      }
    }
  }

  function handleDeleteClick() {
    if (!menuConvId) return;
    setDeleteConfirmId(menuConvId);
    handleMenuClose();
  }

  async function handleDeleteConfirm() {
    if (!deleteConfirmId) return;
    const id = deleteConfirmId;
    setConversations((prev) => prev.filter((c) => c.id !== id));
    setDeleteConfirmId(null);
    // Redirect if deleting the active conversation
    if (pathname === `/chat/${id}`) {
      router.push("/chat");
    }
    try {
      await deleteConversation(id);
    } catch {
      // Re-fetch on error
      getConversations()
        .then((convs) => setConversations(convs ?? []))
        .catch(() => {});
    }
  }

  return (
    <ErrorBoundary>
      <Box sx={{ display: "flex", minHeight: "100vh", bgcolor: "background.default" }}>
        {/* Sidebar */}
        <Box
          component="nav"
          sx={{
            width: sidebarWidth,
            flexShrink: 0,
            borderRight: "1px solid",
            borderColor: "divider",
            display: "flex",
            flexDirection: "column",
            bgcolor: "background.paper",
            overflow: "hidden",
            transition: "width 0.2s ease",
          }}
        >
          {/* Logo */}
          <Box sx={{ px: collapsed ? 0 : 2.5, pt: 2, pb: 1.5, display: "flex", justifyContent: collapsed ? "center" : "flex-start" }}>
            {!collapsed && (
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
            )}
          </Box>

          {/* New chat button */}
          <Box sx={{ px: collapsed ? 0 : 1.5, mb: collapsed ? 2 : 1, display: "flex", justifyContent: "center" }}>
            {collapsed ? (
              <Tooltip title="New chat" placement="right" arrow>
                <IconButton
                  component={Link}
                  href="/chat"
                  sx={{
                    width: 36,
                    height: 36,
                    borderRadius: "10px",
                    border: "1px solid",
                    borderColor: "divider",
                    color: "text.secondary",
                    "&:hover": {
                      borderColor: "action.hover",
                      bgcolor: "background.default",
                      color: "text.primary",
                    },
                  }}
                >
                  <AddOutlined sx={{ fontSize: 16 }} />
                </IconButton>
              </Tooltip>
            ) : (
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
            )}
          </Box>

          {/* Search bar */}
          {!collapsed && isOnChat && (
            <Box sx={{ px: 1.5, mb: 1 }}>
              <TextField
                size="small"
                placeholder="Search chats..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                fullWidth
                slotProps={{
                  input: {
                    startAdornment: (
                      <InputAdornment position="start">
                        <SearchOutlined sx={{ fontSize: 16, color: "text.disabled" }} />
                      </InputAdornment>
                    ),
                    sx: {
                      fontSize: "0.8125rem",
                      borderRadius: "8px",
                      "& fieldset": { borderColor: "divider" },
                    },
                  },
                }}
                sx={{ "& .MuiOutlinedInput-root": { height: 34 } }}
              />
            </Box>
          )}

          {/* Nav icons */}
          <Box
            sx={{
              display: "flex",
              flexDirection: collapsed ? "column" : "row",
              alignItems: "center",
              gap: 0.5,
              px: collapsed ? 0 : 1.5,
              mb: 2,
              justifyContent: collapsed ? "center" : "flex-start",
            }}
          >
            {NAV.map(({ href, label, icon }) => {
              const isActive =
                pathname === href ||
                (href !== "/chat" && pathname?.startsWith(href)) ||
                (href === "/chat" && pathname?.startsWith("/chat"));
              return (
                <Tooltip key={href} title={label} placement={collapsed ? "right" : "bottom"} arrow>
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
          {!collapsed && isOnChat && (
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
                    const isRenaming = renamingId === conv.id;

                    return (
                      <Box
                        key={conv.id}
                        sx={{
                          display: "flex",
                          alignItems: "center",
                          borderRadius: "8px",
                          bgcolor: isActive ? "background.default" : "transparent",
                          mb: "2px",
                          transition: "all 0.12s",
                          "&:hover": {
                            bgcolor: "background.default",
                            "& .chat-menu-btn": { opacity: 1 },
                          },
                        }}
                      >
                        {isRenaming ? (
                          <TextField
                            inputRef={renameInputRef}
                            size="small"
                            value={renameValue}
                            onChange={(e) => setRenameValue(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") handleRenameSubmit(conv.id);
                              if (e.key === "Escape") setRenamingId(null);
                            }}
                            onBlur={() => handleRenameSubmit(conv.id)}
                            fullWidth
                            sx={{
                              mx: 0.5,
                              "& .MuiOutlinedInput-root": {
                                height: 30,
                                fontSize: "0.875rem",
                              },
                            }}
                          />
                        ) : (
                          <>
                            <Box
                              component={Link}
                              href={convPath}
                              sx={{
                                flex: 1,
                                display: "flex",
                                alignItems: "center",
                                gap: 0.5,
                                px: 1.5,
                                py: 0.75,
                                fontSize: "0.875rem",
                                color: isActive ? "text.primary" : "text.secondary",
                                fontWeight: isActive ? 500 : 400,
                                cursor: "pointer",
                                whiteSpace: "nowrap",
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                textDecoration: "none",
                                minWidth: 0,
                              }}
                            >
                              {conv.isStarred && (
                                <StarOutlined sx={{ fontSize: 14, color: "warning.main", flexShrink: 0 }} />
                              )}
                              <Box component="span" sx={{ overflow: "hidden", textOverflow: "ellipsis" }}>
                                {conv.title || "New conversation"}
                              </Box>
                            </Box>
                            <IconButton
                              className="chat-menu-btn"
                              size="small"
                              onClick={(e) => handleMenuOpen(e, conv.id)}
                              sx={{
                                opacity: 0,
                                width: 24,
                                height: 24,
                                mr: 0.5,
                                color: "text.disabled",
                                "&:hover": { color: "text.secondary" },
                                flexShrink: 0,
                              }}
                            >
                              <MoreHoriz sx={{ fontSize: 16 }} />
                            </IconButton>
                          </>
                        )}
                      </Box>
                    );
                  })}
                </Box>
              ))}
            </Box>
          )}

          <Box sx={{ flex: (!collapsed && isOnChat) ? 0 : 1 }} />

          {/* Collapse toggle */}
          <Box sx={{ display: "flex", justifyContent: collapsed ? "center" : "flex-end", px: collapsed ? 0 : 1.5, pb: 0.5 }}>
            <Tooltip title={collapsed ? "Expand sidebar" : "Collapse sidebar"} placement="right" arrow>
              <IconButton
                onClick={toggleSidebar}
                size="small"
                sx={{
                  width: 28,
                  height: 28,
                  color: "text.disabled",
                  "&:hover": {
                    color: "text.secondary",
                    bgcolor: "action.hover",
                  },
                }}
              >
                {collapsed ? <ChevronRight fontSize="small" /> : <ChevronLeft fontSize="small" />}
              </IconButton>
            </Tooltip>
          </Box>

          {/* User area */}
          <Box
            sx={{
              px: 1.5,
              py: 1.5,
              borderTop: "1px solid",
              borderColor: "divider",
              display: "flex",
              alignItems: "center",
              justifyContent: collapsed ? "center" : "flex-start",
              gap: collapsed ? 0 : 1,
            }}
          >
            <Tooltip title={collapsed ? user.email : ""} placement="right" arrow>
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
            </Tooltip>
            {!collapsed && (
              <>
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
              </>
            )}
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

      {/* Three-dot context menu */}
      <Menu
        anchorEl={menuAnchorEl}
        open={Boolean(menuAnchorEl)}
        onClose={handleMenuClose}
        slotProps={{
          paper: {
            sx: {
              bgcolor: "background.paper",
              border: "1px solid",
              borderColor: "divider",
              borderRadius: "8px",
              minWidth: 160,
              boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
            },
          },
        }}
      >
        <MenuItem onClick={handleToggleStar} sx={{ fontSize: "0.875rem" }}>
          <ListItemIcon>
            {conversations.find((c) => c.id === menuConvId)?.isStarred ? (
              <StarOutlined sx={{ fontSize: 18, color: "warning.main" }} />
            ) : (
              <StarBorderOutlined sx={{ fontSize: 18 }} />
            )}
          </ListItemIcon>
          <ListItemText>
            {conversations.find((c) => c.id === menuConvId)?.isStarred ? "Unstar" : "Star"}
          </ListItemText>
        </MenuItem>
        <MenuItem onClick={handleStartRename} sx={{ fontSize: "0.875rem" }}>
          <ListItemIcon>
            <EditOutlined sx={{ fontSize: 18 }} />
          </ListItemIcon>
          <ListItemText>Rename</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleDeleteClick} sx={{ fontSize: "0.875rem", color: "error.main" }}>
          <ListItemIcon>
            <DeleteOutlined sx={{ fontSize: 18, color: "error.main" }} />
          </ListItemIcon>
          <ListItemText>Delete</ListItemText>
        </MenuItem>
      </Menu>

      {/* Delete confirmation dialog */}
      <ConfirmDialog
        isOpen={deleteConfirmId !== null}
        title="Delete Conversation"
        message="Are you sure you want to delete this conversation? This action cannot be undone."
        confirmLabel="Delete"
        cancelLabel="Cancel"
        variant="danger"
        onConfirm={handleDeleteConfirm}
        onCancel={() => setDeleteConfirmId(null)}
      />
    </ErrorBoundary>
  );
}
