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
import Divider from "@mui/material/Divider";
import AddOutlined from "@mui/icons-material/AddOutlined";
import ViewSidebarOutlined from "@mui/icons-material/ViewSidebarOutlined";
import SearchOutlined from "@mui/icons-material/SearchOutlined";
import MoreHoriz from "@mui/icons-material/MoreHoriz";
import StarOutlined from "@mui/icons-material/StarOutlined";
import StarBorderOutlined from "@mui/icons-material/StarBorderOutlined";
import EditOutlined from "@mui/icons-material/EditOutlined";
import DeleteOutlined from "@mui/icons-material/DeleteOutlined";
import LoginOutlined from "@mui/icons-material/LoginOutlined";
import CategoryOutlined from "@mui/icons-material/CategoryOutlined";
import SettingsOutlined from "@mui/icons-material/SettingsOutlined";
import MilitaryTechOutlined from "@mui/icons-material/MilitaryTechOutlined";
import LogoutOutlined from "@mui/icons-material/LogoutOutlined";
import { getConversations, updateConversation, deleteConversation } from "@/api";
import type { Conversation } from "@/api";

const SIDEBAR_EXPANDED = 256;
const SIDEBAR_COLLAPSED = 56;

function formatRelativeDate(dateStr: string): string {
  // Backend sends ISO without Z — treat as UTC if no timezone specified
  const normalized = dateStr.endsWith("Z") || dateStr.includes("+") ? dateStr : dateStr + "Z";
  const date = new Date(normalized);
  if (isNaN(date.getTime())) return "Recent";
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays}d ago`;
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

  const [searchQuery, setSearchQuery] = useState("");
  const [menuAnchorEl, setMenuAnchorEl] = useState<HTMLElement | null>(null);
  const [menuConvId, setMenuConvId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [profileAnchorEl, setProfileAnchorEl] = useState<HTMLElement | null>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);

  const sidebarWidth = collapsed ? SIDEBAR_COLLAPSED : SIDEBAR_EXPANDED;

  useEffect(() => {
    if (loading) return;
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

  useEffect(() => {
    if (!user) return;
    getConversations()
      .then((convs) => setConversations(convs ?? []))
      .catch(() => {});
  }, [user, pathname]);

  useEffect(() => {
    if (renamingId && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [renamingId]);

  useEffect(() => {
    if (!loading && !user) {
      router.replace("/login");
    }
  }, [user, loading, router]);

  if (loading || !user) {
    return (
      <Box
        sx={{
          minHeight: "100dvh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          bgcolor: "#09090B",
        }}
      >
        <CircularProgress size={24} sx={{ color: "#F97316" }} />
      </Box>
    );
  }

  const filtered = searchQuery
    ? conversations.filter((c) =>
        (c.title || "New conversation").toLowerCase().includes(searchQuery.toLowerCase())
      )
    : conversations;

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
    setConversations((prev) =>
      prev.map((c) => (c.id === menuConvId ? { ...c, isStarred: newStarred } : c))
    );
    handleMenuClose();
    try {
      await updateConversation(menuConvId, { isStarred: newStarred });
    } catch {
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
      setRenamingId(null);
      return;
    }
    const original = conversations.find((c) => c.id === convId);
    setConversations((prev) =>
      prev.map((c) => (c.id === convId ? { ...c, title: trimmed } : c))
    );
    setRenamingId(null);
    try {
      await updateConversation(convId, { title: trimmed });
    } catch {
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
    if (pathname === `/chat/${id}`) {
      router.push("/chat");
    }
    try {
      await deleteConversation(id);
    } catch {
      getConversations()
        .then((convs) => setConversations(convs ?? []))
        .catch(() => {});
    }
  }

  return (
    <ErrorBoundary>
      <Box sx={{ display: "flex", height: "100dvh", overflow: "hidden", bgcolor: "#09090B" }}>
        {/* ── Sidebar ─────────────────────────────────────────────── */}
        <Box
          component="nav"
          sx={{
            width: sidebarWidth,
            flexShrink: 0,
            borderRight: "1px solid",
            borderColor: "#1A1A1E",
            display: "flex",
            flexDirection: "column",
            bgcolor: "#0C0C0E",
            overflow: "hidden",
            transition: "width 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
          }}
        >
          {/* Logo + toggle */}
          <Box
            sx={{
              px: collapsed ? 0 : 1.5,
              pt: 1.5,
              pb: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: collapsed ? "center" : "space-between",
            }}
          >
            {!collapsed && (
              <Typography
                sx={{
                  fontFamily: "'Instrument Serif', Georgia, serif",
                  fontStyle: "italic",
                  fontSize: "1.125rem",
                  color: "#FAFAFA",
                  pl: 0.75,
                  letterSpacing: "-0.01em",
                }}
              >
                whitematter
              </Typography>
            )}
            <Tooltip title={collapsed ? "Expand" : "Collapse"} placement="right" arrow>
              <IconButton
                onClick={toggleSidebar}
                size="small"
                sx={{
                  width: 28,
                  height: 28,
                  borderRadius: "6px",
                  color: "#52525B",
                  "&:hover": { color: "#A1A1AA", bgcolor: "rgba(255,255,255,0.04)" },
                }}
              >
                <ViewSidebarOutlined sx={{ fontSize: 16, transform: "scaleX(-1)" }} />
              </IconButton>
            </Tooltip>
          </Box>

          {/* New chat */}
          <Box
            sx={{
              px: collapsed ? 0 : 1.5,
              mb: 0.5,
              display: "flex",
              justifyContent: "center",
            }}
          >
            {collapsed ? (
              <Tooltip title="New chat" placement="right" arrow>
                <IconButton
                  component={Link}
                  href="/chat"
                  sx={{
                    width: 34,
                    height: 34,
                    borderRadius: "8px",
                    border: "1px solid #27272A",
                    color: "#A1A1AA",
                    "&:hover": {
                      borderColor: "#3F3F46",
                      bgcolor: "rgba(255,255,255,0.03)",
                      color: "#FAFAFA",
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
                startIcon={<AddOutlined sx={{ fontSize: "15px !important" }} />}
                fullWidth
                sx={{
                  justifyContent: "flex-start",
                  px: 1.25,
                  py: 0.875,
                  borderRadius: "8px",
                  border: "1px solid #27272A",
                  color: "#A1A1AA",
                  fontSize: "0.8125rem",
                  fontWeight: 400,
                  textTransform: "none",
                  "&:hover": {
                    borderColor: "#3F3F46",
                    bgcolor: "rgba(255,255,255,0.03)",
                    color: "#FAFAFA",
                  },
                }}
              >
                New chat
              </Button>
            )}
          </Box>

          {/* Models nav */}
          <Box
            sx={{
              px: collapsed ? 0 : 1.5,
              mb: 0.75,
              display: "flex",
              justifyContent: "center",
            }}
          >
            {collapsed ? (
              <Tooltip title="Models" placement="right" arrow>
                <IconButton
                  component={Link}
                  href="/models"
                  sx={{
                    width: 34,
                    height: 34,
                    borderRadius: "8px",
                    color: pathname === "/models" ? "#FAFAFA" : "#52525B",
                    bgcolor: pathname === "/models" ? "rgba(255,255,255,0.06)" : "transparent",
                    "&:hover": { bgcolor: "rgba(255,255,255,0.04)", color: "#A1A1AA" },
                  }}
                >
                  <CategoryOutlined sx={{ fontSize: 17 }} />
                </IconButton>
              </Tooltip>
            ) : (
              <Button
                component={Link}
                href="/models"
                fullWidth
                sx={{
                  justifyContent: "flex-start",
                  px: 1.25,
                  py: 0.75,
                  borderRadius: "8px",
                  color: pathname === "/models" ? "#FAFAFA" : "#71717A",
                  bgcolor: pathname === "/models" ? "rgba(255,255,255,0.06)" : "transparent",
                  fontSize: "0.8125rem",
                  fontWeight: pathname === "/models" ? 500 : 400,
                  textTransform: "none",
                  "&:hover": { bgcolor: "rgba(255,255,255,0.04)", color: "#FAFAFA" },
                }}
              >
                Models
              </Button>
            )}
          </Box>

          {/* Settings nav */}
          <Box
            sx={{
              px: collapsed ? 0 : 1.5,
              mb: 0.75,
              display: "flex",
              justifyContent: "center",
            }}
          >
            {collapsed ? (
              <Tooltip title="Settings" placement="right" arrow>
                <IconButton
                  component={Link}
                  href="/settings"
                  sx={{
                    width: 34,
                    height: 34,
                    borderRadius: "8px",
                    color: pathname === "/settings" ? "#FAFAFA" : "#52525B",
                    bgcolor: pathname === "/settings" ? "rgba(255,255,255,0.06)" : "transparent",
                    "&:hover": { bgcolor: "rgba(255,255,255,0.04)", color: "#A1A1AA" },
                  }}
                >
                  <SettingsOutlined sx={{ fontSize: 17 }} />
                </IconButton>
              </Tooltip>
            ) : (
              <Button
                component={Link}
                href="/settings"
                fullWidth
                sx={{
                  justifyContent: "flex-start",
                  px: 1.25,
                  py: 0.75,
                  borderRadius: "8px",
                  color: pathname === "/settings" ? "#FAFAFA" : "#71717A",
                  bgcolor: pathname === "/settings" ? "rgba(255,255,255,0.06)" : "transparent",
                  fontSize: "0.8125rem",
                  fontWeight: pathname === "/settings" ? 500 : 400,
                  textTransform: "none",
                  "&:hover": { bgcolor: "rgba(255,255,255,0.04)", color: "#FAFAFA" },
                }}
              >
                Settings
              </Button>
            )}
          </Box>

          {/* Search */}
          {!collapsed && (
            <Box sx={{ px: 1.5, mb: 1 }}>
              <TextField
                size="small"
                placeholder="Search..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                fullWidth
                slotProps={{
                  input: {
                    startAdornment: (
                      <InputAdornment position="start">
                        <SearchOutlined sx={{ fontSize: 14, color: "#52525B" }} />
                      </InputAdornment>
                    ),
                    sx: {
                      fontSize: "0.75rem",
                      borderRadius: "8px",
                      bgcolor: "transparent",
                      "& fieldset": { borderColor: "#1F1F23" },
                      "&:hover fieldset": { borderColor: "#27272A !important" },
                    },
                  },
                }}
                sx={{ "& .MuiOutlinedInput-root": { height: 32 } }}
              />
            </Box>
          )}

          {/* Conversation list */}
          {!collapsed && (
            <Box sx={{ flex: 1, overflowY: "auto", px: 1 }}>
              {grouped.map(({ label, items }) => (
                <Box key={label} sx={{ mb: 1 }}>
                  <Typography
                    sx={{
                      fontSize: "0.625rem",
                      fontWeight: 600,
                      letterSpacing: "0.08em",
                      textTransform: "uppercase",
                      color: "#3F3F46",
                      px: 0.75,
                      mb: 0.25,
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
                          borderRadius: "6px",
                          bgcolor: isActive ? "rgba(255,255,255,0.06)" : "transparent",
                          mb: "1px",
                          transition: "background-color 0.1s",
                          "&:hover": {
                            bgcolor: isActive
                              ? "rgba(255,255,255,0.06)"
                              : "rgba(255,255,255,0.03)",
                            "& .conv-menu-btn": { opacity: 1 },
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
                                height: 28,
                                fontSize: "0.8125rem",
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
                                px: 1,
                                py: 0.625,
                                fontSize: "0.8125rem",
                                color: isActive ? "#FAFAFA" : "#71717A",
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
                                <StarOutlined
                                  sx={{ fontSize: 12, color: "#EAB308", flexShrink: 0 }}
                                />
                              )}
                              <Box
                                component="span"
                                sx={{ overflow: "hidden", textOverflow: "ellipsis" }}
                              >
                                {conv.title || "New conversation"}
                              </Box>
                            </Box>
                            <IconButton
                              className="conv-menu-btn"
                              size="small"
                              onClick={(e) => handleMenuOpen(e, conv.id)}
                              sx={{
                                opacity: 0,
                                width: 22,
                                height: 22,
                                mr: 0.25,
                                color: "#52525B",
                                borderRadius: "4px",
                                "&:hover": { color: "#A1A1AA", bgcolor: "rgba(255,255,255,0.06)" },
                                flexShrink: 0,
                              }}
                            >
                              <MoreHoriz sx={{ fontSize: 14 }} />
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

          {/* ── Profile ─────────────────────── */}
          <Box
            sx={{
              borderTop: "1px solid #1F1F23",
              p: collapsed ? 1 : 1.5,
              flexShrink: 0,
            }}
          >
            {collapsed ? (
              <Tooltip title={user?.email ?? "Account"} placement="right" arrow>
                <IconButton
                  onClick={(e) => setProfileAnchorEl(e.currentTarget)}
                  sx={{
                    width: 34,
                    height: 34,
                    borderRadius: "8px",
                    bgcolor: "#18181B",
                    border: "1px solid #27272A",
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    fontFamily: "'JetBrains Mono', monospace",
                    color: "#A1A1AA",
                    display: "flex",
                    mx: "auto",
                    "&:hover": { borderColor: "#3F3F46", color: "#FAFAFA" },
                  }}
                >
                  {user?.email?.[0]?.toUpperCase() ?? "U"}
                </IconButton>
              </Tooltip>
            ) : (
              <Box
                onClick={(e) => setProfileAnchorEl(e.currentTarget as HTMLElement)}
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                  px: 0.5,
                  py: 0.5,
                  borderRadius: "8px",
                  cursor: "pointer",
                  transition: "background-color 0.15s",
                  "&:hover": { bgcolor: "rgba(255,255,255,0.04)" },
                }}
              >
                <Box
                  sx={{
                    width: 32,
                    height: 32,
                    borderRadius: "8px",
                    bgcolor: "#18181B",
                    border: "1px solid #27272A",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    fontFamily: "'JetBrains Mono', monospace",
                    color: "#A1A1AA",
                    flexShrink: 0,
                  }}
                >
                  {user?.email?.[0]?.toUpperCase() ?? "U"}
                </Box>
                <Typography
                  sx={{
                    flex: 1,
                    fontSize: "0.8125rem",
                    color: "#A1A1AA",
                    fontFamily: "'Outfit', sans-serif",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    minWidth: 0,
                  }}
                >
                  {user?.email}
                </Typography>
              </Box>
            )}
          </Box>
        </Box>

        {/* ── Main ────────────────────────────────────────────────── */}
        <Box sx={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, overflow: "hidden" }}>
          {/* Content */}
          <Box
            component="main"
            sx={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0, overflow: "hidden" }}
          >
            {children}
          </Box>
        </Box>
      </Box>

      {/* ── Context menu ──────────────────────────────────────────── */}
      <Menu
        anchorEl={menuAnchorEl}
        open={Boolean(menuAnchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleToggleStar} sx={{ fontSize: "0.8125rem" }}>
          <ListItemIcon>
            {conversations.find((c) => c.id === menuConvId)?.isStarred ? (
              <StarOutlined sx={{ fontSize: 16, color: "#EAB308" }} />
            ) : (
              <StarBorderOutlined sx={{ fontSize: 16 }} />
            )}
          </ListItemIcon>
          <ListItemText>
            {conversations.find((c) => c.id === menuConvId)?.isStarred ? "Unstar" : "Star"}
          </ListItemText>
        </MenuItem>
        <MenuItem onClick={handleStartRename} sx={{ fontSize: "0.8125rem" }}>
          <ListItemIcon>
            <EditOutlined sx={{ fontSize: 16 }} />
          </ListItemIcon>
          <ListItemText>Rename</ListItemText>
        </MenuItem>
        <MenuItem
          onClick={handleDeleteClick}
          sx={{ fontSize: "0.8125rem", color: "#EF4444" }}
        >
          <ListItemIcon>
            <DeleteOutlined sx={{ fontSize: 16, color: "#EF4444" }} />
          </ListItemIcon>
          <ListItemText>Delete</ListItemText>
        </MenuItem>
      </Menu>

      {/* ── Profile menu ──────────────────────────────────────────── */}
      <Menu
        anchorEl={profileAnchorEl}
        open={Boolean(profileAnchorEl)}
        onClose={() => setProfileAnchorEl(null)}
        anchorOrigin={{ vertical: "top", horizontal: "right" }}
        transformOrigin={{ vertical: "bottom", horizontal: "left" }}
      >
        <Box sx={{ px: 1.5, py: 1 }}>
          <Typography
            sx={{
              fontSize: "0.625rem",
              fontWeight: 600,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "#52525B",
            }}
          >
            Account
          </Typography>
          <Typography sx={{ fontSize: "0.8125rem", color: "#A1A1AA", mt: 0.25 }}>
            {user?.email}
          </Typography>
        </Box>
        <Divider sx={{ borderColor: "#27272A" }} />
        <MenuItem
          onClick={() => {
            setProfileAnchorEl(null);
            router.push("/settings");
          }}
        >
          <ListItemIcon>
            <MilitaryTechOutlined sx={{ fontSize: 16 }} />
          </ListItemIcon>
          <ListItemText>Upgrade</ListItemText>
        </MenuItem>
        <MenuItem
          onClick={() => {
            setProfileAnchorEl(null);
            router.push("/settings");
          }}
        >
          <ListItemIcon>
            <SettingsOutlined sx={{ fontSize: 16 }} />
          </ListItemIcon>
          <ListItemText>Settings</ListItemText>
        </MenuItem>
        <Divider sx={{ borderColor: "#27272A" }} />
        <MenuItem
          onClick={() => {
            setProfileAnchorEl(null);
            logout();
          }}
          sx={{ color: "#71717A" }}
        >
          <ListItemIcon>
            <LogoutOutlined sx={{ fontSize: 16, color: "#71717A" }} />
          </ListItemIcon>
          <ListItemText>Sign out</ListItemText>
        </MenuItem>
      </Menu>

      {/* ── Delete confirmation ───────────────────────────────────── */}
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
