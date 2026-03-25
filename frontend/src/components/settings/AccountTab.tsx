"use client";

import { useState } from "react";
import { useAuth } from "@/context/AuthContext";
import { useThemeMode } from "@/context/ThemeContext";
import { useToast } from "@/components/Toast";
import { changePassword, exportUserData } from "@/api";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import Alert from "@mui/material/Alert";
import CircularProgress from "@mui/material/CircularProgress";
import Toast from "@/components/Toast";

export default function AccountTab() {
  const { user, token, logout } = useAuth();
  const { mode, setMode } = useThemeMode();
  const { toasts, dismissToast, success, error: showError } = useToast();

  const isOAuth = user?.oauth_provider === "google";

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [passwordError, setPasswordError] = useState("");
  const [changingPassword, setChangingPassword] = useState(false);
  const [exporting, setExporting] = useState(false);

  const validatePassword = (): string | null => {
    if (!currentPassword) return "Current password is required";
    if (newPassword.length < 8) return "New password must be at least 8 characters";
    if (!/[a-zA-Z]/.test(newPassword)) return "New password must contain a letter";
    if (!/\d/.test(newPassword)) return "New password must contain a digit";
    if (newPassword !== confirmPassword) return "Passwords do not match";
    return null;
  };

  const handleChangePassword = async () => {
    const err = validatePassword();
    if (err) { setPasswordError(err); return; }
    setPasswordError("");
    setChangingPassword(true);
    try {
      await changePassword(token!, currentPassword, newPassword);
      success("Password changed successfully");
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    } catch (e) {
      setPasswordError(e instanceof Error ? e.message : "Failed to change password");
    } finally {
      setChangingPassword(false);
    }
  };

  const handleExport = async () => {
    setExporting(true);
    try {
      const blob = await exportUserData(token!);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "whitematter-export.json";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      success("Data exported successfully");
    } catch (e) {
      showError(e instanceof Error ? e.message : "Failed to export data");
    } finally {
      setExporting(false);
    }
  };

  const labelSx = {
    fontSize: "0.6875rem",
    fontWeight: 600,
    fontFamily: "'Outfit', sans-serif",
    color: "#52525B",
    textTransform: "uppercase" as const,
    letterSpacing: "0.08em",
    mb: 1.5,
  };

  return (
    <Box>
      {/* Appearance */}
      <Box sx={{ mb: 5 }}>
        <Typography sx={labelSx}>Appearance</Typography>
        <Box sx={{ display: "flex", gap: 0.5 }}>
          {(["light", "dark"] as const).map((m) => (
            <Box
              key={m}
              onClick={() => setMode(m)}
              sx={{
                px: 2,
                py: 0.75,
                fontSize: "0.8125rem",
                fontFamily: "'Outfit', sans-serif",
                borderRadius: "8px",
                cursor: "pointer",
                fontWeight: mode === m ? 500 : 400,
                color: mode === m ? "#FAFAFA" : "#A1A1AA",
                bgcolor: mode === m ? "#27272A" : "transparent",
                border: "1px solid",
                borderColor: mode === m ? "#3F3F46" : "transparent",
                transition: "all 0.15s ease",
                textTransform: "capitalize",
                userSelect: "none",
                "&:hover": {
                  bgcolor: mode === m ? "#27272A" : "rgba(255,255,255,0.03)",
                },
              }}
            >
              {m}
            </Box>
          ))}
        </Box>
      </Box>

      {/* Account Info */}
      <Box sx={{ mb: 5 }}>
        <Typography sx={labelSx}>Account</Typography>
        <Typography
          variant="body2"
          sx={{ color: "#FAFAFA", fontFamily: "'Outfit', sans-serif" }}
        >
          {user?.email}
        </Typography>
        <Typography
          sx={{
            mt: 0.25,
            fontSize: "0.75rem",
            color: "#52525B",
            fontFamily: "'Outfit', sans-serif",
          }}
        >
          {isOAuth ? "Signed in with Google" : "Email & password"}
        </Typography>
      </Box>

      {/* Change Password */}
      {!isOAuth && (
        <Box sx={{ mb: 5 }}>
          <Typography sx={labelSx}>Password</Typography>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5, maxWidth: 360 }}>
            <TextField
              placeholder="Current password"
              type="password"
              size="small"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              autoComplete="current-password"
            />
            <TextField
              placeholder="New password"
              type="password"
              size="small"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              autoComplete="new-password"
            />
            <TextField
              placeholder="Confirm new password"
              type="password"
              size="small"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              autoComplete="new-password"
            />
            {passwordError && (
              <Alert
                severity="error"
                onClose={() => setPasswordError("")}
                sx={{
                  py: 0.25,
                  borderRadius: "8px",
                  bgcolor: "rgba(239,68,68,0.08)",
                  border: "1px solid rgba(239,68,68,0.2)",
                  "& .MuiAlert-message": {
                    fontSize: "0.8125rem",
                    fontFamily: "'Outfit', sans-serif",
                  },
                }}
              >
                {passwordError}
              </Alert>
            )}
            <Button
              variant="contained"
              size="small"
              onClick={handleChangePassword}
              disabled={changingPassword}
              sx={{
                alignSelf: "flex-start",
                bgcolor: "#F97316",
                color: "#fff",
                fontFamily: "'Outfit', sans-serif",
                fontSize: "0.75rem",
                fontWeight: 500,
                borderRadius: "8px",
                textTransform: "none",
                px: 2,
                py: 0.75,
                "&:hover": { bgcolor: "#EA580C" },
                "&.Mui-disabled": { opacity: 0.5 },
              }}
            >
              {changingPassword ? <CircularProgress size={16} sx={{ color: "#fff" }} /> : "Update password"}
            </Button>
          </Box>
        </Box>
      )}

      {/* Footer actions */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 1.5,
          pt: 4,
          borderTop: "1px solid #27272A",
        }}
      >
        <Button
          variant="outlined"
          size="small"
          onClick={handleExport}
          disabled={exporting}
          sx={{
            borderColor: "#27272A",
            color: "#A1A1AA",
            fontFamily: "'Outfit', sans-serif",
            fontSize: "0.75rem",
            borderRadius: "8px",
            textTransform: "none",
            "&:hover": {
              borderColor: "#3F3F46",
              color: "#FAFAFA",
              bgcolor: "rgba(255,255,255,0.03)",
            },
          }}
        >
          {exporting ? <CircularProgress size={16} /> : "Export data"}
        </Button>
        <Button
          size="small"
          onClick={logout}
          sx={{
            color: "#EF4444",
            fontFamily: "'Outfit', sans-serif",
            fontSize: "0.75rem",
            borderRadius: "8px",
            textTransform: "none",
            "&:hover": {
              bgcolor: "rgba(239,68,68,0.08)",
              color: "#EF4444",
            },
          }}
        >
          Sign out
        </Button>
      </Box>

      <Toast toasts={toasts} onDismiss={dismissToast} />
    </Box>
  );
}
