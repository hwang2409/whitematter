"use client";

import { useState } from "react";
import { useAuth } from "@/context/AuthContext";
import { useThemeMode } from "@/context/ThemeContext";
import { useToast } from "@/components/Toast";
import { changePassword, exportUserData } from "@/api";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import ButtonGroup from "@mui/material/ButtonGroup";
import TextField from "@mui/material/TextField";
import Chip from "@mui/material/Chip";
import Divider from "@mui/material/Divider";
import Alert from "@mui/material/Alert";
import CircularProgress from "@mui/material/CircularProgress";
import Toast from "@/components/Toast";

export default function AccountTab() {
  const { user, token, logout } = useAuth();
  const { mode, setMode } = useThemeMode();
  const { toasts, dismissToast, success, error: showError } = useToast();

  const isOAuth = user?.oauth_provider === "google";

  // Change password state
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [passwordError, setPasswordError] = useState("");
  const [changingPassword, setChangingPassword] = useState(false);

  // Export state
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
    if (err) {
      setPasswordError(err);
      return;
    }
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

  const sectionSx = { mb: 4 };
  const sectionTitle = { mb: 1.5, fontWeight: 600 };

  return (
    <Box>
      <Typography variant="h3" sx={{ mb: 3 }}>
        Account
      </Typography>

      {/* Appearance */}
      <Box sx={sectionSx}>
        <Typography variant="subtitle1" sx={sectionTitle}>
          Appearance
        </Typography>
        <ButtonGroup size="small">
          <Button
            variant={mode === "light" ? "contained" : "outlined"}
            onClick={() => setMode("light")}
          >
            Light
          </Button>
          <Button
            variant={mode === "dark" ? "contained" : "outlined"}
            onClick={() => setMode("dark")}
          >
            Dark
          </Button>
        </ButtonGroup>
      </Box>

      <Divider sx={{ mb: 3 }} />

      {/* Account Info */}
      <Box sx={sectionSx}>
        <Typography variant="subtitle1" sx={sectionTitle}>
          Account Info
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          {user?.email}
        </Typography>
        <Chip
          label={isOAuth ? "Google" : "Email & Password"}
          size="small"
          variant="outlined"
        />
      </Box>

      <Divider sx={{ mb: 3 }} />

      {/* Change Password (non-OAuth only) */}
      {!isOAuth && (
        <>
          <Box sx={sectionSx}>
            <Typography variant="subtitle1" sx={sectionTitle}>
              Change Password
            </Typography>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 2, maxWidth: 400 }}>
              <TextField
                label="Current Password"
                type="password"
                size="small"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                autoComplete="current-password"
              />
              <TextField
                label="New Password"
                type="password"
                size="small"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                autoComplete="new-password"
                helperText="Min 8 characters, must include a letter and a digit"
              />
              <TextField
                label="Confirm New Password"
                type="password"
                size="small"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                autoComplete="new-password"
              />
              {passwordError && (
                <Alert severity="error" onClose={() => setPasswordError("")}>
                  {passwordError}
                </Alert>
              )}
              <Button
                variant="contained"
                onClick={handleChangePassword}
                disabled={changingPassword}
                sx={{ alignSelf: "flex-start" }}
              >
                {changingPassword ? <CircularProgress size={20} /> : "Update Password"}
              </Button>
            </Box>
          </Box>
          <Divider sx={{ mb: 3 }} />
        </>
      )}

      {/* Export Data */}
      <Box sx={sectionSx}>
        <Typography variant="subtitle1" sx={sectionTitle}>
          Export Data
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          Download all your data as a JSON file.
        </Typography>
        <Button
          variant="outlined"
          onClick={handleExport}
          disabled={exporting}
        >
          {exporting ? <CircularProgress size={20} /> : "Export Data"}
        </Button>
      </Box>

      <Divider sx={{ mb: 3 }} />

      {/* Sign Out */}
      <Box>
        <Typography variant="subtitle1" sx={sectionTitle}>
          Sign Out
        </Typography>
        <Button variant="outlined" color="error" onClick={logout}>
          Sign Out
        </Button>
      </Box>

      <Toast toasts={toasts} onDismiss={dismissToast} />
    </Box>
  );
}
