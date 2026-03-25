"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/context/AuthContext";
import Toast from "@/components/Toast";
import { useToast } from "@/components/Toast";
import ConfirmDialog from "@/components/ConfirmDialog";
import {
  getCredentials,
  storeCredentials,
  updateCredentials,
  deleteCredentials,
  type CredentialData,
} from "@/services/aws";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import CircularProgress from "@mui/material/CircularProgress";
import Alert from "@mui/material/Alert";

type ViewState = "loading" | "empty" | "viewing" | "editing";

const REGIONS = ["us-east-1", "us-west-2"];
const PROVIDERS = ["aws", "r2", "b2", "custom"];

interface FormFields {
  access_key: string;
  secret_key: string;
  default_region: string;
  endpoint_url: string;
  provider: string;
}

const emptyForm: FormFields = {
  access_key: "",
  secret_key: "",
  default_region: "us-east-1",
  endpoint_url: "",
  provider: "aws",
};

function maskKey(key: string): string {
  if (key.length <= 4) return key;
  return "\u2022".repeat(8) + key.slice(-4);
}

export default function ConnectTab() {
  const { token } = useAuth();
  const { toasts, dismissToast, success, error: showError } = useToast();

  const [viewState, setViewState] = useState<ViewState>("loading");
  const [form, setForm] = useState<FormFields>(emptyForm);
  const [stored, setStored] = useState<FormFields | null>(null);
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    (async () => {
      try {
        const data = await getCredentials(token);
        if (cancelled) return;
        if (data && data.access_key) {
          setStored({
            access_key: data.access_key || "",
            secret_key: "",
            default_region: data.default_region || "us-east-1",
            endpoint_url: data.endpoint_url || "",
            provider: data.provider || "aws",
          });
          setViewState("viewing");
        } else {
          setViewState("empty");
        }
      } catch {
        if (!cancelled) setViewState("empty");
      }
    })();
    return () => { cancelled = true; };
  }, [token]);

  const updateField = (field: keyof FormFields, value: string) => {
    setForm((prev) => ({ ...prev, [field]: value }));
    setFormError("");
  };

  const validate = (): string | null => {
    if (!form.access_key.trim()) return "Access Key ID is required";
    if (!form.secret_key.trim()) return "Secret Access Key is required";
    return null;
  };

  const handleSave = async () => {
    const err = validate();
    if (err) { setFormError(err); return; }
    if (!token) return;
    setSaving(true);
    setFormError("");
    try {
      const payload: CredentialData = {
        access_key: form.access_key.trim(),
        secret_key: form.secret_key.trim(),
        default_region: form.default_region || undefined,
        endpoint_url: form.endpoint_url.trim() || null,
        provider: form.provider || null,
      };
      const isUpdate = viewState === "editing";
      if (isUpdate) await updateCredentials(token, payload);
      else await storeCredentials(token, payload);
      setStored({
        access_key: form.access_key.trim(),
        secret_key: "",
        default_region: form.default_region,
        endpoint_url: form.endpoint_url.trim(),
        provider: form.provider,
      });
      setViewState("viewing");
      success(isUpdate ? "Credentials updated" : "Credentials saved");
    } catch (e) {
      setFormError(e instanceof Error ? e.message : "Failed to save credentials");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!token) return;
    setConfirmDelete(false);
    try {
      await deleteCredentials(token);
      setStored(null);
      setForm(emptyForm);
      setViewState("empty");
      success("Credentials deleted");
    } catch (e) {
      showError(e instanceof Error ? e.message : "Failed to delete credentials");
    }
  };

  const startEditing = () => {
    setForm({
      access_key: stored?.access_key || "",
      secret_key: "",
      default_region: stored?.default_region || "us-east-1",
      endpoint_url: stored?.endpoint_url || "",
      provider: stored?.provider || "aws",
    });
    setFormError("");
    setViewState("editing");
  };

  const cancelEditing = () => {
    setFormError("");
    setViewState(stored ? "viewing" : "empty");
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

  if (viewState === "loading") {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", py: 8 }}>
        <CircularProgress size={20} sx={{ color: "#F97316" }} />
      </Box>
    );
  }

  return (
    <Box>
      <Typography
        sx={{
          mb: 4,
          fontSize: "0.875rem",
          color: "#A1A1AA",
          fontFamily: "'Outfit', sans-serif",
        }}
      >
        Cloud credentials for storage and deployment.
      </Typography>

      {viewState === "viewing" && stored && (
        <Box>
          <Typography sx={labelSx}>Credentials</Typography>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: "auto 1fr",
              columnGap: 3,
              rowGap: 1,
              mb: 3,
              p: 2.5,
              bgcolor: "#18181B",
              border: "1px solid #27272A",
              borderRadius: "10px",
              fontSize: "0.8125rem",
            }}
          >
            <Typography
              sx={{
                color: "#52525B",
                fontSize: "inherit",
                fontFamily: "'Outfit', sans-serif",
              }}
            >
              Key
            </Typography>
            <Typography
              sx={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: "inherit",
                color: "#FAFAFA",
              }}
            >
              {maskKey(stored.access_key)}
            </Typography>
            <Typography
              sx={{
                color: "#52525B",
                fontSize: "inherit",
                fontFamily: "'Outfit', sans-serif",
              }}
            >
              Region
            </Typography>
            <Typography
              sx={{
                fontSize: "inherit",
                color: "#FAFAFA",
                fontFamily: "'Outfit', sans-serif",
              }}
            >
              {stored.default_region}
            </Typography>
            {stored.provider && (
              <>
                <Typography
                  sx={{
                    color: "#52525B",
                    fontSize: "inherit",
                    fontFamily: "'Outfit', sans-serif",
                  }}
                >
                  Provider
                </Typography>
                <Typography
                  sx={{
                    fontSize: "inherit",
                    textTransform: "uppercase",
                    color: "#FAFAFA",
                    fontFamily: "'Outfit', sans-serif",
                  }}
                >
                  {stored.provider}
                </Typography>
              </>
            )}
            {stored.endpoint_url && (
              <>
                <Typography
                  sx={{
                    color: "#52525B",
                    fontSize: "inherit",
                    fontFamily: "'Outfit', sans-serif",
                  }}
                >
                  Endpoint
                </Typography>
                <Typography
                  sx={{
                    fontSize: "inherit",
                    color: "#FAFAFA",
                    fontFamily: "'JetBrains Mono', monospace",
                  }}
                >
                  {stored.endpoint_url}
                </Typography>
              </>
            )}
          </Box>
          <Box sx={{ display: "flex", gap: 1 }}>
            <Button
              size="small"
              variant="outlined"
              onClick={startEditing}
              sx={{
                borderColor: "#27272A",
                color: "#A1A1AA",
                fontFamily: "'Outfit', sans-serif",
                fontSize: "0.75rem",
                textTransform: "none",
                borderRadius: "8px",
                "&:hover": {
                  borderColor: "#3F3F46",
                  color: "#FAFAFA",
                  bgcolor: "rgba(255,255,255,0.03)",
                },
              }}
            >
              Edit
            </Button>
            <Button
              size="small"
              onClick={() => setConfirmDelete(true)}
              sx={{
                color: "#EF4444",
                fontFamily: "'Outfit', sans-serif",
                fontSize: "0.75rem",
                textTransform: "none",
                borderRadius: "8px",
                "&:hover": {
                  bgcolor: "rgba(239,68,68,0.08)",
                  color: "#EF4444",
                },
              }}
            >
              Delete
            </Button>
          </Box>
        </Box>
      )}

      {(viewState === "empty" || viewState === "editing") && (
        <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5, maxWidth: 400 }}>
          <TextField
            placeholder="Access Key ID"
            size="small"
            value={form.access_key}
            onChange={(e) => updateField("access_key", e.target.value)}
          />
          <TextField
            placeholder="Secret Access Key"
            type="password"
            size="small"
            value={form.secret_key}
            onChange={(e) => updateField("secret_key", e.target.value)}
            autoComplete="off"
          />
          <Box sx={{ display: "flex", gap: 1.5 }}>
            <TextField
              label="Region"
              size="small"
              select
              value={form.default_region}
              onChange={(e) => updateField("default_region", e.target.value)}
              sx={{ flex: 1 }}
            >
              {REGIONS.map((r) => (
                <MenuItem key={r} value={r}>{r}</MenuItem>
              ))}
            </TextField>
            <TextField
              label="Provider"
              size="small"
              select
              value={form.provider}
              onChange={(e) => updateField("provider", e.target.value)}
              sx={{ flex: 1 }}
            >
              {PROVIDERS.map((p) => (
                <MenuItem key={p} value={p}>{p.toUpperCase()}</MenuItem>
              ))}
            </TextField>
          </Box>
          <TextField
            placeholder="S3-compatible endpoint (optional)"
            size="small"
            value={form.endpoint_url}
            onChange={(e) => updateField("endpoint_url", e.target.value)}
          />

          {formError && (
            <Alert
              severity="error"
              onClose={() => setFormError("")}
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
              {formError}
            </Alert>
          )}

          <Box sx={{ display: "flex", gap: 1, pt: 0.5 }}>
            <Button
              variant="contained"
              size="small"
              onClick={handleSave}
              disabled={saving}
              sx={{
                bgcolor: "#F97316",
                color: "#fff",
                fontFamily: "'Outfit', sans-serif",
                fontSize: "0.75rem",
                fontWeight: 500,
                borderRadius: "8px",
                textTransform: "none",
                "&:hover": { bgcolor: "#EA580C" },
                "&.Mui-disabled": { opacity: 0.5 },
              }}
            >
              {saving ? <CircularProgress size={16} sx={{ color: "#fff" }} /> : viewState === "editing" ? "Update" : "Save"}
            </Button>
            {viewState === "editing" && (
              <Button
                size="small"
                variant="outlined"
                onClick={cancelEditing}
                sx={{
                  borderColor: "#27272A",
                  color: "#A1A1AA",
                  fontFamily: "'Outfit', sans-serif",
                  fontSize: "0.75rem",
                  textTransform: "none",
                  borderRadius: "8px",
                  "&:hover": {
                    borderColor: "#3F3F46",
                    color: "#FAFAFA",
                    bgcolor: "rgba(255,255,255,0.03)",
                  },
                }}
              >
                Cancel
              </Button>
            )}
          </Box>
        </Box>
      )}

      <ConfirmDialog
        isOpen={confirmDelete}
        title="Delete Credentials"
        message="Are you sure? This cannot be undone."
        confirmLabel="Delete"
        variant="danger"
        onConfirm={handleDelete}
        onCancel={() => setConfirmDelete(false)}
      />

      <Toast toasts={toasts} onDismiss={dismissToast} />
    </Box>
  );
}
