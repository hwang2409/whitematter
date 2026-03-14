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
          const creds: FormFields = {
            access_key: data.access_key || "",
            secret_key: "",
            default_region: data.default_region || "us-east-1",
            endpoint_url: data.endpoint_url || "",
            provider: data.provider || "aws",
          };
          setStored(creds);
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
    if (err) {
      setFormError(err);
      return;
    }
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
      if (isUpdate) {
        await updateCredentials(token, payload);
      } else {
        await storeCredentials(token, payload);
      }
      const saved: FormFields = {
        access_key: form.access_key.trim(),
        secret_key: "",
        default_region: form.default_region,
        endpoint_url: form.endpoint_url.trim(),
        provider: form.provider,
      };
      setStored(saved);
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
    if (stored) {
      setViewState("viewing");
    } else {
      setViewState("empty");
    }
  };

  if (viewState === "loading") {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", py: 6 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h3" sx={{ mb: 1 }}>
        Connect
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Connect your cloud storage credentials for dataset storage and model deployment.
      </Typography>

      {/* Viewing state — show masked credentials */}
      {viewState === "viewing" && stored && (
        <Box>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5, mb: 3 }}>
            <Box>
              <Typography variant="body2" color="text.secondary">
                Access Key
              </Typography>
              <Typography variant="body1" sx={{ fontFamily: "monospace" }}>
                {maskKey(stored.access_key)}
              </Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">
                Region
              </Typography>
              <Typography variant="body1">{stored.default_region}</Typography>
            </Box>
            {stored.provider && (
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Provider
                </Typography>
                <Typography variant="body1" sx={{ textTransform: "uppercase" }}>
                  {stored.provider}
                </Typography>
              </Box>
            )}
            {stored.endpoint_url && (
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Endpoint
                </Typography>
                <Typography variant="body1">{stored.endpoint_url}</Typography>
              </Box>
            )}
          </Box>
          <Box sx={{ display: "flex", gap: 1.5 }}>
            <Button variant="outlined" size="small" onClick={startEditing}>
              Edit
            </Button>
            <Button
              variant="outlined"
              size="small"
              color="error"
              onClick={() => setConfirmDelete(true)}
            >
              Delete
            </Button>
          </Box>
        </Box>
      )}

      {/* Empty / Editing state — show form */}
      {(viewState === "empty" || viewState === "editing") && (
        <Box sx={{ display: "flex", flexDirection: "column", gap: 2, maxWidth: 440 }}>
          <TextField
            label="Access Key ID"
            size="small"
            value={form.access_key}
            onChange={(e) => updateField("access_key", e.target.value)}
          />
          <TextField
            label="Secret Access Key"
            type="password"
            size="small"
            value={form.secret_key}
            onChange={(e) => updateField("secret_key", e.target.value)}
            autoComplete="off"
          />
          <TextField
            label="Region"
            size="small"
            select
            value={form.default_region}
            onChange={(e) => updateField("default_region", e.target.value)}
          >
            {REGIONS.map((r) => (
              <MenuItem key={r} value={r}>
                {r}
              </MenuItem>
            ))}
          </TextField>
          <TextField
            label="S3-Compatible Endpoint (optional)"
            size="small"
            value={form.endpoint_url}
            onChange={(e) => updateField("endpoint_url", e.target.value)}
            placeholder="https://..."
          />
          <TextField
            label="Provider (optional)"
            size="small"
            select
            value={form.provider}
            onChange={(e) => updateField("provider", e.target.value)}
          >
            {PROVIDERS.map((p) => (
              <MenuItem key={p} value={p}>
                {p.toUpperCase()}
              </MenuItem>
            ))}
          </TextField>

          {formError && (
            <Alert severity="error" onClose={() => setFormError("")}>
              {formError}
            </Alert>
          )}

          <Box sx={{ display: "flex", gap: 1.5 }}>
            <Button
              variant="contained"
              onClick={handleSave}
              disabled={saving}
            >
              {saving ? <CircularProgress size={20} /> : viewState === "editing" ? "Update" : "Save"}
            </Button>
            {viewState === "editing" && (
              <Button variant="outlined" onClick={cancelEditing}>
                Cancel
              </Button>
            )}
          </Box>
        </Box>
      )}

      <ConfirmDialog
        isOpen={confirmDelete}
        title="Delete Credentials"
        message="Are you sure you want to delete your cloud credentials? This cannot be undone."
        confirmLabel="Delete"
        variant="danger"
        onConfirm={handleDelete}
        onCancel={() => setConfirmDelete(false)}
      />

      <Toast toasts={toasts} onDismiss={dismissToast} />
    </Box>
  );
}
