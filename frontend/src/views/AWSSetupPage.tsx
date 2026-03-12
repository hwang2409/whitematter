"use client";

import { useState, useEffect } from "react";
import { useThemeMode } from "@/context/ThemeContext";
import { useAuth } from "@/context/AuthContext";
import * as aws from "@/services/aws";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import InputAdornment from "@mui/material/InputAdornment";
import IconButton from "@mui/material/IconButton";
import FormControlLabel from "@mui/material/FormControlLabel";
import Switch from "@mui/material/Switch";
import Visibility from "@mui/icons-material/Visibility";
import VisibilityOff from "@mui/icons-material/VisibilityOff";
import DarkMode from "@mui/icons-material/DarkMode";
import LightMode from "@mui/icons-material/LightMode";
import Link from "@mui/icons-material/Link";
import LinkOff from "@mui/icons-material/LinkOff";

const REGIONS = ["us-east-1", "us-west-2", "eu-west-1"];
const INSTANCE_TYPES = ["g4dn.xlarge", "g4dn.2xlarge", "g5.xlarge"];

const STORAGE_PROVIDERS = [
  { value: "aws", label: "AWS S3" },
  { value: "r2", label: "Cloudflare R2" },
  { value: "b2", label: "Backblaze B2" },
  { value: "custom", label: "Other S3-compatible" },
] as const;

const R2_ENDPOINT_PLACEHOLDER = "https://<account_id>.r2.cloudflarestorage.com";
const B2_ENDPOINT_PLACEHOLDER = "https://s3.us-west-002.backblazeb2.com";

type CredStatus = {
  has_credentials: boolean;
  default_region?: string;
  default_instance_type?: string;
  endpoint_url?: string | null;
  provider?: string | null;
};

const MASK = "••••••••••••";

export default function AWSSetupPage() {
  const { token } = useAuth();
  const { mode, setMode } = useThemeMode();
  const [accessKey, setAccessKey] = useState("");
  const [secretKey, setSecretKey] = useState("");
  const [region, setRegion] = useState("us-east-1");
  const [instanceType, setInstanceType] = useState("g4dn.xlarge");
  const [provider, setProvider] = useState<string>("aws");
  const [endpointUrl, setEndpointUrl] = useState("");
  const [status, setStatus] = useState<CredStatus | null>(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const [showAccessKey, setShowAccessKey] = useState(false);
  const [showSecretKey, setShowSecretKey] = useState(false);
  const [showEndpoint, setShowEndpoint] = useState(false);

  useEffect(() => {
    if (!token) return;
    aws
      .getCredentials(token)
      .then((s) => {
        setStatus(s as CredStatus);
        if (s?.default_region) setRegion(s.default_region);
        if (s?.default_instance_type) setInstanceType(s.default_instance_type);
        if (s?.provider) setProvider(s.provider);
        if (s?.endpoint_url) setEndpointUrl(s.endpoint_url);
      })
      .catch(() => setStatus({ has_credentials: false }));
  }, [token]);

  const needsEndpoint = provider !== "aws";
  const endpointPlaceholder =
    provider === "r2" ? R2_ENDPOINT_PLACEHOLDER : provider === "b2" ? B2_ENDPOINT_PLACEHOLDER : "https://...";

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token) return;
    setMessage("");
    setLoading(true);
    try {
      const payload: aws.CredentialData = {
        access_key: accessKey,
        secret_key: secretKey,
        default_region: region,
        default_instance_type: instanceType,
        provider: provider === "aws" ? null : provider,
        endpoint_url: needsEndpoint && endpointUrl.trim() ? endpointUrl.trim() : null,
      };
      if (status?.has_credentials) {
        await aws.updateCredentials(token, payload);
        setMessage("Credentials updated.");
      } else {
        await aws.storeCredentials(token, payload);
        setStatus({ ...status, has_credentials: true });
        setMessage("Credentials saved.");
      }
      setAccessKey("");
      setSecretKey("");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Failed");
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!token || !confirm("Remove stored credentials?")) return;
    setLoading(true);
    try {
      await aws.deleteCredentials(token);
      setStatus({ has_credentials: false });
      setMessage("Credentials removed.");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ maxWidth: 480 }}>
      <Typography variant="h2" sx={{ mb: 0.5 }}>
        Settings
      </Typography>

      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
          Theme
        </Typography>
        <FormControlLabel
          control={
            <Switch
              checked={mode === "dark"}
              onChange={(_, checked) => setMode(checked ? "dark" : "light")}
              color="primary"
            />
          }
          label={
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
              {mode === "dark" ? <DarkMode fontSize="small" /> : <LightMode fontSize="small" />}
              <span>{mode === "dark" ? "Dark" : "Light"} mode</span>
            </Box>
          }
        />
      </Box>

      <Typography variant="h3" sx={{ mt: 2, mb: 0.5 }}>
        Cloud storage &amp; BYOC
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
        Store access keys for S3-compatible storage (AWS S3, Cloudflare R2, Backblaze B2) and BYOC training. Keys are
        encrypted.
      </Typography>

      {status && (
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1,
            py: 1.25,
            px: 1.5,
            mb: 2,
            borderRadius: 1,
            border: "1px solid",
            borderColor: status.has_credentials ? "success.main" : "divider",
            bgcolor: status.has_credentials ? "action.hover" : "action.hover",
            "& .connection-dot": {
              bgcolor: status.has_credentials ? "success.main" : "text.disabled",
            },
          }}
        >
          <Box
            className="connection-dot"
            sx={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              flexShrink: 0,
            }}
          />
          {status.has_credentials ? (
            <>
              <Link sx={{ fontSize: 18, opacity: 0.9 }} />
              <Typography variant="body2" fontWeight={500}>
                Connected to {STORAGE_PROVIDERS.find((p) => p.value === (status.provider || "aws"))?.label ?? "S3-compatible"}
                {status.provider === "aws" && status.default_region && (
                  <Typography component="span" variant="body2" color="text.secondary" sx={{ fontWeight: 400, ml: 0.5 }}>
                    · {status.default_region}
                  </Typography>
                )}
                {status.provider && status.provider !== "aws" && status.endpoint_url && (
                  <Typography component="span" variant="body2" color="text.secondary" sx={{ fontWeight: 400, ml: 0.5 }}>
                    · Endpoint configured
                  </Typography>
                )}
              </Typography>
            </>
          ) : (
            <>
              <LinkOff sx={{ fontSize: 18, opacity: 0.7 }} />
              <Typography variant="body2" color="text.secondary">Not connected</Typography>
            </>
          )}
        </Box>
      )}

      <Box component="form" onSubmit={handleSave} sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
        <FormControl fullWidth size="small">
          <InputLabel>Provider</InputLabel>
          <Select
            value={provider}
            label="Provider"
            onChange={(e) => setProvider(e.target.value)}
          >
            {STORAGE_PROVIDERS.map((p) => (
              <MenuItem key={p.value} value={p.value}>
                {p.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <TextField
          fullWidth
          size="small"
          label="Access Key ID"
          placeholder={status?.has_credentials && !accessKey ? MASK : "Access Key ID"}
          type={showAccessKey ? "text" : "password"}
          value={accessKey}
          onChange={(e) => setAccessKey(e.target.value)}
          autoComplete="off"
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  aria-label={showAccessKey ? "Hide" : "Show"}
                  onClick={() => setShowAccessKey((v) => !v)}
                  edge="end"
                  size="small"
                >
                  {showAccessKey ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              </InputAdornment>
            ),
          }}
        />

        <TextField
          fullWidth
          size="small"
          label="Secret Access Key"
          placeholder={status?.has_credentials && !secretKey ? MASK : "Secret Access Key"}
          type={showSecretKey ? "text" : "password"}
          value={secretKey}
          onChange={(e) => setSecretKey(e.target.value)}
          autoComplete="off"
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  aria-label={showSecretKey ? "Hide" : "Show"}
                  onClick={() => setShowSecretKey((v) => !v)}
                  edge="end"
                  size="small"
                >
                  {showSecretKey ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              </InputAdornment>
            ),
          }}
        />

        {needsEndpoint && (
          <TextField
            fullWidth
            size="small"
            label="Endpoint URL"
            placeholder={showEndpoint ? endpointPlaceholder : MASK}
            type={showEndpoint ? "url" : "password"}
            value={endpointUrl}
            onChange={(e) => setEndpointUrl(e.target.value)}
            autoComplete="off"
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    aria-label={showEndpoint ? "Hide" : "Show"}
                    onClick={() => setShowEndpoint((v) => !v)}
                    edge="end"
                    size="small"
                  >
                    {showEndpoint ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
        )}

        <FormControl fullWidth size="small">
          <InputLabel>Region (for BYOC)</InputLabel>
          <Select value={region} label="Region (for BYOC)" onChange={(e) => setRegion(e.target.value)}>
            {REGIONS.map((r) => (
              <MenuItem key={r} value={r}>
                {r}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl fullWidth size="small">
          <InputLabel>Instance type</InputLabel>
          <Select value={instanceType} label="Instance type" onChange={(e) => setInstanceType(e.target.value)}>
            {INSTANCE_TYPES.map((t) => (
              <MenuItem key={t} value={t}>
                {t}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {message && (
          <Typography variant="body2" color="text.secondary">
            {message}
          </Typography>
        )}

        <Box sx={{ display: "flex", gap: 1, mt: 0.5 }}>
          <Button
            type="submit"
            variant="contained"
            disabled={loading || !accessKey || !secretKey || (needsEndpoint && !endpointUrl.trim())}
          >
            {status?.has_credentials ? "Update" : "Save"}
          </Button>
          {status?.has_credentials && (
            <Button type="button" variant="outlined" color="error" onClick={handleDelete} disabled={loading}>
              Remove
            </Button>
          )}
        </Box>
      </Box>
    </Box>
  );
}
