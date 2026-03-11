"use client";
import { useState, useEffect } from "react";
import { useAuth } from "@/context/AuthContext";
import * as aws from "@/services/aws";
import "./AWSSetupPage.css";

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

export default function AWSSetupPage() {
  const { token } = useAuth();
  const [accessKey, setAccessKey] = useState("");
  const [secretKey, setSecretKey] = useState("");
  const [region, setRegion] = useState("us-east-1");
  const [instanceType, setInstanceType] = useState("g4dn.xlarge");
  const [provider, setProvider] = useState<string>("aws");
  const [endpointUrl, setEndpointUrl] = useState("");
  const [status, setStatus] = useState<CredStatus | null>(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!token) return;
    aws.getCredentials(token).then((s) => {
      setStatus(s as CredStatus);
      if (s?.default_region) setRegion(s.default_region);
      if (s?.default_instance_type) setInstanceType(s.default_instance_type);
      if (s?.provider) setProvider(s.provider);
      if (s?.endpoint_url) setEndpointUrl(s.endpoint_url);
    }).catch(() => setStatus({ has_credentials: false }));
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
    <div className="aws-setup-page">
      <h2>Cloud storage &amp; BYOC</h2>
      <p className="aws-setup-desc">
        Store access keys for S3-compatible storage (AWS S3, Cloudflare R2, Backblaze B2) and BYOC training. Keys are encrypted.
      </p>
      {status && (
        <p className="aws-setup-status">
          {status.has_credentials ? "✓ Credentials configured" : "No credentials stored"}
        </p>
      )}
      <form onSubmit={handleSave} className="aws-setup-form">
        <label className="aws-setup-label">Provider</label>
        <select
          value={provider}
          onChange={(e) => setProvider(e.target.value)}
          className="aws-setup-select"
        >
          {STORAGE_PROVIDERS.map((p) => (
            <option key={p.value} value={p.value}>{p.label}</option>
          ))}
        </select>
        <input
          type="text"
          placeholder="Access Key ID"
          value={accessKey}
          onChange={(e) => setAccessKey(e.target.value)}
          autoComplete="off"
        />
        <input
          type="password"
          placeholder="Secret Access Key"
          value={secretKey}
          onChange={(e) => setSecretKey(e.target.value)}
          autoComplete="off"
        />
        {needsEndpoint && (
          <>
            <label className="aws-setup-label">Endpoint URL</label>
            <input
              type="url"
              placeholder={endpointPlaceholder}
              value={endpointUrl}
              onChange={(e) => setEndpointUrl(e.target.value)}
              autoComplete="off"
              className="aws-setup-endpoint"
            />
          </>
        )}
        <label className="aws-setup-label">Region (for BYOC)</label>
        <select value={region} onChange={(e) => setRegion(e.target.value)}>
          {REGIONS.map((r) => (
            <option key={r} value={r}>{r}</option>
          ))}
        </select>
        <select value={instanceType} onChange={(e) => setInstanceType(e.target.value)}>
          {INSTANCE_TYPES.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        {message && <div className="aws-setup-message">{message}</div>}
        <div className="aws-setup-buttons">
          <button
            type="submit"
            disabled={loading || !accessKey || !secretKey || (needsEndpoint && !endpointUrl.trim())}
          >
            {status?.has_credentials ? "Update" : "Save"}
          </button>
          {status?.has_credentials && (
            <button type="button" className="aws-setup-delete" onClick={handleDelete} disabled={loading}>
              Remove
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
