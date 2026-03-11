"use client";
import { useState, useEffect } from "react";
import { useAuth } from "@/context/AuthContext";
import * as aws from "@/services/aws";
import "./AWSSetupPage.css";

const REGIONS = ["us-east-1", "us-west-2", "eu-west-1"];
const INSTANCE_TYPES = ["g4dn.xlarge", "g4dn.2xlarge", "g5.xlarge"];

export default function AWSSetupPage() {
  const { token } = useAuth();
  const [accessKey, setAccessKey] = useState("");
  const [secretKey, setSecretKey] = useState("");
  const [region, setRegion] = useState("us-east-1");
  const [instanceType, setInstanceType] = useState("g4dn.xlarge");
  const [status, setStatus] = useState<{ has_credentials: boolean } | null>(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!token) return;
    aws.getCredentials(token).then(setStatus).catch(() => setStatus({ has_credentials: false }));
  }, [token]);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token) return;
    setMessage("");
    setLoading(true);
    try {
      if (status?.has_credentials) {
        await aws.updateCredentials(token, {
          access_key: accessKey,
          secret_key: secretKey,
          default_region: region,
          default_instance_type: instanceType,
        });
        setMessage("Credentials updated.");
      } else {
        await aws.storeCredentials(token, {
          access_key: accessKey,
          secret_key: secretKey,
          default_region: region,
          default_instance_type: instanceType,
        });
        setStatus({ has_credentials: true });
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
    if (!token || !confirm("Remove stored AWS credentials?")) return;
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
      <h2>AWS credentials</h2>
      <p className="aws-setup-desc">
        Store access keys to use S3 and BYOC training. Keys are encrypted. Use an IAM user with minimal policy (S3 + EC2).
      </p>
      {status && (
        <p className="aws-setup-status">
          {status.has_credentials ? "✓ Credentials configured" : "No credentials stored"}
        </p>
      )}
      <form onSubmit={handleSave} className="aws-setup-form">
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
          <button type="submit" disabled={loading || !accessKey || !secretKey}>
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
