"use client";
import { useState, useEffect } from "react";
import { useAuth } from "@/context/AuthContext";
import * as aws from "@/services/aws";
import "./S3ManagerPage.css";

export default function S3ManagerPage() {
  const { token } = useAuth();
  const [buckets, setBuckets] = useState<{ name: string; created: string }[]>([]);
  const [selectedBucket, setSelectedBucket] = useState<string | null>(null);
  const [prefix, setPrefix] = useState("");
  const [objects, setObjects] = useState<{ key: string; type: string; size?: number; last_modified?: string }[]>([]);
  const [newBucketName, setNewBucketName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!token) return;
    aws.listBuckets(token).then(setBuckets).catch((e) => setError(e.message));
  }, [token]);

  useEffect(() => {
    if (!token || !selectedBucket) {
      setObjects([]);
      return;
    }
    setLoading(true);
    aws
      .listObjects(token, selectedBucket, prefix)
      .then(setObjects)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [token, selectedBucket, prefix]);

  const handleCreateBucket = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token || !newBucketName.trim()) return;
    setError("");
    try {
      await aws.createBucket(token, newBucketName.trim());
      setBuckets((b) => [...b, { name: newBucketName.trim(), created: new Date().toISOString() }]);
      setNewBucketName("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed");
    }
  };

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !token || !selectedBucket) return;
    const key = prefix ? `${prefix}${file.name}` : file.name;
    aws.uploadFile(token, selectedBucket, key, file).then(() => {
      setObjects((o) => [...o, { key, type: "file", size: file.size, last_modified: new Date().toISOString() }]);
    }).catch((err) => setError(err.message));
    e.target.value = "";
  };

  const handleDelete = async (key: string) => {
    if (!token || !selectedBucket || !confirm(`Delete ${key}?`)) return;
    try {
      await aws.deleteObject(token, selectedBucket, key);
      setObjects((o) => o.filter((x) => x.key !== key));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed");
    }
  };

  const handleDownload = async (key: string) => {
    if (!token || !selectedBucket) return;
    try {
      const url = await aws.getPresignedUrl(token, selectedBucket, key);
      window.open(url, "_blank");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed");
    }
  };

  return (
    <div className="s3-manager-page">
      <div className="s3-sidebar">
        <h3>Buckets</h3>
        <form onSubmit={handleCreateBucket} className="s3-new-bucket">
          <input
            value={newBucketName}
            onChange={(e) => setNewBucketName(e.target.value)}
            placeholder="New bucket name"
          />
          <button type="submit">Create</button>
        </form>
        <ul>
          {buckets.map((b) => (
            <li
              key={b.name}
              className={selectedBucket === b.name ? "selected" : ""}
              onClick={() => { setSelectedBucket(b.name); setPrefix(""); }}
            >
              {b.name}
            </li>
          ))}
        </ul>
      </div>
      <div className="s3-main">
        {error && <div className="s3-error">{error}</div>}
        {selectedBucket ? (
          <>
            <div className="s3-toolbar">
              <span className="s3-breadcrumb">{selectedBucket}{prefix && ` / ${prefix}`}</span>
              <label className="s3-upload-btn">
                Upload file
                <input type="file" onChange={handleUpload} hidden />
              </label>
            </div>
            {loading ? (
              <p>Loading…</p>
            ) : (
              <ul className="s3-objects">
                {prefix && (
                  <li
                    className="s3-object folder"
                    onClick={() => setPrefix(prefix.replace(/[^/]+\/$/, ""))}
                  >
                    ../ 
                  </li>
                )}
                {objects.map((obj) =>
                  obj.type === "folder" ? (
                    <li
                      key={obj.key}
                      className="s3-object folder"
                      onClick={() => setPrefix(obj.key)}
                    >
                      {obj.key}
                    </li>
                  ) : (
                    <li key={obj.key} className="s3-object file">
                      <span onClick={() => handleDownload(obj.key)}>{obj.key}</span>
                      <button type="button" onClick={() => handleDelete(obj.key)}>Delete</button>
                    </li>
                  )
                )}
              </ul>
            )}
          </>
        ) : (
          <p className="s3-placeholder">Select a bucket or create one</p>
        )}
      </div>
    </div>
  );
}
