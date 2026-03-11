const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

function authHeaders(token: string) {
  return { Authorization: `Bearer ${token}` };
}

export async function storeCredentials(
  token: string,
  data: { access_key: string; secret_key: string; default_region?: string; default_instance_type?: string }
) {
  const res = await fetch(`${API_BASE}/credentials/aws`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to store credentials");
  }
  return res.json();
}

export async function getCredentials(token: string) {
  const res = await fetch(`${API_BASE}/credentials/aws`, { headers: authHeaders(token) });
  if (!res.ok) throw new Error("Failed to get credentials");
  return res.json();
}

export async function updateCredentials(
  token: string,
  data: { access_key: string; secret_key: string; default_region?: string; default_instance_type?: string }
) {
  const res = await fetch(`${API_BASE}/credentials/aws`, {
    method: "PUT",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to update credentials");
  }
  return res.json();
}

export async function deleteCredentials(token: string) {
  const res = await fetch(`${API_BASE}/credentials/aws`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error("Failed to delete credentials");
}

export async function listBuckets(token: string) {
  const res = await fetch(`${API_BASE}/storage/buckets`, { headers: authHeaders(token) });
  if (!res.ok) throw new Error("Failed to list buckets");
  return res.json();
}

export async function createBucket(token: string, name: string, region?: string) {
  const res = await fetch(`${API_BASE}/storage/buckets`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ name, region }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to create bucket");
  }
  return res.json();
}

export async function listObjects(token: string, bucket: string, prefix = "") {
  const q = prefix ? `?prefix=${encodeURIComponent(prefix)}` : "";
  const res = await fetch(`${API_BASE}/storage/buckets/${encodeURIComponent(bucket)}/objects${q}`, {
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error("Failed to list objects");
  return res.json();
}

export async function uploadFile(token: string, bucket: string, key: string, file: File) {
  const form = new FormData();
  form.append("bucket", bucket);
  form.append("key", key);
  form.append("file", file);
  const res = await fetch(`${API_BASE}/storage/upload`, {
    method: "POST",
    headers: authHeaders(token),
    body: form,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

export async function deleteObject(token: string, bucket: string, key: string) {
  const res = await fetch(`${API_BASE}/storage/objects`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ bucket, key }),
  });
  if (!res.ok) throw new Error("Failed to delete object");
}

export async function getPresignedUrl(token: string, bucket: string, key: string, expires_in = 3600) {
  const res = await fetch(`${API_BASE}/storage/presigned-url`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ bucket, key, expires_in }),
  });
  if (!res.ok) throw new Error("Failed to get download URL");
  const data = await res.json();
  return data.url as string;
}
