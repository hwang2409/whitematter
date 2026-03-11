const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

function authHeaders(token: string) {
  return { Authorization: `Bearer ${token}` };
}

export interface Deployment {
  id: string;
  model_id: string;
  target_type: string;
  status: string;
  instance_id: string | null;
  endpoint_url: string | null;
  region: string | null;
  error_message: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export async function createDeployment(
  token: string,
  params: { model_id: string; target?: string; region?: string; platform_url?: string }
): Promise<{ deployment_id: string; status: string; message: string }> {
  const res = await fetch(`${API_BASE}/deploy`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({
      model_id: params.model_id,
      target: params.target || "ec2",
      region: params.region || "us-east-1",
      platform_url: params.platform_url,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to start deployment");
  }
  return res.json();
}

export async function getDeployment(token: string, deploymentId: string): Promise<Deployment> {
  const res = await fetch(`${API_BASE}/deploy/deployments/${deploymentId}`, {
    headers: authHeaders(token),
  });
  if (!res.ok) {
    if (res.status === 404) throw new Error("Deployment not found");
    throw new Error("Failed to get deployment");
  }
  return res.json();
}

export async function listDeployments(
  token: string,
  modelId?: string
): Promise<Deployment[]> {
  const url = modelId
    ? `${API_BASE}/deploy/deployments?model_id=${encodeURIComponent(modelId)}`
    : `${API_BASE}/deploy/deployments`;
  const res = await fetch(url, { headers: authHeaders(token) });
  if (!res.ok) throw new Error("Failed to list deployments");
  return res.json();
}

export async function terminateDeployment(token: string, deploymentId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/deploy/deployments/${deploymentId}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error("Failed to terminate deployment");
}
