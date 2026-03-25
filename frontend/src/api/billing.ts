import { fetchWithTimeout, API_BASE } from './client';
import type { BillingStatus, BillingUsage } from './types';

export async function getBillingStatus(token: string): Promise<BillingStatus> {
  const res = await fetchWithTimeout(`${API_BASE}/billing/status`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return res.json();
}

export async function getBillingUsage(token: string): Promise<BillingUsage> {
  const res = await fetchWithTimeout(`${API_BASE}/billing/usage`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return res.json();
}

export async function createCheckoutSession(
  token: string,
  plan: string
): Promise<{ url: string }> {
  const res = await fetchWithTimeout(`${API_BASE}/billing/checkout`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ plan }),
  });
  return res.json();
}

export async function createPortalSession(
  token: string
): Promise<{ url: string }> {
  const res = await fetchWithTimeout(`${API_BASE}/billing/portal`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });
  return res.json();
}
