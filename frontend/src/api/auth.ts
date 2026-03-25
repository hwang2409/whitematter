import { fetchWithTimeout, API_BASE } from './client';

export async function changePassword(
  token: string,
  currentPassword: string,
  newPassword: string
): Promise<{ message: string }> {
  const res = await fetchWithTimeout(`${API_BASE}/auth/change-password`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      current_password: currentPassword,
      new_password: newPassword,
    }),
  });
  return res.json();
}

export async function exportUserData(token: string): Promise<Blob> {
  const res = await fetchWithTimeout(`${API_BASE}/auth/export`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return res.blob();
}
