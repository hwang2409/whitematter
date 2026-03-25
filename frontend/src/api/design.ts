import { fetchWithTimeout, API_BASE } from './client';
import type { Architecture, DesignSuggestion, PreviewCodeResponse, DesignHelpRequest } from './types';

export async function suggestArchitecture(
  datasetId: string,
  prompt: string
): Promise<DesignSuggestion> {
  const res = await fetchWithTimeout(`${API_BASE}/design/suggest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId, prompt }),
  }, 60000);
  return res.json();
}

export async function validateArchitecture(
  architecture: Architecture
): Promise<{ valid: boolean; errors: string[]; warnings: string[] }> {
  const res = await fetchWithTimeout(`${API_BASE}/design/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(architecture),
  });
  return res.json();
}

export async function refineArchitecture(
  architecture: Architecture,
  feedback: string
): Promise<DesignSuggestion> {
  const res = await fetchWithTimeout(`${API_BASE}/design/refine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ architecture, feedback }),
  }, 60000);
  return res.json();
}

export async function previewGeneratedCode(
  datasetId: string,
  architecture: Architecture
): Promise<PreviewCodeResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/design/preview-code`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId, architecture }),
  }, 30000);
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Preview failed');
  }
  return res.json();
}

export async function getDesignHelp(request: DesignHelpRequest): Promise<{ response: string }> {
  const res = await fetch(`${API_BASE}/design/help`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Failed to get help');
  }
  return res.json();
}
