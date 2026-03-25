import { fetchWithTimeout, API_BASE } from './client';
import type { CustomDataset, DatasetPreviewSample, TextDatasetOptions } from './types';

export async function uploadDataset(file: File, name: string): Promise<CustomDataset> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('name', name);
  const res = await fetchWithTimeout(`${API_BASE}/datasets/upload`, {
    method: 'POST',
    body: formData,
  }, 60000);
  return res.json();
}

export async function getCustomDatasets(): Promise<CustomDataset[]> {
  const res = await fetchWithTimeout(`${API_BASE}/datasets`);
  const data = await res.json();
  return data.datasets;
}

export async function getCustomDataset(id: string): Promise<CustomDataset> {
  const res = await fetchWithTimeout(`${API_BASE}/datasets/${id}`);
  return res.json();
}

export async function deleteCustomDataset(id: string): Promise<void> {
  await fetchWithTimeout(`${API_BASE}/datasets/${id}`, { method: 'DELETE' });
}

export async function getDatasetPreview(id: string): Promise<{
  metadata: CustomDataset;
  samples: DatasetPreviewSample[];
}> {
  const res = await fetchWithTimeout(`${API_BASE}/datasets/${id}/preview`);
  return res.json();
}

export async function uploadTextDataset(
  file: File,
  name: string,
  options?: TextDatasetOptions
): Promise<CustomDataset> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('name', name);
  formData.append('tokenizer_type', options?.tokenizer_type || 'character');
  formData.append('seq_length', String(options?.seq_length || 128));

  const res = await fetch(`${API_BASE}/datasets/upload/text`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Upload failed');
  }
  return res.json();
}

export async function importDatasetFromUrl(
  url: string,
  name?: string | null
): Promise<CustomDataset> {
  const res = await fetchWithTimeout(`${API_BASE}/datasets/import/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, name: name || null }),
  }, 120000);
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Import failed');
  }
  return res.json();
}

export async function importDatasetFromHuggingFace(
  datasetId: string,
  options?: { name?: string | null; config?: string | null; split?: string }
): Promise<CustomDataset> {
  const res = await fetchWithTimeout(`${API_BASE}/datasets/import/huggingface`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_id: datasetId,
      name: options?.name ?? null,
      config: options?.config ?? null,
      split: options?.split ?? 'train',
    }),
  }, 120000);
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Import failed');
  }
  return res.json();
}
