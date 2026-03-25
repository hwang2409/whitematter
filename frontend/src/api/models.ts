import { fetchWithTimeout, API_BASE } from './client';
import type { Model } from './types';

export async function getModels(): Promise<Model[]> {
  const res = await fetchWithTimeout(`${API_BASE}/models`);
  const data = await res.json();
  return data.models;
}

export async function getModel(id: string): Promise<Model> {
  const res = await fetchWithTimeout(`${API_BASE}/models/${id}`);
  return res.json();
}

export async function deleteModel(id: string): Promise<void> {
  await fetchWithTimeout(`${API_BASE}/models/${id}`, { method: 'DELETE' });
}

export async function getModelInfo(modelId: string): Promise<{
  model_id: string;
  name: string;
  dataset: string;
  input_format: string;
  num_classes: number;
  class_names: string[];
}> {
  const res = await fetch(`${API_BASE}/api/${modelId}/info`);
  return res.json();
}
