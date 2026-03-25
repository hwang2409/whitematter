import { API_BASE } from './client';
import type { PredictResult, GenerateRequest, GenerateResult } from './types';
import { fetchWithTimeout } from './client';

export async function predict(modelId: string, file: File): Promise<PredictResult> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetchWithTimeout(`${API_BASE}/predict?model_id=${modelId}`, {
    method: 'POST',
    body: formData,
  });
  return res.json();
}

// Model inference API (for custom models)
export async function predictCustom(modelId: string, file: File): Promise<PredictResult> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/${modelId}/predict`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Prediction failed');
  }
  return res.json();
}

export async function generateText(
  modelId: string,
  request: GenerateRequest
): Promise<GenerateResult> {
  const res = await fetch(`${API_BASE}/api/${modelId}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: request.prompt,
      max_tokens: request.max_tokens || 100,
      temperature: request.temperature || 0.8,
    }),
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Generation failed');
  }
  return res.json();
}
