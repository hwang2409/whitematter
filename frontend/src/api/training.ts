import { fetchWithTimeout, API_BASE } from './client';
import type { TrainConfig, TrainJob, Architecture, CustomTrainJob } from './types';

export async function startTraining(config: TrainConfig): Promise<{ job_id: string; model_id: string }> {
  const res = await fetchWithTimeout(`${API_BASE}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  return res.json();
}

export async function getTrainingStatus(jobId: string): Promise<TrainJob> {
  const res = await fetchWithTimeout(`${API_BASE}/train/${jobId}`);
  return res.json();
}

export async function cancelTraining(jobId: string): Promise<void> {
  await fetchWithTimeout(`${API_BASE}/train/${jobId}`, { method: 'DELETE' });
}

export async function resumeTraining(modelId: string): Promise<{
  job_id: string;
  model_id: string;
  message: string;
  start_epoch: number;
  total_epochs: number;
}> {
  const res = await fetchWithTimeout(`${API_BASE}/models/${modelId}/resume`, {
    method: 'POST',
  });
  return res.json();
}

export async function startCustomTraining(
  datasetId: string,
  architecture: Architecture
): Promise<CustomTrainJob> {
  const res = await fetch(`${API_BASE}/train/custom`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId, architecture }),
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Training failed to start');
  }
  return res.json();
}

export async function getCustomTrainingStatus(jobId: string): Promise<CustomTrainJob> {
  const res = await fetch(`${API_BASE}/train/${jobId}`);
  return res.json();
}
