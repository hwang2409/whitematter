import { fetchWithTimeout, API_BASE } from './client';
import type { Dataset, Preset, Optimizer, Scheduler, Augmentation, LayerType } from './types';

export async function getDatasets(): Promise<Dataset[]> {
  const res = await fetchWithTimeout(`${API_BASE}/config/datasets`);
  const data = await res.json();
  return data.datasets;
}

export async function getPresets(): Promise<Preset[]> {
  const res = await fetchWithTimeout(`${API_BASE}/config/presets`);
  const data = await res.json();
  return data.presets;
}

export async function getOptimizers(): Promise<Optimizer[]> {
  const res = await fetchWithTimeout(`${API_BASE}/config/optimizers`);
  const data = await res.json();
  return data.optimizers;
}

export async function getSchedulers(): Promise<Scheduler[]> {
  const res = await fetchWithTimeout(`${API_BASE}/config/schedulers`);
  const data = await res.json();
  return data.schedulers;
}

export async function getAugmentations(): Promise<Augmentation[]> {
  const res = await fetchWithTimeout(`${API_BASE}/config/augmentations`);
  const data = await res.json();
  return data.augmentations;
}

export async function getLayerTypes(): Promise<LayerType[]> {
  const res = await fetchWithTimeout(`${API_BASE}/config/layers`);
  const data = await res.json();
  return data.layers;
}
