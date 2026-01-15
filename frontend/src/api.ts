const API_BASE = 'http://localhost:8080';

export interface Dataset {
  id: string;
  name: string;
  description: string;
  available: boolean;
  input_shape: number[];
  num_classes: number;
  classes: string[];
}

export interface Preset {
  id: string;
  name: string;
  dataset: string;
  num_layers: number;
}

export interface LayerType {
  id: string;
  name: string;
  params: string[];
}

export interface Optimizer {
  id: string;
  name: string;
  params: Record<string, number>;
}

export interface Scheduler {
  id: string;
  name: string;
  params: Record<string, number>;
}

export interface Augmentation {
  id: string;
  name: string;
  params: Record<string, number>;
}

export interface Model {
  id: string;
  name: string;
  dataset: string;
  architecture: string;
  created_at: string;
  epochs_trained: number;
  best_accuracy: number;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  training_history: { epoch: number; loss: number; accuracy: number }[];
  config?: TrainConfig;
}

export interface TrainJob {
  id: string;
  model_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  epoch: number;
  total_epochs: number;
  loss: number;
  accuracy: number;
  message: string;
}

export interface OptimizerConfig {
  type: string;
  params: Record<string, number>;
}

export interface SchedulerConfig {
  type: string;
  params: Record<string, number>;
}

export interface AugmentationConfig {
  type: string;
  params: Record<string, number>;
}

export interface TrainConfig {
  dataset: string;
  preset?: string;
  epochs: number;
  batch_size: number;
  optimizer: OptimizerConfig;
  scheduler: SchedulerConfig;
  augmentations: AugmentationConfig[];
  name?: string;
}

export interface PredictResult {
  model_id: string;
  model_name: string;
  predicted_class: number;
  class_name: string;
  confidence: number;
  probabilities: Record<string, number>;
}

// Custom dataset types
export interface CustomDataset {
  id: string;
  name: string;
  data_type: 'image' | 'text' | 'tabular';
  input_shape: number[];
  num_classes: number;
  class_names: string[];
  total_samples: number;
  created_at: string;
  status: 'uploaded' | 'processing' | 'ready' | 'error';
}

export interface ArchitectureLayer {
  type: string;
  params: Record<string, number | string>;
}

export interface TrainingConfig {
  optimizer: { type: string; params: Record<string, number> };
  scheduler: { type: string; params: Record<string, number> };
  epochs: number;
  batch_size: number;
}

export interface Architecture {
  name: string;
  description: string;
  data_type: string;
  input_shape: number[];
  num_classes: number;
  layers: ArchitectureLayer[];
  training: TrainingConfig;
}

export interface DesignSuggestion {
  architecture: Architecture;
  explanation: string;
  raw_response: string;
}

export interface CustomTrainJob {
  job_id: string;
  model_id: string;
  status: 'pending' | 'compiling' | 'training' | 'completed' | 'failed';
  message: string;
  epoch?: number;
  total_epochs?: number;
  loss?: number;
  accuracy?: number;
}

// API calls
export async function getDatasets(): Promise<Dataset[]> {
  const res = await fetch(`${API_BASE}/config/datasets`);
  const data = await res.json();
  return data.datasets;
}

export async function getPresets(): Promise<Preset[]> {
  const res = await fetch(`${API_BASE}/config/presets`);
  const data = await res.json();
  return data.presets;
}

export async function getOptimizers(): Promise<Optimizer[]> {
  const res = await fetch(`${API_BASE}/config/optimizers`);
  const data = await res.json();
  return data.optimizers;
}

export async function getSchedulers(): Promise<Scheduler[]> {
  const res = await fetch(`${API_BASE}/config/schedulers`);
  const data = await res.json();
  return data.schedulers;
}

export async function getAugmentations(): Promise<Augmentation[]> {
  const res = await fetch(`${API_BASE}/config/augmentations`);
  const data = await res.json();
  return data.augmentations;
}

export async function getLayerTypes(): Promise<LayerType[]> {
  const res = await fetch(`${API_BASE}/config/layers`);
  const data = await res.json();
  return data.layers;
}

export async function getModels(): Promise<Model[]> {
  const res = await fetch(`${API_BASE}/models`);
  const data = await res.json();
  return data.models;
}

export async function getModel(id: string): Promise<Model> {
  const res = await fetch(`${API_BASE}/models/${id}`);
  return res.json();
}

export async function deleteModel(id: string): Promise<void> {
  await fetch(`${API_BASE}/models/${id}`, { method: 'DELETE' });
}

export async function startTraining(config: TrainConfig): Promise<{ job_id: string; model_id: string }> {
  const res = await fetch(`${API_BASE}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  return res.json();
}

export async function getTrainingStatus(jobId: string): Promise<TrainJob> {
  const res = await fetch(`${API_BASE}/train/${jobId}`);
  return res.json();
}

export async function cancelTraining(jobId: string): Promise<void> {
  await fetch(`${API_BASE}/train/${jobId}`, { method: 'DELETE' });
}

export async function predict(modelId: string, file: File): Promise<PredictResult> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/predict?model_id=${modelId}`, {
    method: 'POST',
    body: formData,
  });
  return res.json();
}

// Custom dataset API
export async function uploadDataset(file: File, name: string): Promise<CustomDataset> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('name', name);
  const res = await fetch(`${API_BASE}/datasets/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Upload failed');
  }
  return res.json();
}

export async function getCustomDatasets(): Promise<CustomDataset[]> {
  const res = await fetch(`${API_BASE}/datasets`);
  const data = await res.json();
  return data.datasets;
}

export async function getCustomDataset(id: string): Promise<CustomDataset> {
  const res = await fetch(`${API_BASE}/datasets/${id}`);
  return res.json();
}

export async function deleteCustomDataset(id: string): Promise<void> {
  await fetch(`${API_BASE}/datasets/${id}`, { method: 'DELETE' });
}

// Architecture design API
export async function suggestArchitecture(
  datasetId: string,
  prompt: string
): Promise<DesignSuggestion> {
  const res = await fetch(`${API_BASE}/design/suggest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId, prompt }),
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Suggestion failed');
  }
  return res.json();
}

export async function validateArchitecture(
  architecture: Architecture
): Promise<{ valid: boolean; errors: string[]; warnings: string[] }> {
  const res = await fetch(`${API_BASE}/design/validate`, {
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
  const res = await fetch(`${API_BASE}/design/refine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ architecture, feedback }),
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Refinement failed');
  }
  return res.json();
}

// Custom training API
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
