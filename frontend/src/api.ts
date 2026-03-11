// API base URL - empty string uses relative URLs which works with:
// 1. Vite dev server proxy (configured in vite.config.ts)
// 2. Nginx proxy in Docker
// Set VITE_API_BASE environment variable to override
const API_BASE = import.meta.env.VITE_API_BASE || '';

// Request timeout in milliseconds (30 seconds)
const REQUEST_TIMEOUT = 30000;

// Error types for better error handling
export type ApiErrorType = 'timeout' | 'network' | 'server' | 'unknown';

export class ApiError extends Error {
  type: ApiErrorType;
  statusCode?: number;

  constructor(message: string, type: ApiErrorType, statusCode?: number) {
    super(message);
    this.name = 'ApiError';
    this.type = type;
    this.statusCode = statusCode;
  }
}

// Helper to create fetch with timeout
async function fetchWithTimeout(
  url: string,
  options?: RequestInit,
  timeout: number = REQUEST_TIMEOUT
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new ApiError(
        `Server error: ${response.status} ${response.statusText}`,
        'server',
        response.status
      );
    }

    return response;
  } catch (error) {
    clearTimeout(timeoutId);

    if (error instanceof ApiError) {
      throw error;
    }

    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        throw new ApiError(
          'Request timed out. The server took too long to respond.',
          'timeout'
        );
      }

      // Network errors (no internet, server unreachable, etc.)
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        throw new ApiError(
          'Network error. Please check your connection and try again.',
          'network'
        );
      }
    }

    throw new ApiError(
      error instanceof Error ? error.message : 'An unknown error occurred',
      'unknown'
    );
  }
}

// Helper to get user-friendly error message
export function getErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred';
}

// Helper to check if error is retryable
export function isRetryableError(error: unknown): boolean {
  if (error instanceof ApiError) {
    return error.type === 'timeout' || error.type === 'network';
  }
  return false;
}

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
  data_type: 'image' | 'text' | 'tabular' | 'unknown';
  format: string;
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
  status: 'pending' | 'compiling' | 'training' | 'completed' | 'failed' | 'running' | 'cancelled';
  message: string;
  epoch?: number;
  total_epochs?: number;
  loss?: number;
  accuracy?: number;
}

// WebSocket for real-time training updates

function getWsBase(): string {
  const loc = window.location;
  const protocol = loc.protocol === 'https:' ? 'wss:' : 'ws:';
  // In dev with Vite proxy, use the same host/port as the page
  return `${protocol}//${loc.host}`;
}

export function createTrainingWebSocket(
  jobId: string,
  onStatus: (status: CustomTrainJob) => void,
  onError?: (error: Event) => void,
  onClose?: () => void,
): { close: () => void } {
  let ws: WebSocket | null = null;
  let reconnectAttempts = 0;
  const maxReconnects = 5;
  let closed = false;
  let pingInterval: ReturnType<typeof setInterval> | null = null;

  function connect() {
    if (closed) return;

    const url = `${getWsBase()}/ws/train/${jobId}`;
    ws = new WebSocket(url);

    ws.onopen = () => {
      reconnectAttempts = 0;
      // Keepalive ping every 30 seconds
      pingInterval = setInterval(() => {
        if (ws?.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }));
        }
      }, 30000);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'ping') return; // ignore server pings
        onStatus(data as CustomTrainJob);
      } catch {
        // ignore parse errors
      }
    };

    ws.onerror = (event) => {
      onError?.(event);
    };

    ws.onclose = () => {
      if (pingInterval) {
        clearInterval(pingInterval);
        pingInterval = null;
      }
      if (closed) {
        onClose?.();
        return;
      }
      // Auto-reconnect with exponential backoff
      if (reconnectAttempts < maxReconnects) {
        const delay = Math.pow(2, reconnectAttempts) * 1000; // 1s, 2s, 4s, 8s, 16s
        reconnectAttempts++;
        setTimeout(connect, delay);
      } else {
        onClose?.();
      }
    };
  }

  connect();

  return {
    close() {
      closed = true;
      if (pingInterval) {
        clearInterval(pingInterval);
        pingInterval = null;
      }
      if (ws) {
        ws.close();
        ws = null;
      }
    },
  };
}

// API calls
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

export async function predict(modelId: string, file: File): Promise<PredictResult> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetchWithTimeout(`${API_BASE}/predict?model_id=${modelId}`, {
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
  // Use longer timeout for uploads (60 seconds)
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

export interface DatasetPreviewSample {
  image: string;
  label: string;
}

export async function getDatasetPreview(id: string): Promise<{
  metadata: CustomDataset;
  samples: DatasetPreviewSample[];
}> {
  const res = await fetchWithTimeout(`${API_BASE}/datasets/${id}/preview`);
  return res.json();
}

// Architecture design API
export async function suggestArchitecture(
  datasetId: string,
  prompt: string
): Promise<DesignSuggestion> {
  // Use longer timeout for AI suggestions (60 seconds)
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
  // Use longer timeout for AI refinements (60 seconds)
  const res = await fetchWithTimeout(`${API_BASE}/design/refine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ architecture, feedback }),
  }, 60000);
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

// Text dataset upload
export interface TextDatasetOptions {
  tokenizer_type?: 'character' | 'word';
  seq_length?: number;
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

// Text generation
export interface GenerateRequest {
  prompt: string;
  max_tokens?: number;
  temperature?: number;
}

export interface GenerateResult {
  model_id: string;
  prompt: string;
  generated_text: string;
  max_tokens: number;
  temperature: number;
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

// Text dataset preview
export interface TextPreview {
  vocab_size: number;
  sample_text: string;
  sample_tokens: string[];
  tokenizer_type: string;
  seq_length?: number;
  train_sequences?: number;
  test_sequences?: number;
  total_tokens?: number;
  unique_tokens?: number;
}

// Design helper chat
export interface DesignMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface DesignHelpRequest {
  message: string;
  context?: {
    dataset_type?: string;
    current_architecture?: Architecture | null;
  };
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
