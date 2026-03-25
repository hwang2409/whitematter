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

export interface DatasetPreviewSample {
  image: string;
  label: string;
}

// Text dataset upload
export interface TextDatasetOptions {
  tokenizer_type?: 'character' | 'word';
  seq_length?: number;
}

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

export interface PreviewCodeResponse {
  train_cpp: string;
  infer_cpp: string;
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

export type ConversationPhase =
  | "greeting"
  | "exploring"
  | "architecture"
  | "data_needed"
  | "ready"
  | "training"
  | "completed"
  | "predicting";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  type:
    | "text"
    | "architecture"
    | "training_progress"
    | "training_complete"
    | "training_error"
    | "file_upload"
    | "quick_start_chips"
    | "prediction";
  metadata?: Record<string, unknown>;
  createdAt: string;
}

export interface Conversation {
  id: string;
  title: string | null;
  phase: ConversationPhase;
  isStarred: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface TrainingStatus {
  job_id: string;
  status: string;
  epoch: number;
  total_epochs: number;
  loss: number;
  accuracy: number;
  message: string;
}

export interface BillingStatus {
  plan: string;
  plan_details: Record<string, unknown>;
  subscription: Record<string, unknown> | null;
}

export interface BillingUsage {
  models_count: number;
  datasets_count: number;
  conversations_count: number;
}
