"use client";
import { useState, useEffect } from "react";
import * as api from "@/api";
import TrainingChart from "./TrainingChart";
import ArchitectureGraph from "./ArchitectureGraph";
import ParamTooltip from "./ParamTooltip";
import { parseTrainingError } from "@/lib/trainingErrors";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import FormHelperText from "@mui/material/FormHelperText";
import Paper from "@mui/material/Paper";
import FormControlLabel from "@mui/material/FormControlLabel";
import Checkbox from "@mui/material/Checkbox";
import LinearProgress from "@mui/material/LinearProgress";

type TrainMode = 'quick' | 'custom';

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

interface Props {
  datasets?: api.CustomDataset[];
  selectedDataset?: string | null;
  onDatasetChange?: (id: string) => void;
  onTrainingComplete?: () => void;
  helperOpen?: boolean;
  onHelperToggle?: (open: boolean) => void;
  onHelperContextChange?: (context: { datasetType?: string; architecture?: api.Architecture | null }) => void;
}

export default function TrainTab({
  datasets: propDatasets,
  selectedDataset,
  onDatasetChange,
  onTrainingComplete,
  helperOpen,
  onHelperToggle,
  onHelperContextChange,
}: Props) {
  // Mode toggle
  const [mode, setMode] = useState<TrainMode>('quick');

  // Dataset state
  const [localDatasets, setLocalDatasets] = useState<api.CustomDataset[]>([]);
  const datasets = propDatasets?.length ? propDatasets : localDatasets;
  const [selectedDatasetId, setSelectedDatasetId] = useState(selectedDataset || '');

  // Quick Start state (presets)
  const [builtInDatasets, setBuiltInDatasets] = useState<api.Dataset[]>([]);
  const [presets, setPresets] = useState<api.Preset[]>([]);
  const [optimizers, setOptimizers] = useState<api.Optimizer[]>([]);
  const [schedulers, setSchedulers] = useState<api.Scheduler[]>([]);
  const [augmentations, setAugmentations] = useState<api.Augmentation[]>([]);
  const [selectedBuiltInDataset, setSelectedBuiltInDataset] = useState('');
  const [selectedPreset, setSelectedPreset] = useState('');
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(64);
  const [modelName, setModelName] = useState('');
  const [selectedOptimizer, setSelectedOptimizer] = useState('sgd');
  const [learningRate, setLearningRate] = useState(0.01);
  const [momentum, setMomentum] = useState(0.9);
  const [weightDecay, setWeightDecay] = useState(0.0);
  const [selectedScheduler, setSelectedScheduler] = useState('none');
  const [schedulerStepSize, setSchedulerStepSize] = useState(10);
  const [schedulerGamma, setSchedulerGamma] = useState(0.1);
  const [enabledAugs, setEnabledAugs] = useState<Set<string>>(new Set());
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Custom Design state
  const [prompt, setPrompt] = useState('');
  const [architecture, setArchitecture] = useState<api.Architecture | null>(null);
  const [explanation, setExplanation] = useState('');
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [feedback, setFeedback] = useState('');
  const [generating, setGenerating] = useState(false);
  const [refining, setRefining] = useState(false);
  const [codePreview, setCodePreview] = useState<api.PreviewCodeResponse | null>(null);
  const [codePreviewLoading, setCodePreviewLoading] = useState(false);

  // Shared training state
  const [training, setTraining] = useState(false);
  const [trainingJob, setTrainingJob] = useState<api.CustomTrainJob | null>(null);
  const [trainingHistory, setTrainingHistory] = useState<{ epoch: number; loss: number; accuracy: number }[]>([]);
  const [error, setError] = useState('');

  const currentDataset = datasets.find((d) => d.id === selectedDatasetId);

  // Load data on mount
  useEffect(() => {
    loadDatasets();
    loadQuickStartOptions();
  }, []);

  // Shared handler for status updates from either WebSocket or polling
  function handleStatusUpdate(status: api.CustomTrainJob) {
    setTrainingJob(status);

    const epoch = status.epoch ?? 0;
    const loss = status.loss ?? 0;
    const accuracy = status.accuracy ?? 0;

    if (epoch > 0) {
      setTrainingHistory((prev) => {
        const existing = prev.find((h) => h.epoch === epoch);
        if (existing) return prev;
        return [...prev, { epoch, loss, accuracy }];
      });
    }

    if (!['pending', 'compiling', 'training', 'running'].includes(status.status)) {
      setTraining(false);
      if (status.status === 'completed') {
        onTrainingComplete?.();
      }
    }
  }

  // WebSocket-first with HTTP polling fallback
  useEffect(() => {
    if (!trainingJob || !['pending', 'compiling', 'training', 'running'].includes(trainingJob.status)) {
      return;
    }

    const jobId = trainingJob.job_id;
    let pollInterval: number | undefined;
    let wsHandle: { close: () => void } | null = null;

    function startPolling() {
      pollInterval = window.setInterval(async () => {
        try {
          const status = await api.getCustomTrainingStatus(jobId);
          handleStatusUpdate(status);
        } catch (e) {
          console.error('Failed to get training status:', e);
        }
      }, 2000);
    }

    // Try WebSocket first
    wsHandle = api.createTrainingWebSocket(
      jobId,
      (status) => handleStatusUpdate(status),
      () => {
        // On WebSocket error, fall back to polling
        if (!pollInterval) startPolling();
      },
      () => {
        // On WebSocket close (after max reconnects), fall back to polling
        wsHandle = null;
        if (!pollInterval) startPolling();
      },
    );

    return () => {
      wsHandle?.close();
      if (pollInterval) clearInterval(pollInterval);
    };
    // Only re-subscribe when the job_id changes, not on every status update.
    // The early return handles terminal states.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainingJob?.job_id]);

  // Sync external dataset selection
  useEffect(() => {
    if (selectedDataset && selectedDataset !== selectedDatasetId) {
      setSelectedDatasetId(selectedDataset);
    }
  }, [selectedDataset]);

  // Auto-select first custom dataset
  useEffect(() => {
    if (mode === 'custom' && datasets.length > 0 && !selectedDatasetId) {
      const firstId = datasets[0].id;
      setSelectedDatasetId(firstId);
      onDatasetChange?.(firstId);
    }
  }, [datasets, mode]);

  // Update helper context
  useEffect(() => {
    if (mode === 'custom') {
      onHelperContextChange?.({
        datasetType: currentDataset?.data_type,
        architecture: architecture,
      });
    }
  }, [architecture, currentDataset?.data_type, mode]);

  // Filter presets for selected built-in dataset
  const filteredPresets = presets.filter((p) => p.dataset === selectedBuiltInDataset);

  // Update preset when built-in dataset changes
  useEffect(() => {
    const filtered = presets.filter((p) => p.dataset === selectedBuiltInDataset);
    if (filtered.length > 0 && !filtered.find((p) => p.id === selectedPreset)) {
      setSelectedPreset(filtered[0].id);
    }
  }, [selectedBuiltInDataset, presets]);

  async function loadDatasets() {
    if (propDatasets?.length) return;
    try {
      const data = await api.getCustomDatasets();
      const readyDatasets = data.filter((d) => d.status === 'ready');
      setLocalDatasets(readyDatasets);
    } catch (e) {
      console.error('Failed to load datasets:', e);
    }
  }

  async function loadQuickStartOptions() {
    try {
      const [ds, ps, opts, scheds, augs] = await Promise.all([
        api.getDatasets(),
        api.getPresets(),
        api.getOptimizers(),
        api.getSchedulers(),
        api.getAugmentations(),
      ]);
      setBuiltInDatasets(ds);
      setPresets(ps);
      setOptimizers(opts);
      setSchedulers(scheds);
      setAugmentations(augs);

      if (ds.length > 0) setSelectedBuiltInDataset(ds[0].id);
      const filtered = ps.filter((p) => p.dataset === ds[0]?.id);
      if (filtered.length > 0) setSelectedPreset(filtered[0].id);
    } catch (e) {
      console.error('Failed to load options:', e);
    }
  }

  function handleDatasetChange(id: string) {
    setSelectedDatasetId(id);
    onDatasetChange?.(id);
  }

  function toggleAugmentation(id: string) {
    const newSet = new Set(enabledAugs);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setEnabledAugs(newSet);
  }

  // Custom Design handlers
  async function handleGenerate() {
    if (!selectedDatasetId || !prompt.trim()) {
      setError('Please select a dataset and describe what you want');
      return;
    }

    setGenerating(true);
    setError('');
    setArchitecture(null);
    setValidation(null);

    try {
      const result = await api.suggestArchitecture(selectedDatasetId, prompt.trim());
      setArchitecture(result.architecture);
      setExplanation(result.explanation);

      const val = await api.validateArchitecture(result.architecture);
      setValidation(val);
    } catch (e: any) {
      setError(e.message || 'Failed to get suggestion');
    } finally {
      setGenerating(false);
    }
  }

  async function handleRefine() {
    if (!architecture || !feedback.trim()) {
      setError('Please provide feedback for refinement');
      return;
    }

    setRefining(true);
    setError('');

    try {
      const result = await api.refineArchitecture(architecture, feedback.trim());
      setArchitecture(result.architecture);
      setExplanation(result.explanation);
      setFeedback('');

      const val = await api.validateArchitecture(result.architecture);
      setValidation(val);
    } catch (e: any) {
      setError(e.message || 'Failed to refine');
    } finally {
      setRefining(false);
    }
  }

  async function handlePreviewCode() {
    if (!selectedDatasetId || !architecture) return;
    setCodePreviewLoading(true);
    setCodePreview(null);
    setError('');
    try {
      const result = await api.previewGeneratedCode(selectedDatasetId, architecture);
      setCodePreview(result);
    } catch (e: any) {
      setError(e.message || 'Failed to preview code');
    } finally {
      setCodePreviewLoading(false);
    }
  }

  function handleLayerParamChange(
    layerIndex: number,
    paramKey: string,
    value: string | number
  ) {
    if (!architecture) return;
    const prev = architecture.layers[layerIndex].params[paramKey];
    const isNumericParam = typeof prev === 'number';
    const paramValue =
      isNumericParam && typeof value === 'string'
        ? (value === '' ? 0 : parseFloat(value))
        : value;
    const final = isNumericParam && typeof paramValue === 'number' && Number.isNaN(paramValue) ? prev : paramValue;
    setArchitecture({
      ...architecture,
      layers: architecture.layers.map((layer, i) =>
        i === layerIndex
          ? { ...layer, params: { ...layer.params, [paramKey]: final } }
          : layer
      ),
    });
  }

  // Training handlers
  async function handleTrainCustom() {
    if (!selectedDatasetId || !architecture) {
      setError('Missing dataset or architecture');
      return;
    }

    if (validation && !validation.valid) {
      setError('Please fix validation errors before training');
      return;
    }

    setTraining(true);
    setError('');
    setTrainingHistory([]);

    try {
      const job = await api.startCustomTraining(selectedDatasetId, architecture);
      setTrainingJob(job);
    } catch (e: any) {
      setError(e.message || 'Failed to start training');
      setTraining(false);
    }
  }

  async function handleTrainQuickStart() {
    if (!selectedBuiltInDataset || !selectedPreset) {
      setError('Please select dataset and architecture');
      return;
    }

    setError('');
    setTraining(true);
    setTrainingHistory([]);

    const optimizerParams: Record<string, number> = { learning_rate: learningRate };
    if (selectedOptimizer === 'sgd') {
      optimizerParams.momentum = momentum;
    }
    optimizerParams.weight_decay = weightDecay;

    const schedulerParams: Record<string, number> = {};
    if (selectedScheduler === 'step') {
      schedulerParams.step_size = schedulerStepSize;
      schedulerParams.gamma = schedulerGamma;
    } else if (selectedScheduler === 'exponential') {
      schedulerParams.gamma = schedulerGamma;
    }

    const augConfigs: api.AugmentationConfig[] = Array.from(enabledAugs).map((id) => ({
      type: id,
      params: augmentations.find((a) => a.id === id)?.params || {},
    }));

    try {
      const result = await api.startTraining({
        dataset: selectedBuiltInDataset,
        preset: selectedPreset,
        epochs,
        batch_size: batchSize,
        optimizer: { type: selectedOptimizer, params: optimizerParams },
        scheduler: { type: selectedScheduler, params: schedulerParams },
        augmentations: augConfigs,
        name: modelName || undefined,
      });
      setTrainingJob({
        job_id: result.job_id,
        model_id: result.model_id,
        status: 'pending',
        epoch: 0,
        total_epochs: epochs,
        loss: 0,
        accuracy: 0,
        message: 'Starting...',
      });
    } catch (e) {
      setError('Failed to start training');
      setTraining(false);
    }
  }

  async function handleCancelTraining() {
    if (!trainingJob) return;
    try {
      await api.cancelTraining(trainingJob.job_id);
      setTraining(false);
      setTrainingJob(null);
    } catch (e) {
      setError('Failed to cancel training');
    }
  }

  const selectedBuiltInDatasetInfo = builtInDatasets.find((d) => d.id === selectedBuiltInDataset);
  const selectedPresetInfo = presets.find((p) => p.id === selectedPreset);

  const totalEpochs = mode === 'custom'
    ? architecture?.training?.epochs || 10
    : epochs;

  const currentEpoch = trainingJob
    ? ('epoch' in trainingJob ? (trainingJob.epoch ?? 0) : 0)
    : 0;

  return (
    <Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 1, mb: 1.5 }}>
        <Box sx={{ flex: 1 }}>
          <Typography variant="h2" sx={{ mb: 0.5 }}>
            Train a Model
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Use presets for quick training or design a custom architecture with AI.
          </Typography>
        </Box>
        {mode === "custom" && (
          <Button
            variant="outlined"
            size="small"
            onClick={() => onHelperToggle?.(!helperOpen)}
            sx={{
              ...(helperOpen && {
                bgcolor: "action.selected",
                borderColor: "primary.main",
                color: "primary.main",
              }),
            }}
          >
            {helperOpen ? "Close Helper" : "AI Helper"}
          </Button>
        )}
      </Box>

      <Box
        sx={{
          display: "flex",
          bgcolor: "action.hover",
          border: "1px solid",
          borderColor: "divider",
          borderRadius: 1,
          p: 0.25,
          width: "fit-content",
          mb: 1.5,
        }}
      >
        <Button
          size="small"
          onClick={() => setMode("quick")}
          disabled={training}
          sx={{
            ...(mode === "quick" && { bgcolor: "action.selected", color: "text.primary" }),
            color: mode === "quick" ? "text.primary" : "text.secondary",
          }}
        >
          Quick Start
        </Button>
        <Button
          size="small"
          onClick={() => setMode("custom")}
          disabled={training}
          sx={{
            ...(mode === "custom" && { bgcolor: "action.selected", color: "text.primary" }),
            color: mode === "custom" ? "text.primary" : "text.secondary",
          }}
        >
          Custom Design
        </Button>
      </Box>

      {error && (
        <Box
          sx={{
            border: "1px solid",
            borderColor: "error.main",
            color: "error.main",
            p: 1.25,
            borderRadius: 1,
            mb: 1.5,
            fontSize: "0.875rem",
          }}
        >
          {(() => {
            const parsed = parseTrainingError(error);
            return (
              <Box>
                <Typography color="error.main" variant="body2">
                  {parsed.friendly}
                </Typography>
                {parsed.friendly !== parsed.raw && (
                  <details style={{ marginTop: 8 }}>
                    <summary style={{ cursor: "pointer", fontSize: "0.75rem", color: "inherit", opacity: 0.6 }}>
                      Show raw output
                    </summary>
                    <Box
                      component="pre"
                      sx={{
                        fontFamily: '"JetBrains Mono", monospace',
                        fontSize: "0.75rem",
                        p: 1,
                        mt: 0.5,
                        bgcolor: "action.hover",
                        borderRadius: 1,
                        overflow: "auto",
                        whiteSpace: "pre-wrap",
                      }}
                    >
                      {parsed.raw}
                    </Box>
                  </details>
                )}
              </Box>
            );
          })()}
        </Box>
      )}

      {mode === "quick" && (
        <Box sx={{ mb: 2 }}>
          <FormControl fullWidth sx={{ mb: 1.5 }}>
            <InputLabel id="train-dataset-label">Dataset</InputLabel>
            <Select
              labelId="train-dataset-label"
              value={selectedBuiltInDataset}
              label="Dataset"
              onChange={(e) => setSelectedBuiltInDataset(e.target.value)}
              disabled={training}
            >
              {builtInDatasets.map((d) => (
                <MenuItem key={d.id} value={d.id} disabled={!d.available}>
                  {d.name} {!d.available && "(not available)"}
                </MenuItem>
              ))}
            </Select>
            {selectedBuiltInDatasetInfo && (
              <FormHelperText>{selectedBuiltInDatasetInfo.description}</FormHelperText>
            )}
          </FormControl>

          <FormControl fullWidth sx={{ mb: 1.5 }}>
            <InputLabel id="train-preset-label">Architecture</InputLabel>
            <Select
              labelId="train-preset-label"
              value={selectedPreset}
              label="Architecture"
              onChange={(e) => setSelectedPreset(e.target.value)}
              disabled={training}
            >
              {filteredPresets.map((p) => (
                <MenuItem key={p.id} value={p.id}>
                  {p.name}
                </MenuItem>
              ))}
            </Select>
            {selectedPresetInfo && <FormHelperText>{selectedPresetInfo.num_layers} layers</FormHelperText>}
          </FormControl>

          <Box sx={{ display: "flex", gap: 2, mb: 1.5 }}>
            <TextField
              type="number"
              label="Epochs"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value) || 10)}
              inputProps={{ min: 1, max: 100 }}
              disabled={training}
              sx={{ flex: 1 }}
            />
            <FormControl sx={{ flex: 1 }}>
              <Box sx={{ display: "flex", alignItems: "center" }}>
                <InputLabel id="train-batch-label">Batch Size</InputLabel>
                <ParamTooltip paramKey="batch_size" />
              </Box>
              <Select
                labelId="train-batch-label"
                value={batchSize}
                label="Batch Size"
                onChange={(e) => setBatchSize(Number(e.target.value))}
                disabled={training}
              >
                <MenuItem value={32}>32</MenuItem>
                <MenuItem value={64}>64</MenuItem>
                <MenuItem value={128}>128</MenuItem>
                <MenuItem value={256}>256</MenuItem>
              </Select>
            </FormControl>
          </Box>

          <TextField
            fullWidth
            label="Model Name (optional)"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="my_model"
            disabled={training}
            sx={{ mb: 1.5 }}
          />

          <Button
            size="small"
            onClick={() => setShowAdvanced(!showAdvanced)}
            disabled={training}
            sx={{ color: "text.secondary", textTransform: "none", mb: 1 }}
          >
            {showAdvanced ? "Hide" : "Show"} Advanced Settings
          </Button>

          {showAdvanced && (
            <Paper variant="outlined" sx={{ p: 2, mb: 1.5, borderColor: "divider" }}>
              <Box sx={{ display: "flex", alignItems: "center" }}>
                <Typography variant="overline" sx={{ color: "text.secondary", display: "block", mb: 1 }}>
                  Optimizer
                </Typography>
                <ParamTooltip paramKey="optimizer" />
              </Box>
              <Box sx={{ display: "flex", gap: 2, mb: 1.5 }}>
                <FormControl size="small" sx={{ flex: 1 }}>
                  <InputLabel>Type</InputLabel>
                  <Select
                    value={selectedOptimizer}
                    label="Type"
                    onChange={(e) => setSelectedOptimizer(e.target.value)}
                    disabled={training}
                  >
                    {optimizers.map((o) => (
                      <MenuItem key={o.id} value={o.id}>
                        {o.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Box sx={{ display: "flex", alignItems: "center", flex: 1 }}>
                  <TextField
                    type="number"
                    size="small"
                    label="Learning Rate"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.01)}
                    inputProps={{ step: 0.001, min: 0.0001, max: 1 }}
                    disabled={training}
                    sx={{ flex: 1 }}
                  />
                  <ParamTooltip paramKey="learning_rate" />
                </Box>
              </Box>
              {selectedOptimizer === "sgd" && (
                <Box sx={{ display: "flex", gap: 2, mb: 1.5 }}>
                  <TextField
                    type="number"
                    size="small"
                    label="Momentum"
                    value={momentum}
                    onChange={(e) => setMomentum(parseFloat(e.target.value) || 0.9)}
                    inputProps={{ step: 0.1, min: 0, max: 1 }}
                    disabled={training}
                    sx={{ flex: 1 }}
                  />
                  <TextField
                    type="number"
                    size="small"
                    label="Weight Decay"
                    value={weightDecay}
                    onChange={(e) => setWeightDecay(parseFloat(e.target.value) || 0)}
                    inputProps={{ step: 0.0001, min: 0, max: 0.1 }}
                    disabled={training}
                    sx={{ flex: 1 }}
                  />
                </Box>
              )}

              <Box sx={{ display: "flex", alignItems: "center" }}>
                <Typography variant="overline" sx={{ color: "text.secondary", display: "block", mb: 1 }}>
                  Learning Rate Scheduler
                </Typography>
                <ParamTooltip paramKey="scheduler" />
              </Box>
              <FormControl size="small" fullWidth sx={{ mb: 1.5 }}>
                <InputLabel>Type</InputLabel>
                <Select
                  value={selectedScheduler}
                  label="Type"
                  onChange={(e) => setSelectedScheduler(e.target.value)}
                  disabled={training}
                >
                  {schedulers.map((s) => (
                    <MenuItem key={s.id} value={s.id}>
                      {s.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              {selectedScheduler === "step" && (
                <Box sx={{ display: "flex", gap: 2, mb: 1.5 }}>
                  <TextField
                    type="number"
                    size="small"
                    label="Step Size"
                    value={schedulerStepSize}
                    onChange={(e) => setSchedulerStepSize(parseInt(e.target.value) || 10)}
                    inputProps={{ min: 1, max: 50 }}
                    disabled={training}
                    sx={{ flex: 1 }}
                  />
                  <TextField
                    type="number"
                    size="small"
                    label="Gamma"
                    value={schedulerGamma}
                    onChange={(e) => setSchedulerGamma(parseFloat(e.target.value) || 0.1)}
                    inputProps={{ step: 0.1, min: 0.01, max: 1 }}
                    disabled={training}
                    sx={{ flex: 1 }}
                  />
                </Box>
              )}
              {selectedScheduler === "exponential" && (
                <TextField
                  type="number"
                  size="small"
                  fullWidth
                  label="Gamma (decay per epoch)"
                  value={schedulerGamma}
                  onChange={(e) => setSchedulerGamma(parseFloat(e.target.value) || 0.95)}
                  inputProps={{ step: 0.01, min: 0.5, max: 0.99 }}
                  disabled={training}
                  sx={{ mb: 1.5 }}
                />
              )}

              <Box sx={{ display: "flex", alignItems: "center" }}>
                <Typography variant="overline" sx={{ color: "text.secondary", display: "block", mb: 1 }}>
                  Data Augmentation
                </Typography>
                <ParamTooltip paramKey="augmentations" />
              </Box>
              <Box sx={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 0.5 }}>
                {augmentations.map((a) => (
                  <FormControlLabel
                    key={a.id}
                    control={
                      <Checkbox
                        checked={enabledAugs.has(a.id)}
                        onChange={() => toggleAugmentation(a.id)}
                        disabled={training}
                        size="small"
                      />
                    }
                    label={a.name}
                    sx={{ "& .MuiFormControlLabel-label": { fontSize: "0.875rem" } }}
                  />
                ))}
              </Box>
            </Paper>
          )}

          {!training ? (
            <Button variant="contained" fullWidth onClick={handleTrainQuickStart} sx={{ py: 1.25 }}>
              Start Training
            </Button>
          ) : (
            <Button variant="outlined" color="error" fullWidth onClick={handleCancelTraining} sx={{ py: 1.25 }}>
              Cancel Training
            </Button>
          )}
        </Box>
      )}

      {mode === "custom" && (
        <Box
          sx={{
            mb: 2,
            display: training ? "block" : "grid",
            gridTemplateColumns: "60% 1fr",
            gap: 3,
            alignItems: "start",
          }}
        >
          <Box sx={{ minWidth: 0 }}>
          <FormControl fullWidth sx={{ mb: 1.5 }}>
            <InputLabel id="custom-dataset-label">Dataset</InputLabel>
            <Select
              labelId="custom-dataset-label"
              value={selectedDatasetId}
              label="Dataset"
              onChange={(e) => handleDatasetChange(e.target.value)}
              disabled={generating || training}
            >
              {datasets.length === 0 ? (
                <MenuItem value="">No datasets available - upload one first</MenuItem>
              ) : (
                datasets.map((d) => (
                  <MenuItem key={d.id} value={d.id}>
                    {d.name} ({d.data_type}, {d.num_classes} classes)
                  </MenuItem>
                ))
              )}
            </Select>
            {currentDataset && (
              <FormHelperText>
                Input: [{currentDataset.input_shape.join("x")}], {currentDataset.total_samples} samples
              </FormHelperText>
            )}
          </FormControl>

          <TextField
            fullWidth
            multiline
            rows={3}
            placeholder="Describe the model you want to build..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={generating || training}
            sx={{
              mb: 1.5,
              "& .MuiOutlinedInput-root": {
                fontSize: "1rem",
                fontFamily: '"JetBrains Mono", monospace',
                bgcolor: "background.default",
                "&.Mui-focused": {
                  borderColor: "primary.main",
                  "& .MuiOutlinedInput-notchedOutline": { borderColor: "primary.main", borderWidth: 2 },
                },
              },
            }}
          />

          <Button
            variant="contained"
            onClick={handleGenerate}
            disabled={!selectedDatasetId || !prompt.trim() || generating}
            sx={{ mb: 2 }}
          >
            {generating ? "Generating..." : "Generate Architecture"}
          </Button>

          {architecture && (
            <Paper variant="outlined" sx={{ p: 2, mb: 1.5, borderColor: "divider" }}>
              <Typography variant="h3" sx={{ mb: 1 }}>
                Suggested Architecture: {architecture.name}
              </Typography>

              {explanation && (
                <Box
                  sx={{
                    p: 1.25,
                    mb: 1.5,
                    bgcolor: "background.default",
                    borderRadius: 1,
                    borderLeft: "3px solid",
                    borderColor: "divider",
                  }}
                >
                  <Typography variant="body2" color="text.secondary">
                    {explanation}
                  </Typography>
                </Box>
              )}

              {validation && (
                <Box
                  sx={{
                    p: 1.25,
                    mb: 1.5,
                    borderRadius: 1,
                    border: "1px solid",
                    borderColor: validation.valid ? "success.main" : "error.main",
                  }}
                >
                  <Typography variant="body2" fontWeight={600} color={validation.valid ? "success.main" : "error.main"}>
                    {validation.valid ? "Architecture is valid" : "Validation failed"}
                  </Typography>
                  {!validation.valid &&
                    validation.errors.map((err, i) => (
                      <Typography key={i} variant="body2" color="error.main" sx={{ mt: 0.25 }}>
                        {err}
                      </Typography>
                    ))}
                  {validation.warnings.map((warn, i) => (
                    <Typography key={i} variant="body2" color="warning.main" sx={{ mt: 0.25 }}>
                      {warn}
                    </Typography>
                  ))}
                </Box>
              )}

              <Typography variant="overline" sx={{ color: "text.secondary", display: "block", mb: 0.5 }}>
                Layers ({architecture.layers.length})
              </Typography>
              <FormHelperText sx={{ mb: 1 }}>
                Edits here are included when you Refine, Preview code, or Train.
              </FormHelperText>
              <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, overflow: "hidden", mb: 1.5 }}>
                {architecture.layers.map((layer, i) => (
                  <Box
                    key={i}
                    sx={{
                      display: "flex",
                      flexDirection: "column",
                      gap: 0.5,
                      p: 1,
                      borderBottom: i < architecture.layers.length - 1 ? "1px solid" : "none",
                      borderColor: "divider",
                    }}
                  >
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                      <Typography variant="caption" sx={{ color: "text.secondary", fontFamily: '"JetBrains Mono", monospace', minWidth: 24, textAlign: "center" }}>
                        {i + 1}
                      </Typography>
                      <Typography variant="body2" fontWeight={500}>
                        {layer.type}
                      </Typography>
                    </Box>
                    <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, pl: 2 }}>
                      {Object.entries(layer.params).map(([k, v]) => (
                        <Box key={k} sx={{ display: "inline-flex", alignItems: "center", gap: 0.35 }}>
                          <Typography variant="caption" color="text.secondary" sx={{ minWidth: 60 }}>
                            {k}
                          </Typography>
                          <TextField
                            size="small"
                            type={typeof v === "number" ? "number" : "text"}
                            value={v}
                            onChange={(e) =>
                              handleLayerParamChange(
                                i,
                                k,
                                typeof v === "number" ? (e.target.value === "" ? 0 : parseFloat(e.target.value)) : e.target.value
                              )
                            }
                            disabled={training}
                            inputProps={{
                              step: typeof v === "number" && !Number.isInteger(v) ? "any" : 1,
                              style: { width: typeof v === "number" ? 72 : 100 },
                            }}
                            sx={{ "& .MuiInputBase-root": { minHeight: 28 } }}
                          />
                        </Box>
                      ))}
                    </Box>
                  </Box>
                ))}
              </Box>

              <Typography variant="overline" sx={{ color: "text.secondary", display: "block", mb: 0.5 }}>
                Training Configuration
              </Typography>
              <Box sx={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 1, mb: 1.5 }}>
                <Box sx={{ p: 1, bgcolor: "background.default", borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase" }}>
                    Optimizer
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: '"JetBrains Mono", monospace' }}>
                    {architecture.training.optimizer.type} (lr={architecture.training.optimizer.params.learning_rate})
                  </Typography>
                </Box>
                <Box sx={{ p: 1, bgcolor: "background.default", borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase" }}>
                    Scheduler
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: '"JetBrains Mono", monospace' }}>
                    {architecture.training.scheduler.type}
                  </Typography>
                </Box>
                <Box sx={{ p: 1, bgcolor: "background.default", borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase" }}>
                    Epochs
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: '"JetBrains Mono", monospace' }}>
                    {architecture.training.epochs}
                  </Typography>
                </Box>
                <Box sx={{ p: 1, bgcolor: "background.default", borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase" }}>
                    Batch Size
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: '"JetBrains Mono", monospace' }}>
                    {architecture.training.batch_size}
                  </Typography>
                </Box>
              </Box>

              <details style={{ marginBottom: 8 }}>
                <summary style={{ cursor: "pointer", fontSize: "0.8125rem", color: "rgba(255,255,255,0.5)" }}>
                  View Full JSON
                </summary>
                <Box
                  component="pre"
                  sx={{
                    mt: 0.5,
                    p: 1,
                    fontSize: "0.75rem",
                    fontFamily: '"JetBrains Mono", monospace',
                    bgcolor: "background.default",
                    borderRadius: 1,
                    border: "1px solid",
                    borderColor: "divider",
                    overflow: "auto",
                  }}
                >
                  {JSON.stringify(architecture, null, 2)}
                </Box>
              </details>

              <Box sx={{ mb: 1.5 }}>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={handlePreviewCode}
                  disabled={codePreviewLoading || training || (validation !== null && !validation.valid)}
                  sx={{ mb: 0.5 }}
                >
                  {codePreviewLoading ? "Generating..." : "Preview generated code"}
                </Button>
                {codePreview && (
                  <details open style={{ marginTop: 8 }}>
                    <summary style={{ cursor: "pointer", fontWeight: 500, marginBottom: 4 }}>
                      Generated C++ (read-only)
                    </summary>
                    <Box sx={{ display: "flex", flexDirection: "column", gap: 1, pt: 1, borderTop: "1px solid", borderColor: "divider" }}>
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          train.cpp
                        </Typography>
                        <Box
                          component="pre"
                          sx={{
                            m: 0,
                            p: 1,
                            fontSize: "0.75rem",
                            fontFamily: '"JetBrains Mono", monospace',
                            bgcolor: "action.hover",
                            borderRadius: 0.5,
                            overflow: "auto",
                            maxHeight: 320,
                            whiteSpace: "pre",
                          }}
                        >
                          {codePreview.train_cpp}
                        </Box>
                      </Box>
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          infer.cpp
                        </Typography>
                        <Box
                          component="pre"
                          sx={{
                            m: 0,
                            p: 1,
                            fontSize: "0.75rem",
                            fontFamily: '"JetBrains Mono", monospace',
                            bgcolor: "action.hover",
                            borderRadius: 0.5,
                            overflow: "auto",
                            maxHeight: 320,
                            whiteSpace: "pre",
                          }}
                        >
                          {codePreview.infer_cpp}
                        </Box>
                      </Box>
                    </Box>
                  </details>
                )}
              </Box>

              <Box sx={{ pt: 1.5, borderTop: "1px solid", borderColor: "divider", mb: 1.5 }}>
                <Typography variant="overline" sx={{ color: "text.secondary", display: "block", mb: 0.5 }}>
                  Refine Architecture
                </Typography>
                <TextField
                  fullWidth
                  multiline
                  rows={2}
                  placeholder="e.g., Add more dropout layers, use a smaller learning rate..."
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                  disabled={refining || training}
                  sx={{ mb: 1 }}
                />
                <Button variant="outlined" size="small" onClick={handleRefine} disabled={!feedback.trim() || refining}>
                  {refining ? "Refining..." : "Refine"}
                </Button>
              </Box>

              <Box sx={{ pt: 1.5, borderTop: "1px solid", borderColor: "divider" }}>
                {!training ? (
                  <Button
                    variant="contained"
                    fullWidth
                    onClick={handleTrainCustom}
                    disabled={validation !== null && !validation.valid}
                    sx={{ py: 1.25 }}
                  >
                    Start Training
                  </Button>
                ) : (
                  <Button variant="outlined" color="error" fullWidth onClick={handleCancelTraining} sx={{ py: 1.25 }}>
                    Cancel Training
                  </Button>
                )}
              </Box>
            </Paper>
          )}
          </Box>
          {architecture && !training && (
            <Box sx={{ position: "sticky", top: 24, minWidth: 0 }}>
              <ArchitectureGraph architecture={architecture} />
            </Box>
          )}
        </Box>
      )}

      {trainingJob && (
        <Paper variant="outlined" sx={{ mt: 2, p: 2, borderColor: "divider" }}>
          <Typography variant="h3" sx={{ mb: 1.5 }}>
            Training Progress
          </Typography>
          <Box aria-live="polite" role="status">
            <Box sx={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 1, mb: 1.25 }}>
              <Box sx={{ textAlign: "center", p: 1, bgcolor: "background.default", borderRadius: 1 }}>
                <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase", display: "block", mb: 0.5 }}>
                  Status
                </Typography>
                <Typography variant="body1" fontWeight={600} sx={{ fontFamily: '"JetBrains Mono", monospace' }}>
                  {trainingJob.status}
                </Typography>
              </Box>
              <Box sx={{ textAlign: "center", p: 1, bgcolor: "background.default", borderRadius: 1 }}>
                <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase", display: "block", mb: 0.5 }}>
                  Epoch
                </Typography>
                <Typography variant="body1" fontWeight={600} sx={{ fontFamily: '"JetBrains Mono", monospace' }}>
                  {currentEpoch} / {totalEpochs}
                </Typography>
              </Box>
              {"loss" in trainingJob && (
                <Box sx={{ textAlign: "center", p: 1, bgcolor: "background.default", borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase", display: "block", mb: 0.5 }}>
                    Loss
                  </Typography>
                  <Typography variant="body1" fontWeight={600} sx={{ fontFamily: '"JetBrains Mono", monospace' }}>
                    {(trainingJob.loss || 0).toFixed(4)}
                  </Typography>
                </Box>
              )}
              {"accuracy" in trainingJob && (
                <Box sx={{ textAlign: "center", p: 1, bgcolor: "background.default", borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase", display: "block", mb: 0.5 }}>
                    Accuracy
                  </Typography>
                  <Typography variant="body1" fontWeight={600} sx={{ fontFamily: '"JetBrains Mono", monospace' }}>
                    {(trainingJob.accuracy || 0).toFixed(2)}%
                  </Typography>
                </Box>
              )}
            </Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
              {trainingJob.message}
            </Typography>
            {["training", "running"].includes(trainingJob.status) && currentEpoch > 0 && (
              <LinearProgress
                variant="determinate"
                value={(currentEpoch / totalEpochs) * 100}
                sx={{ mt: 1, height: 3, borderRadius: 1, "& .MuiLinearProgress-bar": { borderRadius: 1 } }}
              />
            )}
            {trainingHistory.length > 0 && <TrainingChart data={trainingHistory} />}
          </Box>
          {trainingJob.status === "completed" && (
            <Box
              sx={{
                mt: 1.5,
                p: 1.25,
                border: "1px solid",
                borderColor: "success.main",
                borderRadius: 1,
              }}
            >
              <Typography variant="body2" sx={{ mb: 0.5 }}>
                Training completed! Model ID: <Box component="code" sx={{ fontFamily: '"JetBrains Mono", monospace', bgcolor: "action.hover", px: 0.5, borderRadius: 0.5 }}>{trainingJob.model_id}</Box>
              </Typography>
              <Typography variant="caption" color="text.secondary">
                API endpoint: <Box component="code" sx={{ fontFamily: '"JetBrains Mono", monospace' }}>POST /api/{trainingJob.model_id}/predict</Box>
              </Typography>
            </Box>
          )}
        </Paper>
      )}
    </Box>
  );
}
