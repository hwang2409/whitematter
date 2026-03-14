import { useState, useEffect } from "react";
import * as api from "@/api";
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

interface Props {
  training: boolean;
  onStartTraining: (job: api.CustomTrainJob) => void;
  onCancel: () => void;
  setError: (msg: string) => void;
}

export default function QuickStartForm({ training, onStartTraining, onCancel, setError }: Props) {
  const [builtInDatasets, setBuiltInDatasets] = useState<api.Dataset[]>([]);
  const [presets, setPresets] = useState<api.Preset[]>([]);
  const [optimizers, setOptimizers] = useState<api.Optimizer[]>([]);
  const [schedulers, setSchedulers] = useState<api.Scheduler[]>([]);
  const [augmentations, setAugmentations] = useState<api.Augmentation[]>([]);
  const [selectedBuiltInDataset, setSelectedBuiltInDataset] = useState("");
  const [selectedPreset, setSelectedPreset] = useState("");
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(64);
  const [modelName, setModelName] = useState("");
  const [selectedOptimizer, setSelectedOptimizer] = useState("sgd");
  const [learningRate, setLearningRate] = useState(0.01);
  const [momentum, setMomentum] = useState(0.9);
  const [weightDecay, setWeightDecay] = useState(0.0);
  const [selectedScheduler, setSelectedScheduler] = useState("none");
  const [schedulerStepSize, setSchedulerStepSize] = useState(10);
  const [schedulerGamma, setSchedulerGamma] = useState(0.1);
  const [enabledAugs, setEnabledAugs] = useState<Set<string>>(new Set());
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    loadQuickStartOptions();
  }, []);

  const filteredPresets = presets.filter((p) => p.dataset === selectedBuiltInDataset);

  useEffect(() => {
    const filtered = presets.filter((p) => p.dataset === selectedBuiltInDataset);
    if (filtered.length > 0 && !filtered.find((p) => p.id === selectedPreset)) {
      setSelectedPreset(filtered[0].id);
    }
  }, [selectedBuiltInDataset, presets]);

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
      console.error("Failed to load options:", e);
    }
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

  async function handleTrainQuickStart() {
    if (!selectedBuiltInDataset || !selectedPreset) {
      setError("Please select dataset and architecture");
      return;
    }

    setError("");

    const optimizerParams: Record<string, number> = { learning_rate: learningRate };
    if (selectedOptimizer === "sgd") {
      optimizerParams.momentum = momentum;
    }
    optimizerParams.weight_decay = weightDecay;

    const schedulerParams: Record<string, number> = {};
    if (selectedScheduler === "step") {
      schedulerParams.step_size = schedulerStepSize;
      schedulerParams.gamma = schedulerGamma;
    } else if (selectedScheduler === "exponential") {
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
      onStartTraining({
        job_id: result.job_id,
        model_id: result.model_id,
        status: "pending",
        epoch: 0,
        total_epochs: epochs,
        loss: 0,
        accuracy: 0,
        message: "Starting...",
      });
    } catch {
      setError("Failed to start training");
    }
  }

  const selectedBuiltInDatasetInfo = builtInDatasets.find((d) => d.id === selectedBuiltInDataset);
  const selectedPresetInfo = presets.find((p) => p.id === selectedPreset);

  return (
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
          <InputLabel id="train-batch-label">Batch Size</InputLabel>
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
          <Typography variant="overline" sx={{ color: "text.secondary", display: "block", mb: 1 }}>
            Optimizer
          </Typography>
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

          <Typography variant="overline" sx={{ color: "text.secondary", display: "block", mb: 1 }}>
            Learning Rate Scheduler
          </Typography>
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

          <Typography variant="overline" sx={{ color: "text.secondary", display: "block", mb: 1 }}>
            Data Augmentation
          </Typography>
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
        <Button variant="outlined" color="error" fullWidth onClick={onCancel} sx={{ py: 1.25 }}>
          Cancel Training
        </Button>
      )}
    </Box>
  );
}
