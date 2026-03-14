"use client";
import { useState, useEffect } from "react";
import * as api from "@/api";
import { useTrainingSession } from "@/hooks/useTrainingSession";
import { useArchitectureDesign } from "@/hooks/useArchitectureDesign";
import QuickStartForm from "./QuickStartForm";
import CustomDesignForm from "./CustomDesignForm";
import TrainingProgress from "./TrainingProgress";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";

type TrainMode = 'quick' | 'custom';

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
  const [mode, setMode] = useState<TrainMode>('quick');

  // Dataset state
  const [localDatasets, setLocalDatasets] = useState<api.CustomDataset[]>([]);
  const datasets = propDatasets?.length ? propDatasets : localDatasets;
  const [selectedDatasetId, setSelectedDatasetId] = useState(selectedDataset || '');

  const {
    training, trainingJob, trainingHistory, trainingError,
    setTrainingError, startTrainingJob, cancelTraining,
  } = useTrainingSession(onTrainingComplete);

  const design = useArchitectureDesign(setTrainingError);

  const error = trainingError;
  const setError = setTrainingError;

  // Load datasets on mount
  useEffect(() => {
    if (propDatasets?.length) return;
    (async () => {
      try {
        const data = await api.getCustomDatasets();
        setLocalDatasets(data.filter((d) => d.status === 'ready'));
      } catch (e) {
        console.error('Failed to load datasets:', e);
      }
    })();
  }, []);

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
  const currentDataset = datasets.find((d) => d.id === selectedDatasetId);
  useEffect(() => {
    if (mode === 'custom') {
      onHelperContextChange?.({
        datasetType: currentDataset?.data_type,
        architecture: design.architecture,
      });
    }
  }, [design.architecture, currentDataset?.data_type, mode]);

  function handleDatasetChange(id: string) {
    setSelectedDatasetId(id);
    onDatasetChange?.(id);
  }

  // Custom design training handler
  async function handleTrainCustom() {
    if (!selectedDatasetId || !design.architecture) {
      setError('Missing dataset or architecture');
      return;
    }
    if (design.validation && !design.validation.valid) {
      setError('Please fix validation errors before training');
      return;
    }
    setError('');
    try {
      const job = await api.startCustomTraining(selectedDatasetId, design.architecture);
      startTrainingJob(job);
    } catch (e: any) {
      setError(e.message || 'Failed to start training');
    }
  }

  const totalEpochs = mode === 'custom'
    ? design.architecture?.training?.epochs || 10
    : trainingJob?.total_epochs || 10;

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
          {error}
        </Box>
      )}

      {mode === "quick" && (
        <QuickStartForm
          training={training}
          onStartTraining={startTrainingJob}
          onCancel={cancelTraining}
          setError={setError}
        />
      )}

      {mode === "custom" && (
        <CustomDesignForm
          datasets={datasets}
          selectedDatasetId={selectedDatasetId}
          onDatasetChange={handleDatasetChange}
          training={training}
          prompt={design.prompt}
          setPrompt={design.setPrompt}
          architecture={design.architecture}
          explanation={design.explanation}
          validation={design.validation}
          feedback={design.feedback}
          setFeedback={design.setFeedback}
          generating={design.generating}
          refining={design.refining}
          codePreview={design.codePreview}
          codePreviewLoading={design.codePreviewLoading}
          onGenerate={() => design.handleGenerate(selectedDatasetId)}
          onRefine={design.handleRefine}
          onPreviewCode={() => design.handlePreviewCode(selectedDatasetId)}
          onLayerParamChange={design.handleLayerParamChange}
          onStartTraining={handleTrainCustom}
          onCancelTraining={cancelTraining}
        />
      )}

      {trainingJob && (
        <TrainingProgress
          trainingJob={trainingJob}
          trainingHistory={trainingHistory}
          currentEpoch={currentEpoch}
          totalEpochs={totalEpochs}
        />
      )}
    </Box>
  );
}
