"use client";
import { useState, useEffect, useRef } from "react";
import * as api from "@/api";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import FormHelperText from "@mui/material/FormHelperText";
import { themeTokens } from "@/theme";

interface Props {
  models?: api.Model[];
  selectedModel?: string | null;
  onModelChange?: (id: string) => void;
}

export default function PredictTab({
  models: propModels,
  selectedModel,
  onModelChange,
}: Props) {
  const [localModels, setLocalModels] = useState<api.Model[]>([]);
  const models = propModels?.length ? propModels : localModels;
  const [selectedModelId, setSelectedModelId] = useState(selectedModel || "");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<api.PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (selectedModel && selectedModel !== selectedModelId) {
      setSelectedModelId(selectedModel);
    }
  }, [selectedModel]);

  useEffect(() => {
    if (models.length > 0 && !selectedModelId) {
      const firstId = models[0].id;
      setSelectedModelId(firstId);
      onModelChange?.(firstId);
    }
  }, [models]);

  async function loadModels() {
    if (propModels?.length) return;
    try {
      const data = await api.getModels();
      const completedModels = data.filter((m) => m.status === "completed");
      setLocalModels(completedModels);
      if (completedModels.length > 0 && !selectedModelId) {
        setSelectedModelId(completedModels[0].id);
      }
    } catch {
      setError("Failed to load models");
    }
  }

  function handleModelChange(id: string) {
    setSelectedModelId(id);
    onModelChange?.(id);
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setResult(null);
      setError("");
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result as string);
      reader.readAsDataURL(f);
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f?.type.startsWith("image/")) {
      setFile(f);
      setResult(null);
      setError("");
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result as string);
      reader.readAsDataURL(f);
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
  }

  async function handlePredict() {
    if (!file || !selectedModelId) {
      setError("Please select a model and upload an image");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const prediction = await api.predict(selectedModelId, file);
      setResult(prediction);
    } catch {
      setError("Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  function clearImage() {
    setFile(null);
    setPreview(null);
    setResult(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  const currentModel = models.find((m) => m.id === selectedModelId);

  return (
    <Box>
      <Typography variant="h2" sx={{ mb: 2 }}>
        Make Predictions
      </Typography>

      {error && (
        <Box
          sx={{
            border: "1px solid",
            borderColor: "error.main",
            color: "error.main",
            p: 1.25,
            borderRadius: 1,
            mb: 2,
            fontSize: "0.875rem",
          }}
        >
          {error}
        </Box>
      )}

      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel id="predict-model-label">Model</InputLabel>
        <Select
          labelId="predict-model-label"
          value={selectedModelId}
          label="Model"
          onChange={(e) => handleModelChange(e.target.value)}
          disabled={loading}
          sx={{ bgcolor: "action.hover" }}
        >
          {models.length === 0 ? (
            <MenuItem value="">No trained models available</MenuItem>
          ) : (
            models.map((m) => (
              <MenuItem key={m.id} value={m.id}>
                {m.name} ({m.dataset} - {m.best_accuracy.toFixed(1)}% acc)
              </MenuItem>
            ))
          )}
        </Select>
        {currentModel && (
          <FormHelperText>
            {currentModel.architecture} trained on {currentModel.dataset}
          </FormHelperText>
        )}
      </FormControl>

      <Box
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => !preview && fileInputRef.current?.click()}
        sx={{
          border: "1px dashed",
          borderColor: preview ? themeTokens.accentMuted : "divider",
          borderRadius: 1,
          p: preview ? 2 : 3,
          textAlign: "center",
          cursor: preview ? "default" : "pointer",
          bgcolor: preview ? "background.default" : "action.hover",
          transition: "border-color 0.2s ease-out, background-color 0.2s ease-out",
          "&:hover": {
            borderColor: themeTokens.accentMuted,
          },
        }}
      >
        {preview ? (
          <Box sx={{ position: "relative", display: "inline-block" }}>
            <Box
              component="img"
              src={preview}
              alt="Preview"
              sx={{ maxWidth: "100%", maxHeight: 280, borderRadius: 1 }}
            />
            <Button
              size="small"
              onClick={(e: React.MouseEvent) => {
                e.stopPropagation();
                clearImage();
              }}
              sx={{
                position: "absolute",
                top: -8,
                right: -8,
                minWidth: 24,
                width: 24,
                height: 24,
                borderRadius: 1,
                bgcolor: "background.paper",
                border: "1px solid",
                borderColor: "divider",
                color: "text.primary",
                "&:hover": { bgcolor: "error.main", borderColor: "error.main" },
              }}
            >
              ×
            </Button>
          </Box>
        ) : (
          <Box>
            <Typography variant="body2" color="text.secondary">
              Drop an image here or click to upload
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
              Supports PNG, JPG, JPEG
            </Typography>
          </Box>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          hidden
        />
      </Box>

      <Button
        variant="contained"
        onClick={handlePredict}
        disabled={!file || !selectedModelId || loading}
        sx={{ mt: 2 }}
      >
        {loading ? "Predicting..." : "Predict"}
      </Button>

      {result && (
        <Box
          sx={{
            mt: 3,
            p: 2,
            bgcolor: "background.paper",
            borderRadius: 1,
            border: "1px solid",
            borderColor: "divider",
          }}
        >
          <Typography variant="h3" sx={{ mb: 2 }}>
            Prediction Result
          </Typography>
          <Box
            sx={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              p: 1.5,
              bgcolor: "background.default",
              borderRadius: 1,
              mb: 2,
            }}
          >
            <Typography variant="h6" sx={{ fontSize: "1.5rem", fontWeight: 600 }}>
              {result.class_name}
            </Typography>
            <Typography
              variant="body2"
              sx={{
                fontFamily: '"JetBrains Mono", monospace',
                color: "primary.main",
              }}
            >
              {(result.confidence * 100).toFixed(1)}% confidence
            </Typography>
          </Box>

          <Typography variant="overline" sx={{ display: "block", mb: 1, color: "text.secondary" }}>
            All Probabilities
          </Typography>
          {Object.entries(result.probabilities)
            .sort(([, a], [, b]) => b - a)
            .map(([className, prob]) => (
              <ProbBar key={className} label={className} value={prob} />
            ))}
        </Box>
      )}
    </Box>
  );
}

function ProbBar({ label, value }: { label: string; value: number }) {
  const pct = value * 100;
  return (
    <Box
      sx={{
        display: "grid",
        gridTemplateColumns: "80px 1fr 60px",
        gap: 1,
        alignItems: "center",
        mb: 0.75,
      }}
    >
      <Typography
        variant="body2"
        sx={{
          fontFamily: '"JetBrains Mono", monospace',
          color: "text.secondary",
          textAlign: "right",
        }}
      >
        {label}
      </Typography>
      <Box
        sx={{
          height: 4,
          bgcolor: "action.hover",
          borderRadius: 1,
          overflow: "hidden",
        }}
      >
        <Box
          sx={{
            height: "100%",
            width: `${pct}%`,
            bgcolor: "primary.main",
            borderRadius: 1,
            transition: "width 0.3s ease-out",
          }}
        />
      </Box>
      <Typography
        variant="caption"
        sx={{
          fontFamily: '"JetBrains Mono", monospace',
          color: "text.secondary",
        }}
      >
        {pct.toFixed(1)}%
      </Typography>
    </Box>
  );
}
