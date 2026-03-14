import * as api from "@/api";
import ArchitectureGraph from "./ArchitectureGraph";
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

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

interface Props {
  datasets: api.CustomDataset[];
  selectedDatasetId: string;
  onDatasetChange: (id: string) => void;
  training: boolean;
  // Architecture design state
  prompt: string;
  setPrompt: (v: string) => void;
  architecture: api.Architecture | null;
  explanation: string;
  validation: ValidationResult | null;
  feedback: string;
  setFeedback: (v: string) => void;
  generating: boolean;
  refining: boolean;
  codePreview: api.PreviewCodeResponse | null;
  codePreviewLoading: boolean;
  // Handlers
  onGenerate: () => void;
  onRefine: () => void;
  onPreviewCode: () => void;
  onLayerParamChange: (layerIndex: number, paramKey: string, value: string | number) => void;
  onStartTraining: () => void;
  onCancelTraining: () => void;
}

export default function CustomDesignForm({
  datasets,
  selectedDatasetId,
  onDatasetChange,
  training,
  prompt,
  setPrompt,
  architecture,
  explanation,
  validation,
  feedback,
  setFeedback,
  generating,
  refining,
  codePreview,
  codePreviewLoading,
  onGenerate,
  onRefine,
  onPreviewCode,
  onLayerParamChange,
  onStartTraining,
  onCancelTraining,
}: Props) {
  const currentDataset = datasets.find((d) => d.id === selectedDatasetId);

  return (
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
            onChange={(e) => onDatasetChange(e.target.value)}
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
          onClick={onGenerate}
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
                            onLayerParamChange(
                              i,
                              k,
                              typeof v === "number" ? (e.target.value === "" ? 0 : parseFloat(e.target.value)) : e.target.value,
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
                onClick={onPreviewCode}
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
              <Button variant="outlined" size="small" onClick={onRefine} disabled={!feedback.trim() || refining}>
                {refining ? "Refining..." : "Refine"}
              </Button>
            </Box>

            <Box sx={{ pt: 1.5, borderTop: "1px solid", borderColor: "divider" }}>
              {!training ? (
                <Button
                  variant="contained"
                  fullWidth
                  onClick={onStartTraining}
                  disabled={validation !== null && !validation.valid}
                  sx={{ py: 1.25 }}
                >
                  Start Training
                </Button>
              ) : (
                <Button variant="outlined" color="error" fullWidth onClick={onCancelTraining} sx={{ py: 1.25 }}>
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
  );
}
