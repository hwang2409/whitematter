"use client";
import React, { useState } from "react";
import Link from "next/link";
import * as api from "@/api";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Chip from "@mui/material/Chip";
import Slider from "@mui/material/Slider";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import ConfirmDialog from "../ConfirmDialog";
import Toast from "../Toast";
import ShareCard from "../ShareCard";
import ModelFilters from "./ModelFilters";
import ModelGrid from "./ModelGrid";
import useModelList, { formatModelName, formatDate, getModelLoss, isTextModel } from "./useModelList";
import useDeployPolling from "./useDeployPolling";

/* ── PHOSPHOR design tokens ─────────────────────────────────── */
const P = {
  fontDisplay: "'Instrument Serif', Georgia, serif",
  fontBody: "'Outfit', sans-serif",
  fontMono: "'JetBrains Mono', monospace",
  bg: "#09090B",
  surface: "#18181B",
  border: "#27272A",
  borderHover: "#3F3F46",
  textPrimary: "#FAFAFA",
  textSecondary: "#A1A1AA",
  textMuted: "#52525B",
  accent: "#F97316",
  accentSubtle: "rgba(249,115,22,0.08)",
  error: "#EF4444",
  success: "#22C55E",
  warning: "#EAB308",
} as const;

/* ── Shared button styles ───────────────────────────────────── */
const btnBase = {
  fontFamily: P.fontBody,
  fontSize: "0.8125rem",
  fontWeight: 500,
  textTransform: "none" as const,
  borderRadius: "8px",
  letterSpacing: 0,
  px: 2,
  py: 0.75,
  transition: "all 0.15s ease",
};

const btnPrimary = {
  ...btnBase,
  bgcolor: P.accent,
  color: "#fff",
  "&:hover": { bgcolor: "#EA6C0B" },
  "&.Mui-disabled": { bgcolor: "rgba(249,115,22,0.3)", color: "rgba(255,255,255,0.4)" },
};

const btnOutlined = {
  ...btnBase,
  color: P.textSecondary,
  border: `1px solid ${P.border}`,
  "&:hover": { borderColor: P.borderHover, color: P.textPrimary, bgcolor: "rgba(255,255,255,0.03)" },
};

const btnDanger = {
  ...btnBase,
  color: P.error,
  border: `1px solid rgba(239,68,68,0.25)`,
  "&:hover": { borderColor: P.error, bgcolor: "rgba(239,68,68,0.06)" },
};

/* ── Component ──────────────────────────────────────────────── */

interface Props {
  onModelSelect?: (id: string | null) => void;
  onUpdate?: () => void;
}

export default function ModelsTab({ onModelSelect, onUpdate }: Props) {
  const modelList = useModelList({ onUpdate });
  const [selectedModel, setSelectedModel] = useState<api.Model | null>(null);
  const [expandedModelId, setExpandedModelId] = useState<string | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [deployModalOpen, setDeployModalOpen] = useState(false);

  const [prompt, setPrompt] = useState("");
  const [generatedText, setGeneratedText] = useState("");
  const [generating, setGenerating] = useState(false);
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(100);
  const [copied, setCopied] = useState(false);

  const deploy = useDeployPolling({
    modelId: selectedModel?.id ?? null,
    deployModalOpen,
  });

  function handleRowClick(model: api.Model) {
    if (expandedModelId === model.id) {
      setExpandedModelId(null);
      setSelectedModel(null);
      onModelSelect?.(null);
    } else {
      setExpandedModelId(model.id);
      setSelectedModel(model);
      onModelSelect?.(model.id);
    }
  }

  async function confirmDelete(id: string) {
    try {
      await modelList.deleteModel(id);
      if (selectedModel?.id === id) {
        setSelectedModel(null);
        setExpandedModelId(null);
      }
    } finally {
      setDeleteConfirm(null);
    }
  }

  async function handleResume(id: string) {
    try {
      await modelList.resumeTraining(id);
      if (selectedModel?.id === id) {
        setSelectedModel({ ...selectedModel, status: "running" });
      }
      deploy.toast.success("Resuming training. Check the Train tab for progress.");
    } catch (e: unknown) {
      deploy.toast.error(e instanceof Error ? e.message : "Failed to resume training");
    }
  }

  async function copyEndpoint(modelId: string) {
    const url = `http://localhost:8080/api/${modelId}/predict`;
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      deploy.toast.success("URL copied.");
    } catch {
      deploy.toast.error("Failed to copy");
    }
  }

  async function handleGenerate() {
    if (!selectedModel || !prompt.trim()) return;
    setGenerating(true);
    modelList.setError("");
    setGeneratedText("");
    try {
      const result = await api.generateText(selectedModel.id, {
        prompt: prompt.trim(),
        max_tokens: maxTokens,
        temperature,
      });
      let text = result.generated_text;
      if (text.includes("\n") && text.startsWith("Model loaded:")) {
        text = text.split("\n").slice(1).join("\n");
      }
      setGeneratedText(text);
    } catch (e: unknown) {
      modelList.setError(e instanceof Error ? e.message : "Failed to generate text");
    } finally {
      setGenerating(false);
    }
  }

  /* ── Expanded row content ───────────────────────────────── */

  function renderExpandedContent(model: api.Model) {
    const loss = getModelLoss(model);
    return (
      <Box sx={{ py: 3, px: 3 }}>
        {/* Header: name + accuracy */}
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            flexWrap: "wrap",
            gap: 2,
            mb: 2.5,
          }}
        >
          <Typography
            variant="h2"
            sx={{
              fontFamily: P.fontBody,
              fontSize: "1.5rem",
              fontWeight: 700,
              color: P.textPrimary,
              letterSpacing: "-0.01em",
            }}
          >
            {formatModelName(model.name)}
          </Typography>
          {model.status === "completed" && (
            <Typography
              sx={{
                fontFamily: P.fontMono,
                fontSize: "2.5rem",
                fontWeight: 700,
                color: P.accent,
                lineHeight: 1,
              }}
            >
              {model.best_accuracy.toFixed(1)}%
            </Typography>
          )}
        </Box>

        {/* Architecture chips */}
        <Box sx={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 0.75, mb: 2.5 }}>
          {model.architecture
            .replace(/_/g, " ")
            .split(/[\s,-]+/)
            .filter(Boolean)
            .map((part, i, arr) => (
              <Box key={i} sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                <Chip
                  size="small"
                  label={part}
                  sx={{
                    fontFamily: P.fontMono,
                    fontSize: "0.6875rem",
                    fontWeight: 500,
                    color: P.textSecondary,
                    bgcolor: "rgba(255,255,255,0.04)",
                    border: `1px solid ${P.border}`,
                    borderRadius: "6px",
                    height: 26,
                    "& .MuiChip-label": { px: 1 },
                  }}
                />
                {i < arr.length - 1 && (
                  <Typography
                    component="span"
                    sx={{ fontSize: "0.75rem", color: P.textMuted }}
                  >
                    {"\u2192"}
                  </Typography>
                )}
              </Box>
            ))}
        </Box>

        {/* Stats row */}
        <Box
          sx={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: 1,
            mb: 3,
            fontFamily: P.fontBody,
            color: P.textSecondary,
            fontSize: "0.8125rem",
          }}
        >
          <span>{model.epochs_trained} epochs</span>
          <Box component="span" sx={{ color: P.textMuted }}>&middot;</Box>
          <span>{loss != null ? `${loss.toFixed(4)} final loss` : "\u2014"}</span>
          <Box component="span" sx={{ color: P.textMuted }}>&middot;</Box>
          <span>{model.dataset.startsWith("custom:") ? "Custom dataset" : model.dataset}</span>
          <Box component="span" sx={{ color: P.textMuted }}>&middot;</Box>
          <span>{formatDate(model.created_at)}</span>
        </Box>

        {/* Training history table */}
        {model.training_history.length > 0 && (
          <Box id="training-history-table" sx={{ mb: 3 }}>
            <Typography
              sx={{
                fontFamily: P.fontBody,
                fontSize: "0.625rem",
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                color: P.textMuted,
                mb: 1,
              }}
            >
              Training History
            </Typography>
            <Box
              sx={{
                borderRadius: "8px",
                border: `1px solid ${P.border}`,
                overflow: "hidden",
              }}
            >
              <Table size="small">
                <TableHead>
                  <TableRow>
                    {["Epoch", "Loss", "Accuracy"].map((label) => (
                      <TableCell
                        key={label}
                        sx={{
                          fontFamily: P.fontBody,
                          fontSize: "0.625rem",
                          fontWeight: 600,
                          textTransform: "uppercase",
                          letterSpacing: "0.08em",
                          color: P.textMuted,
                          borderBottom: `1px solid ${P.border}`,
                          bgcolor: "rgba(255,255,255,0.02)",
                          py: 1,
                        }}
                      >
                        {label}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {model.training_history.map((h) => (
                    <TableRow
                      key={h.epoch}
                      sx={{
                        "&:last-child td": { borderBottom: 0 },
                        "&:hover": { bgcolor: "rgba(255,255,255,0.02)" },
                      }}
                    >
                      <TableCell
                        sx={{
                          fontFamily: P.fontMono,
                          fontSize: "0.8125rem",
                          color: P.textSecondary,
                          borderBottom: `1px solid rgba(39,39,42,0.4)`,
                          py: 0.75,
                        }}
                      >
                        {h.epoch}
                      </TableCell>
                      <TableCell
                        sx={{
                          fontFamily: P.fontMono,
                          fontSize: "0.8125rem",
                          color: P.textPrimary,
                          borderBottom: `1px solid rgba(39,39,42,0.4)`,
                          py: 0.75,
                        }}
                      >
                        {h.loss.toFixed(4)}
                      </TableCell>
                      <TableCell
                        sx={{
                          fontFamily: P.fontMono,
                          fontSize: "0.8125rem",
                          color: P.textPrimary,
                          borderBottom: `1px solid rgba(39,39,42,0.4)`,
                          py: 0.75,
                        }}
                      >
                        {h.accuracy.toFixed(2)}%
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Box>
          </Box>
        )}

        {/* Text generation UI (text models) */}
        {model.status === "completed" && isTextModel(model) && (
          <Box
            sx={{
              mt: 1,
              p: 2.5,
              bgcolor: P.bg,
              borderRadius: "8px",
              border: `1px solid ${P.border}`,
            }}
          >
            <Typography
              sx={{
                fontFamily: P.fontBody,
                fontSize: "0.75rem",
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.06em",
                color: P.textSecondary,
                mb: 1.5,
              }}
            >
              Generate Text
            </Typography>
            <TextField
              fullWidth
              multiline
              rows={3}
              placeholder="Enter a starting prompt..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={generating}
              sx={{
                mb: 1.5,
                "& .MuiOutlinedInput-root": {
                  fontFamily: P.fontBody,
                  fontSize: "0.875rem",
                  color: P.textPrimary,
                  bgcolor: "rgba(255,255,255,0.03)",
                  borderRadius: "8px",
                  "& fieldset": { borderColor: P.border },
                  "&:hover fieldset": { borderColor: P.borderHover },
                  "&.Mui-focused fieldset": {
                    borderColor: P.accent,
                    borderWidth: "1px",
                  },
                  "& textarea::placeholder": { color: P.textMuted, opacity: 1 },
                },
              }}
            />
            <Box sx={{ display: "flex", gap: 2, alignItems: "center", mb: 1.5 }}>
              <Box sx={{ flex: 1 }}>
                <Typography
                  sx={{
                    fontFamily: P.fontBody,
                    fontSize: "0.75rem",
                    color: P.textSecondary,
                    mb: 0.5,
                  }}
                >
                  Temperature: {temperature}
                </Typography>
                <Slider
                  size="small"
                  min={0.1}
                  max={1.5}
                  step={0.1}
                  value={temperature}
                  onChange={(_, v) => setTemperature(v as number)}
                  disabled={generating}
                  valueLabelDisplay="auto"
                  sx={{
                    color: P.accent,
                    "& .MuiSlider-track": { bgcolor: P.accent },
                    "& .MuiSlider-thumb": {
                      bgcolor: P.accent,
                      width: 14,
                      height: 14,
                      "&:hover": { boxShadow: `0 0 0 6px ${P.accentSubtle}` },
                    },
                    "& .MuiSlider-rail": { bgcolor: P.border },
                  }}
                />
              </Box>
              <TextField
                type="number"
                size="small"
                label="Max Tokens"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value) || 100)}
                inputProps={{ min: 10, max: 500 }}
                disabled={generating}
                sx={{
                  width: 110,
                  "& .MuiOutlinedInput-root": {
                    fontFamily: P.fontMono,
                    fontSize: "0.8125rem",
                    color: P.textPrimary,
                    borderRadius: "8px",
                    "& fieldset": { borderColor: P.border },
                    "&:hover fieldset": { borderColor: P.borderHover },
                    "&.Mui-focused fieldset": { borderColor: P.accent, borderWidth: "1px" },
                  },
                  "& .MuiInputLabel-root": {
                    fontFamily: P.fontBody,
                    fontSize: "0.75rem",
                    color: P.textMuted,
                  },
                }}
              />
            </Box>
            <Button variant="contained" onClick={handleGenerate} disabled={generating || !prompt.trim()} sx={btnPrimary}>
              {generating ? "Generating..." : "Generate"}
            </Button>
            {generatedText && (
              <Box
                sx={{
                  mt: 2,
                  p: 2,
                  bgcolor: P.surface,
                  borderRadius: "8px",
                  border: `1px solid ${P.border}`,
                }}
              >
                <Typography
                  sx={{
                    fontFamily: P.fontBody,
                    fontSize: "0.625rem",
                    fontWeight: 600,
                    textTransform: "uppercase",
                    letterSpacing: "0.08em",
                    color: P.textMuted,
                    mb: 1,
                  }}
                >
                  Generated Text
                </Typography>
                <Box
                  component="pre"
                  sx={{
                    fontFamily: P.fontMono,
                    fontSize: "0.875rem",
                    color: P.textPrimary,
                    whiteSpace: "pre-wrap",
                    m: 0,
                    lineHeight: 1.6,
                  }}
                >
                  {generatedText}
                </Box>
              </Box>
            )}
          </Box>
        )}

        {/* API endpoint + cURL (non-text models) */}
        {model.status === "completed" && !isTextModel(model) && (
          <>
            <Box sx={{ mt: 2, pt: 2.5, borderTop: `1px solid ${P.border}` }}>
              <Typography
                sx={{
                  fontFamily: P.fontBody,
                  fontSize: "0.625rem",
                  fontWeight: 600,
                  textTransform: "uppercase",
                  letterSpacing: "0.08em",
                  color: P.textMuted,
                  mb: 1,
                }}
              >
                API Endpoint
              </Typography>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.5 }}>
                <Box
                  component="code"
                  sx={{
                    flex: 1,
                    fontFamily: P.fontMono,
                    fontSize: "0.8125rem",
                    color: P.textPrimary,
                    p: 1,
                    bgcolor: "rgba(0,0,0,0.3)",
                    borderRadius: "6px",
                    border: `1px solid ${P.border}`,
                  }}
                >
                  POST /api/{model.id}/predict
                </Box>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => copyEndpoint(model.id)}
                  sx={btnOutlined}
                >
                  {copied ? "Copied!" : "Copy URL"}
                </Button>
              </Box>
              <details style={{ marginTop: 12 }}>
                <summary
                  style={{
                    cursor: "pointer",
                    fontSize: "0.75rem",
                    fontFamily: "'Outfit', sans-serif",
                    color: P.textMuted,
                  }}
                >
                  cURL Example
                </summary>
                <Box
                  component="pre"
                  sx={{
                    fontFamily: P.fontMono,
                    fontSize: "0.75rem",
                    color: P.textSecondary,
                    p: 1.5,
                    bgcolor: "rgba(0,0,0,0.3)",
                    borderRadius: "6px",
                    border: `1px solid ${P.border}`,
                    overflow: "auto",
                    mt: 1,
                  }}
                >
                  {`curl -X POST -F "file=@image.jpg" \\\n  http://localhost:8080/api/${model.id}/predict`}
                </Box>
              </details>
            </Box>
            <Box sx={{ mt: 2 }}>
              <Typography
                sx={{
                  fontFamily: P.fontBody,
                  fontSize: "0.8125rem",
                  color: P.textSecondary,
                  mb: 1,
                  lineHeight: 1.5,
                }}
              >
                Run this model on a small EC2 instance with its own URL. Requires AWS credentials in Settings.
              </Typography>
              <Button variant="contained" onClick={() => setDeployModalOpen(true)} sx={btnPrimary}>
                Deploy to API
              </Button>
              <Typography
                sx={{
                  fontFamily: P.fontBody,
                  fontSize: "0.75rem",
                  color: P.textMuted,
                  display: "block",
                  mt: 1,
                }}
              >
                Requires AWS credentials (optional).{" "}
                <Link href="/settings" style={{ color: P.textSecondary, textDecoration: "underline" }}>
                  Settings
                </Link>
              </Typography>
            </Box>
          </>
        )}

        {/* Action buttons */}
        <Box
          sx={{
            mt: 3,
            pt: 2,
            borderTop: `1px solid ${P.border}`,
            display: "flex",
            flexWrap: "wrap",
            gap: 1,
          }}
        >
          {model.status === "completed" && (
            <>
              <Button
                variant="contained"
                component={Link}
                href="/predict"
                sx={{ ...btnPrimary, textDecoration: "none" }}
              >
                Predict
              </Button>
              <ShareCard model={model} />
              <Button variant="outlined" disabled sx={{ ...btnOutlined, "&.Mui-disabled": { color: P.textMuted, borderColor: "rgba(39,39,42,0.5)" } }}>
                Export ONNX
              </Button>
              <Button
                variant="outlined"
                size="small"
                onClick={() =>
                  document.getElementById("training-history-table")?.scrollIntoView({ behavior: "smooth" })
                }
                sx={btnOutlined}
              >
                View Training Curves
              </Button>
            </>
          )}
          {(model.status === "failed" || model.status === "cancelled") && (
            <Button variant="contained" onClick={() => handleResume(model.id)} sx={btnPrimary}>
              Resume Training
            </Button>
          )}
          <Button variant="outlined" color="error" onClick={() => setDeleteConfirm(model.id)} sx={btnDanger}>
            Delete
          </Button>
        </Box>
      </Box>
    );
  }

  /* ── Loading state ──────────────────────────────────────── */

  if (modelList.loading) {
    return (
      <Box sx={{ py: 6, textAlign: "center" }}>
        <Typography
          sx={{
            fontFamily: P.fontBody,
            fontSize: "0.875rem",
            color: P.textMuted,
          }}
        >
          Loading models...
        </Typography>
      </Box>
    );
  }

  /* ── Main render ────────────────────────────────────────── */

  return (
    <Box sx={{ px: { xs: 2, sm: 3 }, py: 3, maxWidth: 960, mx: "auto", width: "100%" }}>
      <ModelFilters
        modelSearch={modelList.modelSearch}
        onSearchChange={modelList.setModelSearch}
        onRefresh={modelList.loadModels}
      />

      {modelList.error && (
        <Box
          sx={{
            borderLeft: `3px solid ${P.error}`,
            bgcolor: "rgba(239,68,68,0.05)",
            borderRadius: "8px",
            p: 1.5,
            mb: 2.5,
            fontFamily: P.fontBody,
            fontSize: "0.875rem",
            color: P.error,
          }}
        >
          {modelList.error}
        </Box>
      )}

      {modelList.models.length === 0 ? (
        <Box
          sx={{
            py: 8,
            textAlign: "center",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 1,
          }}
        >
          <Typography
            sx={{
              fontFamily: P.fontBody,
              fontSize: "0.9375rem",
              color: P.textSecondary,
            }}
          >
            No models yet
          </Typography>
          <Typography
            sx={{
              fontFamily: P.fontBody,
              fontSize: "0.8125rem",
              color: P.textMuted,
            }}
          >
            Train your first model in the Train tab to get started.
          </Typography>
        </Box>
      ) : (
        <ModelGrid
          models={modelList.sortedFilteredModels}
          expandedModelId={expandedModelId}
          sortColumn={modelList.sortColumn}
          sortDirection={modelList.sortDirection}
          onSortClick={modelList.handleSortClick}
          onRowClick={handleRowClick}
          renderExpandedContent={renderExpandedContent}
        />
      )}

      {/* ── Deploy dialog ──────────────────────────────────── */}
      <Dialog
        open={deployModalOpen && !!selectedModel}
        onClose={() => !deploy.deploying && setDeployModalOpen(false)}
        PaperProps={{
          sx: {
            bgcolor: P.surface,
            border: `1px solid ${P.border}`,
            borderRadius: "12px",
            maxWidth: 440,
            backgroundImage: "none",
          },
        }}
      >
        <DialogTitle
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            fontFamily: P.fontBody,
            fontSize: "1rem",
            fontWeight: 600,
            color: P.textPrimary,
            borderBottom: `1px solid ${P.border}`,
            py: 1.5,
          }}
        >
          Deploy to API
          <Button
            size="small"
            onClick={() => !deploy.deploying && setDeployModalOpen(false)}
            disabled={deploy.deploying}
            sx={{
              minWidth: 32,
              color: P.textMuted,
              fontSize: "1.25rem",
              "&:hover": { color: P.textPrimary },
            }}
          >
            {"\u00d7"}
          </Button>
        </DialogTitle>
        <DialogContent sx={{ pt: "16px !important" }}>
          {selectedModel && (
            <>
              <Typography
                sx={{
                  fontFamily: P.fontBody,
                  fontSize: "0.875rem",
                  color: P.textSecondary,
                  mb: 1.5,
                }}
              >
                {formatModelName(selectedModel.name)}
              </Typography>
              {deploy.deployError && (
                <Box
                  sx={{
                    borderLeft: `3px solid ${P.error}`,
                    bgcolor: "rgba(239,68,68,0.05)",
                    color: P.error,
                    p: 1.25,
                    borderRadius: "8px",
                    mb: 1.5,
                    fontFamily: P.fontBody,
                    fontSize: "0.875rem",
                  }}
                >
                  {deploy.deployError}
                </Box>
              )}
              {deploy.deployments.some((d) => d.status === "live") ? (
                <Box>
                  <Box
                    component="span"
                    sx={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "6px",
                      fontFamily: P.fontMono,
                      fontSize: "0.6875rem",
                      fontWeight: 500,
                      textTransform: "uppercase",
                      color: P.success,
                      bgcolor: "rgba(34,197,94,0.10)",
                      borderRadius: "4px",
                      px: 1,
                      py: 0.5,
                      mb: 1.5,
                    }}
                  >
                    <Box
                      component="span"
                      sx={{
                        width: 6,
                        height: 6,
                        borderRadius: "50%",
                        bgcolor: P.success,
                      }}
                    />
                    Live
                  </Box>
                  {deploy.deployments
                    .filter((d) => d.status === "live")
                    .map((d) => (
                      <Box
                        key={d.id}
                        sx={{
                          display: "flex",
                          alignItems: "center",
                          gap: 0.75,
                          flexWrap: "wrap",
                          mb: 1,
                        }}
                      >
                        <Box
                          component="code"
                          sx={{
                            flex: 1,
                            minWidth: 0,
                            fontFamily: P.fontMono,
                            fontSize: "0.75rem",
                            color: P.textPrimary,
                            p: 1,
                            bgcolor: "rgba(0,0,0,0.3)",
                            borderRadius: "6px",
                            border: `1px solid ${P.border}`,
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                          }}
                        >
                          {d.endpoint_url}/predict
                        </Box>
                        <Button
                          size="small"
                          onClick={() =>
                            d.endpoint_url && deploy.copyDeployEndpoint(`${d.endpoint_url}/predict`)
                          }
                          sx={{ ...btnOutlined, px: 1.5, py: 0.5 }}
                        >
                          {deploy.copied ? "Copied!" : "Copy"}
                        </Button>
                        <Button
                          size="small"
                          onClick={() => deploy.handleDeployTerminate(d.id)}
                          sx={{ ...btnDanger, px: 1.5, py: 0.5 }}
                        >
                          Undeploy
                        </Button>
                      </Box>
                    ))}
                  <Typography
                    sx={{
                      fontFamily: P.fontBody,
                      fontSize: "0.75rem",
                      color: P.textMuted,
                      mt: 1,
                    }}
                  >
                    POST image file as <code style={{ fontFamily: P.fontMono }}>file</code> to the URL above.
                  </Typography>
                </Box>
              ) : deploy.deploying || deploy.deploymentPollId ? (
                <Box>
                  <Typography
                    sx={{
                      fontFamily: P.fontBody,
                      fontSize: "0.875rem",
                      fontWeight: 600,
                      color: P.textPrimary,
                      mb: 0.5,
                    }}
                  >
                    Launching instance...
                  </Typography>
                  <Typography
                    sx={{
                      fontFamily: P.fontBody,
                      fontSize: "0.8125rem",
                      color: P.textSecondary,
                      lineHeight: 1.5,
                    }}
                  >
                    This usually takes 1-2 minutes. The page will update when the endpoint is ready.
                  </Typography>
                </Box>
              ) : (
                <Box sx={{ pt: 0.5 }}>
                  <FormControl
                    fullWidth
                    size="small"
                    sx={{
                      mb: 1.5,
                      "& .MuiOutlinedInput-root": {
                        fontFamily: P.fontMono,
                        fontSize: "0.8125rem",
                        color: P.textPrimary,
                        borderRadius: "8px",
                        "& fieldset": { borderColor: P.border },
                        "&:hover fieldset": { borderColor: P.borderHover },
                        "&.Mui-focused fieldset": { borderColor: P.accent, borderWidth: "1px" },
                      },
                      "& .MuiInputLabel-root": {
                        fontFamily: P.fontBody,
                        fontSize: "0.8125rem",
                        color: P.textMuted,
                      },
                    }}
                  >
                    <InputLabel>Region</InputLabel>
                    <Select
                      value={deploy.deployRegion}
                      label="Region"
                      onChange={(e) => deploy.setDeployRegion(e.target.value)}
                      MenuProps={{
                        PaperProps: {
                          sx: {
                            bgcolor: P.surface,
                            border: `1px solid ${P.border}`,
                            borderRadius: "8px",
                            "& .MuiMenuItem-root": {
                              fontFamily: P.fontMono,
                              fontSize: "0.8125rem",
                              color: P.textSecondary,
                              "&:hover": { bgcolor: "rgba(255,255,255,0.04)" },
                              "&.Mui-selected": { bgcolor: P.accentSubtle, color: P.accent },
                            },
                          },
                        },
                      }}
                    >
                      <MenuItem value="us-east-1">us-east-1</MenuItem>
                      <MenuItem value="us-west-2">us-west-2</MenuItem>
                    </Select>
                  </FormControl>
                  <Button
                    variant="contained"
                    onClick={deploy.handleDeployStart}
                    disabled={!deploy.token}
                    fullWidth
                    sx={btnPrimary}
                  >
                    Deploy to EC2
                  </Button>
                </Box>
              )}
            </>
          )}
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        isOpen={deleteConfirm !== null}
        title="Delete Model"
        message="Are you sure you want to delete this model? This action cannot be undone."
        confirmLabel="Delete"
        cancelLabel="Cancel"
        variant="danger"
        onConfirm={() => deleteConfirm && confirmDelete(deleteConfirm)}
        onCancel={() => setDeleteConfirm(null)}
      />

      <Toast toasts={deploy.toast.toasts} onDismiss={deploy.toast.dismissToast} />
    </Box>
  );
}
