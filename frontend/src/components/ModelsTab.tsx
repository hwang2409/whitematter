"use client";
import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import * as api from "@/api";
import { useAuth } from "@/context/AuthContext";
import * as deployService from "@/services/deploy";
import ConfirmDialog from "./ConfirmDialog";
import Toast, { useToast } from "./Toast";
import ShareCard from "./ShareCard";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Paper from "@mui/material/Paper";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
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

interface Props {
  onModelSelect?: (id: string | null) => void;
  onUpdate?: () => void;
}

const DEPLOY_POLL_INTERVAL_MS = 3000;

function getStatusColor(status: string): "success" | "error" | "default" | "warning" {
  switch (status) {
    case "completed":
      return "success";
    case "running":
      return "warning";
    case "failed":
      return "error";
    case "cancelled":
      return "default";
    default:
      return "default";
  }
}

export default function ModelsTab({ onModelSelect, onUpdate }: Props) {
  const { token } = useAuth();
  const [models, setModels] = useState<api.Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<api.Model | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  const [prompt, setPrompt] = useState("");
  const [generatedText, setGeneratedText] = useState("");
  const [generating, setGenerating] = useState(false);
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(100);

  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [deployModalOpen, setDeployModalOpen] = useState(false);
  const [deployRegion, setDeployRegion] = useState("us-east-1");
  const [deploying, setDeploying] = useState(false);
  const [deploymentPollId, setDeploymentPollId] = useState<string | null>(null);
  const [deployments, setDeployments] = useState<deployService.Deployment[]>([]);
  const [deployError, setDeployError] = useState("");
  const toast = useToast();

  useEffect(() => {
    loadModels();
  }, []);

  const loadDeployments = useCallback(async () => {
    if (!token || !selectedModel) return;
    try {
      const list = await deployService.listDeployments(token, selectedModel.id);
      setDeployments(list);
    } catch {
      setDeployments([]);
    }
  }, [token, selectedModel]);

  useEffect(() => {
    if (deployModalOpen && selectedModel && token) {
      loadDeployments();
      setDeployError("");
    }
  }, [deployModalOpen, selectedModel?.id, token, loadDeployments]);

  useEffect(() => {
    if (!deploymentPollId || !token) return;
    const t = setInterval(async () => {
      try {
        const d = await deployService.getDeployment(token, deploymentPollId);
        setDeployments((prev) => {
          const idx = prev.findIndex((x) => x.id === d.id);
          return idx >= 0 ? [...prev.slice(0, idx), d, ...prev.slice(idx + 1)] : [d, ...prev];
        });
        if (d.status === "live" || d.status === "failed") {
          setDeploymentPollId(null);
          setDeploying(false);
          if (d.status === "live") toast.success("Deployment live! Your API is ready.");
          if (d.status === "failed") toast.error(d.error_message || "Deployment failed.");
        }
      } catch {
        setDeploymentPollId(null);
        setDeploying(false);
      }
    }, DEPLOY_POLL_INTERVAL_MS);
    return () => clearInterval(t);
  }, [deploymentPollId, token, toast]);

  async function handleDeployStart() {
    if (!token || !selectedModel) return;
    setDeploying(true);
    setDeployError("");
    try {
      const res = await deployService.createDeployment(token, {
        model_id: selectedModel.id,
        region: deployRegion,
      });
      setDeploymentPollId(res.deployment_id);
      setDeployments((prev) => [
        {
          id: res.deployment_id,
          model_id: selectedModel.id,
          target_type: "ec2",
          status: res.status,
          instance_id: null,
          endpoint_url: null,
          region: deployRegion,
          error_message: null,
          created_at: null,
          updated_at: null,
        },
        ...prev,
      ]);
    } catch (e: unknown) {
      setDeploying(false);
      setDeployError(e instanceof Error ? e.message : "Failed to start deployment");
    }
  }

  async function handleDeployTerminate(deploymentId: string) {
    if (!token) return;
    try {
      await deployService.terminateDeployment(token, deploymentId);
      await loadDeployments();
      toast.success("Deployment terminated.");
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Failed to terminate");
    }
  }

  async function copyDeployEndpoint(url: string) {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast.success("Endpoint URL copied.");
    } catch {
      toast.error("Failed to copy");
    }
  }

  async function loadModels() {
    setLoading(true);
    try {
      const data = await api.getModels();
      setModels(data);
      onUpdate?.();
    } catch {
      setError("Failed to load models");
    } finally {
      setLoading(false);
    }
  }

  function handleSelectModel(model: api.Model) {
    setSelectedModel(model);
    onModelSelect?.(model.id);
  }

  async function confirmDelete(id: string) {
    try {
      await api.deleteModel(id);
      setModels(models.filter((m) => m.id !== id));
      if (selectedModel?.id === id) setSelectedModel(null);
    } catch {
      setError("Failed to delete model");
    } finally {
      setDeleteConfirm(null);
    }
  }

  async function handleResume(id: string) {
    try {
      setError("");
      await api.resumeTraining(id);
      setModels(models.map((m) => (m.id === id ? { ...m, status: "running" as const } : m)));
      if (selectedModel?.id === id) {
        setSelectedModel({ ...selectedModel, status: "running" });
      }
      toast.success("Resuming training. Check the Train tab for progress.");
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Failed to resume training");
    }
  }

  function formatDate(dateStr: string) {
    return new Date(dateStr).toLocaleString();
  }

  function formatModelName(name: string) {
    const parts = name.replace("custom_", "").split("_");
    if (parts.length > 1) {
      const baseName = parts.slice(0, -1).join("_");
      return baseName
        .split(/[-_]/)
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(" ");
    }
    return name;
  }

  async function copyEndpoint(modelId: string) {
    const url = `http://localhost:8080/api/${modelId}/predict`;
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast.success("URL copied.");
    } catch {
      toast.error("Failed to copy");
    }
  }

  function isTextModel(model: api.Model) {
    const name = model.name.toLowerCase();
    const arch = model.architecture.toLowerCase();
    return (
      model.dataset.startsWith("custom:") &&
      (name.includes("gpt") ||
        name.includes("language") ||
        name.includes("text") ||
        arch.includes("gpt") ||
        arch.includes("language") ||
        arch.includes("text") ||
        arch.includes("lstm"))
    );
  }

  async function handleGenerate() {
    if (!selectedModel || !prompt.trim()) return;
    setGenerating(true);
    setError("");
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
      setError(e instanceof Error ? e.message : "Failed to generate text");
    } finally {
      setGenerating(false);
    }
  }

  if (loading) {
    return (
      <Box sx={{ py: 3 }}>
        <Typography color="text.secondary">Loading models...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
        <Typography variant="h2">Trained Models</Typography>
        <Button variant="outlined" size="small" onClick={loadModels}>
          Refresh
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
            mb: 2,
            fontSize: "0.875rem",
          }}
        >
          {error}
        </Box>
      )}

      {models.length === 0 ? (
        <Typography color="text.secondary" sx={{ py: 4, textAlign: "center" }}>
          No models yet. Train one in the Train tab!
        </Typography>
      ) : (
        <Box sx={{ display: "grid", gridTemplateColumns: "minmax(280px, 320px) 1fr", gap: 2, alignItems: "start" }}>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 1, maxHeight: "min(60vh, 600px)", overflowY: "auto" }}>
            {models.map((model) => (
              <Paper
                key={model.id}
                variant="outlined"
                onClick={() => handleSelectModel(model)}
                sx={{
                  p: 1.25,
                  cursor: "pointer",
                  borderColor: selectedModel?.id === model.id ? "primary.main" : "divider",
                  "&:hover": { borderColor: "primary.main" },
                }}
              >
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 0.5, mb: 0.75 }}>
                  <Typography variant="subtitle2" sx={{ wordBreak: "break-word" }}>
                    {formatModelName(model.name)}
                  </Typography>
                  <Chip size="small" label={model.status} color={getStatusColor(model.status)} sx={{ textTransform: "uppercase", fontSize: "0.6875rem" }} />
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ fontSize: "0.75rem" }}>
                  {model.dataset.startsWith("custom:") ? "Custom Dataset" : model.dataset}
                </Typography>
                <Box sx={{ display: "flex", gap: 1, mt: 0.75, fontSize: "0.8125rem", color: "text.secondary" }}>
                  <span>{model.epochs_trained} epochs</span>
                  <span>{model.best_accuracy.toFixed(2)}% acc</span>
                </Box>
              </Paper>
            ))}
          </Box>

          {selectedModel && (
            <Paper
              variant="outlined"
              sx={{
                p: 3,
                borderColor: "divider",
                minWidth: 0,
                borderBottom: "2px solid",
                borderBottomColor: "primary.main",
              }}
            >
              <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 2, mb: 2 }}>
                <Typography variant="h2" sx={{ fontSize: "1.5rem", fontWeight: 700 }}>
                  {formatModelName(selectedModel.name)}
                </Typography>
                {selectedModel.status === "completed" && (
                  <Typography
                    sx={{
                      fontFamily: '"JetBrains Mono", monospace',
                      fontSize: "2.5rem",
                      fontWeight: 700,
                      color: "primary.main",
                      lineHeight: 1,
                    }}
                  >
                    {selectedModel.best_accuracy.toFixed(1)}%
                  </Typography>
                )}
              </Box>

              <Box sx={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 0.5, mb: 2 }}>
                {selectedModel.architecture
                  .replace(/_/g, " ")
                  .split(/[\s,-]+/)
                  .filter(Boolean)
                  .map((part, i) => (
                    <Box key={i} sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                      <Chip
                        size="small"
                        label={part}
                        sx={{
                          fontFamily: '"JetBrains Mono", monospace',
                          fontSize: "0.6875rem",
                          bgcolor: "action.hover",
                          border: "1px solid",
                          borderColor: "divider",
                        }}
                      />
                      {i < selectedModel.architecture.replace(/_/g, " ").split(/[\s,-]+/).filter(Boolean).length - 1 && (
                        <Typography component="span" color="text.secondary" sx={{ fontSize: "0.75rem" }}>
                          →
                        </Typography>
                      )}
                    </Box>
                  ))}
              </Box>

              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2, mb: 2, color: "text.secondary", fontSize: "0.8125rem" }}>
                <span>{selectedModel.epochs_trained} epochs</span>
                <span>·</span>
                <span>{selectedModel.training_history?.[selectedModel.training_history.length - 1]?.loss != null ? `${selectedModel.training_history[selectedModel.training_history.length - 1].loss.toFixed(4)} final loss` : "—"}</span>
                <span>·</span>
                <span>{selectedModel.dataset.startsWith("custom:") ? "Custom dataset" : selectedModel.dataset}</span>
                <span>·</span>
                <span>{formatDate(selectedModel.created_at)}</span>
              </Box>

              {selectedModel.training_history.length > 0 && (
                <Box id="training-history-table" sx={{ mb: 1.5 }}>
                  <Typography variant="overline" sx={{ color: "text.secondary" }}>
                    Training History
                  </Typography>
                  <Table size="small" sx={{ mt: 0.5 }}>
                    <TableHead>
                      <TableRow>
                        <TableCell>Epoch</TableCell>
                        <TableCell>Loss</TableCell>
                        <TableCell>Accuracy</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {selectedModel.training_history.map((h) => (
                        <TableRow key={h.epoch}>
                          <TableCell>{h.epoch}</TableCell>
                          <TableCell sx={{ fontFamily: '"JetBrains Mono", monospace' }}>{h.loss.toFixed(4)}</TableCell>
                          <TableCell sx={{ fontFamily: '"JetBrains Mono", monospace' }}>{h.accuracy.toFixed(2)}%</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </Box>
              )}

              {selectedModel.status === "completed" && isTextModel(selectedModel) && (
                <Box sx={{ mt: 2, p: 2, bgcolor: "background.default", borderRadius: 1, border: "1px solid", borderColor: "divider" }}>
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>
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
                    sx={{ mb: 1 }}
                  />
                  <Box sx={{ display: "flex", gap: 2, alignItems: "center", mb: 1 }}>
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="caption">Temperature: {temperature}</Typography>
                      <Slider
                        size="small"
                        min={0.1}
                        max={1.5}
                        step={0.1}
                        value={temperature}
                        onChange={(_, v) => setTemperature(v as number)}
                        disabled={generating}
                        valueLabelDisplay="auto"
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
                      sx={{ width: 100 }}
                    />
                  </Box>
                  <Button variant="contained" onClick={handleGenerate} disabled={generating || !prompt.trim()}>
                    {generating ? "Generating..." : "Generate"}
                  </Button>
                  {generatedText && (
                    <Box sx={{ mt: 1.5, p: 1, bgcolor: "background.paper", borderRadius: 1, border: "1px solid", borderColor: "divider" }}>
                      <Typography variant="caption" color="text.secondary">
                        Generated Text
                      </Typography>
                      <Box component="pre" sx={{ fontFamily: '"JetBrains Mono", monospace', fontSize: "0.875rem", whiteSpace: "pre-wrap", mt: 0.5 }}>
                        {generatedText}
                      </Box>
                    </Box>
                  )}
                </Box>
              )}

              {selectedModel.status === "completed" && !isTextModel(selectedModel) && (
                <>
                  <Box sx={{ mt: 1.5, pt: 1.5, borderTop: "1px solid", borderColor: "divider" }}>
                    <Typography variant="overline" sx={{ color: "text.secondary" }}>
                      API Endpoint
                    </Typography>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.5 }}>
                      <Box
                        component="code"
                        sx={{
                          flex: 1,
                          fontSize: "0.8125rem",
                          fontFamily: '"JetBrains Mono", monospace',
                          p: 0.75,
                          bgcolor: "action.hover",
                          borderRadius: 1,
                        }}
                      >
                        POST /api/{selectedModel.id}/predict
                      </Box>
                      <Button size="small" variant="outlined" onClick={() => copyEndpoint(selectedModel.id)}>
                        {copied ? "Copied!" : "Copy URL"}
                      </Button>
                    </Box>
                    <details style={{ marginTop: 8 }}>
                      <summary style={{ cursor: "pointer", fontSize: "0.75rem", color: "rgba(255,255,255,0.5)" }}>
                        cURL Example
                      </summary>
                      <Box
                        component="pre"
                        sx={{
                          fontFamily: '"JetBrains Mono", monospace',
                          fontSize: "0.75rem",
                          p: 1,
                          bgcolor: "action.hover",
                          borderRadius: 1,
                          mt: 0.5,
                          overflow: "auto",
                        }}
                      >
                        {`curl -X POST -F "file=@image.jpg" \\\n  http://localhost:8080/api/${selectedModel.id}/predict`}
                      </Box>
                    </details>
                  </Box>
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                      Run this model on a small EC2 instance with its own URL. Requires AWS credentials in Settings.
                    </Typography>
                    <Button variant="contained" onClick={() => setDeployModalOpen(true)}>
                      Deploy to API
                    </Button>
                    <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
                      Requires AWS credentials (optional).{" "}
                      <Link href="/settings" style={{ color: "inherit", textDecoration: "underline" }}>
                        Settings
                      </Link>
                    </Typography>
                  </Box>
                </>
              )}

              <Box sx={{ mt: 2, pt: 1.5, borderTop: "1px solid", borderColor: "divider", display: "flex", flexWrap: "wrap", gap: 1 }}>
                {selectedModel.status === "completed" && (
                  <>
                    <Button variant="outlined" component={Link} href="/predict" sx={{ textDecoration: "none" }}>
                      Predict
                    </Button>
                    <ShareCard model={selectedModel} />
                    <Button variant="outlined" disabled sx={{ color: "text.secondary" }}>
                      Export ONNX
                    </Button>
                    <Button variant="outlined" size="small" onClick={() => document.getElementById("training-history-table")?.scrollIntoView({ behavior: "smooth" })}>
                      View Training Curves
                    </Button>
                  </>
                )}
                {(selectedModel.status === "failed" || selectedModel.status === "cancelled") && (
                  <Button variant="contained" onClick={() => handleResume(selectedModel.id)}>
                    Resume Training
                  </Button>
                )}
                <Button variant="outlined" color="error" onClick={() => setDeleteConfirm(selectedModel.id)}>
                  Delete
                </Button>
              </Box>
            </Paper>
          )}
        </Box>
      )}

      <Dialog
        open={deployModalOpen && !!selectedModel}
        onClose={() => !deploying && setDeployModalOpen(false)}
        PaperProps={{
          sx: {
            bgcolor: "background.paper",
            border: "1px solid",
            borderColor: "divider",
            maxWidth: 440,
          },
        }}
      >
        <DialogTitle sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          Deploy to API
          <Button
            size="small"
            onClick={() => !deploying && setDeployModalOpen(false)}
            disabled={deploying}
            sx={{ minWidth: 32 }}
          >
            ×
          </Button>
        </DialogTitle>
        <DialogContent>
          {selectedModel && (
            <>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                {formatModelName(selectedModel.name)}
              </Typography>
              {deployError && (
                <Box
                  sx={{
                    border: "1px solid",
                    borderColor: "error.main",
                    color: "error.main",
                    p: 1,
                    borderRadius: 1,
                    mb: 1,
                    fontSize: "0.875rem",
                  }}
                >
                  {deployError}
                </Box>
              )}
              {deployments.some((d) => d.status === "live") ? (
                <Box>
                  <Chip size="small" label="Live" color="success" sx={{ mb: 1 }} />
                  {deployments
                    .filter((d) => d.status === "live")
                    .map((d) => (
                      <Box key={d.id} sx={{ display: "flex", alignItems: "center", gap: 0.5, flexWrap: "wrap", mb: 0.5 }}>
                        <Box
                          component="code"
                          sx={{
                            flex: 1,
                            minWidth: 0,
                            fontSize: "0.8rem",
                            p: 0.5,
                            bgcolor: "action.hover",
                            borderRadius: 1,
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                          }}
                        >
                          {d.endpoint_url}/predict
                        </Box>
                        <Button size="small" onClick={() => d.endpoint_url && copyDeployEndpoint(`${d.endpoint_url}/predict`)}>
                          {copied ? "Copied!" : "Copy"}
                        </Button>
                        <Button size="small" color="error" onClick={() => handleDeployTerminate(d.id)}>
                          Undeploy
                        </Button>
                      </Box>
                    ))}
                  <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                    POST image file as <code>file</code> to the URL above.
                  </Typography>
                </Box>
              ) : deploying || deploymentPollId ? (
                <Box>
                  <Typography variant="body2" fontWeight={600} sx={{ mb: 0.5 }}>
                    Launching instance…
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    This usually takes 1–2 minutes. The page will update when the endpoint is ready.
                  </Typography>
                </Box>
              ) : (
                <Box sx={{ pt: 1 }}>
                  <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                    <InputLabel>Region</InputLabel>
                    <Select
                      value={deployRegion}
                      label="Region"
                      onChange={(e) => setDeployRegion(e.target.value)}
                    >
                      <MenuItem value="us-east-1">us-east-1</MenuItem>
                      <MenuItem value="us-west-2">us-west-2</MenuItem>
                    </Select>
                  </FormControl>
                  <Button variant="contained" onClick={handleDeployStart} disabled={!token} fullWidth>
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

      <Toast toasts={toast.toasts} onDismiss={toast.dismissToast} />
    </Box>
  );
}

function DetailItem({
  label,
  value,
  statusColor,
}: {
  label: string;
  value: string;
  statusColor?: "success" | "error" | "default" | "warning";
}) {
  return (
    <Box>
      <Typography variant="caption" sx={{ color: "text.secondary", textTransform: "uppercase", letterSpacing: "0.08em" }}>
        {label}
      </Typography>
      <Typography variant="body2" fontWeight={500} color={statusColor && statusColor !== "default" ? statusColor : "text.primary"}>
        {value}
      </Typography>
    </Box>
  );
}
