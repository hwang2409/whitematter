"use client";
import { useState, useEffect, useRef } from "react";
import * as api from "@/api";
import ConfirmDialog from "./ConfirmDialog";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import Paper from "@mui/material/Paper";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import FormHelperText from "@mui/material/FormHelperText";
import Chip from "@mui/material/Chip";
import { themeTokens } from "@/theme";

interface Props {
  onDatasetSelect?: (id: string | null) => void;
  onUpdate?: () => void;
}

function getStatusColor(status: string): "success" | "error" | "default" | "warning" {
  switch (status) {
    case "ready":
      return "success";
    case "processing":
      return "warning";
    case "error":
      return "error";
    default:
      return "default";
  }
}

export default function DatasetsTab({ onDatasetSelect, onUpdate }: Props) {
  const [datasets, setDatasets] = useState<api.CustomDataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<api.CustomDataset | null>(null);
  const [preview, setPreview] = useState<api.DatasetPreviewSample[]>([]);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadedDataset, setUploadedDataset] = useState<api.CustomDataset | null>(null);
  const [uploadPreview, setUploadPreview] = useState<api.DatasetPreviewSample[]>([]);
  const [error, setError] = useState("");
  const [datasetName, setDatasetName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [fileType, setFileType] = useState<"zip" | "text">("zip");
  const [tokenizerType, setTokenizerType] = useState<"character" | "word">("character");
  const [seqLength, setSeqLength] = useState(128);
  const [textPreview, setTextPreview] = useState<api.TextPreview | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [importUrl, setImportUrl] = useState("");
  const [importHfId, setImportHfId] = useState("");
  const [importName, setImportName] = useState("");
  const [importHfSplit, setImportHfSplit] = useState<"train" | "validation" | "test">("train");
  const [importing, setImporting] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  useEffect(() => {
    loadDatasets();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      loadPreview(selectedDataset.id);
    } else {
      setPreview([]);
    }
  }, [selectedDataset?.id]);

  async function loadPreview(datasetId: string) {
    setLoadingPreview(true);
    try {
      const data = await api.getDatasetPreview(datasetId);
      setPreview(data.samples || []);
      setTextPreview((data as { text_preview?: api.TextPreview }).text_preview ?? null);
    } catch {
      setPreview([]);
      setTextPreview(null);
    } finally {
      setLoadingPreview(false);
    }
  }

  async function loadDatasets() {
    setLoading(true);
    try {
      const data = await api.getCustomDatasets();
      setDatasets(data);
      onUpdate?.();
    } catch {
      setError("Failed to load datasets");
    } finally {
      setLoading(false);
    }
  }

  function handleSelectDataset(dataset: api.CustomDataset) {
    setSelectedDataset(dataset);
    onDatasetSelect?.(dataset.id);
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setError("");
      if (f.name.endsWith(".txt")) {
        setFileType("text");
        if (!datasetName) setDatasetName(f.name.replace(/\.txt$/i, ""));
      } else {
        setFileType("zip");
        if (!datasetName) setDatasetName(f.name.replace(/\.zip$/i, ""));
      }
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && (f.name.endsWith(".zip") || f.name.endsWith(".txt"))) {
      setFile(f);
      setError("");
      if (f.name.endsWith(".txt")) {
        setFileType("text");
        if (!datasetName) setDatasetName(f.name.replace(/\.txt$/i, ""));
      } else {
        setFileType("zip");
        if (!datasetName) setDatasetName(f.name.replace(/\.zip$/i, ""));
      }
    } else {
      setError("Please upload a ZIP or TXT file");
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
  }

  async function handleImportFromUrl() {
    const url = importUrl.trim();
    if (!url) {
      setError("Enter a URL");
      return;
    }
    setImporting(true);
    setError("");
    try {
      const dataset = await api.importDatasetFromUrl(url, importName.trim() || undefined);
      setDatasets((prev) => [dataset, ...prev]);
      setImportUrl("");
      setImportName("");
      setUploadedDataset(dataset);
      setSelectedDataset(dataset);
      try {
        const previewData = await api.getDatasetPreview(dataset.id);
        setUploadPreview(previewData.samples || []);
        if (dataset.data_type === "text" && (previewData as { text_preview?: api.TextPreview }).text_preview) {
          setTextPreview((previewData as { text_preview?: api.TextPreview }).text_preview ?? null);
        }
      } catch {
        setUploadPreview([]);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Import failed");
    } finally {
      setImporting(false);
    }
  }

  async function handleImportFromHf() {
    const id = importHfId.trim();
    if (!id) {
      setError("Enter a Hugging Face dataset ID (e.g. username/dataset-name)");
      return;
    }
    setImporting(true);
    setError("");
    try {
      const dataset = await api.importDatasetFromHuggingFace(id, {
        name: importName.trim() || undefined,
        split: importHfSplit,
      });
      setDatasets((prev) => [dataset, ...prev]);
      setImportHfId("");
      setImportName("");
      setUploadedDataset(dataset);
      setSelectedDataset(dataset);
      try {
        const previewData = await api.getDatasetPreview(dataset.id);
        setUploadPreview(previewData.samples || []);
        if (dataset.data_type === "text" && (previewData as { text_preview?: api.TextPreview }).text_preview) {
          setTextPreview((previewData as { text_preview?: api.TextPreview }).text_preview ?? null);
        }
      } catch {
        setUploadPreview([]);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Import failed");
    } finally {
      setImporting(false);
    }
  }

  async function handleUpload() {
    if (!file) {
      setError("Please select a file");
      return;
    }
    if (!datasetName.trim()) {
      setError("Please enter a dataset name");
      return;
    }
    setUploading(true);
    setError("");
    try {
      let dataset: api.CustomDataset;
      if (fileType === "text") {
        dataset = await api.uploadTextDataset(file, datasetName.trim(), {
          tokenizer_type: tokenizerType,
          seq_length: seqLength,
        });
      } else {
        dataset = await api.uploadDataset(file, datasetName.trim());
      }
      setDatasets((prev) => [dataset, ...prev]);
      setFile(null);
      setDatasetName("");
      setFileType("zip");
      if (fileInputRef.current) fileInputRef.current.value = "";
      setUploadedDataset(dataset);
      setSelectedDataset(dataset);
      try {
        const previewData = await api.getDatasetPreview(dataset.id);
        setUploadPreview(previewData.samples || []);
        if (dataset.data_type === "text" && (previewData as { text_preview?: api.TextPreview }).text_preview) {
          setTextPreview((previewData as { text_preview?: api.TextPreview }).text_preview ?? null);
        }
      } catch {
        setUploadPreview([]);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function confirmDelete(id: string) {
    try {
      await api.deleteCustomDataset(id);
      setDatasets(datasets.filter((d) => d.id !== id));
      if (selectedDataset?.id === id) setSelectedDataset(null);
    } catch {
      setError("Failed to delete dataset");
    } finally {
      setDeleteConfirm(null);
    }
  }

  function formatDate(dateStr: string) {
    return new Date(dateStr).toLocaleString();
  }

  if (loading) {
    return (
      <Box sx={{ py: 3 }}>
        <Typography color="text.secondary">Loading datasets...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h2" sx={{ mb: 2 }}>
        Datasets
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

      {/* Import from URL */}
      <Paper variant="outlined" sx={{ p: 2, mb: 2, borderColor: "divider" }}>
        <Typography variant="h3" sx={{ mb: 0.5 }}>
          Import from URL
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          Public HTTPS link to a ZIP (images) or TXT (text) file. Max 1 GB.
        </Typography>
        <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", alignItems: "center" }}>
          <TextField
            size="small"
            placeholder="https://example.com/dataset.zip"
            value={importUrl}
            onChange={(e) => setImportUrl(e.target.value)}
            disabled={importing}
            sx={{ flex: 1, minWidth: 200 }}
          />
          <TextField
            size="small"
            placeholder="Dataset name (optional)"
            value={importName}
            onChange={(e) => setImportName(e.target.value)}
            disabled={importing}
            sx={{ width: 180 }}
          />
          <Button
            variant="outlined"
            onClick={handleImportFromUrl}
            disabled={!importUrl.trim() || importing}
          >
            {importing ? "Importing..." : "Import"}
          </Button>
        </Box>
      </Paper>

      {/* Import from Hugging Face */}
      <Paper variant="outlined" sx={{ p: 2, mb: 2, borderColor: "divider" }}>
        <Typography variant="h3" sx={{ mb: 0.5 }}>
          Import from Hugging Face
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          Dataset ID from the Hub (e.g. <code>username/dataset-name</code>). Limits: 50k image rows, 100k text rows.
        </Typography>
        <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", alignItems: "center" }}>
          <TextField
            size="small"
            placeholder="username/dataset-name"
            value={importHfId}
            onChange={(e) => setImportHfId(e.target.value)}
            disabled={importing}
            sx={{ flex: 1, minWidth: 200 }}
          />
          <TextField
            size="small"
            placeholder="Dataset name (optional)"
            value={importName}
            onChange={(e) => setImportName(e.target.value)}
            disabled={importing}
            sx={{ width: 180 }}
          />
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Split</InputLabel>
            <Select
              value={importHfSplit}
              label="Split"
              onChange={(e) => setImportHfSplit(e.target.value as "train" | "validation" | "test")}
              disabled={importing}
            >
              <MenuItem value="train">train</MenuItem>
              <MenuItem value="validation">validation</MenuItem>
              <MenuItem value="test">test</MenuItem>
            </Select>
          </FormControl>
          <Button
            variant="outlined"
            onClick={handleImportFromHf}
            disabled={!importHfId.trim() || importing}
          >
            {importing ? "Importing..." : "Import"}
          </Button>
        </Box>
      </Paper>

      {/* Upload */}
      <Paper variant="outlined" sx={{ p: 2, mb: 2, borderColor: "divider" }}>
        <Typography variant="h3" sx={{ mb: 0.5 }}>
          Upload from device
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          Upload a ZIP file for images, or a TXT file for text/language model training.
        </Typography>

        <TextField
          fullWidth
          label="Dataset Name"
          value={datasetName}
          onChange={(e) => setDatasetName(e.target.value)}
          placeholder="my_dataset"
          disabled={uploading}
          sx={{ mb: 1.5 }}
        />

        <Box
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => !file && fileInputRef.current?.click()}
          sx={{
            border: "1px dashed",
            borderColor: file ? themeTokens.accentMuted : "divider",
            borderRadius: 1,
            p: file ? 2 : 3,
            textAlign: "center",
            cursor: file ? "default" : "pointer",
            bgcolor: "action.hover",
            mb: 1.5,
            "&:hover": { borderColor: themeTokens.accentMuted },
          }}
        >
          {file ? (
            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 1, flexWrap: "wrap" }}>
              <Typography fontWeight={500}>{file.name}</Typography>
              <Typography variant="body2" color="text.secondary">
                ({(file.size / 1024 / 1024).toFixed(2)} MB)
              </Typography>
              <Chip size="small" label={fileType === "text" ? "Text" : "Images"} sx={{ fontFamily: "inherit" }} />
              <Button
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  setFile(null);
                  setFileType("zip");
                  if (fileInputRef.current) fileInputRef.current.value = "";
                }}
                sx={{ minWidth: 24, height: 24 }}
              >
                ×
              </Button>
            </Box>
          ) : (
            <>
              <Typography variant="body2" color="text.secondary">
                Drop a file here or click to upload
              </Typography>
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                ZIP: folder-per-class, MNIST IDX | TXT: text for language models
              </Typography>
            </>
          )}
          <input ref={fileInputRef} type="file" accept=".zip,.txt" onChange={handleFileChange} hidden />
        </Box>

        {fileType === "text" && file && (
          <Box sx={{ mb: 1.5, p: 1.5, bgcolor: "action.hover", borderRadius: 1 }}>
            <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 1 }}>
              <FormControl size="small" fullWidth>
                <InputLabel>Tokenizer</InputLabel>
                <Select
                  value={tokenizerType}
                  label="Tokenizer"
                  onChange={(e) => setTokenizerType(e.target.value as "character" | "word")}
                  disabled={uploading}
                >
                  <MenuItem value="character">Character-level</MenuItem>
                  <MenuItem value="word">Word-level</MenuItem>
                </Select>
              </FormControl>
              <TextField
                size="small"
                type="number"
                label="Sequence Length"
                value={seqLength}
                onChange={(e) => setSeqLength(parseInt(e.target.value) || 128)}
                inputProps={{ min: 32, max: 512 }}
                disabled={uploading}
              />
            </Box>
            <FormHelperText>
              Character-level works well for small datasets. Word-level is better for larger corpora.
            </FormHelperText>
          </Box>
        )}

        <Button
          variant="contained"
          onClick={handleUpload}
          disabled={!file || !datasetName.trim() || uploading}
        >
          {uploading ? "Uploading..." : "Upload Dataset"}
        </Button>

        {uploadedDataset && (
          <Box sx={{ mt: 2, p: 2, bgcolor: "background.default", borderRadius: 1, border: "1px solid", borderColor: "divider" }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
              <Typography variant="subtitle1" color="success.main">
                Upload Successful: {uploadedDataset.name}
              </Typography>
              <Button size="small" onClick={() => { setUploadedDataset(null); setUploadPreview([]); }}>×</Button>
            </Box>
            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mb: 1, fontSize: "0.8125rem", color: "text.secondary" }}>
              <span>{uploadedDataset.data_type}</span>
              <span>{uploadedDataset.format?.replace(/_/g, " ")}</span>
              <span>{uploadedDataset.num_classes} classes</span>
              <span>{uploadedDataset.total_samples?.toLocaleString()} samples</span>
              {uploadedDataset.input_shape?.length > 0 && (
                <span>Shape: [{uploadedDataset.input_shape.join("x")}]</span>
              )}
            </Box>
            {uploadedDataset.class_names?.length > 0 && (
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mb: 1 }}>
                {uploadedDataset.class_names.map((name, i) => (
                  <Chip key={i} size="small" label={name} variant="outlined" />
                ))}
              </Box>
            )}
            {uploadedDataset.data_type === "text" && textPreview && (
              <Box sx={{ mt: 1, p: 1, bgcolor: "background.paper", borderRadius: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  Vocab: {textPreview.vocab_size} · Sequences: {textPreview.train_sequences} / {textPreview.test_sequences} · Total: {textPreview.total_tokens?.toLocaleString()} tokens
                </Typography>
                {textPreview.sample_text && (
                  <Box component="pre" sx={{ fontFamily: '"JetBrains Mono", monospace', fontSize: "0.8125rem", overflow: "auto", mt: 0.5 }}>
                    {textPreview.sample_text.slice(0, 300)}...
                  </Box>
                )}
                {textPreview.sample_tokens?.length > 0 && (
                  <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.25, mt: 0.5 }}>
                    {textPreview.sample_tokens.slice(0, 20).map((t, i) => (
                      <Chip key={i} size="small" label={t === " " ? "␣" : t === "\n" ? "↵" : t} sx={{ fontFamily: '"JetBrains Mono", monospace' }} />
                    ))}
                  </Box>
                )}
              </Box>
            )}
            {uploadPreview.length > 0 && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase" }}>
                  Sample Images
                </Typography>
                <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(80px, 1fr))", gap: 1, mt: 0.5 }}>
                  {uploadPreview.map((sample, i) => (
                    <Box key={i} sx={{ textAlign: "center", bgcolor: "background.paper", borderRadius: 1, overflow: "hidden" }}>
                      <Box component="img" src={`data:image/png;base64,${sample.image}`} alt={sample.label} sx={{ width: "100%", height: 80, objectFit: "contain" }} />
                      <Typography variant="caption" sx={{ display: "block", p: 0.5, fontFamily: '"JetBrains Mono", monospace' }}>
                        {sample.label}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </Box>
            )}
          </Box>
        )}
      </Paper>

      {/* Your Datasets */}
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1.5 }}>
        <Typography variant="h3">Your Datasets</Typography>
        <Button variant="outlined" size="small" onClick={loadDatasets}>
          Refresh
        </Button>
      </Box>

      {datasets.length === 0 ? (
        <Typography color="text.secondary" sx={{ py: 4, textAlign: "center" }}>
          No datasets yet. Upload one above!
        </Typography>
      ) : (
        <Box sx={{ display: "grid", gridTemplateColumns: "minmax(280px, 320px) 1fr", gap: 2, alignItems: "start" }}>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 1, maxHeight: "min(60vh, 600px)", overflowY: "auto" }}>
            {datasets.map((dataset) => (
              <Paper
                key={dataset.id}
                variant="outlined"
                onClick={() => handleSelectDataset(dataset)}
                sx={{
                  p: 1.25,
                  cursor: "pointer",
                  borderColor: selectedDataset?.id === dataset.id ? "primary.main" : "divider",
                  "&:hover": { borderColor: "primary.main" },
                }}
              >
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 0.5, mb: 0.75 }}>
                  <Typography variant="subtitle2" sx={{ wordBreak: "break-word" }}>
                    {dataset.name}
                  </Typography>
                  <Chip size="small" label={dataset.status} color={getStatusColor(dataset.status)} sx={{ textTransform: "uppercase", fontSize: "0.6875rem" }} />
                </Box>
                <Box sx={{ fontSize: "0.75rem", color: "text.secondary" }}>
                  {dataset.data_type} · {dataset.num_classes} classes · {dataset.total_samples} samples
                </Box>
              </Paper>
            ))}
          </Box>

          {selectedDataset && (
            <Paper variant="outlined" sx={{ p: 2, borderColor: "divider", minWidth: 0 }}>
              <Typography variant="h3" sx={{ mb: 1.5 }}>
                {selectedDataset.name}
              </Typography>
              <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1.5, mb: 1.5 }}>
                <DetailItem label="ID" value={selectedDataset.id} />
                <DetailItem label="Type" value={selectedDataset.data_type} />
                <DetailItem label="Format" value={selectedDataset.format?.replace(/_/g, " ") || "unknown"} />
                {selectedDataset.input_shape?.length > 0 && (
                  <DetailItem label="Input Shape" value={`[${selectedDataset.input_shape.join("x")}]`} />
                )}
                <DetailItem label="Classes" value={String(selectedDataset.num_classes)} />
                <DetailItem label="Total Samples" value={selectedDataset.total_samples.toLocaleString()} />
                <DetailItem label="Created" value={formatDate(selectedDataset.created_at)} />
                <DetailItem label="Status" value={selectedDataset.status} statusColor={getStatusColor(selectedDataset.status)} />
              </Box>
              {selectedDataset.class_names?.length > 0 && (
                <Box sx={{ mb: 1.5 }}>
                  <Typography variant="overline" sx={{ color: "text.secondary" }}>Classes</Typography>
                  <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mt: 0.5 }}>
                    {selectedDataset.class_names.map((name, i) => (
                      <Chip key={i} size="small" label={name} variant="outlined" />
                    ))}
                  </Box>
                </Box>
              )}
              <Box sx={{ mt: 1.5, pt: 1.5, borderTop: "1px solid", borderColor: "divider" }}>
                <Typography variant="overline" sx={{ color: "text.secondary" }}>Sample Data</Typography>
                {loadingPreview ? (
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>Loading preview...</Typography>
                ) : selectedDataset.data_type === "text" && textPreview ? (
                  <Box sx={{ mt: 0.5 }}>
                    <Typography variant="body2" color="text.secondary">
                      Vocab: {textPreview.vocab_size} · {textPreview.tokenizer_type} · Seq: {textPreview.seq_length}
                    </Typography>
                    {textPreview.sample_text && (
                      <Box component="pre" sx={{ fontFamily: '"JetBrains Mono", monospace', fontSize: "0.8125rem", mt: 0.5, p: 1, bgcolor: "action.hover", borderRadius: 1 }}>
                        {textPreview.sample_text}
                      </Box>
                    )}
                  </Box>
                ) : preview.length > 0 ? (
                  <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(80px, 1fr))", gap: 1, mt: 0.5 }}>
                    {preview.map((sample, i) => (
                      <Box key={i} sx={{ textAlign: "center", bgcolor: "action.hover", borderRadius: 1, overflow: "hidden" }}>
                        <Box component="img" src={`data:image/png;base64,${sample.image}`} alt={sample.label} sx={{ width: "100%", height: 80, objectFit: "contain" }} />
                        <Typography variant="caption" sx={{ display: "block", p: 0.5, fontFamily: '"JetBrains Mono", monospace' }}>{sample.label}</Typography>
                      </Box>
                    ))}
                  </Box>
                ) : (
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>No preview available</Typography>
                )}
              </Box>
              <Button variant="outlined" color="error" size="small" onClick={() => setDeleteConfirm(selectedDataset.id)} sx={{ mt: 2 }}>
                Delete Dataset
              </Button>
            </Paper>
          )}
        </Box>
      )}

      <ConfirmDialog
        isOpen={deleteConfirm !== null}
        title="Delete Dataset"
        message="Are you sure you want to delete this dataset? This action cannot be undone."
        confirmLabel="Delete"
        cancelLabel="Cancel"
        variant="danger"
        onConfirm={() => deleteConfirm && confirmDelete(deleteConfirm)}
        onCancel={() => setDeleteConfirm(null)}
      />
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
