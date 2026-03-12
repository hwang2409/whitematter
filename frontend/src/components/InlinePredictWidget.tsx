"use client";
import { useState, useCallback, useEffect, useRef } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import CircularProgress from "@mui/material/CircularProgress";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

interface Prediction {
  label: string;
  confidence: number;
}

interface InlinePredictWidgetProps {
  modelId: string;
}

export default function InlinePredictWidget({ modelId }: InlinePredictWidgetProps) {
  const [predictions, setPredictions] = useState<Prediction[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const prevUrlRef = useRef<string | null>(null);

  // Cleanup object URLs to prevent memory leaks
  useEffect(() => {
    return () => {
      if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current);
    };
  }, []);

  const handleFile = useCallback(async (file: File) => {
    setLoading(true);
    setError("");
    setPredictions(null);
    if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current);
    const newUrl = URL.createObjectURL(file);
    prevUrlRef.current = newUrl;
    setPreviewUrl(newUrl);

    const token = localStorage.getItem("access_token");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/predict?model_id=${encodeURIComponent(modelId)}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();
      setPredictions(data.predictions || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }, [modelId]);

  return (
    <Box>
      {/* Drop zone */}
      <Box
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const file = e.dataTransfer.files[0];
          if (file) handleFile(file);
        }}
        onClick={() => {
          const input = document.createElement("input");
          input.type = "file";
          input.accept = "image/*";
          input.onchange = (e) => {
            const file = (e.target as HTMLInputElement).files?.[0];
            if (file) handleFile(file);
          };
          input.click();
        }}
        sx={{
          border: "2px dashed",
          borderColor: dragOver ? "primary.main" : "divider",
          borderRadius: 1,
          p: 2,
          textAlign: "center",
          cursor: "pointer",
          bgcolor: dragOver ? "action.hover" : "transparent",
          transition: "all 0.2s",
        }}
      >
        <Typography variant="body2" color="text.secondary">
          Drop an image here or click to upload
        </Typography>
      </Box>

      {/* Preview + Results */}
      {(previewUrl || loading || predictions || error) && (
        <Box sx={{ mt: 1.5, display: "flex", gap: 2, alignItems: "flex-start" }}>
          {previewUrl && (
            <Box
              component="img"
              src={previewUrl}
              sx={{ width: 80, height: 80, objectFit: "cover", borderRadius: 1 }}
            />
          )}
          <Box sx={{ flex: 1 }}>
            {loading && <CircularProgress size={20} />}
            {error && <Typography color="error" variant="body2">{error}</Typography>}
            {predictions && predictions.map((p, i) => (
              <Box key={i} sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
                <Typography variant="body2">{p.label}</Typography>
                <Typography variant="body2" fontWeight="bold">
                  {(p.confidence * 100).toFixed(1)}%
                </Typography>
              </Box>
            ))}
          </Box>
        </Box>
      )}
    </Box>
  );
}
