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
          borderColor: dragOver ? "#F97316" : "#27272A",
          borderRadius: "10px",
          p: 2.5,
          textAlign: "center",
          cursor: "pointer",
          bgcolor: dragOver ? "rgba(249,115,22,0.04)" : "transparent",
          transition: "all 0.15s ease-out",
          "&:hover": {
            borderColor: "#3F3F46",
          },
        }}
      >
        <Typography
          variant="body2"
          sx={{
            color: "#A1A1AA",
            fontFamily: "'Outfit', sans-serif",
          }}
        >
          Drop an image here or click to upload
        </Typography>
      </Box>

      {(previewUrl || loading || predictions || error) && (
        <Box sx={{ mt: 2, display: "flex", gap: 2, alignItems: "flex-start" }}>
          {previewUrl && (
            <Box
              component="img"
              src={previewUrl}
              sx={{
                width: 72,
                height: 72,
                objectFit: "cover",
                borderRadius: "8px",
                border: "1px solid #27272A",
              }}
            />
          )}
          <Box sx={{ flex: 1 }}>
            {loading && <CircularProgress size={20} sx={{ color: "#F97316" }} />}
            {error && (
              <Typography
                variant="body2"
                sx={{
                  color: "#EF4444",
                  fontFamily: "'Outfit', sans-serif",
                }}
              >
                {error}
              </Typography>
            )}
            {predictions && predictions.map((p, i) => (
              <Box key={i} sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
                <Typography
                  variant="body2"
                  sx={{
                    color: "#A1A1AA",
                    fontFamily: "'Outfit', sans-serif",
                  }}
                >
                  {p.label}
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    fontWeight: 600,
                    color: "#FAFAFA",
                    fontFamily: "'JetBrains Mono', monospace",
                  }}
                >
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
