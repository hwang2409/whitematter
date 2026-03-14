"use client";
import { useRef, useState } from "react";
import type { Model } from "@/api";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import ShareOutlined from "@mui/icons-material/ShareOutlined";
import Toast, { useToast } from "./Toast";

interface Props {
  model: Model;
}

export default function ShareCard({ model }: Props) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [generating, setGenerating] = useState(false);
  const toast = useToast();

  async function handleShare() {
    if (!cardRef.current) return;
    setGenerating(true);
    try {
      const html2canvas = (await import("html2canvas")).default;
      const canvas = await html2canvas(cardRef.current, {
        backgroundColor: "#0a0a0a",
        scale: 2,
      });
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, "image/png")
      );
      if (!blob) throw new Error("Failed to generate image");

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${model.name.replace(/\s+/g, "-")}-results.png`;
      a.click();
      URL.revokeObjectURL(url);

      try {
        await navigator.clipboard.write([
          new ClipboardItem({ "image/png": blob }),
        ]);
        toast.success("Image downloaded and copied to clipboard!");
      } catch {
        toast.success("Image downloaded!");
      }
    } catch {
      toast.error("Failed to generate share image");
    } finally {
      setGenerating(false);
    }
  }

  const lastHistory =
    model.training_history?.[model.training_history.length - 1];
  const archParts = model.architecture
    .replace(/_/g, " ")
    .split(/[\s,-]+/)
    .filter(Boolean);

  return (
    <>
      <Box
        ref={cardRef}
        sx={{
          position: "absolute",
          left: "-9999px",
          width: 600,
          height: 400,
          bgcolor: "#0a0a0a",
          p: 4,
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          border: "1px solid rgba(126,184,255,0.2)",
          borderRadius: 2,
        }}
      >
        <Box>
          <Typography
            sx={{
              fontFamily: "'DM Mono', monospace",
              fontSize: "0.875rem",
              color: "#7EB8FF",
              mb: 1,
            }}
          >
            wm
          </Typography>
          <Typography
            sx={{
              fontSize: "1.5rem",
              fontWeight: 700,
              color: "#fff",
              mb: 0.5,
            }}
          >
            {model.name.replace(/^custom_/, "").replace(/_/g, " ")}
          </Typography>
          <Typography
            sx={{
              fontFamily: "'DM Mono', monospace",
              fontSize: "3rem",
              fontWeight: 700,
              color: "#7EB8FF",
              lineHeight: 1,
              mb: 2,
            }}
          >
            {model.best_accuracy.toFixed(1)}%
          </Typography>
        </Box>
        <Box>
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mb: 2 }}>
            {archParts.slice(0, 8).map((part, i) => (
              <Chip
                key={i}
                size="small"
                label={part}
                sx={{
                  fontFamily: "'DM Mono', monospace",
                  fontSize: "0.625rem",
                  bgcolor: "rgba(126,184,255,0.15)",
                  color: "#7EB8FF",
                  border: "1px solid rgba(126,184,255,0.3)",
                }}
              />
            ))}
          </Box>
          <Box
            sx={{
              display: "flex",
              gap: 2,
              fontSize: "0.75rem",
              color: "rgba(255,255,255,0.5)",
            }}
          >
            <span>{model.epochs_trained} epochs</span>
            {lastHistory && <span>Loss: {lastHistory.loss.toFixed(4)}</span>}
            <span>
              {model.dataset.startsWith("custom:")
                ? "Custom dataset"
                : model.dataset}
            </span>
          </Box>
        </Box>
        <Typography
          sx={{
            fontSize: "0.6875rem",
            color: "rgba(255,255,255,0.3)",
            fontFamily: "'DM Mono', monospace",
          }}
        >
          Built with whitematter
        </Typography>
      </Box>

      <Button
        variant="outlined"
        size="small"
        startIcon={<ShareOutlined />}
        onClick={handleShare}
        disabled={generating}
      >
        {generating ? "Generating..." : "Share"}
      </Button>

      <Toast toasts={toast.toasts} onDismiss={toast.dismissToast} />
    </>
  );
}
