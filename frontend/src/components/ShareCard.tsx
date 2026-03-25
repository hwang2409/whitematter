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
        backgroundColor: "#09090B",
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
      {/* Offscreen card for capture */}
      <Box
        ref={cardRef}
        sx={{
          position: "absolute",
          left: "-9999px",
          width: 600,
          height: 400,
          bgcolor: "#09090B",
          p: 4,
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          border: "1px solid #27272A",
          borderRadius: "12px",
        }}
      >
        {/* Top section */}
        <Box>
          <Typography
            sx={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: "0.875rem",
              fontWeight: 600,
              color: "#F97316",
              mb: 1.5,
            }}
          >
            wm
          </Typography>
          <Typography
            sx={{
              fontFamily: "'Outfit', sans-serif",
              fontSize: "1.5rem",
              fontWeight: 600,
              color: "#FAFAFA",
              mb: 0.5,
              lineHeight: 1.2,
            }}
          >
            {model.name.replace(/^custom_/, "").replace(/_/g, " ")}
          </Typography>
          <Typography
            sx={{
              fontFamily: "'Instrument Serif', Georgia, serif",
              fontSize: "3rem",
              fontWeight: 400,
              color: "#F97316",
              lineHeight: 1,
              mb: 2,
            }}
          >
            {model.best_accuracy.toFixed(1)}%
          </Typography>
        </Box>

        {/* Bottom section */}
        <Box>
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mb: 2 }}>
            {archParts.slice(0, 8).map((part, i) => (
              <Chip
                key={i}
                size="small"
                label={part}
                sx={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: "0.625rem",
                  fontWeight: 500,
                  bgcolor: "rgba(249,115,22,0.08)",
                  color: "#F97316",
                  border: "1px solid rgba(249,115,22,0.2)",
                  borderRadius: "6px",
                  height: 22,
                }}
              />
            ))}
          </Box>
          <Box
            sx={{
              display: "flex",
              gap: 2,
              fontSize: "0.75rem",
              fontFamily: "'Outfit', sans-serif",
              color: "#52525B",
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

        {/* Footer */}
        <Typography
          sx={{
            fontSize: "0.6875rem",
            color: "#52525B",
            fontFamily: "'JetBrains Mono', monospace",
          }}
        >
          Built with whitematter
        </Typography>
      </Box>

      {/* Share button */}
      <Button
        variant="outlined"
        size="small"
        startIcon={<ShareOutlined />}
        onClick={handleShare}
        disabled={generating}
        sx={{
          borderColor: "#27272A",
          color: "#A1A1AA",
          fontFamily: "'Outfit', sans-serif",
          fontSize: "0.75rem",
          textTransform: "none",
          borderRadius: "8px",
          "&:hover": {
            borderColor: "#3F3F46",
            color: "#FAFAFA",
            bgcolor: "rgba(255,255,255,0.03)",
          },
        }}
      >
        {generating ? "Generating..." : "Share"}
      </Button>

      <Toast toasts={toast.toasts} onDismiss={toast.dismissToast} />
    </>
  );
}
