"use client";
import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import InlinePredictWidget from "./InlinePredictWidget";

interface CompletedModelCardProps {
  modelId: string;
  accuracy: number;
  params: string;
  trainingTime: string;
  architecture: string;
  datasetName: string;
}

export default function CompletedModelCard({
  modelId,
  accuracy,
  params,
  trainingTime,
  architecture,
  datasetName,
}: CompletedModelCardProps) {
  const [showPredict, setShowPredict] = useState(false);

  const accuracyPct = (accuracy * 100).toFixed(1);

  function handleShare() {
    const text = encodeURIComponent(
      `Just trained a model with ${accuracyPct}% accuracy (${params} params) in ${trainingTime} on @whitematter`
    );
    const url = encodeURIComponent("https://whitematter.com");
    window.open(
      `https://twitter.com/intent/tweet?text=${text}&url=${url}`,
      "_blank",
      "width=550,height=420"
    );
  }

  function handleSaveImage() {
    import("html2canvas")
      .then((html2canvas) => {
        const el = document.getElementById(`model-card-${modelId}`);
        if (!el) return;
        return html2canvas.default(el);
      })
      .then((canvas) => {
        if (!canvas) return;
        const link = document.createElement("a");
        link.download = `whitematter-${modelId}.png`;
        link.href = canvas.toDataURL();
        link.click();
      })
      .catch((err) => {
        console.error("Failed to save image:", err);
      });
  }

  return (
    <Box
      id={`model-card-${modelId}`}
      sx={{
        bgcolor: "#18181B",
        border: "1px solid #27272A",
        borderRadius: "12px",
        p: 3.5,
        mt: 1,
        textAlign: "center",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <Box
        sx={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: 2,
          bgcolor: "#F97316",
        }}
      />

      <Typography
        sx={{
          fontSize: "0.8125rem",
          fontWeight: 500,
          fontFamily: "'Outfit', sans-serif",
          color: "#22C55E",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          mb: 2,
        }}
      >
        Training complete
      </Typography>

      <Typography
        sx={{
          fontFamily: "'Instrument Serif', Georgia, serif",
          fontSize: "3.5rem",
          color: "#F97316",
          letterSpacing: "-0.03em",
          lineHeight: 1,
        }}
      >
        {accuracyPct}%
      </Typography>
      <Typography
        sx={{
          fontSize: "0.875rem",
          fontFamily: "'Outfit', sans-serif",
          color: "#52525B",
          mt: 0.5,
          mb: 2.5,
        }}
      >
        Accuracy
      </Typography>

      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          gap: 4,
          pb: 2.5,
          borderBottom: "1px solid #27272A",
          mb: 2.5,
        }}
      >
        <Box>
          <Typography
            sx={{
              fontSize: "1.25rem",
              fontWeight: 600,
              fontFamily: "'JetBrains Mono', monospace",
              color: "#FAFAFA",
              letterSpacing: "-0.02em",
            }}
          >
            {params}
          </Typography>
          <Typography
            sx={{
              fontSize: "0.75rem",
              fontFamily: "'Outfit', sans-serif",
              color: "#52525B",
              mt: 0.25,
            }}
          >
            Parameters
          </Typography>
        </Box>
        <Box>
          <Typography
            sx={{
              fontSize: "1.25rem",
              fontWeight: 600,
              fontFamily: "'JetBrains Mono', monospace",
              color: "#FAFAFA",
              letterSpacing: "-0.02em",
            }}
          >
            {trainingTime}
          </Typography>
          <Typography
            sx={{
              fontSize: "0.75rem",
              fontFamily: "'Outfit', sans-serif",
              color: "#52525B",
              mt: 0.25,
            }}
          >
            Training time
          </Typography>
        </Box>
      </Box>

      <Typography
        sx={{
          fontSize: "0.8125rem",
          fontFamily: "'Outfit', sans-serif",
          color: "#A1A1AA",
          mb: 2.5,
        }}
      >
        {architecture} on {datasetName}
      </Typography>

      <Box sx={{ display: "flex", gap: 1, justifyContent: "center", flexWrap: "wrap" }}>
        <Button
          size="small"
          onClick={() => setShowPredict(!showPredict)}
          sx={{
            px: 3,
            py: 1.25,
            borderRadius: "8px",
            fontSize: "0.875rem",
            fontWeight: 500,
            fontFamily: "'Outfit', sans-serif",
            bgcolor: "#F97316",
            color: "#FFFFFF",
            textTransform: "none",
            "&:hover": {
              bgcolor: "#EA580C",
            },
          }}
        >
          {showPredict ? "Close" : "Try it"}
        </Button>
        <Button
          size="small"
          variant="outlined"
          disabled
          sx={{
            px: 3,
            py: 1.25,
            borderRadius: "8px",
            fontSize: "0.875rem",
            fontWeight: 500,
            fontFamily: "'Outfit', sans-serif",
            color: "#A1A1AA",
            borderColor: "#27272A",
            textTransform: "none",
            "&:hover": {
              borderColor: "#3F3F46",
              bgcolor: "transparent",
            },
            "&.Mui-disabled": {
              color: "#52525B",
              borderColor: "#27272A",
            },
          }}
        >
          Deploy API (coming soon)
        </Button>
        <Button
          size="small"
          variant="outlined"
          onClick={handleShare}
          sx={{
            px: 2,
            py: 1.25,
            borderRadius: "8px",
            fontSize: "0.875rem",
            fontFamily: "'Outfit', sans-serif",
            color: "#A1A1AA",
            borderColor: "#27272A",
            textTransform: "none",
            "&:hover": {
              borderColor: "#3F3F46",
              color: "#FAFAFA",
              bgcolor: "transparent",
            },
          }}
        >
          Share
        </Button>
        <Button
          size="small"
          onClick={handleSaveImage}
          sx={{
            px: 2,
            py: 1.25,
            borderRadius: "8px",
            fontSize: "0.875rem",
            fontFamily: "'Outfit', sans-serif",
            color: "#A1A1AA",
            textTransform: "none",
            "&:hover": {
              color: "#FAFAFA",
              bgcolor: "transparent",
            },
          }}
        >
          Save image
        </Button>
      </Box>

      {showPredict && (
        <Box sx={{ mt: 2.5, textAlign: "left" }}>
          <InlinePredictWidget modelId={modelId} />
        </Box>
      )}
    </Box>
  );
}
