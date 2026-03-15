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
        bgcolor: "background.paper",
        border: "1px solid",
        borderColor: "divider",
        borderRadius: "16px",
        p: 3.5,
        mt: 1,
        textAlign: "center",
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Accent bar — solid violet, no gradient */}
      <Box
        sx={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: 3,
          bgcolor: "primary.main",
          borderRadius: "16px 16px 0 0",
        }}
      />

      {/* Status */}
      <Typography
        sx={{
          fontSize: "0.8125rem",
          fontWeight: 500,
          color: "success.main",
          mb: 2,
        }}
      >
        Training complete
      </Typography>

      {/* Big accuracy number */}
      <Typography
        sx={{
          fontFamily: "'DM Serif Display', Georgia, serif",
          fontSize: "3.25rem",
          color: "text.primary",
          letterSpacing: "-0.03em",
          lineHeight: 1,
        }}
      >
        {accuracyPct}%
      </Typography>
      <Typography
        sx={{
          fontSize: "0.875rem",
          color: "text.disabled",
          mt: 0.5,
          mb: 2.5,
        }}
      >
        Accuracy
      </Typography>

      {/* Stats row */}
      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          gap: 4,
          pb: 2.5,
          borderBottom: "1px solid",
          borderColor: "divider",
          mb: 2.5,
        }}
      >
        <Box>
          <Typography sx={{ fontSize: "1.25rem", fontWeight: 600, color: "text.primary", letterSpacing: "-0.02em" }}>
            {params}
          </Typography>
          <Typography sx={{ fontSize: "0.75rem", color: "text.disabled", mt: 0.25 }}>
            Parameters
          </Typography>
        </Box>
        <Box>
          <Typography sx={{ fontSize: "1.25rem", fontWeight: 600, color: "text.primary", letterSpacing: "-0.02em" }}>
            {trainingTime}
          </Typography>
          <Typography sx={{ fontSize: "0.75rem", color: "text.disabled", mt: 0.25 }}>
            Training time
          </Typography>
        </Box>
      </Box>

      {/* Architecture info */}
      <Typography
        sx={{
          fontSize: "0.8125rem",
          color: "text.secondary",
          mb: 2.5,
        }}
      >
        {architecture} on {datasetName}
      </Typography>

      {/* Actions */}
      <Box sx={{ display: "flex", gap: 1, justifyContent: "center", flexWrap: "wrap" }}>
        <Button
          size="small"
          onClick={() => setShowPredict(!showPredict)}
          sx={{
            px: 3,
            py: 1.25,
            borderRadius: "10px",
            fontSize: "0.875rem",
            fontWeight: 500,
            bgcolor: "text.primary",
            color: (theme) => (theme.palette.mode === "dark" ? "#141311" : "#FFFFFF"),
            "&:hover": {
              bgcolor: (theme) => (theme.palette.mode === "dark" ? "#d4d3d0" : "#333"),
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
            borderRadius: "10px",
            fontSize: "0.875rem",
            fontWeight: 500,
            color: "text.secondary",
            borderColor: "divider",
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
            borderRadius: "10px",
            fontSize: "0.875rem",
            color: "text.secondary",
            borderColor: "divider",
            "&:hover": {
              borderColor: (theme) =>
                theme.palette.mode === "dark"
                  ? "rgba(255,255,255,0.14)"
                  : "rgba(0,0,0,0.12)",
              color: "text.primary",
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
            borderRadius: "10px",
            fontSize: "0.875rem",
            color: "text.secondary",
            "&:hover": { color: "text.primary" },
          }}
        >
          Save image
        </Button>
      </Box>

      {/* Inline prediction widget */}
      {showPredict && (
        <Box sx={{ mt: 2.5, textAlign: "left" }}>
          <InlinePredictWidget modelId={modelId} />
        </Box>
      )}
    </Box>
  );
}
