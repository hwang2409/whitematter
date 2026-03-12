"use client";
import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import Divider from "@mui/material/Divider";
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
    import("html2canvas").then((html2canvas) => {
      const el = document.getElementById(`model-card-${modelId}`);
      if (!el) return;
      html2canvas.default(el).then((canvas) => {
        const link = document.createElement("a");
        link.download = `whitematter-${modelId}.png`;
        link.href = canvas.toDataURL();
        link.click();
      });
    });
  }

  return (
    <Box
      id={`model-card-${modelId}`}
      sx={{
        border: 1,
        borderColor: "divider",
        borderRadius: 2,
        p: 2,
        my: 1,
        bgcolor: "background.paper",
      }}
    >
      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
        Training Complete
      </Typography>

      <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", mb: 1.5 }}>
        <Chip label={`${accuracyPct}% accuracy`} color="success" size="small" />
        <Chip label={params} size="small" variant="outlined" />
        <Chip label={trainingTime} size="small" variant="outlined" />
      </Box>

      <Typography variant="body2" color="text.secondary" gutterBottom>
        {architecture} on {datasetName}
      </Typography>

      <Divider sx={{ my: 1.5 }} />

      <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
        <Button
          size="small"
          variant="contained"
          onClick={() => setShowPredict(!showPredict)}
        >
          {showPredict ? "Close" : "Try it"}
        </Button>
        <Button size="small" variant="outlined" disabled>
          Deploy as API (coming soon)
        </Button>
        <Button size="small" variant="outlined" onClick={handleShare}>
          Share
        </Button>
        <Button size="small" variant="text" onClick={handleSaveImage}>
          Save image
        </Button>
      </Box>

      {showPredict && (
        <Box sx={{ mt: 2 }}>
          <InlinePredictWidget modelId={modelId} />
        </Box>
      )}
    </Box>
  );
}
