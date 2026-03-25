"use client";

import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";

interface ModelCardProps {
  name: string;
  description: string;
  /** Human-readable layer summary — string like "Conv2D → Linear" or array of layer objects */
  layers: string | unknown[];
  /** Human-readable training config summary */
  trainingConfig: string;
  userPlan?: string;
  onApprove: (compute?: "cpu" | "gpu") => void;
  onRequestChanges: () => void;
}

export default function ModelCard({
  name,
  description,
  layers,
  trainingConfig,
  userPlan,
  onApprove,
  onRequestChanges,
}: ModelCardProps) {
  const [compute, setCompute] = useState<"cpu" | "gpu">("gpu");

  // Parse layers into individual chips
  const layerChips: string[] = typeof layers === "string"
    ? layers.split(/\s*->\s*|\s*→\s*|\s*,\s*/).map((l) => l.trim()).filter(Boolean)
    : Array.isArray(layers)
      ? layers.map((l) => typeof l === "string" ? l : (l as any).type ?? JSON.stringify(l))
      : [];

  return (
    <Box
      sx={{
        bgcolor: "#18181B",
        border: "1px solid #27272A",
        borderRadius: "12px",
        p: 3,
        mt: 1,
        transition: "border-color 0.15s ease-out",
        "&:hover": {
          borderColor: "#3F3F46",
        },
      }}
    >
      {/* Header */}
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1.5 }}>
        <Typography
          sx={{
            fontSize: "1rem",
            fontWeight: 600,
            color: "#FAFAFA",
            fontFamily: "'Outfit', sans-serif",
          }}
        >
          {name}
        </Typography>
        <Box
          sx={{
            fontSize: "0.75rem",
            px: 1.25,
            py: 0.375,
            borderRadius: "99px",
            bgcolor: "rgba(249,115,22,0.08)",
            color: "#F97316",
            fontWeight: 500,
            fontFamily: "'JetBrains Mono', monospace",
          }}
        >
          CNN
        </Box>
      </Box>

      {/* Description */}
      <Typography
        variant="body2"
        sx={{
          color: "#A1A1AA",
          mb: 2,
          lineHeight: 1.5,
          fontFamily: "'Outfit', sans-serif",
        }}
      >
        {description}
      </Typography>

      {/* Layer chips */}
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.75, mb: 2 }}>
        {layerChips.map((layer, i) => (
          <Box
            key={i}
            sx={{
              fontSize: "0.6875rem",
              px: 1.25,
              py: 0.5,
              borderRadius: "4px",
              bgcolor: "rgba(255,255,255,0.04)",
              border: "1px solid #27272A",
              color: "#A1A1AA",
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            {layer}
          </Box>
        ))}
      </Box>

      {/* Training config */}
      <Typography
        sx={{
          fontSize: "0.8125rem",
          fontFamily: "'JetBrains Mono', monospace",
          color: "#A1A1AA",
          mb: 2.5,
          p: 1.5,
          borderRadius: "8px",
          bgcolor: "rgba(0,0,0,0.3)",
          lineHeight: 1.6,
        }}
      >
        {trainingConfig}
      </Typography>

      {/* CPU/GPU selector (Scale users only) */}
      {userPlan === "scale" && (
        <Box sx={{ display: "flex", gap: 1, mb: 2 }}>
          <Chip
            label="CPU (instant)"
            onClick={() => setCompute("cpu")}
            size="small"
            sx={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: "0.75rem",
              bgcolor: compute === "cpu" ? "#F97316" : "transparent",
              color: compute === "cpu" ? "#FAFAFA" : "#A1A1AA",
              border: compute === "cpu" ? "1px solid #F97316" : "1px solid #27272A",
              "&:hover": {
                bgcolor: compute === "cpu" ? "#EA580C" : "rgba(255,255,255,0.04)",
                borderColor: compute === "cpu" ? "#EA580C" : "#3F3F46",
              },
            }}
          />
          <Chip
            label="GPU (faster, ~60s startup)"
            onClick={() => setCompute("gpu")}
            size="small"
            sx={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: "0.75rem",
              bgcolor: compute === "gpu" ? "#F97316" : "transparent",
              color: compute === "gpu" ? "#FAFAFA" : "#A1A1AA",
              border: compute === "gpu" ? "1px solid #F97316" : "1px solid #27272A",
              "&:hover": {
                bgcolor: compute === "gpu" ? "#EA580C" : "rgba(255,255,255,0.04)",
                borderColor: compute === "gpu" ? "#EA580C" : "#3F3F46",
              },
            }}
          />
        </Box>
      )}

      {/* Actions */}
      <Box sx={{ display: "flex", gap: 1 }}>
        <Button
          size="small"
          onClick={() => onApprove(userPlan === "scale" ? compute : "cpu")}
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
          Train this model
        </Button>
        <Button
          size="small"
          variant="outlined"
          onClick={onRequestChanges}
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
              color: "#FAFAFA",
              bgcolor: "transparent",
            },
          }}
        >
          Modify architecture
        </Button>
      </Box>
    </Box>
  );
}
