"use client";

import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";

interface ModelCardProps {
  name: string;
  description: string;
  /** Human-readable layer summary, e.g. "3 Conv2D -> Flatten -> 2 Linear" */
  layers: string;
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

  // Parse layers string into individual chips
  const layerChips = layers
    .split(/\s*->\s*|\s*→\s*|\s*,\s*/)
    .map((l) => l.trim())
    .filter(Boolean);

  return (
    <Box
      sx={{
        bgcolor: "background.paper",
        border: "1px solid",
        borderColor: "divider",
        borderRadius: "16px",
        p: 3,
        mt: 1,
        transition: "border-color 0.15s ease-out",
        "&:hover": {
          borderColor: (theme) =>
            theme.palette.mode === "dark"
              ? "rgba(255,255,255,0.14)"
              : "rgba(0,0,0,0.12)",
        },
      }}
    >
      {/* Header */}
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1.5 }}>
        <Typography sx={{ fontSize: "1rem", fontWeight: 600, color: "text.primary" }}>
          {name}
        </Typography>
        <Box
          sx={{
            fontSize: "0.75rem",
            px: 1.25,
            py: 0.375,
            borderRadius: "99px",
            bgcolor: "rgba(120,113,108,0.08)",
            color: "primary.main",
            fontWeight: 500,
          }}
        >
          CNN
        </Box>
      </Box>

      {/* Description */}
      <Typography
        variant="body2"
        sx={{ color: "text.secondary", mb: 2, lineHeight: 1.5 }}
      >
        {description}
      </Typography>

      {/* Layer chips */}
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.75, mb: 2 }}>
        {layerChips.map((layer, i) => (
          <Box
            key={i}
            sx={{
              fontSize: "0.75rem",
              px: 1.25,
              py: 0.5,
              borderRadius: "6px",
              bgcolor: "background.default",
              border: "1px solid",
              borderColor: "divider",
              color: "text.secondary",
              fontFamily: "'DM Mono', monospace",
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
          fontFamily: "'DM Mono', monospace",
          color: "text.secondary",
          mb: 2.5,
          p: 1.5,
          borderRadius: "10px",
          bgcolor: "background.default",
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
            color={compute === "cpu" ? "primary" : "default"}
            variant={compute === "cpu" ? "filled" : "outlined"}
            size="small"
          />
          <Chip
            label="GPU (faster, ~60s startup)"
            onClick={() => setCompute("gpu")}
            color={compute === "gpu" ? "primary" : "default"}
            variant={compute === "gpu" ? "filled" : "outlined"}
            size="small"
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
          Train this model
        </Button>
        <Button
          size="small"
          variant="outlined"
          onClick={onRequestChanges}
          sx={{
            px: 3,
            py: 1.25,
            borderRadius: "10px",
            fontSize: "0.875rem",
            fontWeight: 500,
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
          Modify architecture
        </Button>
      </Box>
    </Box>
  );
}
