"use client";

import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import Divider from "@mui/material/Divider";
import { themeTokens } from "@/theme";

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
  return (
    <Box
      sx={{
        bgcolor: "background.paper",
        border: "1px solid",
        borderColor: "divider",
        borderRadius: 2,
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <Box
        sx={{
          px: 2,
          py: 1.5,
          bgcolor: "rgba(126,184,255,0.06)",
          borderBottom: "1px solid",
          borderColor: "divider",
        }}
      >
        <Typography
          variant="body2"
          sx={{
            fontWeight: 600,
            color: themeTokens.accent,
            fontSize: "0.8125rem",
            letterSpacing: "0.01em",
          }}
        >
          Proposed Architecture
        </Typography>
      </Box>

      {/* Body */}
      <Box sx={{ px: 2, py: 1.5 }}>
        <Typography variant="body1" sx={{ fontWeight: 600, mb: 0.5 }}>
          {name}
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          {description}
        </Typography>

        <Divider sx={{ my: 1 }} />

        {/* Layers */}
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.06em", mb: 0.5 }}
        >
          Layers
        </Typography>
        <Typography
          variant="body2"
          sx={{
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: "0.8125rem",
            mb: 1.5,
          }}
        >
          {layers}
        </Typography>

        {/* Training config */}
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.06em", mb: 0.5 }}
        >
          Training Config
        </Typography>
        <Typography
          variant="body2"
          sx={{
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: "0.8125rem",
            mb: 2,
          }}
        >
          {trainingConfig}
        </Typography>

        {/* CPU/GPU selector (Scale users only) */}
        {userPlan === "scale" && (
          <Box sx={{ display: "flex", gap: 1, mb: 1.5 }}>
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
            variant="contained"
            size="small"
            onClick={() => onApprove(userPlan === "scale" ? compute : "cpu")}
            sx={{
              bgcolor: themeTokens.accent,
              color: "#0a0a0a",
              fontWeight: 600,
              "&:hover": { bgcolor: "#9ecaff" },
            }}
          >
            Looks good, train it!
          </Button>
          <Button
            variant="outlined"
            size="small"
            onClick={onRequestChanges}
          >
            Make changes
          </Button>
        </Box>
      </Box>
    </Box>
  );
}
