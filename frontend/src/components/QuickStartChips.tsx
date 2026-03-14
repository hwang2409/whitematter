"use client";

import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

const SUGGESTIONS = [
  "Classify images",
  "Analyze text",
  "Find anomalies",
  "Generate text",
];

interface QuickStartChipsProps {
  onSelect?: (text: string) => void;
}

export default function QuickStartChips({ onSelect }: QuickStartChipsProps) {
  const [selected, setSelected] = useState<string | null>(null);

  function handleClick(label: string) {
    if (selected === label) {
      setSelected(null);
    } else {
      setSelected(label);
      onSelect?.(label);
    }
  }

  return (
    <Box
      sx={{
        display: "flex",
        flexWrap: "nowrap",
        gap: 1,
        justifyContent: "center",
        width: "100%",
      }}
    >
      {SUGGESTIONS.map((label) => {
        const isSelected = selected === label;
        return (
          <Box
            key={label}
            onClick={() => handleClick(label)}
            sx={{
              borderRadius: "9999px",
              border: "1px solid",
              borderColor: "divider",
              bgcolor: "transparent",
              color: "text.secondary",
              px: 2,
              py: 1,
              cursor: "pointer",
              transition: "all 0.15s ease-out",
              "&:hover": {
                borderColor: "primary.main",
                boxShadow: "0 0 12px rgba(139, 92, 246, 0.1)",
                color: "text.primary",
              },
              "@media (prefers-reduced-motion: reduce)": {
                animation: "none",
                transition: "none",
              },
              ...(isSelected && {
                bgcolor: "rgba(139, 92, 246, 0.08)",
                borderColor: "primary.main",
                color: "text.primary",
              }),
            }}
          >
            <Typography
              sx={{
                fontSize: "0.8125rem",
                color: "inherit",
                whiteSpace: "nowrap",
              }}
            >
              {label}
            </Typography>
          </Box>
        );
      })}
    </Box>
  );
}
