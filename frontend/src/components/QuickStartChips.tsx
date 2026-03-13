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
              px: 2,
              py: 0.75,
              borderRadius: "9999px",
              border: "1px solid",
              borderColor: isSelected ? "text.secondary" : "divider",
              bgcolor: isSelected
                ? (theme) =>
                    theme.palette.mode === "dark"
                      ? "rgba(255,255,255,0.08)"
                      : "rgba(0,0,0,0.06)"
                : "transparent",
              cursor: "pointer",
              transition: "all 0.15s ease-out",
              "&:hover": {
                bgcolor: (theme) =>
                  theme.palette.mode === "dark"
                    ? "rgba(255,255,255,0.06)"
                    : "rgba(0,0,0,0.04)",
                borderColor: (theme) =>
                  theme.palette.mode === "dark"
                    ? "rgba(255,255,255,0.14)"
                    : "rgba(0,0,0,0.12)",
              },
            }}
          >
            <Typography
              sx={{
                fontSize: "0.8125rem",
                color: isSelected ? "text.primary" : "text.secondary",
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
