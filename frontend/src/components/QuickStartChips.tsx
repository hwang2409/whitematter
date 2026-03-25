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
              borderRadius: "999px",
              border: "1px solid",
              borderColor: isSelected ? "#F97316" : "#27272A",
              bgcolor: isSelected ? "rgba(249,115,22,0.08)" : "transparent",
              cursor: "pointer",
              transition: "all 0.15s ease-out",
              "&:hover": {
                bgcolor: isSelected
                  ? "rgba(249,115,22,0.08)"
                  : "rgba(255,255,255,0.03)",
                borderColor: isSelected ? "#F97316" : "#3F3F46",
              },
            }}
          >
            <Typography
              sx={{
                fontSize: "0.8125rem",
                fontFamily: "'Outfit', sans-serif",
                color: isSelected ? "#F97316" : "#A1A1AA",
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
