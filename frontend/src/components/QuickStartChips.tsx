"use client";

import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Typography from "@mui/material/Typography";

const QUICK_STARTS = [
  {
    label: "Quick Start (MNIST)",
    message: "I want to try the Quick Start with MNIST",
  },
  {
    label: "I want to build something",
    message: "I want to build a custom neural network",
  },
];

interface QuickStartChipsProps {
  onSelect: (message: string) => void;
}

export default function QuickStartChips({ onSelect }: QuickStartChipsProps) {
  return (
    <Box>
      <Typography
        variant="body2"
        color="text.secondary"
        sx={{ mb: 1.5, fontSize: "0.8125rem" }}
      >
        Quick start
      </Typography>
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
        {QUICK_STARTS.map(({ label, message }) => (
          <Chip
            key={label}
            label={label}
            variant="outlined"
            onClick={() => onSelect(message)}
            sx={{
              cursor: "pointer",
              fontSize: "0.8125rem",
              fontFamily: "inherit",
              py: 0.5,
              "&:hover": {
                bgcolor: "rgba(126,184,255,0.1)",
                borderColor: "primary.main",
                color: "primary.main",
              },
              transition: "all 0.2s ease-out",
            }}
          />
        ))}
      </Box>
    </Box>
  );
}
