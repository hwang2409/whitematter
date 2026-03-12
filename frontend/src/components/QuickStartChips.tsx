"use client";

import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Typography from "@mui/material/Typography";

const QUICK_STARTS = [
  { label: "Classify images (CIFAR-10)", message: "I want to classify images using the CIFAR-10 dataset" },
  { label: "Generate text (Shakespeare)", message: "I want to generate text in the style of Shakespeare" },
  { label: "Detect sentiment", message: "I want to build a sentiment detection model" },
  { label: "Custom dataset", message: "I have my own dataset I'd like to use" },
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
