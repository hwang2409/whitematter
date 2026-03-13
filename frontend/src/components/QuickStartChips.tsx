"use client";

import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

const SUGGESTIONS = [
  {
    title: "Quick Start",
    desc: "Train MNIST in 18 seconds",
    message: "I want to try the Quick Start with MNIST",
  },
  {
    title: "Image classifier",
    desc: "Sort photos into categories",
    message: "I want to build an image classifier",
  },
  {
    title: "Text generator",
    desc: "Generate text like Shakespeare",
    message: "I want to build a text generation model",
  },
  {
    title: "Help me decide",
    desc: "Not sure what I need",
    message: "I want to build a custom neural network",
  },
];

interface QuickStartChipsProps {
  onSelect: (message: string) => void;
}

export default function QuickStartChips({ onSelect }: QuickStartChipsProps) {
  return (
    <Box
      sx={{
        display: "grid",
        gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" },
        gap: 1.25,
        maxWidth: 420,
        width: "100%",
      }}
    >
      {SUGGESTIONS.map(({ title, desc, message }) => (
        <Box
          key={title}
          onClick={() => onSelect(message)}
          sx={{
            px: 2.25,
            py: 1.75,
            borderRadius: "12px",
            bgcolor: "background.paper",
            border: "1px solid",
            borderColor: "divider",
            cursor: "pointer",
            textAlign: "left",
            transition: "all 0.15s ease-out",
            "&:hover": {
              borderColor: (theme) =>
                theme.palette.mode === "dark"
                  ? "rgba(255,255,255,0.14)"
                  : "rgba(0,0,0,0.12)",
              boxShadow: "0 2px 12px rgba(0,0,0,0.03)",
            },
          }}
        >
          <Typography
            sx={{
              fontSize: "0.875rem",
              fontWeight: 500,
              color: "text.primary",
              mb: 0.25,
            }}
          >
            {title}
          </Typography>
          <Typography
            sx={{
              fontSize: "0.75rem",
              color: "text.disabled",
              lineHeight: 1.4,
            }}
          >
            {desc}
          </Typography>
        </Box>
      ))}
    </Box>
  );
}
