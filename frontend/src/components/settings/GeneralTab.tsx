"use client";

import { useThemeMode } from "@/context/ThemeContext";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import LightModeIcon from "@mui/icons-material/LightMode";
import DarkModeIcon from "@mui/icons-material/DarkMode";

export default function GeneralTab() {
  const { mode, setMode } = useThemeMode();

  return (
    <Box>
      <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
        Appearance
      </Typography>

      <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
        <Typography variant="body2" color="text.secondary">
          Theme
        </Typography>
        <ToggleButtonGroup
          value={mode}
          exclusive
          onChange={(_, value) => {
            if (value) setMode(value);
          }}
          size="small"
        >
          <ToggleButton value="light" sx={{ gap: 0.5, textTransform: "none" }}>
            <LightModeIcon fontSize="small" />
            Light
          </ToggleButton>
          <ToggleButton value="dark" sx={{ gap: 0.5, textTransform: "none" }}>
            <DarkModeIcon fontSize="small" />
            Dark
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>
    </Box>
  );
}
