"use client";

import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

export default function SettingsRoute() {
  return (
    <Box sx={{ p: { xs: 2, sm: 3 }, maxWidth: 800, mx: "auto", width: "100%" }}>
      <Typography variant="h2" sx={{ mb: 2 }}>
        Settings
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Settings page coming soon.
      </Typography>
    </Box>
  );
}
