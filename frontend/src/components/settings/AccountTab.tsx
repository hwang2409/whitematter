"use client";

import { useAuth } from "@/context/AuthContext";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";

export default function AccountTab() {
  const { user, logout } = useAuth();

  return (
    <Box>
      <Typography variant="h3" sx={{ mb: 2 }}>
        Account
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        {user?.email}
      </Typography>
      <Button variant="outlined" color="error" onClick={logout}>
        Sign Out
      </Button>
    </Box>
  );
}
