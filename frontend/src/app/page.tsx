"use client";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";

export default function HomePage() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/chat");
  }, [router]);

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        bgcolor: "background.default",
      }}
    >
      <CircularProgress size={32} sx={{ color: "text.secondary" }} />
    </Box>
  );
}
