"use client";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";
import LandingPage from "@/views/LandingPage";

export default function HomePage() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/chat");
  }, [router]);

  if (loading) {
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

  if (user) {
    return null;
  }

  return <LandingPage />;
}
