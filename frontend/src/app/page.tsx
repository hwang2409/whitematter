"use client";
import { useAuth } from "@/context/AuthContext";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";

export default function HomePage() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (loading) return;
    if (user) {
      router.replace("/dashboard");
    } else {
      router.replace("/login");
    }
  }, [user, loading, router]);

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
