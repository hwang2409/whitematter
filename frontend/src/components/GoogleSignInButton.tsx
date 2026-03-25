"use client";
import { useGoogleLogin } from "@react-oauth/google";
import { useAuth } from "@/context/AuthContext";
import { useRouter } from "next/navigation";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

interface GoogleSignInButtonProps {
  label?: string;
  fullWidth?: boolean;
}

function GoogleSignInButtonInner({
  label = "Continue with Google",
  fullWidth = true,
}: GoogleSignInButtonProps) {
  const { loginWithTokens } = useAuth();
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const googleLogin = useGoogleLogin({
    onSuccess: async (tokenResponse) => {
      setLoading(true);
      setError("");
      try {
        const res = await fetch(`${API_BASE}/auth/google`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ access_token: tokenResponse.access_token }),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.detail || "Google sign-in failed");
        }
        const tokens = await res.json();
        loginWithTokens(tokens);
        router.replace("/chat");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Google sign-in failed");
      } finally {
        setLoading(false);
      }
    },
    onError: () => setError("Google sign-in was cancelled"),
  });

  return (
    <>
      <Button
        onClick={() => googleLogin()}
        disabled={loading}
        fullWidth={fullWidth}
        variant="outlined"
        sx={{
          textTransform: "none",
          borderColor: "#27272A",
          color: "#FAFAFA",
          fontFamily: "'Outfit', sans-serif",
          fontSize: "0.875rem",
          fontWeight: 500,
          borderRadius: "8px",
          py: 1.2,
          transition: "all 0.15s ease",
          "&:hover": {
            borderColor: "#3F3F46",
            bgcolor: "rgba(255,255,255,0.03)",
          },
          "&.Mui-disabled": {
            borderColor: "#27272A",
            color: "#52525B",
          },
        }}
      >
        {loading ? "Signing in..." : label}
      </Button>
      {error && (
        <Typography
          sx={{
            color: "#EF4444",
            fontSize: "0.8125rem",
            fontFamily: "'Outfit', sans-serif",
            mt: 0.5,
          }}
        >
          {error}
        </Typography>
      )}
    </>
  );
}

export default function GoogleSignInButton(props: GoogleSignInButtonProps) {
  if (!process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID) return null;
  return <GoogleSignInButtonInner {...props} />;
}
