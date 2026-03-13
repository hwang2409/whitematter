"use client";
import { useGoogleLogin } from "@react-oauth/google";
import { useAuth } from "@/context/AuthContext";
import { useRouter } from "next/navigation";
import Button from "@mui/material/Button";
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
        size="large"
        sx={{
          textTransform: "none",
          borderColor: "divider",
          color: "text.primary",
          py: 1.2,
        }}
      >
        {loading ? "Signing in..." : label}
      </Button>
      {error && (
        <p style={{ color: "red", fontSize: "0.85rem", marginTop: 4 }}>
          {error}
        </p>
      )}
    </>
  );
}

export default function GoogleSignInButton(props: GoogleSignInButtonProps) {
  if (!process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID) return null;
  return <GoogleSignInButtonInner {...props} />;
}
