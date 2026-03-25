"use client";
import { useState, useEffect } from "react";
import NextLink from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@/context/AuthContext";
import { login } from "@/services/auth";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import Link from "@mui/material/Link";
import Alert from "@mui/material/Alert";
import GoogleSignInButton from "@/components/GoogleSignInButton";

const inputSx = {
  "& .MuiOutlinedInput-root": {
    fontFamily: "'Outfit', sans-serif",
    fontSize: "0.9375rem",
    color: "#FAFAFA",
    bgcolor: "rgba(255,255,255,0.03)",
    borderRadius: "10px",
    "& fieldset": { borderColor: "#27272A", transition: "border-color 0.2s" },
    "&:hover fieldset": { borderColor: "#3F3F46" },
    "&.Mui-focused fieldset": { borderColor: "#F97316", borderWidth: "1.5px" },
  },
  "& .MuiInputBase-input::placeholder": { color: "#52525B", opacity: 1 },
} as const;

export default function LoginPage() {
  const { user, loginWithTokens } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (user) router.replace("/chat");
  }, [user, router]);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const tokens = await login(email, password);
      loginWithTokens(tokens);
      router.replace("/chat");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Login failed";
      setError(
        msg.includes("fetch") || msg.includes("Failed to fetch") || msg.includes("NetworkError")
          ? "Login failed. Is the API server running? (See frontend README.)"
          : msg
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        bgcolor: "#09090B",
        px: 3,
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Atmospheric glow */}
      <Box
        sx={{
          position: "absolute",
          top: "32%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: 700,
          height: 500,
          background:
            "radial-gradient(ellipse, rgba(249,115,22,0.04) 0%, transparent 65%)",
          pointerEvents: "none",
        }}
      />

      <Box
        sx={{
          width: "100%",
          maxWidth: 360,
          position: "relative",
          zIndex: 1,
          animation: "fadeIn 0.5s ease-out",
        }}
      >
        {/* Brand mark */}
        <Box sx={{ textAlign: "center", mb: 5 }}>
          <Typography
            sx={{
              fontFamily: "'Instrument Serif', Georgia, serif",
              fontStyle: "italic",
              fontSize: "1.375rem",
              color: "#FAFAFA",
              letterSpacing: "-0.01em",
            }}
          >
            whitematter
          </Typography>
          <Box
            sx={{
              width: 24,
              height: 1.5,
              bgcolor: "#F97316",
              mx: "auto",
              mt: 1,
              borderRadius: 1,
            }}
          />
        </Box>

        {/* Heading */}
        <Typography
          sx={{
            fontFamily: "'Outfit', sans-serif",
            fontSize: "1.375rem",
            fontWeight: 600,
            color: "#FAFAFA",
            mb: 0.5,
            textAlign: "center",
          }}
        >
          Welcome back
        </Typography>
        <Typography
          sx={{
            fontFamily: "'Outfit', sans-serif",
            fontSize: "0.875rem",
            color: "#52525B",
            mb: 4,
            textAlign: "center",
          }}
        >
          Sign in to continue building
        </Typography>

        <form onSubmit={handleSubmit}>
          {error && (
            <Alert
              severity="error"
              onClose={() => setError("")}
              sx={{
                mb: 2.5,
                borderRadius: "10px",
                fontFamily: "'Outfit', sans-serif",
                fontSize: "0.8125rem",
                bgcolor: "rgba(239,68,68,0.08)",
                color: "#FCA5A5",
                border: "1px solid rgba(239,68,68,0.15)",
                "& .MuiAlert-icon": { color: "#F87171" },
              }}
            >
              {error}
            </Alert>
          )}

          <GoogleSignInButton />

          {/* Divider */}
          <Box sx={{ display: "flex", alignItems: "center", my: 2.5 }}>
            <Box sx={{ flex: 1, height: "1px", bgcolor: "#1F1F23" }} />
            <Typography
              sx={{
                px: 2,
                fontFamily: "'Outfit', sans-serif",
                color: "#3F3F46",
                fontSize: "0.75rem",
              }}
            >
              or
            </Typography>
            <Box sx={{ flex: 1, height: "1px", bgcolor: "#1F1F23" }} />
          </Box>

          <TextField
            fullWidth
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            autoComplete="email"
            sx={{ mb: 1.5, ...inputSx }}
          />

          <TextField
            fullWidth
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            autoComplete="current-password"
            sx={{ mb: 3, ...inputSx }}
          />

          <Button
            type="submit"
            fullWidth
            variant="contained"
            disabled={loading}
            sx={{
              py: 1.25,
              textTransform: "none",
              fontFamily: "'Outfit', sans-serif",
              fontWeight: 600,
              fontSize: "0.9375rem",
              borderRadius: "10px",
              bgcolor: "#F97316",
              color: "#fff",
              boxShadow: "none",
              "&:hover": {
                bgcolor: "#EA580C",
                boxShadow: "0 0 20px rgba(249,115,22,0.2)",
              },
              "&.Mui-disabled": {
                bgcolor: "#27272A",
                color: "#52525B",
              },
            }}
          >
            {loading ? "Signing in\u2026" : "Sign in"}
          </Button>
        </form>

        <Typography
          sx={{
            mt: 3.5,
            textAlign: "center",
            fontFamily: "'Outfit', sans-serif",
            fontSize: "0.8125rem",
            color: "#52525B",
          }}
        >
          Don&apos;t have an account?{" "}
          <Link
            component={NextLink}
            href="/register"
            sx={{
              color: "#F97316",
              fontWeight: 500,
              textDecoration: "none",
              "&:hover": { textDecoration: "underline" },
            }}
          >
            Sign up
          </Link>
        </Typography>
      </Box>
    </Box>
  );
}
