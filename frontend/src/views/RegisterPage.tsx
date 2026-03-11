"use client";
import { useState } from "react";
import NextLink from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@/context/AuthContext";
import { register } from "@/services/auth";
import Box from "@mui/material/Box";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import Link from "@mui/material/Link";
import Alert from "@mui/material/Alert";
import InputAdornment from "@mui/material/InputAdornment";
import EmailOutlined from "@mui/icons-material/EmailOutlined";
import LockOutlined from "@mui/icons-material/LockOutlined";

export default function RegisterPage() {
  const { loginWithTokens } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const tokens = await register(email, password);
      loginWithTokens(tokens);
      router.replace("/dashboard");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Registration failed";
      setError(
        msg.includes("fetch") || msg.includes("Failed to fetch") || msg.includes("NetworkError")
          ? "Registration failed. Is the API server running? (See frontend README.)"
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
        bgcolor: "background.default",
        px: 2,
      }}
    >
      <Paper
        elevation={0}
        sx={{
          width: "100%",
          maxWidth: 400,
          p: 3,
          py: 3.5,
        }}
      >
        <Typography variant="h1" sx={{ mb: 0.5 }}>
          whitematter
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Create your account
        </Typography>
        <form onSubmit={handleSubmit}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError("")}>
              {error}
            </Alert>
          )}
          <TextField
            fullWidth
            type="email"
            label="Email"
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            autoComplete="email"
            sx={{ mb: 2 }}
            slotProps={{
              input: {
                startAdornment: (
                  <InputAdornment position="start" sx={{ color: "text.secondary" }}>
                    <EmailOutlined fontSize="small" />
                  </InputAdornment>
                ),
              },
            }}
          />
          <TextField
            fullWidth
            type="password"
            label="Password"
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            autoComplete="new-password"
            sx={{ mb: 2.5 }}
            slotProps={{
              input: {
                startAdornment: (
                  <InputAdornment position="start" sx={{ color: "text.secondary" }}>
                    <LockOutlined fontSize="small" />
                  </InputAdornment>
                ),
              },
            }}
          />
          <Button
            type="submit"
            fullWidth
            variant="contained"
            size="large"
            disabled={loading}
            sx={{ py: 1.5, textTransform: "none" }}
          >
            {loading ? "Creating account…" : "Sign up"}
          </Button>
        </form>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2.5, textAlign: "center" }}>
          Already have an account?{" "}
          <Link component={NextLink} href="/login" color="primary" fontWeight={500}>
            Sign in
          </Link>
        </Typography>
      </Paper>
    </Box>
  );
}
