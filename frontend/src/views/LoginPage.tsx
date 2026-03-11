"use client";
import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@/context/AuthContext";
import { login } from "@/services/auth";
import "./AuthPages.css";

export default function LoginPage() {
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
      const tokens = await login(email, password);
      loginWithTokens(tokens);
      router.replace("/dashboard");
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
    <div className="auth-page">
      <div className="auth-card">
        <h1>whitematter</h1>
        <h2>Sign in</h2>
        <form onSubmit={handleSubmit}>
          {error && <div className="auth-error">{error}</div>}
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            autoComplete="email"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            autoComplete="current-password"
          />
          <button type="submit" disabled={loading}>
            {loading ? "Signing in…" : "Sign in"}
          </button>
        </form>
        <p className="auth-switch">
          Don’t have an account? <Link href="/register" className="auth-link">Sign up</Link>
        </p>
      </div>
    </div>
  );
}
