"use client";

import { useState } from "react";
import { useAuth } from "@/context/AuthContext";
import { getStoredToken } from "@/services/auth";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Paper from "@mui/material/Paper";
import Button from "@mui/material/Button";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import Alert from "@mui/material/Alert";
import CheckIcon from "@mui/icons-material/Check";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

type Interval = "month" | "year";

interface PlanTier {
  id: string;
  name: string;
  monthlyPrice: number;
  annualPrice: number;
  features: string[];
  recommended?: boolean;
}

const PLANS: PlanTier[] = [
  {
    id: "pro",
    name: "Pro",
    monthlyPrice: 29,
    annualPrice: 23,
    features: [
      "Unlimited training runs",
      "Unlimited models",
      "20 GB storage",
      "1 deploy endpoint",
    ],
  },
  {
    id: "scale",
    name: "Scale",
    monthlyPrice: 59,
    annualPrice: 47,
    recommended: true,
    features: [
      "Everything in Pro",
      "100 GB storage",
      "GPU access ($0.99/hr)",
      "5 deploy endpoints",
    ],
  },
];

export default function PricingPage() {
  const { user } = useAuth();
  const [interval, setInterval] = useState<Interval>("month");
  const [loadingPlan, setLoadingPlan] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const currentPlan = user?.plan || "free";

  async function handleUpgrade(planId: string) {
    setError(null);
    setLoadingPlan(planId);
    try {
      const token = getStoredToken();
      const res = await fetch(
        `${API_BASE}/billing/checkout?plan=${planId}&interval=${interval}`,
        {
          method: "POST",
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        },
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to create checkout session");
      }
      const { url } = await res.json();
      window.location.href = url;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Please try again.");
      setLoadingPlan(null);
    }
  }

  return (
    <Box sx={{ pt: { xs: 1, sm: 1.5 }, px: { xs: 2, sm: 3 }, pb: { xs: 2, sm: 3 }, maxWidth: 900, mx: "auto", width: "100%" }}>
      {/* Header */}
      <Box sx={{ textAlign: "center", mb: 5 }}>
        <Typography
          variant="h1"
          sx={{
            fontFamily: "'DM Serif Display', Georgia, serif",
            fontSize: { xs: "1.75rem", sm: "2.25rem" },
            fontWeight: 400,
            letterSpacing: "-0.02em",
            mb: 1,
          }}
        >
          Choose your plan
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Scale your ML workflow with the right plan for your team.
        </Typography>

        {/* Billing toggle */}
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
          <ToggleButtonGroup
            value={interval}
            exclusive
            onChange={(_, v) => { if (v) setInterval(v); }}
            size="small"
          >
            <ToggleButton value="month" sx={{ textTransform: "none", px: 2.5 }}>
              Monthly
            </ToggleButton>
            <ToggleButton value="year" sx={{ textTransform: "none", px: 2.5 }}>
              Annual
            </ToggleButton>
          </ToggleButtonGroup>
          <Chip
            label="Save 20%"
            size="small"
            sx={{
              position: "absolute",
              left: "calc(50% + 105px)",
              bgcolor: "primary.main",
              color: "#fff",
              fontWeight: 600,
              fontSize: "0.75rem",
              opacity: interval === "year" ? 1 : 0,
              transition: "opacity 0.2s ease",
            }}
          />
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Pricing cards */}
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" },
          gap: 3,
          maxWidth: 720,
          mx: "auto",
        }}
      >
        {PLANS.map((plan) => {
          const price = interval === "year" ? plan.annualPrice : plan.monthlyPrice;
          const isCurrentPlan = currentPlan === plan.id;
          const isDowngrade = currentPlan === "scale" && plan.id === "pro";
          const isLoading = loadingPlan === plan.id;

          return (
            <Paper
              key={plan.id}
              elevation={0}
              sx={(theme) => ({
                p: 3.5,
                borderRadius: "16px",
                border: "1px solid",
                borderColor: plan.recommended
                  ? "primary.main"
                  : "divider",
                position: "relative",
                display: "flex",
                flexDirection: "column",
                ...(plan.recommended && {
                  boxShadow: theme.palette.mode === "dark"
                    ? "0 0 0 1px rgba(139, 92, 246, 0.3)"
                    : "0 0 0 1px rgba(139, 92, 246, 0.2)",
                }),
              })}
            >
              {/* Recommended badge */}
              {plan.recommended && (
                <Chip
                  label="Recommended"
                  size="small"
                  sx={{
                    position: "absolute",
                    top: -30,
                    left: "50%",
                    transform: "translateX(-50%)",
                    bgcolor: "primary.main",
                    color: "#fff",
                    fontWeight: 600,
                    fontSize: "0.75rem",
                  }}
                />
              )}

              {/* Plan name */}
              <Typography variant="h3" sx={{ fontWeight: 600, fontSize: "1.1rem", mb: 1.5 }}>
                {plan.name}
              </Typography>

              {/* Price */}
              <Box sx={{ mb: 0.5 }}>
                <Typography
                  component="span"
                  sx={{
                    fontFamily: "'DM Serif Display', Georgia, serif",
                    fontSize: "2.5rem",
                    fontWeight: 400,
                    letterSpacing: "-0.02em",
                    lineHeight: 1,
                  }}
                >
                  ${price}
                </Typography>
                <Typography
                  component="span"
                  variant="body2"
                  color="text.secondary"
                  sx={{ ml: 0.5 }}
                >
                  /mo
                </Typography>
              </Box>

              <Typography
                variant="body2"
                color="text.secondary"
                sx={{
                  mb: 2,
                  visibility: interval === "year" ? "visible" : "hidden",
                }}
              >
                Billed ${price * 12}/year
              </Typography>

              {/* Features */}
              <Box sx={{ flex: 1, mb: 3 }}>
                {plan.features.map((feature) => (
                  <Box
                    key={feature}
                    sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}
                  >
                    <CheckIcon sx={{ fontSize: 18, color: "primary.main" }} />
                    <Typography variant="body2">{feature}</Typography>
                  </Box>
                ))}
              </Box>

              {/* CTA */}
              {isCurrentPlan ? (
                <Button
                  variant="outlined"
                  disabled
                  fullWidth
                  sx={{ textTransform: "none", py: 1.2 }}
                >
                  Current plan
                </Button>
              ) : isDowngrade ? (
                <Button
                  variant="outlined"
                  disabled
                  fullWidth
                  sx={{ textTransform: "none", py: 1.2 }}
                >
                  Current plan: Scale
                </Button>
              ) : (
                <Button
                  variant="contained"
                  fullWidth
                  onClick={() => handleUpgrade(plan.id)}
                  disabled={isLoading || loadingPlan !== null}
                  sx={{ textTransform: "none", py: 1.2 }}
                >
                  {isLoading ? (
                    <CircularProgress size={20} sx={{ color: "inherit" }} />
                  ) : (
                    `Upgrade to ${plan.name}`
                  )}
                </Button>
              )}
            </Paper>
          );
        })}
      </Box>
    </Box>
  );
}
