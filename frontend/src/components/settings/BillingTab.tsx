"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/context/AuthContext";
import Toast from "@/components/Toast";
import { useToast } from "@/components/Toast";
import {
  getBillingStatus,
  getBillingUsage,
  createCheckoutSession,
  createPortalSession,
  type BillingStatus,
  type BillingUsage,
} from "@/api";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";

interface PlanDef {
  id: string;
  name: string;
  price: string;
  features: string[];
}

const PLANS: PlanDef[] = [
  {
    id: "free",
    name: "Free",
    price: "$0",
    features: ["5 trains/day", "3 models", "1 GB storage"],
  },
  {
    id: "pro",
    name: "Pro",
    price: "$29",
    features: ["Unlimited trains", "Unlimited models", "20 GB", "1 endpoint"],
  },
  {
    id: "scale",
    name: "Scale",
    price: "$59",
    features: ["Everything in Pro", "GPU access", "100 GB", "5 endpoints"],
  },
];

export default function BillingTab() {
  const { token } = useAuth();
  const { toasts, dismissToast, error: showError } = useToast();

  const [status, setStatus] = useState<BillingStatus | null>(null);
  const [usage, setUsage] = useState<BillingUsage | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    (async () => {
      try {
        const [s, u] = await Promise.all([getBillingStatus(token), getBillingUsage(token)]);
        if (!cancelled) { setStatus(s); setUsage(u); }
      } catch (e) {
        if (!cancelled) showError(e instanceof Error ? e.message : "Failed to load billing");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const currentPlan = status?.plan || "free";

  const handleCheckout = async (planId: string) => {
    if (!token) return;
    setActionLoading(planId);
    try {
      const { url } = await createCheckoutSession(token, planId);
      window.location.href = url;
    } catch (e) {
      showError(e instanceof Error ? e.message : "Failed to start checkout");
    } finally {
      setActionLoading(null);
    }
  };

  const handlePortal = async () => {
    if (!token) return;
    setActionLoading("portal");
    try {
      const { url } = await createPortalSession(token);
      window.location.href = url;
    } catch (e) {
      showError(e instanceof Error ? e.message : "Failed to open billing portal");
    } finally {
      setActionLoading(null);
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", py: 8 }}>
        <CircularProgress size={20} sx={{ color: "#F97316" }} />
      </Box>
    );
  }

  const labelSx = {
    fontSize: "0.6875rem",
    fontWeight: 600,
    fontFamily: "'Outfit', sans-serif",
    color: "#52525B",
    textTransform: "uppercase" as const,
    letterSpacing: "0.08em",
    mb: 1.5,
  };

  const planRank = (id: string) => PLANS.findIndex((p) => p.id === id);

  return (
    <Box>
      {usage && (
        <Box sx={{ mb: 5 }}>
          <Typography sx={labelSx}>Usage</Typography>
          <Box sx={{ display: "flex", gap: 5 }}>
            {[
              { label: "Models", value: usage.models_count },
              { label: "Datasets", value: usage.datasets_count },
              { label: "Chats", value: usage.conversations_count },
            ].map((item) => (
              <Box key={item.label}>
                <Typography
                  sx={{
                    fontSize: "1.5rem",
                    fontFamily: "'Instrument Serif', Georgia, serif",
                    fontWeight: 400,
                    color: "#FAFAFA",
                    lineHeight: 1,
                  }}
                >
                  {item.value}
                </Typography>
                <Typography
                  sx={{
                    fontSize: "0.75rem",
                    color: "#52525B",
                    fontFamily: "'Outfit', sans-serif",
                    mt: 0.5,
                  }}
                >
                  {item.label}
                </Typography>
              </Box>
            ))}
          </Box>
        </Box>
      )}

      <Box sx={{ mb: 5 }}>
        <Typography sx={labelSx}>Plan</Typography>
        <Box sx={{ display: "flex", gap: 1.5, flexWrap: "wrap" }}>
          {PLANS.map((plan) => {
            const isCurrent = plan.id === currentPlan;
            const rank = planRank(plan.id);
            const curRank = planRank(currentPlan);
            const action = rank > curRank ? "Upgrade" : "Downgrade";

            return (
              <Box
                key={plan.id}
                sx={{
                  flex: "1 1 180px",
                  maxWidth: 240,
                  p: 2.5,
                  borderRadius: "10px",
                  border: "1px solid",
                  borderColor: isCurrent ? "#3F3F46" : "#27272A",
                  bgcolor: "#18181B",
                  display: "flex",
                  flexDirection: "column",
                  transition: "border-color 0.15s ease",
                  "&:hover": {
                    borderColor: "#3F3F46",
                  },
                }}
              >
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "baseline",
                    mb: 1.5,
                  }}
                >
                  <Typography
                    sx={{
                      fontSize: "0.8125rem",
                      fontWeight: 500,
                      fontFamily: "'Outfit', sans-serif",
                      color: "#FAFAFA",
                    }}
                  >
                    {plan.name}
                  </Typography>
                  <Typography
                    sx={{
                      fontSize: "0.8125rem",
                      fontFamily: "'Outfit', sans-serif",
                      color: "#A1A1AA",
                    }}
                  >
                    {plan.price}
                    <Typography
                      component="span"
                      sx={{
                        fontSize: "0.6875rem",
                        color: "#52525B",
                        fontFamily: "'Outfit', sans-serif",
                      }}
                    >
                      {plan.id !== "free" ? "/mo" : ""}
                    </Typography>
                  </Typography>
                </Box>
                <Box sx={{ flex: 1, mb: 2 }}>
                  {plan.features.map((f) => (
                    <Typography
                      key={f}
                      sx={{
                        fontSize: "0.75rem",
                        color: "#A1A1AA",
                        fontFamily: "'Outfit', sans-serif",
                        lineHeight: 1.8,
                      }}
                    >
                      {f}
                    </Typography>
                  ))}
                </Box>
                {isCurrent ? (
                  <Typography
                    sx={{
                      fontSize: "0.6875rem",
                      color: "#52525B",
                      fontFamily: "'Outfit', sans-serif",
                      textAlign: "center",
                    }}
                  >
                    Current
                  </Typography>
                ) : (
                  <Button
                    size="small"
                    variant="outlined"
                    disabled={actionLoading === plan.id}
                    onClick={() => handleCheckout(plan.id)}
                    sx={{
                      fontSize: "0.75rem",
                      fontFamily: "'Outfit', sans-serif",
                      textTransform: "none",
                      borderColor: "#27272A",
                      color: "#A1A1AA",
                      borderRadius: "8px",
                      "&:hover": {
                        borderColor: "#3F3F46",
                        color: "#FAFAFA",
                        bgcolor: "rgba(255,255,255,0.03)",
                      },
                    }}
                  >
                    {actionLoading === plan.id ? <CircularProgress size={14} /> : action}
                  </Button>
                )}
              </Box>
            );
          })}
        </Box>
      </Box>

      {currentPlan !== "free" && (
        <Box sx={{ pt: 4, borderTop: "1px solid #27272A" }}>
          <Button
            variant="outlined"
            size="small"
            onClick={handlePortal}
            disabled={actionLoading === "portal"}
            sx={{
              borderColor: "#27272A",
              color: "#A1A1AA",
              fontFamily: "'Outfit', sans-serif",
              fontSize: "0.75rem",
              textTransform: "none",
              borderRadius: "8px",
              "&:hover": {
                borderColor: "#3F3F46",
                color: "#FAFAFA",
                bgcolor: "rgba(255,255,255,0.03)",
              },
            }}
          >
            {actionLoading === "portal" ? <CircularProgress size={16} /> : "Manage billing"}
          </Button>
        </Box>
      )}

      <Toast toasts={toasts} onDismiss={dismissToast} />
    </Box>
  );
}
