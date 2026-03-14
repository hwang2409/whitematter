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
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import Divider from "@mui/material/Divider";
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
    features: [
      "5 training runs / day",
      "3 saved models",
      "1 GB storage",
      "No GPU access",
      "No deploy endpoints",
    ],
  },
  {
    id: "pro",
    name: "Pro",
    price: "$29/mo",
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
    price: "$59/mo",
    features: [
      "Everything in Pro",
      "GPU access",
      "100 GB storage",
      "5 deploy endpoints",
    ],
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
        const [s, u] = await Promise.all([
          getBillingStatus(token),
          getBillingUsage(token),
        ]);
        if (!cancelled) {
          setStatus(s);
          setUsage(u);
        }
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

  const handleUpgradeDowngrade = async (planId: string) => {
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

  const handleManageBilling = async () => {
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
      <Box sx={{ display: "flex", justifyContent: "center", py: 6 }}>
        <CircularProgress />
      </Box>
    );
  }

  const planRank = (id: string) => PLANS.findIndex((p) => p.id === id);

  return (
    <Box>
      <Typography variant="h3" sx={{ mb: 3 }}>
        Billing
      </Typography>

      {/* Current Plan */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="subtitle1" sx={{ mb: 1.5, fontWeight: 600 }}>
          Current Plan
        </Typography>
        <Card
          variant="outlined"
          sx={{ p: 2, display: "inline-flex", alignItems: "center", gap: 1.5 }}
        >
          <Typography variant="h6">
            {PLANS.find((p) => p.id === currentPlan)?.name || currentPlan}
          </Typography>
          <Chip label="Current Plan" size="small" color="primary" />
        </Card>
      </Box>

      {/* Usage */}
      {usage && (
        <Box sx={{ mb: 4 }}>
          <Typography variant="subtitle1" sx={{ mb: 1.5, fontWeight: 600 }}>
            Usage
          </Typography>
          <Box sx={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
            {[
              { label: "Models", value: usage.models_count },
              { label: "Datasets", value: usage.datasets_count },
              { label: "Conversations", value: usage.conversations_count },
            ].map((item) => (
              <Card key={item.label} variant="outlined" sx={{ p: 2, minWidth: 120, textAlign: "center" }}>
                <Typography variant="h5">{item.value}</Typography>
                <Typography variant="body2" color="text.secondary">
                  {item.label}
                </Typography>
              </Card>
            ))}
          </Box>
        </Box>
      )}

      <Divider sx={{ mb: 3 }} />

      {/* Plan Comparison */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="subtitle1" sx={{ mb: 1.5, fontWeight: 600 }}>
          Plans
        </Typography>
        <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
          {PLANS.map((plan) => {
            const isCurrent = plan.id === currentPlan;
            const rank = planRank(plan.id);
            const currentRank = planRank(currentPlan);
            const actionLabel = rank > currentRank ? "Upgrade" : "Downgrade";

            return (
              <Card
                key={plan.id}
                variant="outlined"
                sx={{
                  p: 2.5,
                  flex: "1 1 220px",
                  maxWidth: 280,
                  display: "flex",
                  flexDirection: "column",
                  border: isCurrent ? 2 : 1,
                  borderColor: isCurrent ? "primary.main" : "divider",
                }}
              >
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
                  <Typography variant="h6">{plan.name}</Typography>
                  {isCurrent && <Chip label="Current Plan" size="small" color="primary" />}
                </Box>
                <Typography variant="h5" sx={{ mb: 1.5 }}>
                  {plan.price}
                </Typography>
                <Box component="ul" sx={{ pl: 2, mb: 2, flex: 1 }}>
                  {plan.features.map((f) => (
                    <Typography component="li" variant="body2" color="text.secondary" key={f} sx={{ mb: 0.5 }}>
                      {f}
                    </Typography>
                  ))}
                </Box>
                {!isCurrent && (
                  <Button
                    variant="contained"
                    size="small"
                    disabled={actionLoading === plan.id}
                    onClick={() => handleUpgradeDowngrade(plan.id)}
                  >
                    {actionLoading === plan.id ? (
                      <CircularProgress size={18} />
                    ) : (
                      actionLabel
                    )}
                  </Button>
                )}
              </Card>
            );
          })}
        </Box>
      </Box>

      {/* Manage Subscription */}
      {currentPlan !== "free" && (
        <>
          <Divider sx={{ mb: 3 }} />
          <Box>
            <Typography variant="subtitle1" sx={{ mb: 1.5, fontWeight: 600 }}>
              Manage Subscription
            </Typography>
            <Button
              variant="outlined"
              onClick={handleManageBilling}
              disabled={actionLoading === "portal"}
            >
              {actionLoading === "portal" ? (
                <CircularProgress size={20} />
              ) : (
                "Manage Billing"
              )}
            </Button>
          </Box>
        </>
      )}

      <Toast toasts={toasts} onDismiss={dismissToast} />
    </Box>
  );
}
