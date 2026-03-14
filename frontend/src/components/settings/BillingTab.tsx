"use client";

import { useState } from "react";
import { useAuth } from "@/context/AuthContext";
import { useRouter } from "next/navigation";
import { getStoredToken } from "@/services/auth";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

const PLAN_LABELS: Record<string, string> = {
  free: "Free",
  pro: "Pro",
  scale: "Scale",
};

export default function BillingTab() {
  const { user } = useAuth();
  const router = useRouter();
  const [portalLoading, setPortalLoading] = useState(false);

  const currentPlan = user?.plan || "free";

  async function handleManageSubscription() {
    setPortalLoading(true);
    try {
      const token = getStoredToken();
      const res = await fetch(`${API_BASE}/billing/portal`, {
        method: "POST",
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok) throw new Error("Failed to open billing portal");
      const { url } = await res.json();
      window.location.href = url;
    } catch {
      setPortalLoading(false);
    }
  }

  return (
    <Box>
      <Typography variant="h3" sx={{ mb: 3 }}>
        Billing
      </Typography>

      {/* Current plan */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          Current plan
        </Typography>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
          <Typography sx={{ fontWeight: 600, fontSize: "1.1rem" }}>
            {PLAN_LABELS[currentPlan] || currentPlan}
          </Typography>
          {currentPlan !== "free" && (
            <Chip
              label="Active"
              size="small"
              color="success"
              variant="outlined"
              sx={{ fontSize: "0.75rem" }}
            />
          )}
        </Box>
      </Box>

      {/* Actions */}
      <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
        <Button
          variant="contained"
          onClick={() => router.push("/pricing")}
          sx={{ textTransform: "none" }}
        >
          {currentPlan === "free" ? "Upgrade plan" : "Change plan"}
        </Button>

        {currentPlan !== "free" && (
          <Button
            variant="outlined"
            onClick={handleManageSubscription}
            disabled={portalLoading}
            sx={{ textTransform: "none" }}
          >
            {portalLoading ? (
              <CircularProgress size={18} sx={{ color: "inherit", mr: 1 }} />
            ) : null}
            Manage subscription
          </Button>
        )}
      </Box>
    </Box>
  );
}
