"use client";
import { useState, useEffect, useRef } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import LinearProgress from "@mui/material/LinearProgress";
import Chip from "@mui/material/Chip";
import { themeTokens } from "@/theme";

interface TrainingStatus {
  job_id: string;
  status: string;
  epoch: number;
  total_epochs: number;
  loss: number;
  accuracy: number;
  message: string;
}

interface Props {
  conversationId: string;
  jobId: string;
  onComplete?: (status: TrainingStatus) => void;
}

export default function TrainingProgress({ conversationId, jobId, onComplete }: Props) {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [history, setHistory] = useState<{ epoch: number; loss: number; accuracy: number }[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const token = typeof window !== "undefined" ? localStorage.getItem("access_token") : null;
        const res = await fetch(`/chat/conversations/${conversationId}/training`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (!res.ok) return;
        const data: TrainingStatus = await res.json();
        setStatus(data);

        if (data.epoch > 0) {
          setHistory((prev) => {
            const exists = prev.some((h) => h.epoch === data.epoch);
            if (exists) return prev;
            return [...prev, { epoch: data.epoch, loss: data.loss, accuracy: data.accuracy }];
          });
        }

        if (["completed", "failed", "cancelled"].includes(data.status)) {
          if (intervalRef.current) clearInterval(intervalRef.current);
          onComplete?.(data);
        }
      } catch {
        // ignore polling errors
      }
    };

    poll();
    intervalRef.current = setInterval(poll, 2000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [conversationId, jobId, onComplete]);

  const progress = status && status.total_epochs > 0
    ? (status.epoch / status.total_epochs) * 100
    : 0;

  const statusColor = status?.status === "completed"
    ? "success.main"
    : status?.status === "failed"
    ? "error.main"
    : "primary.main";

  return (
    <Box
      sx={{
        bgcolor: "background.paper",
        border: "1px solid",
        borderColor: "divider",
        borderRadius: 2,
        p: 2,
        maxWidth: 480,
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
        <Typography variant="body2" fontWeight={600}>
          Training
        </Typography>
        <Chip
          size="small"
          label={status?.status || "pending"}
          sx={{
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: "0.6875rem",
            bgcolor: themeTokens.accentLight,
            color: statusColor,
          }}
        />
      </Box>

      <LinearProgress
        variant="determinate"
        value={progress}
        sx={{
          height: 6,
          borderRadius: 3,
          mb: 1.5,
          bgcolor: "action.hover",
          "& .MuiLinearProgress-bar": { borderRadius: 3 },
        }}
      />

      <Box sx={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem" }}>
        <Typography variant="caption" color="text.secondary">
          Epoch {status?.epoch || 0}/{status?.total_epochs || "?"}
        </Typography>
        {status && status.accuracy > 0 && (
          <Typography variant="caption" sx={{ color: themeTokens.accent, fontWeight: 600 }}>
            {status.accuracy.toFixed(1)}%
          </Typography>
        )}
        {status && status.loss > 0 && (
          <Typography variant="caption" color="text.secondary">
            Loss: {status.loss.toFixed(4)}
          </Typography>
        )}
      </Box>

      {status?.message && (
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ mt: 1, display: "block", fontFamily: '"JetBrains Mono", monospace', fontSize: "0.6875rem" }}
        >
          {status.message}
        </Typography>
      )}
    </Box>
  );
}
