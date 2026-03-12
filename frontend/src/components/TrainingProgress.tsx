"use client";
import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import LinearProgress from "@mui/material/LinearProgress";
import Chip from "@mui/material/Chip";
import { themeTokens } from "@/theme";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

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

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    const controller = new AbortController();

    async function connectSSE() {
      try {
        const res = await fetch(
          `${API_BASE}/chat/conversations/${conversationId}/training/stream`,
          {
            headers: token ? { Authorization: `Bearer ${token}` } : {},
            signal: controller.signal,
          },
        );
        const reader = res.body?.getReader();
        const decoder = new TextDecoder();

        while (reader) {
          const { done, value } = await reader.read();
          if (done) break;
          const text = decoder.decode(value);
          const lines = text.split("\n");
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data: TrainingStatus = JSON.parse(line.slice(6));
                setStatus(data);
                setHistory((prev) => [...prev, {
                  epoch: data.epoch,
                  loss: data.loss,
                  accuracy: data.accuracy,
                }]);
                if (["completed", "failed", "cancelled"].includes(data.status)) {
                  onComplete?.(data);
                  return;
                }
              } catch {
                // Skip partial SSE chunks
              }
            }
          }
        }
      } catch (err) {
        if (!controller.signal.aborted) console.error("SSE error:", err);
      }
    }

    connectSSE();
    return () => controller.abort();
  }, [conversationId, jobId]);

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
