"use client";
import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

import type { TrainingStatus } from "@/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

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

  return (
    <Box
      sx={{
        bgcolor: "#18181B",
        border: "1px solid #27272A",
        borderRadius: "12px",
        p: 3,
        mt: 1,
      }}
    >
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1.5 }}>
        <Typography
          sx={{
            fontSize: "0.9375rem",
            fontWeight: 600,
            fontFamily: "'Outfit', sans-serif",
            color: "#FAFAFA",
          }}
        >
          Training {status?.status === "completed" ? "Complete" : "in Progress"}
        </Typography>
        <Typography
          sx={{
            fontSize: "0.875rem",
            fontFamily: "'JetBrains Mono', monospace",
            color: "#F97316",
            fontWeight: 500,
          }}
        >
          {Math.round(progress)}%
        </Typography>
      </Box>

      <Box
        sx={{
          height: 4,
          bgcolor: "rgba(255,255,255,0.06)",
          borderRadius: "2px",
          overflow: "hidden",
          mb: 2,
        }}
      >
        <Box
          sx={{
            height: "100%",
            width: `${progress}%`,
            bgcolor: "#F97316",
            borderRadius: "2px",
            transition: "width 0.3s ease-out",
          }}
        />
      </Box>

      {history.length > 0 && (
        <Box sx={{ display: "flex", flexDirection: "column" }}>
          {history.map((entry, i) => {
            const isCurrent = i === history.length - 1 && status?.status !== "completed";
            return (
              <Box
                key={i}
                sx={{
                  display: "flex",
                  justifyContent: "space-between",
                  py: 1,
                  borderBottom: i < history.length - 1 ? "1px solid #27272A" : "none",
                  fontSize: "0.8125rem",
                  fontFamily: "'Outfit', sans-serif",
                  color: isCurrent ? "#FAFAFA" : "#A1A1AA",
                  fontWeight: isCurrent ? 500 : 400,
                }}
              >
                <span>Epoch {entry.epoch}/{status?.total_epochs || "?"}</span>
                <Box
                  component="span"
                  sx={{ fontFamily: "'JetBrains Mono', monospace" }}
                >
                  Loss: {entry.loss.toFixed(3)} · Acc: {entry.accuracy.toFixed(1)}%
                </Box>
              </Box>
            );
          })}
        </Box>
      )}

      {status?.message && (
        <Typography
          sx={{
            mt: 1.5,
            fontSize: "0.75rem",
            color: "#52525B",
            fontFamily: "'JetBrains Mono', monospace",
          }}
        >
          {status.message}
        </Typography>
      )}
    </Box>
  );
}
