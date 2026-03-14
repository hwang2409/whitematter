"use client";
import { useState, useEffect } from "react";
import TrainingCard from './training/TrainingCard';
import { TrainingStage } from './training/StageIndicator';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

interface TrainingStatus {
  job_id: string;
  status: string;
  epoch: number;
  total_epochs: number;
  loss: number;
  accuracy: number;
  message: string;
  model_name?: string;
}

interface Props {
  conversationId: string;
  jobId: string;
  onComplete?: (status: TrainingStatus) => void;
  isLatestActive?: boolean;
}

function mapStatusToStage(status?: string): TrainingStage {
  switch (status) {
    case 'preparing': return 'preparing';
    case 'training': return 'training';
    case 'evaluating': return 'evaluating';
    case 'completed': return 'complete';
    case 'ready': return 'ready';
    default: return 'preparing';
  }
}

export default function TrainingProgress({ conversationId, jobId, onComplete, isLatestActive }: Props) {
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

  return (
    <TrainingCard
      modelName={status?.model_name ?? 'Model'}
      stage={mapStatusToStage(status?.status)}
      epoch={status?.epoch ?? 0}
      totalEpochs={status?.total_epochs ?? 0}
      loss={status?.loss ?? 0}
      accuracy={status?.accuracy ?? 0}
      lossHistory={history.map((e) => ({ epoch: e.epoch, loss: e.loss, accuracy: e.accuracy }))}
      isLatestActive={isLatestActive ?? false}
    />
  );
}
