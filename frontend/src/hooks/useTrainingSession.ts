import { useState, useEffect, useCallback } from "react";
import * as api from "@/api";

interface TrainingHistoryEntry {
  epoch: number;
  loss: number;
  accuracy: number;
}

export function useTrainingSession(onTrainingComplete?: () => void) {
  const [training, setTraining] = useState(false);
  const [trainingJob, setTrainingJob] = useState<api.CustomTrainJob | null>(null);
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistoryEntry[]>([]);
  const [trainingError, setTrainingError] = useState("");

  function handleStatusUpdate(status: api.CustomTrainJob) {
    setTrainingJob(status);

    const epoch = status.epoch ?? 0;
    const loss = status.loss ?? 0;
    const accuracy = status.accuracy ?? 0;

    if (epoch > 0) {
      setTrainingHistory((prev) => {
        const existing = prev.find((h) => h.epoch === epoch);
        if (existing) return prev;
        return [...prev, { epoch, loss, accuracy }];
      });
    }

    if (!["pending", "compiling", "training", "running"].includes(status.status)) {
      setTraining(false);
      if (status.status === "completed") {
        onTrainingComplete?.();
      }
    }
  }

  // WebSocket-first with HTTP polling fallback
  useEffect(() => {
    if (!trainingJob || !["pending", "compiling", "training", "running"].includes(trainingJob.status)) {
      return;
    }

    const jobId = trainingJob.job_id;
    let pollInterval: number | undefined;
    let wsHandle: { close: () => void } | null = null;

    function startPolling() {
      pollInterval = window.setInterval(async () => {
        try {
          const status = await api.getCustomTrainingStatus(jobId);
          handleStatusUpdate(status);
        } catch (e) {
          console.error("Failed to get training status:", e);
        }
      }, 2000);
    }

    wsHandle = api.createTrainingWebSocket(
      jobId,
      (status) => handleStatusUpdate(status),
      () => {
        if (!pollInterval) startPolling();
      },
      () => {
        wsHandle = null;
        if (!pollInterval) startPolling();
      },
    );

    return () => {
      wsHandle?.close();
      if (pollInterval) clearInterval(pollInterval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainingJob?.job_id]);

  const startTrainingJob = useCallback(
    (job: api.CustomTrainJob) => {
      setTraining(true);
      setTrainingError("");
      setTrainingHistory([]);
      setTrainingJob(job);
    },
    [],
  );

  const cancelTraining = useCallback(async () => {
    if (!trainingJob) return;
    try {
      await api.cancelTraining(trainingJob.job_id);
      setTraining(false);
      setTrainingJob(null);
    } catch {
      setTrainingError("Failed to cancel training");
    }
  }, [trainingJob]);

  return {
    training,
    trainingJob,
    trainingHistory,
    trainingError,
    setTrainingError,
    startTrainingJob,
    cancelTraining,
  };
}
