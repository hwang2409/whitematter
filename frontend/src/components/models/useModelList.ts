"use client";
import { useState, useEffect, useCallback, useMemo } from "react";
import * as api from "@/api";

export type SortColumn = "name" | "architecture" | "status" | "accuracy" | "loss" | "created";
export type SortDirection = "asc" | "desc";

export function formatModelName(name: string) {
  const parts = name.replace("custom_", "").split("_");
  if (parts.length > 1) {
    const baseName = parts.slice(0, -1).join("_");
    return baseName
      .split(/[-_]/)
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ");
  }
  return name;
}

export function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleString();
}

export function getModelLoss(model: api.Model): number | null {
  return model.training_history?.[model.training_history.length - 1]?.loss ?? null;
}

export function isTextModel(model: api.Model) {
  const name = model.name.toLowerCase();
  const arch = model.architecture.toLowerCase();
  return (
    model.dataset.startsWith("custom:") &&
    (name.includes("gpt") ||
      name.includes("language") ||
      name.includes("text") ||
      arch.includes("gpt") ||
      arch.includes("language") ||
      arch.includes("text") ||
      arch.includes("lstm"))
  );
}

interface UseModelListOptions {
  onUpdate?: () => void;
}

export default function useModelList({ onUpdate }: UseModelListOptions = {}) {
  const [models, setModels] = useState<api.Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [sortColumn, setSortColumn] = useState<SortColumn>("created");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [modelSearch, setModelSearch] = useState("");

  const loadModels = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.getModels();
      setModels(data);
      onUpdate?.();
    } catch {
      setModels([]);
    } finally {
      setLoading(false);
    }
  }, [onUpdate]);

  useEffect(() => {
    loadModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleSortClick(column: SortColumn) {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortColumn(column);
      setSortDirection("asc");
    }
  }

  const sortedFilteredModels = useMemo(() => {
    const sorted = [...models].sort((a, b) => {
      let cmp = 0;
      switch (sortColumn) {
        case "name":
          cmp = formatModelName(a.name).localeCompare(formatModelName(b.name));
          break;
        case "architecture":
          cmp = a.architecture.localeCompare(b.architecture);
          break;
        case "status":
          cmp = a.status.localeCompare(b.status);
          break;
        case "accuracy":
          cmp = a.best_accuracy - b.best_accuracy;
          break;
        case "loss": {
          const aLoss = getModelLoss(a);
          const bLoss = getModelLoss(b);
          if (aLoss === null && bLoss === null) cmp = 0;
          else if (aLoss === null) cmp = 1;
          else if (bLoss === null) cmp = -1;
          else cmp = aLoss - bLoss;
          break;
        }
        case "created":
          cmp = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
          break;
      }
      return sortDirection === "asc" ? cmp : -cmp;
    });

    return sorted.filter((m) =>
      modelSearch
        ? formatModelName(m.name).toLowerCase().includes(modelSearch.toLowerCase()) ||
          m.architecture.toLowerCase().includes(modelSearch.toLowerCase())
        : true
    );
  }, [models, sortColumn, sortDirection, modelSearch]);

  async function deleteModel(id: string) {
    try {
      await api.deleteModel(id);
      setModels((prev) => prev.filter((m) => m.id !== id));
    } catch {
      setError("Failed to delete model");
    }
  }

  async function resumeTraining(id: string) {
    setError("");
    await api.resumeTraining(id);
    setModels((prev) => prev.map((m) => (m.id === id ? { ...m, status: "running" as const } : m)));
  }

  return {
    models,
    setModels,
    loading,
    error,
    setError,
    sortColumn,
    sortDirection,
    modelSearch,
    setModelSearch,
    handleSortClick,
    sortedFilteredModels,
    loadModels,
    deleteModel,
    resumeTraining,
  };
}
