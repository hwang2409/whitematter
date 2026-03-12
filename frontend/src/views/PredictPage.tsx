"use client";
import { useState, useEffect } from "react";
import PredictTab from "@/components/PredictTab";
import { getModels } from "@/api";
import type { Model } from "@/api";

export default function PredictPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  useEffect(() => {
    getModels()
      .then((data) => setModels(data.filter((m) => m.status === "completed")))
      .catch(() => {});
  }, []);

  return (
    <PredictTab
      models={models}
      selectedModel={selectedModel}
      onModelChange={setSelectedModel}
    />
  );
}
