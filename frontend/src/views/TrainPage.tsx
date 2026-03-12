"use client";
import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import TrainTab from "@/components/TrainTab";
import { getCustomDatasets } from "@/api";
import type { CustomDataset } from "@/api";
import Box from "@mui/material/Box";

export default function TrainPage() {
  const searchParams = useSearchParams();
  const [datasets, setDatasets] = useState<CustomDataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(
    searchParams.get("dataset") || null
  );

  const loadData = async () => {
    try {
      const datasetsData = await getCustomDatasets();
      setDatasets(datasetsData);
    } catch (err) {
      console.error("Failed to load data:", err);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const readyDatasets = datasets.filter((d) => d.status === "ready");

  return (
    <Box sx={{ width: "100%", maxWidth: "100%", p: 0 }}>
      <TrainTab
        datasets={readyDatasets}
        selectedDataset={selectedDataset}
        onDatasetChange={setSelectedDataset}
        onTrainingComplete={loadData}
      />
    </Box>
  );
}
