"use client";
import { useState, useEffect } from "react";
import TrainTab from "@/components/TrainTab";
import DesignHelper from "@/components/DesignHelper";
import { getCustomDatasets, getModels } from "@/api";
import type { CustomDataset, Architecture } from "@/api";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Paper from "@mui/material/Paper";

export default function TrainRoute() {
  const [datasets, setDatasets] = useState<CustomDataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [designHelperOpen, setDesignHelperOpen] = useState(false);
  const [designHelperContext, setDesignHelperContext] = useState<{
    datasetType?: string;
    architecture?: Architecture | null;
  }>({});
  const [chatMessages, setChatMessages] = useState<{ role: "user" | "assistant"; content: string }[]>([]);

  const loadData = async () => {
    try {
      const [datasetsData] = await Promise.all([getCustomDatasets(), getModels()]);
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
      <Box sx={{ display: "flex", gap: 2, alignItems: "flex-start", width: "100%" }}>
        <Box component="main" sx={{ flex: 1, minWidth: 0 }}>
          <TrainTab
            datasets={readyDatasets}
            selectedDataset={selectedDataset}
            onDatasetChange={setSelectedDataset}
            onTrainingComplete={loadData}
            helperOpen={designHelperOpen}
            onHelperToggle={setDesignHelperOpen}
            onHelperContextChange={setDesignHelperContext}
          />
        </Box>
        {designHelperOpen && (
          <Paper
            variant="outlined"
            sx={{
              width: 450,
              flexShrink: 0,
              borderColor: "divider",
              display: "flex",
              flexDirection: "column",
              position: "sticky",
              top: 24,
              overflow: "hidden",
            }}
          >
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                p: 1.25,
                borderBottom: "1px solid",
                borderColor: "divider",
                bgcolor: "background.paper",
              }}
            >
              <Typography variant="subtitle1" fontWeight={600}>
                AI Design Assistant
              </Typography>
              <Button
                size="small"
                onClick={() => setDesignHelperOpen(false)}
                sx={{
                  minWidth: 28,
                  height: 28,
                  color: "text.secondary",
                  "&:hover": { bgcolor: "action.hover", color: "text.primary" },
                }}
              >
                ×
              </Button>
            </Box>
            <Box sx={{ display: "flex", flexDirection: "column", overflow: "hidden", flex: 1 }}>
              <DesignHelper
                datasetType={designHelperContext.datasetType}
                currentArchitecture={designHelperContext.architecture}
                messages={chatMessages}
                onMessagesChange={setChatMessages}
              />
            </Box>
          </Paper>
        )}
      </Box>
    </Box>
  );
}
