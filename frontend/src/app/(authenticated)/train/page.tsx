"use client";
import { useState, useEffect } from "react";
import TrainTab from "@/components/TrainTab";
import DesignHelper from "@/components/DesignHelper";
import { getCustomDatasets, getModels } from "@/api";
import type { CustomDataset, Architecture } from "@/api";

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
    <div className={designHelperOpen ? "app with-sidebar" : ""} style={{ maxWidth: "100%", padding: 0 }}>
      <div className="app-body" style={{ width: "100%" }}>
        <main className="content">
          <TrainTab
            datasets={readyDatasets}
            selectedDataset={selectedDataset}
            onDatasetChange={setSelectedDataset}
            onTrainingComplete={loadData}
            helperOpen={designHelperOpen}
            onHelperToggle={setDesignHelperOpen}
            onHelperContextChange={setDesignHelperContext}
          />
        </main>
        {designHelperOpen && (
          <aside className="app-sidebar">
            <div className="sidebar-header">
              <h3>AI Design Assistant</h3>
              <button className="sidebar-close" onClick={() => setDesignHelperOpen(false)}>
                &times;
              </button>
            </div>
            <div className="sidebar-body">
              <DesignHelper
                datasetType={designHelperContext.datasetType}
                currentArchitecture={designHelperContext.architecture}
                messages={chatMessages}
                onMessagesChange={setChatMessages}
              />
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
