"use client";
import { useState } from "react";
import Box from "@mui/material/Box";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Typography from "@mui/material/Typography";
import DatasetsTab from "@/components/DatasetsTab";
import S3ManagerPage from "@/views/S3ManagerPage";

type DataTab = "datasets" | "storage";

export default function DataPage() {
  const [tab, setTab] = useState<DataTab>("datasets");

  return (
    <Box>
      <Typography variant="h2" sx={{ mb: 1 }}>
        Data
      </Typography>
      <Tabs
        value={tab}
        onChange={(_, v: DataTab) => setTab(v)}
        sx={{
          mb: 2,
          minHeight: 40,
          "& .MuiTab-root": { minHeight: 40, textTransform: "none", fontSize: "0.9375rem" },
        }}
      >
        <Tab value="datasets" label="Datasets" />
        <Tab value="storage" label="S3 Storage" />
      </Tabs>
      {tab === "datasets" && (
        <DatasetsTab
          onDatasetSelect={() => {}}
          onUpdate={() => {}}
        />
      )}
      {tab === "storage" && <S3ManagerPage />}
    </Box>
  );
}
