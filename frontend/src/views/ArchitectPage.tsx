// frontend/src/views/ArchitectPage.tsx
"use client";
import { useState, useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import * as api from "@/api";
import { useDesign } from "@/context/DesignContext";
import DesignHelper from "@/components/DesignHelper";
import ArchitectureGraph from "@/components/ArchitectureGraph";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Paper from "@mui/material/Paper";
import Chip from "@mui/material/Chip";
import Alert from "@mui/material/Alert";

export default function ArchitectPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { architecture, setArchitecture } = useDesign();

  const [datasets, setDatasets] = useState<api.CustomDataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>(
    searchParams.get("dataset") || ""
  );
  const [loading, setLoading] = useState(true);
  const [suggesting, setSuggesting] = useState(false);
  const [error, setError] = useState("");
  const [chatMessages, setChatMessages] = useState<
    { role: "user" | "assistant"; content: string }[]
  >([]);

  useEffect(() => {
    api
      .getCustomDatasets()
      .then((ds) => {
        setDatasets(ds.filter((d) => d.status === "ready"));
      })
      .catch(() => setError("Failed to load datasets"))
      .finally(() => setLoading(false));
  }, []);

  const selectedDataset = datasets.find((d) => d.id === selectedDatasetId);

  async function handleSuggest() {
    if (!selectedDatasetId) return;
    setSuggesting(true);
    setError("");
    try {
      const result = await api.suggestArchitecture(
        selectedDatasetId,
        "Suggest a good architecture for this dataset"
      );
      setArchitecture(result.architecture);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Failed to suggest architecture"
      );
    } finally {
      setSuggesting(false);
    }
  }

  function handleSendToTraining() {
    if (!architecture || !selectedDatasetId) return;
    router.push(`/train?dataset=${selectedDatasetId}`);
  }

  return (
    <Box>
      <Typography variant="h2" sx={{ mb: 0.5 }}>
        AI Architecture Designer
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Describe what you want to build. Claude designs the neural network.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError("")}>
          {error}
        </Alert>
      )}

      <Box sx={{ display: "flex", gap: 3, alignItems: "flex-start" }}>
        {/* Left column: dataset + architecture */}
        <Box sx={{ flex: "0 0 60%", minWidth: 0 }}>
          <FormControl fullWidth size="small" sx={{ mb: 2 }}>
            <InputLabel>Dataset</InputLabel>
            <Select
              value={selectedDatasetId}
              label="Dataset"
              onChange={(e) => setSelectedDatasetId(e.target.value)}
              disabled={loading}
            >
              {datasets.map((d) => (
                <MenuItem key={d.id} value={d.id}>
                  {d.name} ({d.data_type}, {d.num_classes} classes)
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {!selectedDatasetId && (
            <Typography
              color="text.secondary"
              sx={{ py: 4, textAlign: "center" }}
            >
              Select a dataset to start designing an architecture.
            </Typography>
          )}

          {selectedDatasetId && !architecture && (
            <Paper
              variant="outlined"
              sx={{ p: 3, textAlign: "center", borderColor: "divider" }}
            >
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ mb: 2 }}
              >
                Use the chat on the right to describe your model, or let AI
                suggest one.
              </Typography>
              <Button
                variant="contained"
                onClick={handleSuggest}
                disabled={suggesting}
              >
                {suggesting ? "Generating..." : "Suggest Architecture"}
              </Button>
            </Paper>
          )}

          {architecture && (
            <Box>
              <Paper
                variant="outlined"
                sx={{ p: 2, mb: 2, borderColor: "divider" }}
              >
                <Typography variant="h3" sx={{ mb: 1 }}>
                  {architecture.name}
                </Typography>
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ mb: 1.5 }}
                >
                  {architecture.description}
                </Typography>
                <Box
                  sx={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: 0.5,
                    mb: 1.5,
                  }}
                >
                  {architecture.layers.map((layer, i) => (
                    <Box
                      key={i}
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 0.5,
                      }}
                    >
                      <Chip
                        size="small"
                        label={layer.type}
                        sx={{
                          fontFamily: '"JetBrains Mono", monospace',
                          fontSize: "0.6875rem",
                        }}
                      />
                      {i < architecture.layers.length - 1 && (
                        <Typography
                          component="span"
                          color="text.secondary"
                          sx={{ fontSize: "0.75rem" }}
                        >
                          →
                        </Typography>
                      )}
                    </Box>
                  ))}
                </Box>
                <ArchitectureGraph architecture={architecture} />
              </Paper>

              <Box sx={{ display: "flex", gap: 1 }}>
                <Button
                  variant="contained"
                  onClick={handleSendToTraining}
                  disabled={!selectedDatasetId}
                >
                  Send to Training
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => setArchitecture(null)}
                >
                  Clear
                </Button>
              </Box>
            </Box>
          )}
        </Box>

        {/* Right column: chat */}
        <Box
          sx={{
            flex: "0 0 40%",
            minWidth: 0,
            position: "sticky",
            top: 24,
          }}
        >
          <Paper
            variant="outlined"
            sx={{ borderColor: "divider", overflow: "hidden" }}
          >
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                p: 1.25,
                borderBottom: "1px solid",
                borderColor: "divider",
              }}
            >
              <Typography variant="subtitle1" fontWeight={600}>
                AI Design Assistant
              </Typography>
            </Box>
            <DesignHelper
              datasetType={selectedDataset?.data_type}
              currentArchitecture={architecture}
              messages={chatMessages}
              onMessagesChange={setChatMessages}
            />
          </Paper>
        </Box>
      </Box>
    </Box>
  );
}
