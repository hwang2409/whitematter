"use client";
import { useState } from "react";
import NextLink from "next/link";
import * as api from "@/api";
import {
  QUICK_START_DATASET_HF_ID,
  QUICK_START_DATASET_NAME,
} from "@/lib/quickStart";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Stepper from "@mui/material/Stepper";
import Step from "@mui/material/Step";
import StepLabel from "@mui/material/StepLabel";
import Alert from "@mui/material/Alert";
import CircularProgress from "@mui/material/CircularProgress";
import Link from "@mui/material/Link";

const STEPS = [
  "Upload your first dataset",
  "Design your architecture",
  "Train your model",
  "Make a prediction",
];

interface Props {
  userId: string;
  onDatasetImported: (dataset: api.CustomDataset) => void;
}

function getStorageKey(userId: string) {
  return `wm_onboarding_${userId}`;
}

function getSavedStep(userId: string): number {
  if (typeof window === "undefined") return 0;
  const saved = localStorage.getItem(getStorageKey(userId));
  return saved ? parseInt(saved, 10) : 0;
}

function saveStep(userId: string, step: number) {
  if (typeof window === "undefined") return;
  localStorage.setItem(getStorageKey(userId), String(step));
}

export default function OnboardingWizard({
  userId,
  onDatasetImported,
}: Props) {
  const [activeStep, setActiveStep] = useState(getSavedStep(userId));
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState("");
  const [importedDataset, setImportedDataset] =
    useState<api.CustomDataset | null>(null);

  function goToStep(step: number) {
    setActiveStep(step);
    saveStep(userId, step);
  }

  async function handleImportSample() {
    setImporting(true);
    setError("");
    try {
      const dataset = await api.importDatasetFromHuggingFace(
        QUICK_START_DATASET_HF_ID,
        {
          name: QUICK_START_DATASET_NAME,
          split: "train",
        }
      );
      setImportedDataset(dataset);
      onDatasetImported(dataset);
      goToStep(1);
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Failed to import sample dataset. Please try again."
      );
    } finally {
      setImporting(false);
    }
  }

  return (
    <Box>
      <Typography variant="h2" sx={{ mb: 0.5 }}>
        Welcome to whitematter
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Let&apos;s get you up and running in a few steps.
      </Typography>

      <Stepper activeStep={activeStep} sx={{ mb: 3 }}>
        {STEPS.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError("")}>
          {error}
          <Box sx={{ mt: 1 }}>
            <Button
              size="small"
              variant="outlined"
              onClick={handleImportSample}
              sx={{ mr: 1 }}
            >
              Try again
            </Button>
            <Link
              component={NextLink}
              href="/data"
              color="inherit"
              sx={{ fontSize: "0.875rem" }}
            >
              Upload manually
            </Link>
          </Box>
        </Alert>
      )}

      {activeStep === 0 && (
        <Box sx={{ textAlign: "center", py: 3 }}>
          <Typography variant="body1" sx={{ mb: 2 }}>
            Start by uploading a dataset, or use our sample MNIST dataset to get
            started instantly.
          </Typography>
          <Box
            sx={{
              display: "flex",
              gap: 2,
              justifyContent: "center",
              flexWrap: "wrap",
            }}
          >
            <Button
              variant="contained"
              onClick={handleImportSample}
              disabled={importing}
              startIcon={
                importing ? <CircularProgress size={16} /> : undefined
              }
            >
              {importing ? "Importing MNIST..." : "Use sample dataset"}
            </Button>
            <Button variant="outlined" component={NextLink} href="/data">
              Upload your own
            </Button>
          </Box>
          <Button
            size="small"
            sx={{ mt: 2, color: "text.secondary" }}
            onClick={() => goToStep(1)}
          >
            Skip
          </Button>
        </Box>
      )}

      {activeStep === 1 && (
        <Box sx={{ textAlign: "center", py: 3 }}>
          <Typography variant="body1" sx={{ mb: 1 }}>
            Use the AI-powered architecture designer to create a neural network
            for your data.
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Describe what you want to build and Claude will design the
            architecture.
          </Typography>
          <Button
            variant="contained"
            component={NextLink}
            href={
              importedDataset
                ? `/architect?dataset=${importedDataset.id}`
                : "/architect"
            }
          >
            Open AI Architect
          </Button>
          <br />
          <Button
            size="small"
            sx={{ mt: 2, color: "text.secondary" }}
            onClick={() => goToStep(2)}
          >
            Skip
          </Button>
        </Box>
      )}

      {activeStep === 2 && (
        <Box sx={{ textAlign: "center", py: 3 }}>
          <Typography variant="body1" sx={{ mb: 1 }}>
            Configure training parameters and watch your model train in
            real-time.
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            You&apos;ll see a live chart with loss and accuracy as training
            progresses.
          </Typography>
          <Button variant="contained" component={NextLink} href="/train">
            Start Training
          </Button>
          <br />
          <Button
            size="small"
            sx={{ mt: 2, color: "text.secondary" }}
            onClick={() => goToStep(3)}
          >
            Skip
          </Button>
        </Box>
      )}

      {activeStep === 3 && (
        <Box sx={{ textAlign: "center", py: 3 }}>
          <Typography variant="body1" sx={{ mb: 1 }}>
            Test your trained model by uploading an image for prediction.
          </Typography>
          <Button variant="contained" component={NextLink} href="/predict">
            Try Prediction
          </Button>
          <br />
          <Button
            size="small"
            sx={{ mt: 2, color: "text.secondary" }}
            onClick={() => goToStep(4)}
          >
            Finish
          </Button>
        </Box>
      )}
    </Box>
  );
}
