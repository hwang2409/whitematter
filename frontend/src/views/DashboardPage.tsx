"use client";
import Link from "next/link";
import { useAuth } from "@/context/AuthContext";
import { useEffect, useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import * as api from "@/api";
import { getCustomDatasets, getModels } from "@/api";
import OnboardingWizard from "@/components/OnboardingWizard";
import {
  QUICK_START_ARCHITECTURE,
  QUICK_START_DATASET_HF_ID,
  QUICK_START_DATASET_NAME,
} from "@/lib/quickStart";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import CardContent from "@mui/material/CardContent";
import Button from "@mui/material/Button";
import UploadFileOutlined from "@mui/icons-material/UploadFileOutlined";
import PlayArrowOutlined from "@mui/icons-material/PlayArrowOutlined";
import AutoAwesomeOutlined from "@mui/icons-material/AutoAwesomeOutlined";
import RocketLaunchOutlined from "@mui/icons-material/RocketLaunchOutlined";
import { themeTokens } from "@/theme";

type ActivityItem = {
  id: string;
  type: "dataset_uploaded" | "training_started" | "model_completed";
  label: string;
  at: string;
};

function formatRelativeTime(dateStr: string): string {
  const d = new Date(dateStr);
  const now = new Date();
  const sec = Math.floor((now.getTime() - d.getTime()) / 1000);
  if (sec < 60) return "Just now";
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
  if (sec < 604800) return `${Math.floor(sec / 86400)}d ago`;
  return d.toLocaleDateString();
}

export default function DashboardPage() {
  const { user } = useAuth();
  const router = useRouter();
  const [datasets, setDatasets] = useState<Awaited<ReturnType<typeof getCustomDatasets>>>([]);
  const [models, setModels] = useState<Awaited<ReturnType<typeof getModels>>>([]);
  const [loading, setLoading] = useState(true);
  const [quickStarting, setQuickStarting] = useState(false);
  const [quickStartError, setQuickStartError] = useState("");

  useEffect(() => {
    Promise.all([getCustomDatasets(), getModels()])
      .then(([ds, ms]) => {
        setDatasets(ds);
        setModels(ms);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  async function handleQuickStart() {
    setQuickStarting(true);
    setQuickStartError("");
    try {
      let mnistDataset = datasets.find(
        (d) =>
          d.name === QUICK_START_DATASET_NAME ||
          d.name.toLowerCase().includes("mnist")
      );
      if (!mnistDataset) {
        mnistDataset = await api.importDatasetFromHuggingFace(
          QUICK_START_DATASET_HF_ID,
          {
            name: QUICK_START_DATASET_NAME,
            split: "train",
          }
        );
      }
      await api.startCustomTraining(mnistDataset.id, QUICK_START_ARCHITECTURE);
      router.push("/train");
    } catch (e) {
      setQuickStartError(
        e instanceof Error ? e.message : "Quick start failed"
      );
    } finally {
      setQuickStarting(false);
    }
  }

  const readyDatasets = useMemo(() => datasets.filter((d) => d.status === "ready"), [datasets]);
  const completedModels = useMemo(() => models.filter((m) => m.status === "completed"), [models]);
  const activeTraining = useMemo(() => models.filter((m) => m.status === "running").length, [models]);

  const activity: ActivityItem[] = useMemo(() => {
    const items: ActivityItem[] = [];
    datasets.forEach((d) => {
      items.push({
        id: `ds-${d.id}`,
        type: "dataset_uploaded",
        label: `Dataset "${d.name}" uploaded`,
        at: d.created_at,
      });
    });
    models.forEach((m) => {
      items.push({
        id: `m-${m.id}`,
        type: m.status === "completed" ? "model_completed" : "training_started",
        label: m.status === "completed" ? `Model "${m.name}" completed` : `Training started for "${m.name}"`,
        at: m.created_at,
      });
    });
    items.sort((a, b) => new Date(b.at).getTime() - new Date(a.at).getTime());
    return items.slice(0, 15);
  }, [datasets, models]);

  return (
    <Box>
      <Typography variant="h2" sx={{ mb: 0.5 }}>
        Dashboard
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Welcome back, {user?.email}
      </Typography>

      {!loading && datasets.length === 0 && models.length === 0 && (
        <OnboardingWizard
          userId={user?.id || "anonymous"}
          onDatasetImported={(dataset) => {
            setDatasets((prev) => [dataset, ...prev]);
          }}
        />
      )}

      {(loading || datasets.length > 0 || models.length > 0) && (
      <>
      {/* Stat cards */}
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 2,
          mb: 3,
        }}
      >
        <Card
          variant="outlined"
          sx={{
            borderColor: "divider",
            borderBottom: "2px solid",
            borderBottomColor: themeTokens.accent,
            borderRadius: 1,
          }}
        >
          <CardContent sx={{ py: 2, "&:last-child": { pb: 2 } }}>
            <Typography
              variant="h4"
              sx={{
                fontFamily: '"JetBrains Mono", monospace',
                fontWeight: 700,
                color: "primary.main",
                fontSize: "2rem",
              }}
            >
              {loading ? "—" : readyDatasets.length}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Datasets
            </Typography>
          </CardContent>
        </Card>
        <Card
          variant="outlined"
          sx={{
            borderColor: "divider",
            borderBottom: "2px solid",
            borderBottomColor: themeTokens.accent,
            borderRadius: 1,
          }}
        >
          <CardContent sx={{ py: 2, "&:last-child": { pb: 2 } }}>
            <Typography
              variant="h4"
              sx={{
                fontFamily: '"JetBrains Mono", monospace',
                fontWeight: 700,
                color: "primary.main",
                fontSize: "2rem",
              }}
            >
              {loading ? "—" : completedModels.length}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Models
            </Typography>
          </CardContent>
        </Card>
        <Card
          variant="outlined"
          sx={{
            borderColor: "divider",
            borderBottom: "2px solid",
            borderBottomColor: themeTokens.accent,
            borderRadius: 1,
          }}
        >
          <CardContent sx={{ py: 2, "&:last-child": { pb: 2 } }}>
            <Typography
              variant="h4"
              sx={{
                fontFamily: '"JetBrains Mono", monospace',
                fontWeight: 700,
                color: "primary.main",
                fontSize: "2rem",
              }}
            >
              {loading ? "—" : activeTraining > 0 ? activeTraining : "Idle"}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Training
            </Typography>
          </CardContent>
        </Card>
      </Box>

      {/* Recent Activity */}
      <Typography variant="h3" sx={{ mb: 1.5 }}>
        Recent Activity
      </Typography>
      <Card variant="outlined" sx={{ borderColor: "divider", borderRadius: 1, mb: 3 }}>
        <CardContent sx={{ py: 2, "&:last-child": { pb: 2 } }}>
          {loading ? (
            <Typography variant="body2" color="text.secondary">
              Loading…
            </Typography>
          ) : activity.length === 0 ? (
            <Box sx={{ textAlign: "center", py: 2 }}>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
                No activity yet — upload a dataset to get started.
              </Typography>
              <Button variant="outlined" component={Link} href="/data" sx={{ borderColor: "primary.main", color: "primary.main" }}>
                Upload Dataset
              </Button>
            </Box>
          ) : (
            <Box component="ul" sx={{ m: 0, p: 0, listStyle: "none" }}>
              {activity.map((item) => (
                <Box
                  component="li"
                  key={item.id}
                  sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    py: 1,
                    borderBottom: "1px solid",
                    borderColor: "divider",
                    "&:last-child": { borderBottom: "none" },
                  }}
                >
                  <Typography variant="body2">{item.label}</Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ fontFamily: '"JetBrains Mono", monospace' }}>
                    {formatRelativeTime(item.at)}
                  </Typography>
                </Box>
              ))}
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Typography variant="h3" sx={{ mb: 1.5 }}>
        Quick Actions
      </Typography>
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
        {models.length === 0 && (
          <Card
            variant="outlined"
            sx={{
              flex: "1 1 240px",
              maxWidth: 320,
              borderColor: "primary.main",
              borderRadius: 1,
              borderWidth: 2,
            }}
          >
            <CardActionArea
              onClick={handleQuickStart}
              disabled={quickStarting}
              sx={{ display: "block" }}
            >
              <CardContent sx={{ py: 2, "&:last-child": { pb: 2 } }}>
                <RocketLaunchOutlined
                  sx={{ color: "primary.main", fontSize: 28, mb: 0.5 }}
                />
                <Typography
                  variant="subtitle2"
                  fontWeight={600}
                  color="primary.main"
                >
                  {quickStarting
                    ? "Starting..."
                    : "Train a digit classifier in 60 seconds"}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  One click → MNIST + CNN → live training
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        )}
        <Card
          variant="outlined"
          sx={{
            flex: "1 1 180px",
            maxWidth: 240,
            borderColor: "divider",
            borderRadius: 1,
            transition: "border-color 0.2s ease-out, color 0.2s ease-out",
            "&:hover": {
              borderColor: themeTokens.accent,
              "& .quick-action-icon": { color: "primary.main" },
              "& .quick-action-label": { color: "primary.main" },
            },
          }}
        >
          <CardActionArea component={Link} href="/data" sx={{ display: "block" }}>
            <CardContent sx={{ py: 2, "&:last-child": { pb: 2 } }}>
              <UploadFileOutlined className="quick-action-icon" sx={{ color: "text.secondary", fontSize: 28, mb: 0.5 }} />
              <Typography className="quick-action-label" variant="subtitle2" fontWeight={600}>
                Upload Dataset
              </Typography>
            </CardContent>
          </CardActionArea>
        </Card>
        <Card
          variant="outlined"
          sx={{
            flex: "1 1 180px",
            maxWidth: 240,
            borderColor: "divider",
            borderRadius: 1,
            transition: "border-color 0.2s ease-out, color 0.2s ease-out",
            "&:hover": {
              borderColor: themeTokens.accent,
              "& .quick-action-icon": { color: "primary.main" },
              "& .quick-action-label": { color: "primary.main" },
            },
          }}
        >
          <CardActionArea component={Link} href="/train" sx={{ display: "block" }}>
            <CardContent sx={{ py: 2, "&:last-child": { pb: 2 } }}>
              <PlayArrowOutlined className="quick-action-icon" sx={{ color: "text.secondary", fontSize: 28, mb: 0.5 }} />
              <Typography className="quick-action-label" variant="subtitle2" fontWeight={600}>
                New Training Run
              </Typography>
            </CardContent>
          </CardActionArea>
        </Card>
        <Card
          variant="outlined"
          sx={{
            flex: "1 1 180px",
            maxWidth: 240,
            borderColor: "divider",
            borderRadius: 1,
            transition: "border-color 0.2s ease-out, color 0.2s ease-out",
            "&:hover": {
              borderColor: themeTokens.accent,
              "& .quick-action-icon": { color: "primary.main" },
              "& .quick-action-label": { color: "primary.main" },
            },
          }}
        >
          <CardActionArea component={Link} href="/predict" sx={{ display: "block" }}>
            <CardContent sx={{ py: 2, "&:last-child": { pb: 2 } }}>
              <AutoAwesomeOutlined className="quick-action-icon" sx={{ color: "text.secondary", fontSize: 28, mb: 0.5 }} />
              <Typography className="quick-action-label" variant="subtitle2" fontWeight={600}>
                Try Prediction
              </Typography>
            </CardContent>
          </CardActionArea>
        </Card>
      </Box>
      </>
      )}
    </Box>
  );
}
