"use client";

import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "@/api";
import ModelCard from "@/components/ModelCard";
import TrainingProgress from "@/components/TrainingProgress";
import CompletedModelCard from "./CompletedModelCard";

interface ChatMessageBubbleProps {
  message: ChatMessage;
  onRetry?: () => void;
  onTrainingComplete?: (status: any) => void;
}

function AiAvatar() {
  return (
    <Box
      sx={{
        width: 30,
        height: 30,
        borderRadius: "50%",
        bgcolor: "rgba(120,113,108,0.08)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: "0.6875rem",
        fontWeight: 700,
        color: "primary.main",
        flexShrink: 0,
        mt: 0.25,
        fontFamily: "'DM Sans', sans-serif",
      }}
    >
      wm
    </Box>
  );
}

export default function ChatMessageBubble({ message, onRetry, onTrainingComplete }: ChatMessageBubbleProps) {
  const isUser = message.role === "user";

  // Architecture type: render ModelCard
  if (message.type === "architecture" && message.metadata) {
    return (
      <Box sx={{ px: 4, mb: 3 }}>
        <Box sx={{ display: "flex", gap: 1.5, alignItems: "flex-start" }}>
          <AiAvatar />
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <ModelCard
              name={message.metadata.name as string}
              description={message.metadata.description as string}
              layers={message.metadata.layers as string}
              trainingConfig={message.metadata.trainingConfig as string}
              onApprove={() => {}}
              onRequestChanges={() => {}}
            />
          </Box>
        </Box>
      </Box>
    );
  }

  // Training progress
  if (message.type === "training_progress" && message.metadata) {
    const jobId = message.metadata.job_id as string;
    const convId = message.metadata.conversation_id as string;
    if (jobId && convId) {
      return (
        <Box sx={{ px: 4, mb: 3 }}>
          <Box sx={{ display: "flex", gap: 1.5, alignItems: "flex-start" }}>
            <AiAvatar />
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <TrainingProgress conversationId={convId} jobId={jobId} onComplete={onTrainingComplete} />
            </Box>
          </Box>
        </Box>
      );
    }
    return (
      <Box sx={{ px: 4, mb: 3 }}>
        <Box sx={{ display: "flex", gap: 1.5, alignItems: "flex-start" }}>
          <AiAvatar />
          <Box
            sx={{
              bgcolor: "background.paper",
              border: "1px solid",
              borderColor: "divider",
              borderRadius: "16px",
              px: 2,
              py: 1.5,
            }}
          >
            <Typography variant="body2" color="text.secondary">
              {message.content}
            </Typography>
          </Box>
        </Box>
      </Box>
    );
  }

  // Training error
  if (message.type === "training_error") {
    const errorMsg = (message.metadata?.friendlyMessage as string) || message.content;
    const suggestion = message.metadata?.suggestion as string | undefined;
    return (
      <Box sx={{ px: 4, mb: 3 }}>
        <Box sx={{ display: "flex", gap: 1.5, alignItems: "flex-start" }}>
          <AiAvatar />
          <Box
            sx={{
              bgcolor: "rgba(239,68,68,0.08)",
              border: "1px solid rgba(239,68,68,0.2)",
              borderRadius: "16px",
              px: 2.5,
              py: 1.5,
            }}
          >
            <Typography
              sx={{
                fontSize: "0.8125rem",
                fontWeight: 600,
                color: "error.main",
                mb: 0.5,
              }}
            >
              Training Failed
            </Typography>
            <Typography variant="body2" sx={{ color: "text.secondary" }}>
              {errorMsg}
            </Typography>
            {suggestion && (
              <Typography variant="body2" sx={{ mt: 1, fontStyle: "italic", color: "text.secondary" }}>
                {suggestion}
              </Typography>
            )}
          </Box>
        </Box>
      </Box>
    );
  }

  // Training complete
  if (message.type === "training_complete") {
    const meta = message.metadata || {};
    return (
      <Box sx={{ px: 4, mb: 3 }}>
        <Box sx={{ display: "flex", gap: 1.5, alignItems: "flex-start" }}>
          <AiAvatar />
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <CompletedModelCard
              modelId={meta.model_id as string}
              accuracy={meta.accuracy as number}
              params={meta.params as string}
              trainingTime={meta.training_time as string}
              architecture={meta.architecture as string}
              datasetName={meta.dataset_name as string}
            />
          </Box>
        </Box>
      </Box>
    );
  }

  // File upload
  if (message.type === "file_upload") {
    return (
      <Box sx={{ px: 4, mb: 3 }}>
        <Box sx={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start" }}>
          <Box
            sx={{
              bgcolor: "background.paper",
              border: "1px solid",
              borderColor: "divider",
              borderRadius: "16px",
              px: 2.5,
              py: 1.5,
              maxWidth: "70%",
            }}
          >
            <Typography variant="body2" color="text.secondary">
              {message.content}
            </Typography>
          </Box>
        </Box>
      </Box>
    );
  }

  // Prediction placeholder
  if (message.type === "prediction") {
    return (
      <Box sx={{ px: 4, mb: 3 }}>
        <Box sx={{ display: "flex", gap: 1.5, alignItems: "flex-start" }}>
          <AiAvatar />
          <Box
            sx={{
              bgcolor: "background.paper",
              border: "1px solid",
              borderColor: "divider",
              borderRadius: "16px",
              px: 2.5,
              py: 1.5,
            }}
          >
            <Typography variant="body2" color="text.secondary">
              Prediction result placeholder
            </Typography>
          </Box>
        </Box>
      </Box>
    );
  }

  // User message
  if (isUser) {
    return (
      <Box sx={{ px: 4, mb: 3 }}>
        <Box sx={{ display: "flex", justifyContent: "flex-end" }}>
          <Box
            sx={{
              maxWidth: "70%",
              bgcolor: (theme) =>
                theme.palette.mode === "dark"
                  ? "rgba(255,255,255,0.06)"
                  : "#F2F1EE",
              borderRadius: "18px 18px 4px 18px",
              px: 2.5,
              py: 1.5,
              "& p": { m: 0 },
              "& p + p": { mt: 1 },
            }}
          >
            <Typography
              component="div"
              sx={{
                fontSize: "0.9375rem",
                lineHeight: 1.6,
                color: "text.primary",
              }}
            >
              <Markdown remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
            </Typography>
          </Box>
        </Box>
      </Box>
    );
  }

  // Assistant text message
  return (
    <Box sx={{ px: 4, mb: 3 }}>
      <Box sx={{ display: "flex", gap: 1.5, alignItems: "flex-start" }}>
        <AiAvatar />
        <Box
          sx={{
            flex: 1,
            minWidth: 0,
            "& p": { m: 0 },
            "& p + p": { mt: 1.5 },
            "& code": {
              fontFamily: "'DM Mono', monospace",
              fontSize: "0.8125rem",
              bgcolor: (theme) =>
                theme.palette.mode === "dark"
                  ? "rgba(120,113,108,0.1)"
                  : "rgba(120,113,108,0.06)",
              px: 0.75,
              py: 0.25,
              borderRadius: 0.75,
            },
            "& pre": {
              bgcolor: (theme) =>
                theme.palette.mode === "dark"
                  ? "rgba(0,0,0,0.3)"
                  : "rgba(0,0,0,0.03)",
              borderRadius: 1.5,
              p: 2,
              overflow: "auto",
              my: 1.5,
              "& code": {
                bgcolor: "transparent",
                px: 0,
                py: 0,
              },
            },
            "& ul, & ol": { pl: 2.5, my: 0.5 },
            "& li": { mb: 0.25 },
            "& a": { color: "primary.main", textDecoration: "underline" },
            "& strong": { fontWeight: 500, color: "text.primary" },
          }}
        >
          <Typography
            component="div"
            sx={{
              fontSize: "0.9375rem",
              lineHeight: 1.7,
              color: "text.secondary",
            }}
          >
            <Markdown remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
          </Typography>
          {Boolean(message.metadata?.error) && onRetry && (
            <Button
              size="small"
              onClick={onRetry}
              sx={{ mt: 1, color: "primary.main" }}
            >
              Try again
            </Button>
          )}
        </Box>
      </Box>
    </Box>
  );
}
