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
  onSend?: (text: string) => void;
}

function AiAvatar() {
  return (
    <Box
      sx={{
        width: 28,
        height: 28,
        borderRadius: "6px",
        bgcolor: "#27272A",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: "0.625rem",
        fontWeight: 700,
        color: "#F97316",
        flexShrink: 0,
        mt: 0.25,
        fontFamily: "'JetBrains Mono', monospace",
      }}
    >
      wm
    </Box>
  );
}

export default function ChatMessageBubble({ message, onRetry, onTrainingComplete, onSend }: ChatMessageBubbleProps) {
  const isUser = message.role === "user";

  // Architecture type: render ModelCard
  if (message.type === "architecture" && message.metadata) {
    // Metadata may be the architecture directly or wrapped in {"architecture": ...}
    const arch = (message.metadata.architecture ?? message.metadata) as Record<string, unknown>;
    return (
      <Box sx={{ px: 4, mb: 3 }}>
        <Box sx={{ display: "flex", gap: 1.5, alignItems: "flex-start" }}>
          <AiAvatar />
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <ModelCard
              name={(arch.name as string) ?? "Model"}
              description={(arch.description as string) ?? ""}
              layers={(arch.layers as string) ?? ""}
              trainingConfig={(arch.trainingConfig as string) ?? ""}
              onApprove={() => onSend?.("Train it!")}
              onRequestChanges={() => onSend?.("I'd like to modify the architecture.")}
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
              px: 2,
              py: 1.5,
            }}
          >
            <Typography
              variant="body2"
              sx={{
                color: "#A1A1AA",
                fontFamily: "'Outfit', sans-serif",
              }}
            >
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
              bgcolor: "rgba(239,68,68,0.05)",
              borderLeft: "3px solid #EF4444",
              borderRadius: "4px",
              px: 2.5,
              py: 1.5,
            }}
          >
            <Typography
              sx={{
                fontSize: "0.8125rem",
                fontWeight: 600,
                color: "#EF4444",
                mb: 0.5,
                fontFamily: "'Outfit', sans-serif",
              }}
            >
              Training Failed
            </Typography>
            <Typography
              variant="body2"
              sx={{
                color: "#A1A1AA",
                fontFamily: "'Outfit', sans-serif",
              }}
            >
              {errorMsg}
            </Typography>
            {suggestion && (
              <Typography
                variant="body2"
                sx={{
                  mt: 1,
                  fontStyle: "italic",
                  color: "#A1A1AA",
                  fontFamily: "'Outfit', sans-serif",
                }}
              >
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
              bgcolor: "rgba(255,255,255,0.05)",
              borderRadius: "16px 16px 4px 16px",
              px: 2.5,
              py: 1.5,
              maxWidth: "70%",
            }}
          >
            <Typography
              variant="body2"
              sx={{
                color: "#A1A1AA",
                fontFamily: "'Outfit', sans-serif",
              }}
            >
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
              px: 2,
              py: 1.5,
            }}
          >
            <Typography
              variant="body2"
              sx={{
                color: "#A1A1AA",
                fontFamily: "'Outfit', sans-serif",
              }}
            >
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
              bgcolor: "rgba(255,255,255,0.05)",
              borderRadius: "16px 16px 4px 16px",
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
                color: "#FAFAFA",
                fontFamily: "'Outfit', sans-serif",
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
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: "0.8125rem",
              bgcolor: "rgba(255,255,255,0.06)",
              px: 0.75,
              py: 0.25,
              borderRadius: "4px",
            },
            "& pre": {
              bgcolor: "rgba(0,0,0,0.4)",
              borderRadius: "8px",
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
            "& li": { mb: 0.5 },
            "& a": { color: "#F97316", textDecoration: "underline" },
            "& strong": { fontWeight: 500, color: "#FAFAFA" },
          }}
        >
          <Typography
            component="div"
            sx={{
              fontSize: "0.9375rem",
              lineHeight: 1.7,
              color: "#A1A1AA",
              fontFamily: "'Outfit', sans-serif",
            }}
          >
            <Markdown remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
          </Typography>
          {Boolean(message.metadata?.error) && onRetry && (
            <Button
              size="small"
              onClick={onRetry}
              sx={{
                mt: 1,
                color: "#F97316",
                fontFamily: "'Outfit', sans-serif",
                "&:hover": {
                  bgcolor: "rgba(249,115,22,0.08)",
                },
              }}
            >
              Try again
            </Button>
          )}
        </Box>
      </Box>
    </Box>
  );
}
