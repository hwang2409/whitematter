"use client";

import React from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "@/api";
import ModelCard from "@/components/ModelCard";
import TrainingProgress from "@/components/TrainingProgress";
import CompletedModelCard from "./CompletedModelCard";
import TrainingCard from './training/TrainingCard';
import { TrainingStage } from './training/StageIndicator';

interface ChatMessageBubbleProps {
  message: ChatMessage;
  onRetry?: () => void;
  onTrainingComplete?: (status: any) => void;
}

function AiAvatar() {
  return (
    <Box
      sx={{
        width: 28,
        height: 28,
        borderRadius: "50%",
        bgcolor: "primary.main",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
      }}
    >
      <Typography
        sx={{
          fontFamily: "'DM Serif Display', Georgia, serif",
          fontSize: "0.8rem",
          fontWeight: 400,
          color: "#FFFFFF",
          lineHeight: 1,
        }}
      >
        W
      </Typography>
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
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 1 }}>
          <TrainingProgress
            conversationId={convId}
            jobId={jobId}
            onComplete={message.metadata?.onComplete as (() => void) | undefined}
            isLatestActive={message.metadata?.isLatestActive as boolean ?? false}
          />
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
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 1 }}>
        <TrainingCard
          modelName={message.metadata?.model_name as string ?? 'Model'}
          stage={message.metadata?.stage as TrainingStage ?? 'training'}
          epoch={message.metadata?.epoch as number ?? 0}
          totalEpochs={message.metadata?.total_epochs as number ?? 0}
          loss={message.metadata?.loss as number ?? 0}
          accuracy={message.metadata?.accuracy as number ?? 0}
          lossHistory={message.metadata?.loss_history as { epoch: number; loss: number; accuracy?: number }[] ?? []}
          isLatestActive={false}
          error={message.metadata?.error as string ?? message.content}
        />
      </Box>
    );
  }

  // Training complete
  if (message.type === "training_complete") {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 1 }}>
        <TrainingCard
          modelName={message.metadata?.model_name as string ?? 'Model'}
          stage="complete"
          epoch={message.metadata?.total_epochs as number ?? 0}
          totalEpochs={message.metadata?.total_epochs as number ?? 0}
          loss={message.metadata?.loss as number ?? 0}
          accuracy={message.metadata?.accuracy as number ?? 0}
          lossHistory={message.metadata?.loss_history as { epoch: number; loss: number; accuracy?: number }[] ?? []}
          isLatestActive={false}
          modelId={message.metadata?.model_id as string}
          optimizer={message.metadata?.optimizer as string}
          learningRate={message.metadata?.learning_rate as number}
          batchSize={message.metadata?.batch_size as number}
        />
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
            bgcolor: "transparent",
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
              backgroundColor: "#161514",
              border: "1px solid rgba(255,255,255,0.07)",
              borderLeft: "4px solid",
              borderLeftColor: "primary.main",
              borderRadius: "8px",
              padding: "12px 16px",
              overflow: "auto",
              fontFamily: "'DM Mono', monospace",
              fontSize: "0.8125rem",
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
