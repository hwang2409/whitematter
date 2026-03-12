"use client";

import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "@/api";
import { themeTokens } from "@/theme";
import ModelCard from "@/components/ModelCard";
import TrainingProgress from "@/components/TrainingProgress";

interface ChatMessageBubbleProps {
  message: ChatMessage;
  onRetry?: () => void;
}

export default function ChatMessageBubble({ message, onRetry }: ChatMessageBubbleProps) {
  const isUser = message.role === "user";

  // Architecture type: render ModelCard
  if (message.type === "architecture" && message.metadata) {
    return (
      <Box sx={{ display: "flex", justifyContent: "flex-start", width: "100%" }}>
        <Box sx={{ maxWidth: "85%" }}>
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
    );
  }

  // Training progress: render live TrainingProgress component
  if (message.type === "training_progress" && message.metadata) {
    const jobId = message.metadata.job_id as string;
    const convId = message.metadata.conversation_id as string;
    if (jobId && convId) {
      return (
        <Box sx={{ display: "flex", justifyContent: "flex-start", width: "100%" }}>
          <Box sx={{ maxWidth: "85%" }}>
            <TrainingProgress conversationId={convId} jobId={jobId} />
          </Box>
        </Box>
      );
    }
    return (
      <Box sx={{ display: "flex", justifyContent: "flex-start", width: "100%" }}>
        <Box
          sx={{
            bgcolor: "background.paper",
            border: "1px solid",
            borderColor: "divider",
            borderRadius: 2,
            px: 2,
            py: 1.5,
            maxWidth: "85%",
          }}
        >
          <Typography variant="body2" color="text.secondary">
            {message.content}
          </Typography>
        </Box>
      </Box>
    );
  }

  // Training complete placeholder
  if (message.type === "training_complete") {
    return (
      <Box sx={{ display: "flex", justifyContent: "flex-start", width: "100%" }}>
        <Box
          sx={{
            bgcolor: "background.paper",
            border: "1px solid",
            borderColor: "divider",
            borderRadius: 2,
            px: 2,
            py: 1.5,
            maxWidth: "85%",
          }}
        >
          <Typography variant="body2" color="text.secondary">
            Training complete placeholder
          </Typography>
        </Box>
      </Box>
    );
  }

  // File upload placeholder
  if (message.type === "file_upload") {
    return (
      <Box sx={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start", width: "100%" }}>
        <Box
          sx={{
            bgcolor: "background.paper",
            border: "1px solid",
            borderColor: "divider",
            borderRadius: 2,
            px: 2,
            py: 1.5,
            maxWidth: "85%",
          }}
        >
          <Typography variant="body2" color="text.secondary">
            File upload placeholder
          </Typography>
        </Box>
      </Box>
    );
  }

  // Prediction placeholder
  if (message.type === "prediction") {
    return (
      <Box sx={{ display: "flex", justifyContent: "flex-start", width: "100%" }}>
        <Box
          sx={{
            bgcolor: "background.paper",
            border: "1px solid",
            borderColor: "divider",
            borderRadius: 2,
            px: 2,
            py: 1.5,
            maxWidth: "85%",
          }}
        >
          <Typography variant="body2" color="text.secondary">
            Prediction result placeholder
          </Typography>
        </Box>
      </Box>
    );
  }

  // Default: text message
  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        width: "100%",
      }}
    >
      <Box
        sx={{
          maxWidth: "85%",
          bgcolor: isUser ? themeTokens.accent : "background.paper",
          color: isUser ? "#0a0a0a" : "text.primary",
          borderRadius: 2,
          px: 2,
          py: 1.5,
          border: isUser ? "none" : "1px solid",
          borderColor: isUser ? undefined : "divider",
          "& p": { m: 0 },
          "& p + p": { mt: 1 },
          "& code": {
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: "0.8125rem",
            bgcolor: isUser
              ? "rgba(0,0,0,0.1)"
              : "rgba(126,184,255,0.1)",
            px: 0.5,
            py: 0.25,
            borderRadius: 0.5,
          },
          "& pre": {
            bgcolor: isUser
              ? "rgba(0,0,0,0.15)"
              : "rgba(255,255,255,0.04)",
            borderRadius: 1,
            p: 1.5,
            overflow: "auto",
            my: 1,
            "& code": {
              bgcolor: "transparent",
              px: 0,
              py: 0,
            },
          },
          "& ul, & ol": { pl: 2.5, my: 0.5 },
          "& li": { mb: 0.25 },
          "& a": { color: isUser ? "#0a0a0a" : themeTokens.accent, textDecoration: "underline" },
          "& strong": { fontWeight: 600 },
        }}
      >
        <Markdown remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
        {message.metadata?.error && onRetry && (
          <Button size="small" onClick={onRetry} sx={{ mt: 1 }}>
            Try again
          </Button>
        )}
      </Box>
    </Box>
  );
}
