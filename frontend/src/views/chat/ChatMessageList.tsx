"use client";

import Box from "@mui/material/Box";
import LinearProgress from "@mui/material/LinearProgress";
import Typography from "@mui/material/Typography";
import SearchOutlined from "@mui/icons-material/SearchOutlined";
import DownloadOutlined from "@mui/icons-material/DownloadOutlined";
import ChatMessageBubble from "@/components/ChatMessage";
import type { ChatMessage } from "@/api";

interface ChatMessageListProps {
  messages: ChatMessage[];
  streaming: boolean;
  bottomRef: React.RefObject<HTMLDivElement | null>;
  onTrainingComplete: (status: any) => void;
  toolStatus?: string | null;
  onSend?: (text: string) => void;
}

export default function ChatMessageList({
  messages,
  streaming,
  bottomRef,
  onTrainingComplete,
  toolStatus,
  onSend,
}: ChatMessageListProps) {
  const isDownloading = toolStatus?.startsWith("Downloading") ?? false;
  const isSearching = toolStatus != null && !isDownloading;

  return (
    <Box
      sx={{
        maxWidth: 700,
        width: "100%",
        mx: "auto",
        py: 3,
        display: "flex",
        flexDirection: "column",
        gap: 0,
        flex: 1,
      }}
    >
      {messages.map((msg) => (
        <ChatMessageBubble
          key={msg.id}
          message={msg}
          onTrainingComplete={onTrainingComplete}
          onSend={onSend}
        />
      ))}

      {toolStatus && (
        <Box sx={{ px: 4, py: 1 }}>
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 1.5,
            }}
          >
            <Box
              sx={{
                width: 28,
                height: 28,
                borderRadius: "6px",
                bgcolor: "#27272A",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
              }}
            >
              {isDownloading ? (
                <DownloadOutlined
                  sx={{
                    fontSize: 14,
                    color: "#F97316",
                    animation: "pulse 1.2s ease-in-out infinite",
                  }}
                />
              ) : (
                <SearchOutlined
                  sx={{
                    fontSize: 14,
                    color: "#F97316",
                    animation: "pulse 1.2s ease-in-out infinite",
                  }}
                />
              )}
            </Box>
            <Typography
              variant="body2"
              sx={{ color: "#A1A1AA", fontSize: "0.8125rem" }}
            >
              {toolStatus}
            </Typography>
          </Box>
          {isDownloading && (
            <Box sx={{ mt: 1, ml: 5.5 }}>
              <LinearProgress
                variant="indeterminate"
                sx={{
                  height: 3,
                  borderRadius: 2,
                  bgcolor: "#1F1F23",
                  "& .MuiLinearProgress-bar": {
                    bgcolor: "#F97316",
                    borderRadius: 2,
                  },
                }}
              />
            </Box>
          )}
        </Box>
      )}

      {streaming && !toolStatus && (
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1.5,
            px: 4,
            py: 1,
          }}
        >
          <Box
            sx={{
              width: 28,
              height: 28,
              borderRadius: "6px",
              bgcolor: "#27272A",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Box
              sx={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                bgcolor: "#F97316",
                animation: "pulse 1.2s ease-in-out infinite",
              }}
            />
          </Box>
          <Typography
            variant="body2"
            sx={{ color: "#52525B", fontSize: "0.8125rem" }}
          >
            Thinking...
          </Typography>
        </Box>
      )}

      <div ref={bottomRef} />
    </Box>
  );
}
