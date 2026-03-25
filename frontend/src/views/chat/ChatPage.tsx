"use client";

import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import CircularProgress from "@mui/material/CircularProgress";
import ChatInput from "@/components/ChatInput";
import QuickStartChips from "@/components/QuickStartChips";
import { useAuth } from "@/context/AuthContext";
import type { ConversationPhase } from "@/api";
import { useChatSession } from "./useChatSession";
import { useChatTraining } from "./useChatTraining";
import { useChatUpload, DragOverlay } from "./ChatUploadArea";
import ChatMessageList from "./ChatMessageList";

const UPLOAD_LIMITS: Record<string, number> = {
  free: 200,
  pro: 1000,
  scale: 5000,
};

function placeholderForPhase(p: ConversationPhase): string {
  switch (p) {
    case "greeting":
    case "exploring":
      return "Describe what you want to build...";
    case "architecture":
      return "Ask questions or request changes...";
    case "data_needed":
      return "Upload your dataset or choose a built-in one...";
    case "ready":
      return "Say 'train it!' or adjust parameters...";
    case "training":
      return "Training in progress...";
    case "completed":
      return "Ask about results or start a new model...";
    case "predicting":
      return "Upload a file to predict...";
    default:
      return "Type a message...";
  }
}

interface ChatPageProps {
  conversationId?: string;
}

export default function ChatPage({ conversationId }: ChatPageProps) {
  const { token, user } = useAuth();
  const uploadLimitMB = UPLOAD_LIMITS[user?.plan ?? "free"] ?? 200;

  const {
    messages,
    setMessages,
    phase,
    streaming,
    loading,
    currentConversationId,
    bottomRef,
    handleSend,
    toolStatus,
  } = useChatSession({ conversationId, token });

  const { handleTrainingComplete } = useChatTraining({
    setMessages,
    handleSend,
  });

  const {
    dragOver,
    messagesContainerRef,
    handleFileUpload,
    handleDragOver,
    handleDragLeave,
    handleDrop,
  } = useChatUpload({
    currentConversationId,
    token,
    uploadLimitMB,
    setMessages,
    handleSend,
  });

  if (loading) {
    return (
      <Box
        sx={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <CircularProgress size={24} sx={{ color: "#F97316" }} />
      </Box>
    );
  }

  const showEmptyState = phase === "greeting" && messages.length <= 1;

  return (
    <Box
      sx={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        height: "100%",
        minHeight: 0,
      }}
    >
      {/* Messages area */}
      <Box
        ref={messagesContainerRef}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        sx={{
          flex: 1,
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          position: "relative",
        }}
      >
        {dragOver && <DragOverlay uploadLimitMB={uploadLimitMB} />}

        {showEmptyState ? (
          <Box
            sx={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              px: 4,
              animation: "fadeIn 0.5s ease-out",
            }}
          >
            <Typography
              sx={{
                fontFamily: "'Instrument Serif', Georgia, serif",
                fontSize: "3rem",
                fontWeight: 400,
                color: "#FAFAFA",
                mb: 1,
                textAlign: "center",
                letterSpacing: "-0.02em",
                lineHeight: 1.1,
                "& em": {
                  fontStyle: "italic",
                  color: "#F97316",
                },
              }}
            >
              What will you <em>build?</em>
            </Typography>
            <Typography
              sx={{
                fontSize: "0.9375rem",
                color: "#52525B",
                mb: 4,
                textAlign: "center",
                maxWidth: 420,
                lineHeight: 1.5,
              }}
            >
              Describe a problem and I&apos;ll design a neural network for it.
            </Typography>
            <Box sx={{ maxWidth: 640, width: "100%", mt: 1 }}>
              <ChatInput
                onSend={handleSend}
                onFileUpload={handleFileUpload}
                maxUploadMB={uploadLimitMB}
                disabled={streaming}
                placeholder={placeholderForPhase(phase)}
              />
            </Box>
            <Box sx={{ maxWidth: 640, width: "100%", mt: 1.5 }}>
              <QuickStartChips onSelect={handleSend} />
            </Box>
          </Box>
        ) : (
          <ChatMessageList
            messages={messages}
            streaming={streaming}
            bottomRef={bottomRef}
            onTrainingComplete={handleTrainingComplete}
            toolStatus={toolStatus}
            onSend={handleSend}
          />
        )}
      </Box>

      {/* Input bar */}
      {!showEmptyState && (
        <Box
          sx={{
            px: 3,
            pb: 2.5,
            pt: 1.5,
            display: "flex",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          <Box sx={{ maxWidth: 640, width: "100%" }}>
            <ChatInput
              onSend={handleSend}
              onFileUpload={handleFileUpload}
              maxUploadMB={uploadLimitMB}
              disabled={streaming || phase === "training"}
              placeholder={placeholderForPhase(phase)}
            />
          </Box>
        </Box>
      )}
    </Box>
  );
}
