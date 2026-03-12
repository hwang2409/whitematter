"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import CircularProgress from "@mui/material/CircularProgress";
import ChatMessageBubble from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";
import QuickStartChips from "@/components/QuickStartChips";
import type { ChatMessage, ConversationPhase } from "@/api";

// ---------------------------------------------------------------------------
// Mock helpers -- will be replaced by real API calls later
// ---------------------------------------------------------------------------

const GREETING_MESSAGE: ChatMessage = {
  id: "greeting",
  role: "assistant",
  content:
    "Hi! I'm **WhiteMatter**, your AI assistant for building and training neural networks.\n\nTell me what you'd like to build, or pick one of the quick starts below to get going.",
  type: "text",
  createdAt: new Date().toISOString(),
};

function mockAssistantReply(userText: string): Promise<ChatMessage> {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        id: crypto.randomUUID(),
        role: "assistant",
        content: `Thanks! You said: "${userText}". (This is a placeholder response -- real SSE streaming will be wired up in a later chunk.)`,
        type: "text",
        createdAt: new Date().toISOString(),
      });
    }, 800);
  });
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface ChatPageProps {
  conversationId?: string;
}

export default function ChatPage({ conversationId }: ChatPageProps) {
  const router = useRouter();
  const [messages, setMessages] = useState<ChatMessage[]>([GREETING_MESSAGE]);
  const [phase, setPhase] = useState<ConversationPhase>("greeting");
  const [streaming, setStreaming] = useState(false);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // If there is a conversationId, we'd load the conversation here.
  // For now this is a placeholder.
  useEffect(() => {
    if (conversationId) {
      setLoading(true);
      // TODO: load conversation from API via getConversation(conversationId)
      // For now just reset with greeting
      setMessages([GREETING_MESSAGE]);
      setPhase("greeting");
      setLoading(false);
    }
  }, [conversationId]);

  const handleSend = useCallback(
    async (text: string) => {
      if (!text.trim() || streaming) return;

      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
        type: "text",
        createdAt: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setPhase("exploring");
      setStreaming(true);

      try {
        const reply = await mockAssistantReply(text);
        setMessages((prev) => [...prev, reply]);
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "Sorry, something went wrong. Please try again.",
            type: "text",
            createdAt: new Date().toISOString(),
          },
        ]);
      } finally {
        setStreaming(false);
      }
    },
    [streaming],
  );

  const handleQuickStart = useCallback(
    (text: string) => {
      handleSend(text);
    },
    [handleSend],
  );

  const placeholderForPhase = (p: ConversationPhase): string => {
    switch (p) {
      case "greeting":
      case "exploring":
        return "Describe the model you want to build...";
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
  };

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
        <CircularProgress size={32} sx={{ color: "primary.main" }} />
      </Box>
    );
  }

  const showQuickStart = phase === "greeting" && messages.length <= 1;

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
        sx={{
          flex: 1,
          overflowY: "auto",
          px: { xs: 2, sm: 4 },
          py: 3,
          display: "flex",
          flexDirection: "column",
          gap: 2,
        }}
      >
        <Box sx={{ maxWidth: 780, width: "100%", mx: "auto", display: "flex", flexDirection: "column", gap: 2 }}>
          {messages.map((msg) => (
            <ChatMessageBubble key={msg.id} message={msg} />
          ))}

          {streaming && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, pl: 1 }}>
              <CircularProgress size={16} sx={{ color: "primary.main" }} />
              <Typography variant="body2" color="text.secondary">
                Thinking...
              </Typography>
            </Box>
          )}

          {showQuickStart && (
            <Box sx={{ mt: 2 }}>
              <QuickStartChips onSelect={handleQuickStart} />
            </Box>
          )}

          <div ref={bottomRef} />
        </Box>
      </Box>

      {/* Input bar */}
      <Box
        sx={{
          borderTop: "1px solid",
          borderColor: "divider",
          bgcolor: "background.paper",
          px: { xs: 2, sm: 4 },
          py: 1.5,
        }}
      >
        <Box sx={{ maxWidth: 780, mx: "auto" }}>
          <ChatInput
            onSend={handleSend}
            disabled={streaming}
            placeholder={placeholderForPhase(phase)}
          />
        </Box>
      </Box>
    </Box>
  );
}
