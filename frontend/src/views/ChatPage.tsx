"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import CircularProgress from "@mui/material/CircularProgress";
import ChatMessageBubble from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";
import QuickStartChips from "@/components/QuickStartChips";
import { useAuth } from "@/context/AuthContext";
import {
  createConversation,
  getConversation,
  sendChatMessage,
} from "@/api";
import type { ChatMessage, ConversationPhase } from "@/api";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface ChatPageProps {
  conversationId?: string;
}

export default function ChatPage({ conversationId }: ChatPageProps) {
  const router = useRouter();
  const { token } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [phase, setPhase] = useState<ConversationPhase>("greeting");
  const [streaming, setStreaming] = useState(false);
  const [loading, setLoading] = useState(true);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(conversationId || null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<{ abort: () => void } | null>(null);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Create or load conversation on mount
  useEffect(() => {
    if (!token) return;

    let cancelled = false;

    async function init() {
      try {
        if (conversationId) {
          // Load existing conversation
          const data = await getConversation(token!, conversationId);
          if (cancelled) return;
          setCurrentConversationId(conversationId);
          setPhase(data.conversation?.phase ?? "greeting");
          setMessages(
            (data.messages ?? []).map((m: any) => ({
              id: m.id ?? crypto.randomUUID(),
              role: m.role,
              content: m.content,
              type: m.type ?? m.message_type ?? "text",
              metadata: m.metadata,
              createdAt: m.created_at ?? m.createdAt ?? new Date().toISOString(),
            })),
          );
        } else {
          // Create a new conversation, then fetch it to get the greeting
          const conv = await createConversation(token!);
          if (cancelled) return;
          setCurrentConversationId(conv.id);

          const data = await getConversation(token!, conv.id);
          if (cancelled) return;
          setPhase(data.conversation?.phase ?? "greeting");
          setMessages(
            (data.messages ?? []).map((m: any) => ({
              id: m.id ?? crypto.randomUUID(),
              role: m.role,
              content: m.content,
              type: m.type ?? m.message_type ?? "text",
              metadata: m.metadata,
              createdAt: m.created_at ?? m.createdAt ?? new Date().toISOString(),
            })),
          );
        }
      } catch (err) {
        console.error("Failed to initialise conversation:", err);
        if (!cancelled) {
          setMessages([
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content:
                "I'm having trouble connecting right now. Please refresh to try again.",
              type: "text",
              createdAt: new Date().toISOString(),
            },
          ]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    init();
    return () => {
      cancelled = true;
    };
  }, [token, conversationId]);

  // Cleanup SSE abort on unmount
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const handleSend = useCallback(
    (text: string) => {
      if (!text.trim() || streaming || !currentConversationId) return;

      // Add user message immediately
      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
        type: "text",
        createdAt: new Date().toISOString(),
      };

      // Add empty assistant placeholder for streaming
      const assistantPlaceholderId = crypto.randomUUID();
      const assistantPlaceholder: ChatMessage = {
        id: assistantPlaceholderId,
        role: "assistant",
        content: "",
        type: "text",
        createdAt: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
      setStreaming(true);

      const handle = sendChatMessage(
        currentConversationId,
        text,
        // onChunk: append chunk text to last (assistant) message
        (chunkText: string) => {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            updated[updated.length - 1] = {
              ...last,
              content: last.content + chunkText,
            };
            return updated;
          });
        },
        // onDone: replace last message with full response
        (doneMessage: ChatMessage) => {
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              ...doneMessage,
              id: doneMessage.id || assistantPlaceholderId,
              type: doneMessage.type ?? (doneMessage as any).message_type ?? "text",
              createdAt: doneMessage.createdAt ?? new Date().toISOString(),
            };
            return updated;
          });

          // Update phase if the response carries metadata.phase
          const newPhase =
            (doneMessage.metadata?.phase as ConversationPhase | undefined) ??
            undefined;
          if (newPhase) {
            setPhase(newPhase);
          }

          setStreaming(false);
          abortRef.current = null;
        },
        // onError
        (error: Error) => {
          console.error("Chat SSE error:", error);
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              id: assistantPlaceholderId,
              role: "assistant",
              content:
                "I'm having trouble thinking right now, try again in a moment.",
              type: "text",
              createdAt: new Date().toISOString(),
            };
            return updated;
          });
          setStreaming(false);
          abortRef.current = null;
        },
        token ?? undefined,
      );

      abortRef.current = handle;
    },
    [streaming, currentConversationId, token],
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
