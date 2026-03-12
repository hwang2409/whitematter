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

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface ChatPageProps {
  conversationId?: string;
}

export default function ChatPage({ conversationId }: ChatPageProps) {
  const router = useRouter();
  const { token, user } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [phase, setPhase] = useState<ConversationPhase>("greeting");
  const [streaming, setStreaming] = useState(false);
  const [loading, setLoading] = useState(true);
  const [dragOver, setDragOver] = useState(false);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(conversationId || null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<{ abort: () => void } | null>(null);

  const UPLOAD_LIMITS: Record<string, number> = { free: 200, pro: 1000, scale: 5000 };
  const uploadLimitMB = UPLOAD_LIMITS[user?.plan ?? "free"] ?? 200;

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

  const handleFileUpload = useCallback(
    async (file: File) => {
      if (!currentConversationId || !token) return;

      const uploadMsgId = crypto.randomUUID();
      const uploadMessage: ChatMessage = {
        id: uploadMsgId,
        role: "user",
        content: `Uploading dataset: ${file.name}...`,
        type: "file_upload",
        createdAt: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, uploadMessage]);

      try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("name", file.name);

        const res = await fetch(`${API_BASE}/datasets/upload`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
          },
          body: formData,
        });

        if (!res.ok) {
          const errBody = await res.text().catch(() => "Upload failed");
          throw new Error(errBody);
        }

        const data = await res.json();

        // Update the upload message to show success
        setMessages((prev) =>
          prev.map((m) =>
            m.id === uploadMsgId
              ? { ...m, content: `Dataset uploaded: ${file.name}` }
              : m,
          ),
        );

        // Notify the chat about the uploaded dataset
        handleSend(
          `I uploaded a dataset: ${file.name} (id: ${data.id ?? data.dataset_id ?? "unknown"})`,
        );
      } catch (err) {
        console.error("File upload error:", err);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === uploadMsgId
              ? {
                  ...m,
                  content: `Failed to upload ${file.name}: ${err instanceof Error ? err.message : "Unknown error"}`,
                }
              : m,
          ),
        );
      }
    },
    [currentConversationId, token, handleSend],
  );

  const handleQuickStart = useCallback(
    (text: string) => {
      handleSend(text);
    },
    [handleSend],
  );

  // Drag-and-drop handlers
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragOver(false);

      const file = e.dataTransfer.files?.[0];
      if (!file) return;

      // Only accept .zip files
      if (!file.name.endsWith(".zip")) {
        alert("Only .zip files are accepted.");
        return;
      }

      const maxBytes = uploadLimitMB * 1024 * 1024;
      if (file.size > maxBytes) {
        alert(`File too large. Maximum size is ${uploadLimitMB} MB.`);
        return;
      }

      handleFileUpload(file);
    },
    [handleFileUpload, uploadLimitMB],
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
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        sx={{
          flex: 1,
          overflowY: "auto",
          px: { xs: 2, sm: 4 },
          py: 3,
          display: "flex",
          flexDirection: "column",
          gap: 2,
          position: "relative",
        }}
      >
        {/* Drag-and-drop overlay */}
        {dragOver && (
          <Box
            sx={{
              position: "absolute",
              inset: 0,
              zIndex: 10,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              bgcolor: "rgba(0, 0, 0, 0.5)",
              borderRadius: 2,
              border: "2px dashed",
              borderColor: "primary.main",
              pointerEvents: "none",
            }}
          >
            <Typography variant="h6" sx={{ color: "common.white" }}>
              Drop your dataset here (max {uploadLimitMB} MB)
            </Typography>
          </Box>
        )}

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
            onFileUpload={handleFileUpload}
            maxUploadMB={uploadLimitMB}
            disabled={streaming}
            placeholder={placeholderForPhase(phase)}
          />
        </Box>
      </Box>
    </Box>
  );
}
