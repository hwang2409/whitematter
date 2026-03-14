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
import { parseTrainingError } from "@/lib/trainingErrors";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";
const UPLOAD_LIMITS: Record<string, number> = { free: 200, pro: 1000, scale: 5000 };

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

      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
        type: "text",
        createdAt: new Date().toISOString(),
      };

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

          const newPhase =
            (doneMessage.metadata?.phase as ConversationPhase | undefined) ??
            undefined;
          if (newPhase) {
            setPhase(newPhase);
          }

          setStreaming(false);
          abortRef.current = null;
        },
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

        setMessages((prev) =>
          prev.map((m) =>
            m.id === uploadMsgId
              ? { ...m, content: `Dataset uploaded: ${file.name}` }
              : m,
          ),
        );

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

  const handleTrainingComplete = useCallback(
    (status: any) => {
      if (status.status === "failed") {
        const parsed = parseTrainingError(status.message || "Unknown error");
        const errorMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          type: "training_error",
          content: parsed.friendly,
          metadata: { friendlyMessage: parsed.friendly },
          createdAt: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, errorMsg]);
        handleSend(`Training failed with error: ${parsed.friendly}. Can you suggest a fix?`);
      }
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
        <CircularProgress size={28} sx={{ color: "primary.main" }} />
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
              bgcolor: (theme) =>
                theme.palette.mode === "dark"
                  ? "rgba(0, 0, 0, 0.6)"
                  : "rgba(255, 255, 255, 0.85)",
              borderRadius: 2,
              border: "2px dashed",
              borderColor: "primary.main",
              pointerEvents: "none",
            }}
          >
            <Typography
              variant="h3"
              sx={{ color: "primary.main", fontWeight: 500 }}
            >
              Drop your dataset here (max {uploadLimitMB} MB)
            </Typography>
          </Box>
        )}

        {showEmptyState ? (
          /* Empty state — shown when phase is greeting and no messages */
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              flex: 1,
              gap: 3,
              px: 2,
            }}
          >
            {/* Watermark logomark */}
            <Typography
              sx={{
                fontFamily: "'DM Serif Display', Georgia, serif",
                fontSize: "6rem",
                fontWeight: 400,
                color: "text.primary",
                opacity: 0.06,
                lineHeight: 1,
                userSelect: "none",
              }}
            >
              W
            </Typography>

            {/* Heading */}
            <Typography
              variant="h1"
              sx={{
                fontFamily: "'DM Serif Display', Georgia, serif",
                fontSize: "1.5rem",
                fontWeight: 400,
                color: "text.secondary",
                letterSpacing: "-0.02em",
                mt: -2,
              }}
            >
              What would you like to train?
            </Typography>

            {/* Chat input */}
            <Box sx={{ width: "100%", maxWidth: 600 }}>
              <ChatInput
                onSend={handleSend}
                onFileUpload={handleFileUpload}
                maxUploadMB={uploadLimitMB}
                disabled={streaming}
                placeholder="Describe what you want to build..."
              />
            </Box>

            {/* Quick start chips */}
            <QuickStartChips onSelect={handleSend} />
          </Box>
        ) : (
          /* Messages */
          <Box
            sx={{
              maxWidth: 720,
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
              <ChatMessageBubble key={msg.id} message={msg} onTrainingComplete={handleTrainingComplete} />
            ))}

            {streaming && (
              <Box sx={{ display: "flex", alignItems: "flex-start", gap: 1.5, px: 2, py: 1 }}>
                {/* AI Avatar */}
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
                    animation: "pulse 1.5s ease-in-out infinite",
                    "@keyframes pulse": {
                      "0%, 100%": { opacity: 1 },
                      "50%": { opacity: 0.5 },
                    },
                  }}
                >
                  <Typography
                    sx={{
                      fontFamily: "'DM Serif Display', Georgia, serif",
                      fontSize: "0.8rem",
                      color: "#FFFFFF",
                      lineHeight: 1,
                    }}
                  >
                    W
                  </Typography>
                </Box>
                {/* Sliding bar indicator */}
                <Box
                  sx={{
                    width: 40,
                    height: 3,
                    borderRadius: 2,
                    bgcolor: "primary.main",
                    mt: "12px",
                    animation: "slideBar 1.2s ease-in-out infinite",
                    "@keyframes slideBar": {
                      "0%": { transform: "translateX(0)", opacity: 0.4 },
                      "50%": { transform: "translateX(12px)", opacity: 1 },
                      "100%": { transform: "translateX(0)", opacity: 0.4 },
                    },
                  }}
                />
              </Box>
            )}

            <div ref={bottomRef} />
          </Box>
        )}
      </Box>

      {/* Input bar — only shown after empty state */}
      {!showEmptyState && (
        <Box
          sx={{
            px: 3,
            pb: 3,
            pt: 2,
            display: "flex",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          <Box sx={{ maxWidth: 660, width: "100%" }}>
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
