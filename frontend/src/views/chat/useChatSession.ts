"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  createConversation,
  getConversation,
  sendChatMessage,
} from "@/api";
import type { ChatMessage, ConversationPhase } from "@/api";

export interface UseChatSessionOptions {
  conversationId?: string;
  token: string | null;
}

export interface UseChatSessionReturn {
  messages: ChatMessage[];
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  phase: ConversationPhase;
  streaming: boolean;
  loading: boolean;
  currentConversationId: string | null;
  bottomRef: React.RefObject<HTMLDivElement | null>;
  handleSend: (text: string) => void;
  toolStatus: string | null;
}

export function useChatSession({
  conversationId,
  token,
}: UseChatSessionOptions): UseChatSessionReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [phase, setPhase] = useState<ConversationPhase>("greeting");
  const [streaming, setStreaming] = useState(false);
  const [toolStatus, setToolStatus] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [currentConversationId, setCurrentConversationId] = useState<
    string | null
  >(conversationId || null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<{ abort: () => void } | null>(null);
  const scrollRafRef = useRef<number>(0);
  const streamingRef = useRef(false);

  streamingRef.current = streaming;

  useEffect(() => {
    if (scrollRafRef.current) cancelAnimationFrame(scrollRafRef.current);
    scrollRafRef.current = requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({
        behavior: streamingRef.current ? "instant" : "smooth",
      });
    });
  }, [messages]);

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
          setMessages(data.messages ?? []);
        } else {
          // No conversationId — start as a draft (no backend call).
          // The conversation will be created when the user sends their first message.
          setCurrentConversationId(null);
          setPhase("greeting");
          setMessages([]);
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

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const handleSend = useCallback(
    async (text: string) => {
      if (!text.trim() || streaming) return;

      // Create conversation on first message if we're in draft state
      let convId = currentConversationId;
      if (!convId) {
        if (!token) return;
        try {
          const conv = await createConversation(token);
          convId = conv.id;
          setCurrentConversationId(convId);
          // Update URL without triggering Next.js navigation/remount
          window.history.replaceState(null, "", `/chat/${convId}`);
        } catch (err) {
          console.error("Failed to create conversation:", err);
          return;
        }
      }

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
        convId,
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
              type: doneMessage.type ?? "text",
              createdAt:
                doneMessage.createdAt ?? new Date().toISOString(),
            };

            // Append training progress message if training started
            const training = doneMessage.metadata?.training as
              | { job_id: string; conversation_id: string }
              | undefined;
            if (training?.job_id) {
              updated.push({
                id: crypto.randomUUID(),
                role: "assistant",
                content: "Training started!",
                type: "training_progress",
                metadata: {
                  job_id: training.job_id,
                  conversation_id: training.conversation_id,
                  status: "started",
                },
                createdAt: new Date().toISOString(),
              });
            }

            return updated;
          });

          const newPhase =
            (doneMessage.metadata?.phase as
              | ConversationPhase
              | undefined) ?? undefined;
          if (newPhase) {
            setPhase(newPhase);
          }

          setStreaming(false);
          setToolStatus(null);
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
          setToolStatus(null);
          abortRef.current = null;
        },
        token ?? undefined,
        (name) =>
          setToolStatus(
            name === "search_datasets"
              ? "Searching HuggingFace..."
              : "Importing dataset...",
          ),
        () => setToolStatus(null),
        (_name, message) => setToolStatus(message),
      );

      abortRef.current = handle;
    },
    [streaming, currentConversationId, token],
  );

  return {
    messages,
    setMessages,
    phase,
    streaming,
    loading,
    currentConversationId,
    bottomRef,
    handleSend,
    toolStatus,
  };
}
