"use client";

import { useCallback } from "react";
import type { ChatMessage } from "@/api";
import { parseTrainingError } from "@/lib/trainingErrors";

export interface UseChatTrainingOptions {
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  handleSend: (text: string) => void;
}

export function useChatTraining({
  setMessages,
  handleSend,
}: UseChatTrainingOptions) {
  const handleTrainingComplete = useCallback(
    (status: any) => {
      if (status.status === "failed") {
        const parsed = parseTrainingError(status.error_message || status.message || "Unknown error");
        const errorMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          type: "training_error",
          content: parsed.friendly,
          metadata: {
            friendlyMessage: parsed.friendly,
            suggestion: parsed.suggestion,
          },
          createdAt: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, errorMsg]);
        handleSend(
          `Training failed with error: ${parsed.friendly}. Can you suggest a fix?`,
        );
      }
    },
    [handleSend, setMessages],
  );

  return { handleTrainingComplete };
}
