"use client";

import { useState, useRef, useCallback } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import type { ChatMessage } from "@/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export interface ChatUploadAreaOptions {
  currentConversationId: string | null;
  token: string | null;
  uploadLimitMB: number;
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  handleSend: (text: string) => void;
}

export function useChatUpload({
  currentConversationId,
  token,
  uploadLimitMB,
  setMessages,
  handleSend,
}: ChatUploadAreaOptions) {
  const [dragOver, setDragOver] = useState(false);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

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
    [currentConversationId, token, handleSend, setMessages],
  );

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

  return {
    dragOver,
    messagesContainerRef,
    handleFileUpload,
    handleDragOver,
    handleDragLeave,
    handleDrop,
  };
}

interface DragOverlayProps {
  uploadLimitMB: number;
}

export function DragOverlay({ uploadLimitMB }: DragOverlayProps) {
  return (
    <Box
      sx={{
        position: "absolute",
        inset: 0,
        zIndex: 10,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        bgcolor: "rgba(9, 9, 11, 0.85)",
        backdropFilter: "blur(8px)",
        borderRadius: 1.5,
        border: "2px dashed #F97316",
        pointerEvents: "none",
      }}
    >
      <Box sx={{ textAlign: "center" }}>
        <Typography
          sx={{
            fontSize: "1rem",
            fontWeight: 600,
            color: "#F97316",
            mb: 0.5,
          }}
        >
          Drop your dataset
        </Typography>
        <Typography sx={{ fontSize: "0.8125rem", color: "#52525B" }}>
          .zip files up to {uploadLimitMB} MB
        </Typography>
      </Box>
    </Box>
  );
}
