"use client";

import { useState, useCallback, useRef, type KeyboardEvent, type ChangeEvent } from "react";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import AttachFileOutlined from "@mui/icons-material/AttachFileOutlined";
import ArrowUpwardRounded from "@mui/icons-material/ArrowUpwardRounded";

interface ChatInputProps {
  onSend: (text: string) => void;
  onFileUpload?: (file: File) => void;
  disabled?: boolean;
  placeholder?: string;
  maxUploadMB?: number;
}

export default function ChatInput({
  onSend,
  onFileUpload,
  disabled,
  placeholder,
  maxUploadMB = 200,
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSend = useCallback(() => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
    setTimeout(() => inputRef.current?.focus(), 0);
  }, [value, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const handleAttachClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      e.target.value = "";

      const maxBytes = maxUploadMB * 1024 * 1024;
      if (file.size > maxBytes) {
        alert(`File too large. Maximum size is ${maxUploadMB} MB.`);
        return;
      }

      onFileUpload?.(file);
    },
    [onFileUpload, maxUploadMB],
  );

  const hasText = value.trim().length > 0;

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 0.75,
        bgcolor: "background.paper",
        border: "1px solid",
        borderColor: "divider",
        borderRadius: "14px",
        px: 0.75,
        pl: 2.5,
        py: 0.75,
        transition: "all 0.2s ease-out",
        boxShadow: "0 2px 12px rgba(0,0,0,0.02)",
        "&:focus-within": {
          borderColor: (theme) =>
            theme.palette.mode === "dark"
              ? "rgba(255,255,255,0.14)"
              : "rgba(0,0,0,0.12)",
          boxShadow: "0 2px 20px rgba(0,0,0,0.04)",
        },
      }}
    >
      <Box
        component="textarea"
        ref={inputRef}
        value={value}
        onChange={(e: any) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder ?? "Type a message..."}
        disabled={disabled}
        rows={1}
        sx={{
          flex: 1,
          background: "none",
          border: "none",
          outline: "none",
          color: "text.primary",
          fontSize: "0.9375rem",
          fontFamily: "'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif",
          py: 1.25,
          px: 0,
          resize: "none",
          maxHeight: "120px",
          overflow: "auto",
          lineHeight: 1.5,
          "&::placeholder": {
            color: "text.disabled",
          },
          "&:disabled": {
            opacity: 0.5,
          },
        }}
      />

      <input
        ref={fileInputRef}
        type="file"
        accept=".zip"
        hidden
        onChange={handleFileChange}
      />

      <IconButton
        size="small"
        disabled={disabled}
        onClick={handleAttachClick}
        sx={{
          width: 36,
          height: 36,
          borderRadius: "10px",
          color: "text.disabled",
          "&:hover": {
            color: "text.secondary",
            bgcolor: "background.default",
          },
        }}
        aria-label="Attach file"
      >
        <AttachFileOutlined sx={{ fontSize: 18 }} />
      </IconButton>

      <IconButton
        size="small"
        onClick={handleSend}
        disabled={disabled || !hasText}
        sx={{
          width: 36,
          height: 36,
          borderRadius: "10px",
          bgcolor: hasText ? "text.primary" : "transparent",
          color: hasText
            ? (theme) => (theme.palette.mode === "dark" ? "#141311" : "#FFFFFF")
            : "text.disabled",
          "&:hover": {
            bgcolor: hasText
              ? (theme) => (theme.palette.mode === "dark" ? "#d4d3d0" : "#333")
              : "transparent",
          },
          "&.Mui-disabled": {
            color: "text.disabled",
            bgcolor: "transparent",
          },
          transition: "all 0.15s ease-out",
        }}
        aria-label="Send message"
      >
        <ArrowUpwardRounded sx={{ fontSize: 18 }} />
      </IconButton>
    </Box>
  );
}
