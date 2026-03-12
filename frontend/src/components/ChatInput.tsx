"use client";

import { useState, useCallback, useRef, type KeyboardEvent } from "react";
import Box from "@mui/material/Box";
import TextField from "@mui/material/TextField";
import IconButton from "@mui/material/IconButton";
import SendOutlined from "@mui/icons-material/SendOutlined";
import AttachFileOutlined from "@mui/icons-material/AttachFileOutlined";
import { themeTokens } from "@/theme";

interface ChatInputProps {
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function ChatInput({ onSend, disabled, placeholder }: ChatInputProps) {
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
    // Re-focus input after sending
    setTimeout(() => inputRef.current?.focus(), 0);
  }, [value, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "flex-end",
        gap: 1,
      }}
    >
      <IconButton
        size="small"
        disabled={disabled}
        sx={{
          color: "text.secondary",
          mb: 0.5,
          "&:hover": { color: "primary.main" },
        }}
        aria-label="Attach file"
      >
        <AttachFileOutlined fontSize="small" />
      </IconButton>

      <TextField
        inputRef={inputRef}
        multiline
        maxRows={6}
        fullWidth
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder ?? "Type a message..."}
        disabled={disabled}
        slotProps={{
          input: {
            sx: {
              py: 1,
              px: 1.5,
              fontSize: "0.9375rem",
              lineHeight: 1.5,
            },
          },
        }}
        sx={{
          "& .MuiOutlinedInput-root": {
            borderRadius: 2,
          },
        }}
      />

      <IconButton
        size="small"
        onClick={handleSend}
        disabled={disabled || !value.trim()}
        sx={{
          mb: 0.5,
          color: value.trim() ? themeTokens.accent : "text.disabled",
          bgcolor: value.trim() ? "rgba(126,184,255,0.12)" : "transparent",
          "&:hover": {
            bgcolor: "rgba(126,184,255,0.2)",
          },
          "&.Mui-disabled": {
            color: "text.disabled",
          },
          transition: "all 0.2s ease-out",
        }}
        aria-label="Send message"
      >
        <SendOutlined fontSize="small" />
      </IconButton>
    </Box>
  );
}
