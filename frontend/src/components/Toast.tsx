"use client";
import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import Typography from "@mui/material/Typography";

export interface ToastMessage {
  id: string;
  type: "success" | "error" | "info" | "warning";
  message: string;
}

interface Props {
  toasts: ToastMessage[];
  onDismiss: (id: string) => void;
}

const typeColors: Record<ToastMessage["type"], string> = {
  success: "#22C55E",
  error: "#EF4444",
  warning: "#EAB308",
  info: "#3B82F6",
};

export default function Toast({ toasts, onDismiss }: Props) {
  return (
    <Box
      sx={{
        position: "fixed",
        bottom: 24,
        right: 24,
        display: "flex",
        flexDirection: "column",
        gap: 1,
        zIndex: 1001,
      }}
    >
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
      ))}
    </Box>
  );
}

function ToastItem({
  toast,
  onDismiss,
}: {
  toast: ToastMessage;
  onDismiss: (id: string) => void;
}) {
  const [isExiting, setIsExiting] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => {
      setIsExiting(true);
      setTimeout(() => onDismiss(toast.id), 250);
    }, 4000);
    return () => clearTimeout(t);
  }, [toast.id, onDismiss]);

  const handleDismiss = () => {
    setIsExiting(true);
    setTimeout(() => onDismiss(toast.id), 250);
  };

  const color = typeColors[toast.type];

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 1.25,
        py: 1.25,
        px: 1.5,
        pr: 1,
        minWidth: 280,
        maxWidth: 400,
        bgcolor: "#18181B",
        border: "1px solid #27272A",
        borderLeft: `3px solid ${color}`,
        borderRadius: "8px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
        animation: isExiting
          ? "phosphorToastExit 0.25s ease-out forwards"
          : "phosphorToastEnter 0.3s ease-out",
        "@keyframes phosphorToastEnter": {
          from: { opacity: 0, transform: "translateX(40px)" },
          to: { opacity: 1, transform: "translateX(0)" },
        },
        "@keyframes phosphorToastExit": {
          from: { opacity: 1, transform: "translateX(0)" },
          to: { opacity: 0, transform: "translateX(40px)" },
        },
      }}
    >
      {/* Colored dot indicator */}
      <Box
        sx={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          bgcolor: color,
          flexShrink: 0,
        }}
      />
      <Typography
        sx={{
          flex: 1,
          fontSize: "0.875rem",
          fontFamily: "'Outfit', sans-serif",
          color: "#FAFAFA",
          lineHeight: 1.5,
        }}
      >
        {toast.message}
      </Typography>
      <IconButton
        size="small"
        onClick={handleDismiss}
        sx={{
          color: "#52525B",
          p: 0.5,
          fontSize: "0.75rem",
          "&:hover": { color: "#A1A1AA" },
        }}
        aria-label="Dismiss"
      >
        &#10005;
      </IconButton>
    </Box>
  );
}

export function useToast() {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const addToast = (type: ToastMessage["type"], message: string) => {
    const id = Math.random().toString(36).slice(2);
    setToasts((prev) => [...prev, { id, type, message }]);
  };

  const dismissToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  return {
    toasts,
    addToast,
    dismissToast,
    success: (message: string) => addToast("success", message),
    error: (message: string) => addToast("error", message),
    info: (message: string) => addToast("info", message),
    warning: (message: string) => addToast("warning", message),
  };
}
