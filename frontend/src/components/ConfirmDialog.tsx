"use client";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogActions from "@mui/material/DialogActions";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";

interface Props {
  isOpen: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: "danger" | "warning" | "default";
  onConfirm: () => void;
  onCancel: () => void;
}

export default function ConfirmDialog({
  isOpen,
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  variant = "default",
  onConfirm,
  onCancel,
}: Props) {
  const isDanger = variant === "danger";
  const isWarning = variant === "warning";

  const confirmBg = isDanger
    ? "#EF4444"
    : isWarning
      ? "#EAB308"
      : "#F97316";

  const confirmHoverBg = isDanger
    ? "#DC2626"
    : isWarning
      ? "#CA8A04"
      : "#EA580C";

  return (
    <Dialog
      open={isOpen}
      onClose={onCancel}
      PaperProps={{
        sx: {
          bgcolor: "#18181B",
          border: "1px solid #27272A",
          borderRadius: "12px",
          maxWidth: 400,
          backgroundImage: "none",
          boxShadow: "0 16px 48px rgba(0,0,0,0.6)",
        },
      }}
      slotProps={{
        backdrop: {
          sx: {
            bgcolor: "rgba(0,0,0,0.8)",
            backdropFilter: "blur(8px)",
          },
        },
      }}
    >
      <DialogTitle
        sx={{
          fontSize: "1.125rem",
          fontWeight: 600,
          fontFamily: "'Outfit', sans-serif",
          color: "#FAFAFA",
          pb: 0.5,
        }}
      >
        {title}
      </DialogTitle>
      <DialogContent>
        <Typography
          sx={{
            fontSize: "0.875rem",
            fontFamily: "'Outfit', sans-serif",
            color: "#A1A1AA",
            lineHeight: 1.5,
          }}
        >
          {message}
        </Typography>
      </DialogContent>
      <DialogActions
        sx={{
          px: 2.5,
          py: 1.5,
          borderTop: "1px solid #27272A",
          bgcolor: "#09090B",
          borderBottomLeftRadius: "12px",
          borderBottomRightRadius: "12px",
          gap: 1,
        }}
      >
        <Button
          variant="outlined"
          onClick={onCancel}
          sx={{
            borderColor: "#27272A",
            color: "#A1A1AA",
            fontFamily: "'Outfit', sans-serif",
            fontSize: "0.8125rem",
            fontWeight: 500,
            textTransform: "none",
            borderRadius: "8px",
            "&:hover": {
              borderColor: "#3F3F46",
              color: "#FAFAFA",
              bgcolor: "rgba(255,255,255,0.03)",
            },
          }}
        >
          {cancelLabel}
        </Button>
        <Button
          variant="contained"
          onClick={onConfirm}
          sx={{
            bgcolor: confirmBg,
            color: "#fff",
            fontFamily: "'Outfit', sans-serif",
            fontSize: "0.8125rem",
            fontWeight: 500,
            textTransform: "none",
            borderRadius: "8px",
            minWidth: 80,
            boxShadow: "none",
            "&:hover": {
              bgcolor: confirmHoverBg,
              boxShadow: "none",
            },
          }}
        >
          {confirmLabel}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
