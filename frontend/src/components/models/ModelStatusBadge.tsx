"use client";
import Box from "@mui/material/Box";

const STATUS_STYLES: Record<string, { color: string; bg: string; pulse?: boolean }> = {
  completed: { color: "#22C55E", bg: "rgba(34,197,94,0.10)" },
  running: { color: "#F97316", bg: "rgba(249,115,22,0.10)", pulse: true },
  failed: { color: "#EF4444", bg: "rgba(239,68,68,0.10)" },
  cancelled: { color: "#52525B", bg: "rgba(82,82,91,0.10)" },
};

interface ModelStatusBadgeProps {
  status: string;
}

export default function ModelStatusBadge({ status }: ModelStatusBadgeProps) {
  const style = STATUS_STYLES[status] ?? STATUS_STYLES.cancelled;

  return (
    <Box
      component="span"
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: "0.6875rem",
        fontWeight: 500,
        lineHeight: 1,
        textTransform: "uppercase",
        letterSpacing: "0.025em",
        color: style.color,
        bgcolor: style.bg,
        borderRadius: "4px",
        px: 1,
        py: 0.5,
        ...(style.pulse && {
          "@keyframes statusPulse": {
            "0%, 100%": { opacity: 1 },
            "50%": { opacity: 0.5 },
          },
        }),
      }}
    >
      {style.pulse && (
        <Box
          component="span"
          sx={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            bgcolor: style.color,
            animation: "statusPulse 2s ease-in-out infinite",
            "@keyframes statusPulse": {
              "0%, 100%": { opacity: 1 },
              "50%": { opacity: 0.4 },
            },
          }}
        />
      )}
      {status}
    </Box>
  );
}
