"use client";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  ComposedChart,
} from "recharts";
import Box from "@mui/material/Box";
import { useTheme } from "@mui/material/styles";
import { themeTokens } from "@/theme";

interface DataPoint {
  epoch: number;
  loss: number;
  accuracy: number;
}

interface TrainingChartProps {
  data: DataPoint[];
  /** If true, use single-accent styling (loss + accuracy in accent shades) for design system */
  singleAccent?: boolean;
}

const MUTED = "rgba(255,255,255,0.35)";
const GRID = "rgba(255,255,255,0.06)";

export default function TrainingChart({ data, singleAccent = true }: TrainingChartProps) {
  const theme = useTheme();
  if (data.length === 0) {
    return null;
  }

  const accent = themeTokens.accent;
  const accentLight = themeTokens.accentMuted;  // violet at 0.15 opacity
  const accentLighter = themeTokens.completion;  // #F59E0B amber for accuracy line

  return (
    <Box
      aria-label="Training progress chart"
      role="img"
      sx={{
        mt: 2,
        bgcolor: "background.default",
        borderRadius: 1,
        p: 1.5,
        border: "1px solid",
        borderColor: "divider",
      }}
    >
      <ResponsiveContainer width="100%" height={250}>
        <ComposedChart data={data} margin={{ top: 10, right: 24, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="lossFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={accent} stopOpacity={0.15} />
              <stop offset="100%" stopColor={accent} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
          <XAxis
            dataKey="epoch"
            stroke={MUTED}
            tick={{ fill: MUTED, fontSize: 11, fontFamily: "'DM Mono', monospace" }}
            label={{
              value: "Epoch",
              position: "insideBottom",
              offset: -5,
              fill: MUTED,
              fontSize: 11,
            }}
          />
          <YAxis
            yAxisId="left"
            stroke={MUTED}
            tick={{ fill: MUTED, fontSize: 11, fontFamily: "'DM Mono', monospace" }}
            label={{
              value: "Loss",
              angle: -90,
              position: "insideLeft",
              fill: accent,
              fontSize: 11,
            }}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            stroke={MUTED}
            tick={{ fill: MUTED, fontSize: 11, fontFamily: "'DM Mono', monospace" }}
            domain={[0, 100]}
            label={{
              value: "Accuracy %",
              angle: 90,
              position: "insideRight",
              fill: singleAccent ? accentLighter : MUTED,
              fontSize: 11,
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: 8,
              fontFamily: "'DM Mono', monospace",
            }}
            labelStyle={{ color: "#fff" }}
            formatter={(value, name) => [
              value == null ? "" : name === "Accuracy" ? `${Number(value).toFixed(2)}%` : Number(value).toFixed(4),
              name ?? "",
            ]}
          />
          <Area
            yAxisId="left"
            type="monotone"
            dataKey="loss"
            fill="url(#lossFill)"
            stroke="none"
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="loss"
            stroke={accent}
            strokeWidth={2}
            dot={{ fill: accent, r: 3 }}
            activeDot={{ r: 5, fill: accent, stroke: "none" }}
            name="Loss"
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="accuracy"
            stroke={singleAccent ? accentLighter : accent}
            strokeWidth={2}
            dot={{ fill: singleAccent ? accentLighter : accent, r: 3 }}
            activeDot={{ r: 5 }}
            name="Accuracy"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </Box>
  );
}
