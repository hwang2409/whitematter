// frontend/src/components/training/SparklineChart.tsx
'use client';

import React from 'react';
import { Box } from '@mui/material';
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import { themeTokens } from '../../theme';

interface SparklineDataPoint {
  epoch: number;
  loss: number;
}

interface SparklineChartProps {
  data: SparklineDataPoint[];
  height?: number;
  width?: number;
}

export default function SparklineChart({ data, height = 48, width = 200 }: SparklineChartProps) {
  if (data.length === 0) return null;

  return (
    <Box sx={{ width: '100%', height }}>
      <LineChart data={data} width={width} height={height}>
        <Line
          type="monotone"
          dataKey="loss"
          stroke={themeTokens.accent}
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </Box>
  );
}
