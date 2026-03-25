'use client';

import { Box, Typography } from '@mui/material';
import { themeTokens } from '../../theme';

interface ProgressRingProps {
  value: number;
  size?: number;
  strokeWidth?: number;
  completed?: boolean;
}

export default function ProgressRing({
  value,
  size = 64,
  strokeWidth = 4,
  completed = false,
}: ProgressRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;
  const color = completed ? themeTokens.completion : themeTokens.accent;

  return (
    <Box
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={100}
      sx={{
        position: 'relative',
        width: size,
        height: size,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        '& circle.progress-circle': {
          transition: 'stroke-dashoffset 0.5s ease-out, stroke 0.5s ease-out',
        },
        '@media (prefers-reduced-motion: reduce)': {
          '& circle.progress-circle': {
            transition: 'none',
          },
        },
      }}
    >
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.07)"
          strokeWidth={strokeWidth}
        />
        <circle
          className="progress-circle"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <Typography
        sx={{
          position: 'absolute',
          fontFamily: "'DM Mono', monospace",
          fontWeight: 700,
          fontSize: size * 0.22,
          color: 'text.primary',
        }}
      >
        {value}%
      </Typography>
    </Box>
  );
}
