// frontend/src/components/training/StageIndicator.tsx
'use client';

import React from 'react';
import { Box, Typography } from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import { themeTokens } from '../../theme';

export type TrainingStage = 'preparing' | 'training' | 'evaluating' | 'complete' | 'ready';

const STAGES: { key: TrainingStage; label: string }[] = [
  { key: 'preparing', label: 'Preparing' },
  { key: 'training', label: 'Training' },
  { key: 'evaluating', label: 'Evaluating' },
  { key: 'complete', label: 'Complete' },
  { key: 'ready', label: 'Ready' },
];

const STAGE_INDEX: Record<TrainingStage, number> = {
  preparing: 0,
  training: 1,
  evaluating: 2,
  complete: 3,
  ready: 4,
};

interface StageIndicatorProps {
  currentStage: TrainingStage;
}

export default function StageIndicator({ currentStage }: StageIndicatorProps) {
  const currentIdx = STAGE_INDEX[currentStage];

  return (
    <Box
      role="status"
      aria-label={`Current stage: ${STAGES[currentIdx].label}`}
      sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}
    >
      {STAGES.map((stage, idx) => {
        const isCompleted = idx < currentIdx;
        const isActive = idx === currentIdx;
        const isAmber = idx >= 3 && isCompleted;
        const isAmberActive = idx >= 3 && isActive && currentIdx >= 3;

        const dotColor = isCompleted || isActive
          ? (isAmber || isAmberActive ? themeTokens.completion : themeTokens.accent)
          : themeTokens.textMuted;

        return (
          <Box
            key={stage.key}
            sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.5, minWidth: 56 }}
          >
            <Box
              sx={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                bgcolor: isCompleted || isActive ? dotColor : 'transparent',
                border: isCompleted || isActive ? 'none' : `2px solid ${themeTokens.textMuted}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                ...(isActive && !isCompleted && {
                  animation: 'stagePulse 1.5s ease-in-out infinite',
                  '@keyframes stagePulse': {
                    '0%, 100%': { opacity: 1 },
                    '50%': { opacity: 0.5 },
                  },
                  '@media (prefers-reduced-motion: reduce)': {
                    animation: 'none',
                    transition: 'none',
                  },
                }),
              }}
            >
              {isCompleted && (
                <CheckIcon
                  data-testid="stage-check"
                  sx={{ fontSize: 8, color: '#FFFFFF' }}
                />
              )}
            </Box>
            <Typography
              sx={{
                fontSize: '0.625rem',
                fontFamily: "'DM Mono', monospace",
                color: isActive ? 'text.primary' : themeTokens.textMuted,
                fontWeight: isActive ? 600 : 400,
              }}
            >
              {stage.label}
            </Typography>
          </Box>
        );
      })}
    </Box>
  );
}
