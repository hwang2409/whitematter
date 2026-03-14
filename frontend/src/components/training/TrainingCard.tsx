'use client';

import React, { useState } from 'react';
import { Box, Typography, Button, Collapse } from '@mui/material';
import { themeTokens } from '../../theme';
import ProgressRing from './ProgressRing';
import StageIndicator, { TrainingStage } from './StageIndicator';
import SparklineChart from './SparklineChart';

interface TrainingCardProps {
  modelName: string;
  stage: TrainingStage;
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  lossHistory: { epoch: number; loss: number; accuracy?: number }[];
  isLatestActive: boolean;
  modelId?: string;
  error?: string;
  architecture?: string[];
  datasetName?: string;
  learningRate?: number;
  batchSize?: number;
  optimizer?: string;
  onCancel?: () => void;
  onRetry?: () => void;
}

export default function TrainingCard({
  modelName,
  stage,
  epoch,
  totalEpochs,
  loss,
  accuracy,
  lossHistory,
  isLatestActive,
  modelId,
  error,
  learningRate,
  batchSize,
  optimizer,
  onCancel,
  onRetry,
}: TrainingCardProps) {
  const [expanded, setExpanded] = useState(false);

  const percentage = totalEpochs > 0 ? Math.round((epoch / totalEpochs) * 100) : 0;
  const isCompleted = stage === 'complete' || stage === 'ready';
  const isError = !!error;
  const isActive = !isCompleted && !isError;

  const borderColor = isError
    ? themeTokens.error
    : isCompleted
      ? themeTokens.completion
      : themeTokens.accent;

  const glowColor = isError
    ? 'rgba(239, 68, 68, 0.12)'
    : isCompleted
      ? themeTokens.completionMuted
      : themeTokens.accentMuted;

  return (
    <Box
      aria-live="polite"
      sx={{
        width: '90%',
        maxWidth: 640,
        mx: 'auto',
        bgcolor: themeTokens.card,
        border: `1px solid ${borderColor}`,
        borderRadius: '16px',
        p: 2.5,
        position: 'relative',
        overflow: 'hidden',
        boxShadow: `0 0 16px ${glowColor}`,
        ...(isActive && isLatestActive && {
          '&::after': {
            content: '""',
            position: 'absolute',
            inset: -1,
            borderRadius: '16px',
            boxShadow: `0 0 20px ${themeTokens.accentMuted}`,
            animation: 'glowPulse 2s ease-in-out infinite',
            pointerEvents: 'none',
            '@keyframes glowPulse': {
              '0%, 100%': { opacity: 1 },
              '50%': { opacity: 0.3 },
            },
          },
        }),
      }}
    >
      {/* Header */}
      <Typography
        sx={{
          fontFamily: "'DM Serif Display', Georgia, serif",
          fontSize: '1rem',
          fontWeight: 400,
          color: 'text.primary',
          mb: 1.5,
        }}
      >
        {isError ? 'Training Failed' : isCompleted ? 'Training Complete' : 'Training'}: {modelName}
      </Typography>

      {/* Stage Indicator */}
      <Box sx={{ mb: 2 }}>
        <StageIndicator currentStage={stage} />
      </Box>

      {/* Error state */}
      {isError && (
        <Box sx={{ mb: 2 }}>
          <Typography sx={{ color: themeTokens.error, fontSize: '0.875rem', mb: 1 }}>
            {error}
          </Typography>
          {onRetry && (
            <Button size="small" variant="outlined" onClick={onRetry} sx={{ borderColor: themeTokens.error, color: themeTokens.error }}>
              Retry
            </Button>
          )}
        </Box>
      )}

      {/* Progress section — hidden on error */}
      {!isError && (
        <>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2.5, mb: 2 }}>
            <ProgressRing value={isCompleted ? 100 : percentage} completed={isCompleted} />
            <Box sx={{ flex: 1 }}>
              <Typography sx={{ fontFamily: "'DM Mono', monospace", fontSize: '0.875rem', color: 'text.primary' }}>
                Epoch {epoch}/{totalEpochs}
              </Typography>
              <Typography sx={{ fontFamily: "'DM Mono', monospace", fontSize: '0.8125rem', color: 'text.secondary' }}>
                Loss: {loss.toFixed(4)}
              </Typography>
              <Typography sx={{ fontFamily: "'DM Mono', monospace", fontSize: '0.8125rem', color: 'text.secondary' }}>
                Accuracy: {accuracy.toFixed(1)}%
              </Typography>
            </Box>
          </Box>

          {/* Sparkline */}
          {lossHistory.length > 1 && (
            <Box sx={{ mb: 2 }}>
              <SparklineChart data={lossHistory} />
            </Box>
          )}
        </>
      )}

      {/* Actions */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Button
          size="small"
          onClick={() => setExpanded(!expanded)}
          sx={{ color: 'text.secondary', textTransform: 'none', fontSize: '0.8125rem' }}
        >
          {expanded ? 'Collapse' : 'Expand Details'}
        </Button>
        {isActive && onCancel && (
          <Button
            size="small"
            onClick={onCancel}
            sx={{ color: themeTokens.textMuted, textTransform: 'none', fontSize: '0.8125rem' }}
          >
            Cancel
          </Button>
        )}
      </Box>

      {/* Expanded details */}
      <Collapse in={expanded} timeout={200} easing="ease-out">
        <Box
          sx={{
            mt: 2,
            pt: 2,
            borderTop: `1px solid ${themeTokens.border}`,
          }}
        >
          {/* Hyperparameters */}
          {(learningRate || batchSize || optimizer) && (
            <Box sx={{ mb: 2 }}>
              <Typography sx={{ fontSize: '0.75rem', color: themeTokens.textMuted, mb: 0.5, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                Hyperparameters
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                {optimizer && (
                  <Typography sx={{ fontFamily: "'DM Mono', monospace", fontSize: '0.8125rem', color: 'text.secondary' }}>
                    Optimizer: {optimizer}
                  </Typography>
                )}
                {learningRate && (
                  <Typography sx={{ fontFamily: "'DM Mono', monospace", fontSize: '0.8125rem', color: 'text.secondary' }}>
                    LR: {learningRate}
                  </Typography>
                )}
                {batchSize && (
                  <Typography sx={{ fontFamily: "'DM Mono', monospace", fontSize: '0.8125rem', color: 'text.secondary' }}>
                    Batch: {batchSize}
                  </Typography>
                )}
              </Box>
            </Box>
          )}
        </Box>
      </Collapse>
    </Box>
  );
}
