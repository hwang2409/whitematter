// frontend/src/__tests__/TrainingCard.test.tsx
import { describe, it, expect, vi } from 'vitest';
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import TrainingCard from '../components/training/TrainingCard';

const theme = getTheme('dark');
const wrap = (ui: React.ReactElement) =>
  render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

describe('TrainingCard', () => {
  const defaultProps = {
    modelName: 'MNIST Classifier',
    stage: 'training' as const,
    epoch: 12,
    totalEpochs: 50,
    loss: 0.12,
    accuracy: 94.2,
    lossHistory: [
      { epoch: 1, loss: 2.3 },
      { epoch: 5, loss: 0.8 },
      { epoch: 12, loss: 0.12 },
    ],
    isLatestActive: true,
  };

  it('renders model name', () => {
    wrap(<TrainingCard {...defaultProps} />);
    expect(screen.getByText(/MNIST Classifier/)).toBeTruthy();
  });

  it('renders progress percentage', () => {
    wrap(<TrainingCard {...defaultProps} />);
    expect(screen.getByText('24%')).toBeTruthy(); // 12/50 = 24%
  });

  it('renders epoch count', () => {
    wrap(<TrainingCard {...defaultProps} />);
    expect(screen.getByText(/12\/50/)).toBeTruthy();
  });

  it('renders loss and accuracy stats', () => {
    wrap(<TrainingCard {...defaultProps} />);
    expect(screen.getByText(/0\.1200/)).toBeTruthy();
    expect(screen.getByText(/94\.2/)).toBeTruthy();
  });

  it('has aria-live for screen readers', () => {
    const { container } = wrap(<TrainingCard {...defaultProps} />);
    const liveRegion = container.querySelector('[aria-live="polite"]');
    expect(liveRegion).toBeTruthy();
  });

  it('shows expand button', () => {
    wrap(<TrainingCard {...defaultProps} />);
    expect(screen.getByText('Expand Details')).toBeTruthy();
  });

  it('expands on click to show collapse button', () => {
    wrap(<TrainingCard {...defaultProps} />);
    fireEvent.click(screen.getByText('Expand Details'));
    expect(screen.getByText('Collapse')).toBeTruthy();
  });

  it('renders completed state', () => {
    wrap(
      <TrainingCard
        {...defaultProps}
        stage="complete"
        epoch={50}
        accuracy={97.5}
      />
    );
    expect(screen.getByText(/97\.5/)).toBeTruthy();
  });

  it('renders error state', () => {
    wrap(
      <TrainingCard
        {...defaultProps}
        stage="training"
        error="Out of memory"
      />
    );
    expect(screen.getByText(/Out of memory/)).toBeTruthy();
  });

  it('hides progress ring in error state', () => {
    wrap(
      <TrainingCard
        {...defaultProps}
        error="Out of memory"
      />
    );
    expect(screen.queryByRole('progressbar')).toBeNull();
  });

  it('shows cancel button when onCancel provided and active', () => {
    const onCancel = vi.fn();
    wrap(<TrainingCard {...defaultProps} onCancel={onCancel} />);
    expect(screen.getByText('Cancel')).toBeTruthy();
  });

  it('hides cancel button in completed state', () => {
    const onCancel = vi.fn();
    wrap(
      <TrainingCard
        {...defaultProps}
        stage="complete"
        epoch={50}
        onCancel={onCancel}
      />
    );
    expect(screen.queryByText('Cancel')).toBeNull();
  });
});
