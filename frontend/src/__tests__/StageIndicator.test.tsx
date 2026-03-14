import { describe, it, expect } from 'vitest';
import React from 'react';
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import StageIndicator from '../components/training/StageIndicator';

const theme = getTheme('dark');
const wrap = (ui: React.ReactElement) =>
  render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

describe('StageIndicator', () => {
  it('renders all 5 stage labels', () => {
    wrap(<StageIndicator currentStage="preparing" />);
    expect(screen.getByText('Preparing')).toBeTruthy();
    expect(screen.getByText('Training')).toBeTruthy();
    expect(screen.getByText('Evaluating')).toBeTruthy();
    expect(screen.getByText('Complete')).toBeTruthy();
    expect(screen.getByText('Ready')).toBeTruthy();
  });

  it('marks completed stages with checkmarks', () => {
    wrap(<StageIndicator currentStage="evaluating" />);
    const checks = screen.getAllByTestId('stage-check');
    expect(checks.length).toBe(2);
  });

  it('has aria-label describing current stage', () => {
    wrap(<StageIndicator currentStage="training" />);
    const indicator = screen.getByRole('status');
    expect(indicator.getAttribute('aria-label')).toContain('Training');
  });
});
