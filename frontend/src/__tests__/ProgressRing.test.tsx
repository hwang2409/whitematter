import { describe, it, expect } from 'vitest';
import React from 'react';
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import ProgressRing from '../components/training/ProgressRing';

const theme = getTheme('dark');
const wrap = (ui: React.ReactElement) =>
  render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

describe('ProgressRing', () => {
  it('renders percentage text', () => {
    wrap(<ProgressRing value={72} />);
    expect(screen.getByText('72%')).toBeTruthy();
  });

  it('renders 0% when value is 0', () => {
    wrap(<ProgressRing value={0} />);
    expect(screen.getByText('0%')).toBeTruthy();
  });

  it('renders completed state with amber color', () => {
    wrap(<ProgressRing value={100} completed />);
    expect(screen.getByText('100%')).toBeTruthy();
  });

  it('has progressbar role with correct aria attributes', () => {
    wrap(<ProgressRing value={72} />);
    const ring = screen.getByRole('progressbar');
    expect(ring.getAttribute('aria-valuenow')).toBe('72');
    expect(ring.getAttribute('aria-valuemin')).toBe('0');
    expect(ring.getAttribute('aria-valuemax')).toBe('100');
  });
});
