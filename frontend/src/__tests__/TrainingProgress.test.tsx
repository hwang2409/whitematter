// frontend/src/__tests__/TrainingProgress.test.tsx
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import React from 'react';
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import TrainingProgress from '../components/TrainingProgress';

const theme = getTheme('dark');
const wrap = (ui: React.ReactElement) =>
  render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

// Prevent the SSE useEffect from actually fetching — return a promise that
// never resolves so the component stays in its initial (loading) state.
const originalFetch = globalThis.fetch;

beforeEach(() => {
  globalThis.fetch = vi.fn(() => new Promise<Response>(() => {}));
  vi.spyOn(Storage.prototype, 'getItem').mockReturnValue(null);
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe('TrainingProgress', () => {
  const defaultProps = {
    conversationId: 'conv-1',
    jobId: 'job-1',
  };

  it('renders "Training in Progress" on initial load', () => {
    wrap(<TrainingProgress {...defaultProps} />);
    expect(screen.getByText(/Training.*in Progress/)).toBeTruthy();
  });

  it('renders 0% progress before any SSE data arrives', () => {
    wrap(<TrainingProgress {...defaultProps} />);
    expect(screen.getByText('0%')).toBeTruthy();
  });
});
