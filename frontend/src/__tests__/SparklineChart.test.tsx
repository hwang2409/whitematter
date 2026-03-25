import { describe, it, expect } from 'vitest';
import React from 'react';
import { render } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import SparklineChart from '../components/training/SparklineChart';

const theme = getTheme('dark');
const wrap = (ui: React.ReactElement) =>
  render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

describe('SparklineChart', () => {
  it('renders without crashing', () => {
    const data = [
      { epoch: 1, loss: 2.3 },
      { epoch: 2, loss: 1.8 },
      { epoch: 3, loss: 1.2 },
    ];
    const { container } = wrap(<SparklineChart data={data} />);
    // Recharts renders SVG elements
    expect(container.querySelector('svg')).toBeTruthy();
  });

  it('renders nothing when data is empty', () => {
    const { container } = wrap(<SparklineChart data={[]} />);
    expect(container.querySelector('svg')).toBeNull();
  });
});
