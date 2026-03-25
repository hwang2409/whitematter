// frontend/src/__tests__/Toast.test.tsx
import { describe, it, expect, vi } from 'vitest';
import React from 'react';
import { render, screen, fireEvent, act, renderHook } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import Toast, { ToastMessage, useToast } from '../components/Toast';

const theme = getTheme('dark');
const wrap = (ui: React.ReactElement) =>
  render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

describe('Toast', () => {
  const defaultToasts: ToastMessage[] = [
    { id: '1', type: 'success', message: 'Model saved successfully' },
  ];
  const noop = () => {};

  it('renders toast message with correct text', () => {
    wrap(<Toast toasts={defaultToasts} onDismiss={noop} />);
    expect(screen.getByText('Model saved successfully')).toBeTruthy();
  });

  it('renders dismiss button', () => {
    wrap(<Toast toasts={defaultToasts} onDismiss={noop} />);
    expect(screen.getByLabelText('Dismiss')).toBeTruthy();
  });

  it('calls onDismiss with correct id when dismiss is clicked', () => {
    vi.useFakeTimers();
    const onDismiss = vi.fn();
    wrap(<Toast toasts={defaultToasts} onDismiss={onDismiss} />);
    fireEvent.click(screen.getByLabelText('Dismiss'));
    act(() => {
      vi.advanceTimersByTime(250);
    });
    expect(onDismiss).toHaveBeenCalledWith('1');
    vi.useRealTimers();
  });

  it('renders multiple toasts simultaneously', () => {
    const toasts: ToastMessage[] = [
      { id: '1', type: 'success', message: 'Training started' },
      { id: '2', type: 'error', message: 'GPU memory exceeded' },
      { id: '3', type: 'info', message: 'Checkpoint saved' },
    ];
    wrap(<Toast toasts={toasts} onDismiss={noop} />);
    expect(screen.getByText('Training started')).toBeTruthy();
    expect(screen.getByText('GPU memory exceeded')).toBeTruthy();
    expect(screen.getByText('Checkpoint saved')).toBeTruthy();
  });
});

describe('useToast', () => {
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <ThemeProvider theme={theme}>{children}</ThemeProvider>
  );

  it('starts with no toasts', () => {
    const { result } = renderHook(() => useToast(), { wrapper });
    expect(result.current.toasts).toEqual([]);
  });

  it('success adds a toast with type success', () => {
    const { result } = renderHook(() => useToast(), { wrapper });
    act(() => {
      result.current.success('Model deployed');
    });
    expect(result.current.toasts).toHaveLength(1);
    expect(result.current.toasts[0].type).toBe('success');
    expect(result.current.toasts[0].message).toBe('Model deployed');
  });

  it('dismissToast removes the toast by id', () => {
    const { result } = renderHook(() => useToast(), { wrapper });
    act(() => {
      result.current.success('First');
      result.current.error('Second');
    });
    expect(result.current.toasts).toHaveLength(2);
    const idToRemove = result.current.toasts[0].id;
    act(() => {
      result.current.dismissToast(idToRemove);
    });
    expect(result.current.toasts).toHaveLength(1);
    expect(result.current.toasts[0].message).toBe('Second');
  });
});
