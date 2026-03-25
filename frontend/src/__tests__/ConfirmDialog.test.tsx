// frontend/src/__tests__/ConfirmDialog.test.tsx
import { describe, it, expect, vi } from 'vitest';
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import ConfirmDialog from '../components/ConfirmDialog';

const theme = getTheme('dark');
const wrap = (ui: React.ReactElement) =>
  render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

describe('ConfirmDialog', () => {
  const defaultProps = {
    isOpen: true,
    title: 'Delete Model',
    message: 'Are you sure you want to delete this model?',
    onConfirm: vi.fn(),
    onCancel: vi.fn(),
  };

  it('renders title and message when isOpen=true', () => {
    wrap(<ConfirmDialog {...defaultProps} />);
    expect(screen.getByText('Delete Model')).toBeTruthy();
    expect(screen.getByText('Are you sure you want to delete this model?')).toBeTruthy();
  });

  it('does not render content when isOpen=false', () => {
    wrap(<ConfirmDialog {...defaultProps} isOpen={false} />);
    expect(screen.queryByText('Delete Model')).toBeNull();
    expect(screen.queryByText('Are you sure you want to delete this model?')).toBeNull();
  });

  it('renders custom confirm/cancel labels', () => {
    wrap(
      <ConfirmDialog
        {...defaultProps}
        confirmLabel="Yes, delete"
        cancelLabel="Go back"
      />
    );
    expect(screen.getByText('Yes, delete')).toBeTruthy();
    expect(screen.getByText('Go back')).toBeTruthy();
  });

  it('clicking confirm button calls onConfirm', () => {
    const onConfirm = vi.fn();
    wrap(<ConfirmDialog {...defaultProps} onConfirm={onConfirm} />);
    fireEvent.click(screen.getByText('Confirm'));
    expect(onConfirm).toHaveBeenCalledOnce();
  });

  it('clicking cancel button calls onCancel', () => {
    const onCancel = vi.fn();
    wrap(<ConfirmDialog {...defaultProps} onCancel={onCancel} />);
    fireEvent.click(screen.getByText('Cancel'));
    expect(onCancel).toHaveBeenCalledOnce();
  });

  it('danger variant styles apply', () => {
    wrap(<ConfirmDialog {...defaultProps} variant="danger" />);
    const confirmButton = screen.getByText('Confirm');
    const style = window.getComputedStyle(confirmButton);
    expect(style.backgroundColor).toBe('rgb(239, 68, 68)');
  });
});
