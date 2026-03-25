// frontend/src/__tests__/ChatInput.test.tsx
import { describe, it, expect, vi } from 'vitest';
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import ChatInput from '../components/ChatInput';

const theme = getTheme('dark');
const wrap = (ui: React.ReactElement) =>
  render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

describe('ChatInput', () => {
  const defaultProps = {
    onSend: vi.fn(),
  };

  it('renders textarea with placeholder', () => {
    wrap(<ChatInput {...defaultProps} />);
    expect(screen.getByPlaceholderText('Type a message...')).toBeTruthy();
  });

  it('renders with custom placeholder when provided', () => {
    wrap(<ChatInput {...defaultProps} placeholder="Ask anything..." />);
    expect(screen.getByPlaceholderText('Ask anything...')).toBeTruthy();
  });

  it('calls onSend with trimmed text on send button click', () => {
    const onSend = vi.fn();
    wrap(<ChatInput onSend={onSend} />);
    const textarea = screen.getByPlaceholderText('Type a message...');
    fireEvent.change(textarea, { target: { value: '  hello world  ' } });
    fireEvent.click(screen.getByLabelText('Send message'));
    expect(onSend).toHaveBeenCalledWith('hello world');
  });

  it('clears input after sending', () => {
    const onSend = vi.fn();
    wrap(<ChatInput onSend={onSend} />);
    const textarea = screen.getByPlaceholderText('Type a message...');
    fireEvent.change(textarea, { target: { value: 'hello' } });
    fireEvent.click(screen.getByLabelText('Send message'));
    expect((textarea as HTMLTextAreaElement).value).toBe('');
  });

  it('does not send when input is empty', () => {
    const onSend = vi.fn();
    wrap(<ChatInput onSend={onSend} />);
    fireEvent.click(screen.getByLabelText('Send message'));
    expect(onSend).not.toHaveBeenCalled();
  });

  it('does not send when disabled prop is true', () => {
    const onSend = vi.fn();
    wrap(<ChatInput onSend={onSend} disabled />);
    const textarea = screen.getByPlaceholderText('Type a message...');
    fireEvent.change(textarea, { target: { value: 'hello' } });
    fireEvent.click(screen.getByLabelText('Send message'));
    expect(onSend).not.toHaveBeenCalled();
  });

  it('Enter key triggers send, Shift+Enter does not', () => {
    const onSend = vi.fn();
    wrap(<ChatInput onSend={onSend} />);
    const textarea = screen.getByPlaceholderText('Type a message...');
    fireEvent.change(textarea, { target: { value: 'hello' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });
    expect(onSend).not.toHaveBeenCalled();
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });
    expect(onSend).toHaveBeenCalledWith('hello');
  });
});
