import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import ChatMessageBubble from '../components/ChatMessage';

const theme = getTheme('dark');

const wrap = (ui: React.ReactElement) =>
  render(<ThemeProvider theme={theme}>{ui}</ThemeProvider>);

describe('ChatMessageBubble', () => {
  it('renders AI avatar with logo', () => {
    wrap(
      <ChatMessageBubble
        message={{ id: 'test-1', role: 'assistant', content: 'Hello', type: 'text', createdAt: new Date().toISOString() }}
      />
    );
    const logo = screen.getByAltText('WhiteMatter');
    expect(logo).toBeTruthy();
    expect(logo.getAttribute('src')).toBe('/logo.png');
  });

  it('does not render avatar for user messages', () => {
    wrap(
      <ChatMessageBubble
        message={{ id: 'test-2', role: 'user', content: 'Hi', type: 'text', createdAt: new Date().toISOString() }}
      />
    );
    expect(screen.queryByAltText('WhiteMatter')).toBeNull();
  });
});
