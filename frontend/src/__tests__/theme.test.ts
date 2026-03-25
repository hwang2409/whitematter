import { describe, it, expect } from 'vitest';
import { getTheme, themeTokens } from '../theme';

describe('theme color tokens', () => {
  it('dark theme uses orange as primary', () => {
    const theme = getTheme('dark');
    expect(theme.palette.primary.main).toBe('#F97316');
    expect(theme.palette.primary.light).toBe('#F97316');
    expect(theme.palette.primary.dark).toBe('#EA580C');
  });

  it('light theme uses darker orange for contrast', () => {
    const theme = getTheme('light');
    expect(theme.palette.primary.main).toBe('#EA580C');
    expect(theme.palette.primary.light).toBe('#F97316');
    expect(theme.palette.primary.dark).toBe('#EA580C');
  });

  it('success stays green', () => {
    const darkTheme = getTheme('dark');
    expect(darkTheme.palette.success.main).toBe('#22C55E');
  });

  it('themeTokens exports correct accent and completion values', () => {
    expect(themeTokens.accent).toBe('#F97316');
    expect(themeTokens.accentHover).toBe('#EA580C');
    expect(themeTokens.accentMuted).toBe('rgba(249, 115, 22, 0.35)');
    expect(themeTokens.completion).toBe('#22C55E');
    expect(themeTokens.completionMuted).toBe('rgba(34, 197, 94, 0.12)');
    expect(themeTokens.fontMono).toBe("'JetBrains Mono', monospace");
  });

  it('dark card background matches surface', () => {
    expect(themeTokens.card).toBe('#18181B');
  });
});
