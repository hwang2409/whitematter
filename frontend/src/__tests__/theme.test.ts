import { describe, it, expect } from 'vitest';
import { getTheme, themeTokens } from '../theme';

describe('theme color tokens', () => {
  it('dark theme uses violet as primary', () => {
    const theme = getTheme('dark');
    expect(theme.palette.primary.main).toBe('#8B5CF6');
    expect(theme.palette.primary.light).toBe('#A78BFA');
    expect(theme.palette.primary.dark).toBe('#7C3AED');
  });

  it('light theme uses darker violet for contrast', () => {
    const theme = getTheme('light');
    expect(theme.palette.primary.main).toBe('#7C3AED');
    expect(theme.palette.primary.light).toBe('#8B5CF6');
    expect(theme.palette.primary.dark).toBe('#6D28D9');
  });

  it('success stays green (not amber)', () => {
    const darkTheme = getTheme('dark');
    expect(darkTheme.palette.success.main).toBe('#22c55e');
  });

  it('themeTokens exports correct accent and completion values', () => {
    expect(themeTokens.accent).toBe('#8B5CF6');
    expect(themeTokens.accentLight).toBe('#A78BFA');
    expect(themeTokens.accentMuted).toBe('rgba(139, 92, 246, 0.15)');
    expect(themeTokens.completion).toBe('#F59E0B');
    expect(themeTokens.completionLight).toBe('#FBBF24');
    expect(themeTokens.completionMuted).toBe('rgba(245, 158, 11, 0.12)');
    expect(themeTokens.fontMono).toBe("'DM Mono', monospace");
  });

  it('dark card background is warm-tinted', () => {
    const theme = getTheme('dark');
    expect(themeTokens.card).toBe('#1C1A18');
  });
});
