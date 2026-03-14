interface ParsedError {
  friendly: string;
  raw: string;
}

const ERROR_PATTERNS: { pattern: RegExp; message: string }[] = [
  {
    pattern: /dimension mismatch/i,
    message:
      "Your architecture has incompatible layer dimensions. Check that output sizes match input sizes between layers.",
  },
  {
    pattern: /out of memory|oom/i,
    message:
      "Not enough memory for this configuration. Try reducing batch size or simplifying the architecture.",
  },
  {
    pattern: /cuda error/i,
    message: "GPU error occurred. Try restarting training or switching to CPU.",
  },
  {
    pattern: /loss is nan|nan/i,
    message: "Training diverged (loss became NaN). Try a lower learning rate.",
  },
  {
    pattern: /invalid argument/i,
    message: "Invalid training parameter. Check your architecture configuration.",
  },
];

export function parseTrainingError(raw: string): ParsedError {
  for (const { pattern, message } of ERROR_PATTERNS) {
    if (pattern.test(raw)) {
      return { friendly: message, raw };
    }
  }
  return { friendly: raw, raw };
}
