interface ParsedError {
  friendly: string;
  suggestion: string;
}

const ERROR_PATTERNS: [RegExp, string, string][] = [
  [/\[compile\]/i, "Code compilation failed.", "Try simplifying your architecture or removing unsupported layer types."],
  [/\[codegen\]/i, "Code generation failed.", "Check that your layer configuration is valid."],
  [/\[store\]/i, "Failed to save model artifacts.", "Try training again — this is usually a temporary issue."],
  [/dimension mismatch/i, "Layer dimensions are incompatible.", "Check that output dimensions of each layer match the input of the next."],
  [/out of memory|oom/i, "Ran out of memory during training.", "Try reducing batch size or simplifying the model."],
  [/cuda error/i, "GPU error occurred.", "Try switching to CPU training or reducing model size."],
  [/loss is nan|nan detected/i, "Training diverged (loss became NaN).", "Try lowering the learning rate or using gradient clipping."],
  [/invalid argument/i, "Invalid layer or parameter configuration.", "Check your architecture parameters."],
  [/cannot open/i, "Dataset file not found.", "Try re-uploading your dataset."],
  [/segmentation fault|segfault/i, "Training process crashed.", "Try reducing model size or batch size."],
  [/undefined reference/i, "Compilation error — unsupported operation.", "Some layer combinations may not be supported yet."],
  [/shape mismatch/i, "Tensor shape mismatch between layers.", "Verify that layer dimensions are consistent."],
];

export function parseTrainingError(rawMessage: string): ParsedError {
  for (const [pattern, friendly, suggestion] of ERROR_PATTERNS) {
    if (pattern.test(rawMessage)) {
      return { friendly, suggestion };
    }
  }
  return {
    friendly: rawMessage.length > 200 ? rawMessage.slice(0, 200) + "..." : rawMessage,
    suggestion: "Check the training configuration and try again.",
  };
}
