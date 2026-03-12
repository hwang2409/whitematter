export interface ParamTooltipData {
  description: string;
  range: string;
}

export const PARAM_TOOLTIPS: Record<string, ParamTooltipData> = {
  optimizer: {
    description:
      "Controls how model weights are updated during training. Adam is a good default for most tasks.",
    range: "Adam, SGD, AdamW",
  },
  scheduler: {
    description:
      "Adjusts the learning rate during training to improve convergence.",
    range: "StepLR, CosineAnnealing, None",
  },
  batch_size: {
    description:
      "Number of samples processed before updating weights. Larger batches train faster but use more memory.",
    range: "16\u2013128, default 32",
  },
  learning_rate: {
    description:
      "Step size for weight updates. Too high causes instability, too low causes slow training.",
    range: "0.0001\u20130.01, default 0.001",
  },
  augmentations: {
    description:
      "Random transforms applied to training data to prevent overfitting and improve generalization.",
    range: "RandomFlip, RandomRotation for images",
  },
};
