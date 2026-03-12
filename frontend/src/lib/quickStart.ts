import type { Architecture } from "@/api";

export const QUICK_START_DATASET_HF_ID = "ylecun/mnist";
export const QUICK_START_DATASET_NAME = "MNIST Sample";

export const QUICK_START_ARCHITECTURE: Architecture = {
  name: "MNIST Digit Classifier",
  description: "Simple CNN for handwritten digit recognition",
  data_type: "image",
  input_shape: [1, 28, 28],
  num_classes: 10,
  layers: [
    { type: "conv2d", params: { in_channels: 1, out_channels: 16, kernel_size: 3 } },
    { type: "relu", params: {} },
    { type: "maxpool2d", params: { kernel_size: 2 } },
    { type: "conv2d", params: { in_channels: 16, out_channels: 32, kernel_size: 3 } },
    { type: "relu", params: {} },
    { type: "maxpool2d", params: { kernel_size: 2 } },
    { type: "flatten", params: {} },
    { type: "linear", params: { in_features: 800, out_features: 10 } },
  ],
  training: {
    optimizer: { type: "adam", params: { lr: 0.001 } },
    scheduler: { type: "step_lr", params: { step_size: 5, gamma: 0.5 } },
    epochs: 10,
    batch_size: 32,
  },
};
