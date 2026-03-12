import { describe, it, expect } from "vitest";
import { QUICK_START_ARCHITECTURE, QUICK_START_DATASET_HF_ID } from "@/lib/quickStart";

describe("QUICK_START_ARCHITECTURE", () => {
  it("has required Architecture fields", () => {
    expect(QUICK_START_ARCHITECTURE.name).toBe("MNIST Digit Classifier");
    expect(QUICK_START_ARCHITECTURE.data_type).toBe("image");
    expect(QUICK_START_ARCHITECTURE.input_shape).toEqual([1, 28, 28]);
    expect(QUICK_START_ARCHITECTURE.num_classes).toBe(10);
    expect(QUICK_START_ARCHITECTURE.layers.length).toBeGreaterThan(0);
    expect(QUICK_START_ARCHITECTURE.training.optimizer.type).toBe("adam");
    expect(QUICK_START_ARCHITECTURE.training.epochs).toBe(10);
    expect(QUICK_START_ARCHITECTURE.training.batch_size).toBe(32);
  });

  it("exports HuggingFace dataset ID", () => {
    expect(QUICK_START_DATASET_HF_ID).toBe("ylecun/mnist");
  });
});
