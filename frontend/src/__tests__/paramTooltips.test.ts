import { describe, it, expect } from "vitest";
import { PARAM_TOOLTIPS } from "@/lib/paramTooltips";

describe("PARAM_TOOLTIPS", () => {
  it("has entries for all training params", () => {
    const keys = [
      "optimizer",
      "scheduler",
      "batch_size",
      "learning_rate",
      "augmentations",
    ];
    for (const key of keys) {
      expect(PARAM_TOOLTIPS[key]).toBeDefined();
      expect(PARAM_TOOLTIPS[key].description).toBeTruthy();
      expect(PARAM_TOOLTIPS[key].range).toBeTruthy();
    }
  });
});
