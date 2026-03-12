import { describe, it, expect } from "vitest";
import { parseTrainingError } from "@/lib/trainingErrors";

describe("parseTrainingError", () => {
  it("parses dimension mismatch", () => {
    const result = parseTrainingError("Error: dimension mismatch at layer 3");
    expect(result.friendly).toContain("incompatible layer dimensions");
    expect(result.raw).toContain("dimension mismatch");
  });

  it("parses out of memory", () => {
    const result = parseTrainingError("CUDA out of memory");
    expect(result.friendly).toContain("memory");
  });

  it("parses OOM", () => {
    const result = parseTrainingError("OOM when allocating tensor");
    expect(result.friendly).toContain("memory");
  });

  it("parses CUDA error", () => {
    const result = parseTrainingError("CUDA error: device-side assert");
    expect(result.friendly).toContain("GPU error");
  });

  it("parses NaN loss", () => {
    const result = parseTrainingError("loss is nan at epoch 5");
    expect(result.friendly).toContain("diverged");
  });

  it("parses invalid argument", () => {
    const result = parseTrainingError("invalid argument for kernel_size");
    expect(result.friendly).toContain("Invalid training parameter");
  });

  it("falls back to raw message for unknown errors", () => {
    const result = parseTrainingError("something unexpected happened");
    expect(result.friendly).toBe("something unexpected happened");
    expect(result.raw).toBe("something unexpected happened");
  });

  it("is case-insensitive", () => {
    const result = parseTrainingError("DIMENSION MISMATCH error");
    expect(result.friendly).toContain("incompatible layer dimensions");
  });
});
