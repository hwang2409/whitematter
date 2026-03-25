import { describe, it, expect } from "vitest";
import { parseTrainingError } from "@/lib/trainingErrors";

describe("parseTrainingError", () => {
  it("parses dimension mismatch", () => {
    const result = parseTrainingError("Error: dimension mismatch at layer 3");
    expect(result.friendly).toBe("Layer dimensions are incompatible.");
    expect(result.suggestion).toContain("output dimensions");
  });

  it("parses out of memory", () => {
    const result = parseTrainingError("CUDA out of memory");
    expect(result.friendly).toContain("memory");
    expect(result.suggestion).toContain("batch size");
  });

  it("parses OOM", () => {
    const result = parseTrainingError("OOM when allocating tensor");
    expect(result.friendly).toContain("memory");
    expect(result.suggestion).toContain("batch size");
  });

  it("parses CUDA error", () => {
    const result = parseTrainingError("CUDA error: device-side assert");
    expect(result.friendly).toBe("GPU error occurred.");
    expect(result.suggestion).toContain("CPU");
  });

  it("parses NaN loss", () => {
    const result = parseTrainingError("loss is nan at epoch 5");
    expect(result.friendly).toContain("diverged");
    expect(result.suggestion).toContain("learning rate");
  });

  it("parses invalid argument", () => {
    const result = parseTrainingError("invalid argument for kernel_size");
    expect(result.friendly).toBe("Invalid layer or parameter configuration.");
    expect(result.suggestion).toContain("architecture parameters");
  });

  it("parses [compile] prefix", () => {
    const result = parseTrainingError("[compile] error: cannot find -lmylib");
    expect(result.friendly).toBe("Code compilation failed.");
    expect(result.suggestion).toContain("simplifying");
  });

  it("parses [codegen] prefix", () => {
    const result = parseTrainingError("[codegen] failed to generate forward pass");
    expect(result.friendly).toBe("Code generation failed.");
    expect(result.suggestion).toContain("layer configuration");
  });

  it("parses cannot open", () => {
    const result = parseTrainingError("cannot open file /data/train.csv");
    expect(result.friendly).toBe("Dataset file not found.");
    expect(result.suggestion).toContain("re-uploading");
  });

  it("parses segmentation fault", () => {
    const result = parseTrainingError("segmentation fault (core dumped)");
    expect(result.friendly).toBe("Training process crashed.");
    expect(result.suggestion).toContain("reducing model size");
  });

  it("parses segfault shorthand", () => {
    const result = parseTrainingError("Process exited with segfault");
    expect(result.friendly).toBe("Training process crashed.");
    expect(result.suggestion).toContain("batch size");
  });

  it("parses undefined reference", () => {
    const result = parseTrainingError("undefined reference to `customOp`");
    expect(result.friendly).toBe("Compilation error — unsupported operation.");
    expect(result.suggestion).toContain("not be supported");
  });

  it("parses shape mismatch", () => {
    const result = parseTrainingError("RuntimeError: shape mismatch at index 0");
    expect(result.friendly).toBe("Tensor shape mismatch between layers.");
    expect(result.suggestion).toContain("layer dimensions");
  });

  it("falls back to raw message for unknown errors", () => {
    const result = parseTrainingError("something unexpected happened");
    expect(result.friendly).toBe("something unexpected happened");
    expect(result.suggestion).toBe("Check the training configuration and try again.");
  });

  it("truncates long unknown messages to 200 characters", () => {
    const longMessage = "A".repeat(250);
    const result = parseTrainingError(longMessage);
    expect(result.friendly).toBe("A".repeat(200) + "...");
    expect(result.friendly.length).toBe(203);
    expect(result.suggestion).toBe("Check the training configuration and try again.");
  });

  it("is case-insensitive", () => {
    const result = parseTrainingError("DIMENSION MISMATCH error");
    expect(result.friendly).toBe("Layer dimensions are incompatible.");
  });
});
