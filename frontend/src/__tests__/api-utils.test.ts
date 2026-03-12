import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  ApiError,
  getErrorMessage,
  isRetryableError,
  getDatasets,
  getModels,
  startTraining,
  cancelTraining,
  predict,
} from "@/api";

// Mock safeParseJson
vi.mock("@/lib/safeJson", () => ({
  safeParseJson: async (res: Response) => {
    const text = await res.text();
    const trimmed = text.trim();
    if (!trimmed) return {};
    return JSON.parse(text);
  },
}));

const mockFetch = vi.fn();
global.fetch = mockFetch;

beforeEach(() => {
  mockFetch.mockReset();
});

// ---------------------------------------------------------------------------
// ApiError class
// ---------------------------------------------------------------------------
describe("ApiError", () => {
  it("creates error with type, message, and statusCode", () => {
    const err = new ApiError("Server error: 500", "server", 500);
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.name).toBe("ApiError");
    expect(err.message).toBe("Server error: 500");
    expect(err.type).toBe("server");
    expect(err.statusCode).toBe(500);
  });

  it("creates error without statusCode", () => {
    const err = new ApiError("timeout!", "timeout");
    expect(err.type).toBe("timeout");
    expect(err.statusCode).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// getErrorMessage
// ---------------------------------------------------------------------------
describe("getErrorMessage", () => {
  it("returns message from ApiError", () => {
    const err = new ApiError("Server down", "server", 500);
    expect(getErrorMessage(err)).toBe("Server down");
  });

  it("returns message from plain Error", () => {
    expect(getErrorMessage(new Error("oops"))).toBe("oops");
  });

  it("returns fallback for non-Error values", () => {
    expect(getErrorMessage("string")).toBe("An unexpected error occurred");
    expect(getErrorMessage(42)).toBe("An unexpected error occurred");
    expect(getErrorMessage(null)).toBe("An unexpected error occurred");
  });
});

// ---------------------------------------------------------------------------
// isRetryableError
// ---------------------------------------------------------------------------
describe("isRetryableError", () => {
  it("returns true for timeout errors", () => {
    expect(isRetryableError(new ApiError("timed out", "timeout"))).toBe(true);
  });

  it("returns true for network errors", () => {
    expect(isRetryableError(new ApiError("network", "network"))).toBe(true);
  });

  it("returns false for server errors", () => {
    expect(isRetryableError(new ApiError("500", "server", 500))).toBe(false);
  });

  it("returns false for unknown errors", () => {
    expect(isRetryableError(new ApiError("???", "unknown"))).toBe(false);
  });

  it("returns false for non-ApiError", () => {
    expect(isRetryableError(new Error("regular error"))).toBe(false);
    expect(isRetryableError("string")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// API functions using fetchWithTimeout
// ---------------------------------------------------------------------------
describe("getDatasets", () => {
  it("fetches datasets and returns the datasets array", async () => {
    const datasets = [
      { id: "mnist", name: "MNIST", description: "Handwritten digits", available: true, input_shape: [1, 28, 28], num_classes: 10, classes: [] },
    ];
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ datasets }), { status: 200 })
    );

    const result = await getDatasets();
    expect(result).toEqual(datasets);
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/config/datasets"),
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    );
  });

  it("throws ApiError on server error", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response("Internal Server Error", { status: 500, statusText: "Internal Server Error" })
    );

    await expect(getDatasets()).rejects.toThrow(ApiError);
    try {
      await getDatasets();
    } catch (err) {
      // Already threw above, just verifying the first assertion
    }
  });
});

describe("getModels", () => {
  it("fetches models and returns the models array", async () => {
    const models = [
      {
        id: "m1",
        name: "Test Model",
        dataset: "mnist",
        architecture: "cnn",
        created_at: "2025-01-01",
        epochs_trained: 10,
        best_accuracy: 0.95,
        status: "completed" as const,
        training_history: [],
      },
    ];
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify({ models }), { status: 200 })
    );

    const result = await getModels();
    expect(result).toEqual(models);
  });
});

describe("startTraining", () => {
  it("sends POST with config and returns job info", async () => {
    const config = {
      dataset: "mnist",
      epochs: 10,
      batch_size: 32,
      optimizer: { type: "adam", params: { lr: 0.001 } },
      scheduler: { type: "step", params: { step_size: 5 } },
      augmentations: [],
    };
    const response = { job_id: "job-1", model_id: "model-1" };

    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify(response), { status: 200 })
    );

    const result = await startTraining(config);
    expect(result).toEqual(response);
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/train"),
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
        signal: expect.any(AbortSignal),
      })
    );
  });
});

describe("cancelTraining", () => {
  it("sends DELETE to cancel a training job", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response("", { status: 200 })
    );

    await cancelTraining("job-1");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/train/job-1"),
      expect.objectContaining({
        method: "DELETE",
        signal: expect.any(AbortSignal),
      })
    );
  });
});

describe("predict", () => {
  it("sends POST with FormData and returns prediction result", async () => {
    const prediction = {
      model_id: "m1",
      model_name: "Test",
      predicted_class: 7,
      class_name: "7",
      confidence: 0.98,
      probabilities: { "7": 0.98, "1": 0.01 },
    };
    const file = new File(["data"], "test.png", { type: "image/png" });

    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify(prediction), { status: 200 })
    );

    const result = await predict("m1", file);
    expect(result).toEqual(prediction);
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/predict?model_id=m1"),
      expect.objectContaining({
        method: "POST",
        signal: expect.any(AbortSignal),
      })
    );
  });
});

// ---------------------------------------------------------------------------
// Timeout and network error handling
// ---------------------------------------------------------------------------
describe("fetchWithTimeout error handling", () => {
  it("throws timeout ApiError when fetch is aborted", async () => {
    mockFetch.mockImplementationOnce(() => {
      const error = new Error("The operation was aborted");
      error.name = "AbortError";
      return Promise.reject(error);
    });

    try {
      await getDatasets();
      expect.fail("Should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      expect((err as ApiError).type).toBe("timeout");
      expect((err as ApiError).message).toContain("timed out");
    }
  });

  it("throws network ApiError on 'Failed to fetch'", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Failed to fetch"));

    try {
      await getDatasets();
      expect.fail("Should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      expect((err as ApiError).type).toBe("network");
    }
  });

  it("throws network ApiError on NetworkError", async () => {
    mockFetch.mockRejectedValueOnce(new Error("NetworkError when attempting to fetch resource"));

    try {
      await getDatasets();
      expect.fail("Should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      expect((err as ApiError).type).toBe("network");
    }
  });

  it("throws server ApiError on non-ok response", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response("Not Found", { status: 404, statusText: "Not Found" })
    );

    try {
      await getDatasets();
      expect.fail("Should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      expect((err as ApiError).type).toBe("server");
      expect((err as ApiError).statusCode).toBe(404);
    }
  });

  it("throws unknown ApiError for other errors", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Something unexpected"));

    try {
      await getDatasets();
      expect.fail("Should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      expect((err as ApiError).type).toBe("unknown");
    }
  });
});
