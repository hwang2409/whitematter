import { useState, useCallback } from "react";
import * as api from "@/api";

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export function useArchitectureDesign(setError: (msg: string) => void) {
  const [prompt, setPrompt] = useState("");
  const [architecture, setArchitecture] = useState<api.Architecture | null>(null);
  const [explanation, setExplanation] = useState("");
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [feedback, setFeedback] = useState("");
  const [generating, setGenerating] = useState(false);
  const [refining, setRefining] = useState(false);
  const [codePreview, setCodePreview] = useState<api.PreviewCodeResponse | null>(null);
  const [codePreviewLoading, setCodePreviewLoading] = useState(false);

  const handleGenerate = useCallback(
    async (selectedDatasetId: string) => {
      if (!selectedDatasetId || !prompt.trim()) {
        setError("Please select a dataset and describe what you want");
        return;
      }

      setGenerating(true);
      setError("");
      setArchitecture(null);
      setValidation(null);

      try {
        const result = await api.suggestArchitecture(selectedDatasetId, prompt.trim());
        setArchitecture(result.architecture);
        setExplanation(result.explanation);

        const val = await api.validateArchitecture(result.architecture);
        setValidation(val);
      } catch (e: any) {
        setError(e.message || "Failed to get suggestion");
      } finally {
        setGenerating(false);
      }
    },
    [prompt, setError],
  );

  const handleRefine = useCallback(async () => {
    if (!architecture || !feedback.trim()) {
      setError("Please provide feedback for refinement");
      return;
    }

    setRefining(true);
    setError("");

    try {
      const result = await api.refineArchitecture(architecture, feedback.trim());
      setArchitecture(result.architecture);
      setExplanation(result.explanation);
      setFeedback("");

      const val = await api.validateArchitecture(result.architecture);
      setValidation(val);
    } catch (e: any) {
      setError(e.message || "Failed to refine");
    } finally {
      setRefining(false);
    }
  }, [architecture, feedback, setError]);

  const handlePreviewCode = useCallback(
    async (selectedDatasetId: string) => {
      if (!selectedDatasetId || !architecture) return;
      setCodePreviewLoading(true);
      setCodePreview(null);
      setError("");
      try {
        const result = await api.previewGeneratedCode(selectedDatasetId, architecture);
        setCodePreview(result);
      } catch (e: any) {
        setError(e.message || "Failed to preview code");
      } finally {
        setCodePreviewLoading(false);
      }
    },
    [architecture, setError],
  );

  const handleLayerParamChange = useCallback(
    (layerIndex: number, paramKey: string, value: string | number) => {
      if (!architecture) return;
      const prev = architecture.layers[layerIndex].params[paramKey];
      const isNumericParam = typeof prev === "number";
      const paramValue =
        isNumericParam && typeof value === "string"
          ? value === ""
            ? 0
            : parseFloat(value)
          : value;
      const final =
        isNumericParam && typeof paramValue === "number" && Number.isNaN(paramValue)
          ? prev
          : paramValue;
      setArchitecture({
        ...architecture,
        layers: architecture.layers.map((layer, i) =>
          i === layerIndex
            ? { ...layer, params: { ...layer.params, [paramKey]: final } }
            : layer,
        ),
      });
    },
    [architecture],
  );

  return {
    prompt,
    setPrompt,
    architecture,
    setArchitecture,
    explanation,
    validation,
    feedback,
    setFeedback,
    generating,
    refining,
    codePreview,
    codePreviewLoading,
    handleGenerate,
    handleRefine,
    handlePreviewCode,
    handleLayerParamChange,
  };
}
