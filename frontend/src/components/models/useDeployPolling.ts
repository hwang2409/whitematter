"use client";
import { useState, useEffect, useCallback } from "react";
import { useAuth } from "@/context/AuthContext";
import * as deployService from "@/services/deploy";
import { useToast } from "../Toast";

const DEPLOY_POLL_INTERVAL_MS = 3000;

interface UseDeployPollingOptions {
  modelId: string | null;
  deployModalOpen: boolean;
}

export default function useDeployPolling({ modelId, deployModalOpen }: UseDeployPollingOptions) {
  const { token } = useAuth();
  const toast = useToast();
  const [deployRegion, setDeployRegion] = useState("us-east-1");
  const [deploying, setDeploying] = useState(false);
  const [deploymentPollId, setDeploymentPollId] = useState<string | null>(null);
  const [deployments, setDeployments] = useState<deployService.Deployment[]>([]);
  const [deployError, setDeployError] = useState("");
  const [copied, setCopied] = useState(false);

  const loadDeployments = useCallback(async () => {
    if (!token || !modelId) return;
    try {
      const list = await deployService.listDeployments(token, modelId);
      setDeployments(list);
    } catch {
      setDeployments([]);
    }
  }, [token, modelId]);

  useEffect(() => {
    if (deployModalOpen && modelId && token) {
      loadDeployments();
      setDeployError("");
    }
  }, [deployModalOpen, modelId, token, loadDeployments]);

  useEffect(() => {
    if (!deploymentPollId || !token) return;
    const t = setInterval(async () => {
      try {
        const d = await deployService.getDeployment(token, deploymentPollId);
        setDeployments((prev) => {
          const idx = prev.findIndex((x) => x.id === d.id);
          return idx >= 0 ? [...prev.slice(0, idx), d, ...prev.slice(idx + 1)] : [d, ...prev];
        });
        if (d.status === "live" || d.status === "failed") {
          setDeploymentPollId(null);
          setDeploying(false);
          if (d.status === "live") toast.success("Deployment live! Your API is ready.");
          if (d.status === "failed") toast.error(d.error_message || "Deployment failed.");
        }
      } catch {
        setDeploymentPollId(null);
        setDeploying(false);
      }
    }, DEPLOY_POLL_INTERVAL_MS);
    return () => clearInterval(t);
  }, [deploymentPollId, token, toast]);

  async function handleDeployStart() {
    if (!token || !modelId) return;
    setDeploying(true);
    setDeployError("");
    try {
      const res = await deployService.createDeployment(token, {
        model_id: modelId,
        region: deployRegion,
      });
      setDeploymentPollId(res.deployment_id);
      setDeployments((prev) => [
        {
          id: res.deployment_id,
          model_id: modelId,
          target_type: "ec2",
          status: res.status,
          instance_id: null,
          endpoint_url: null,
          region: deployRegion,
          error_message: null,
          created_at: null,
          updated_at: null,
        },
        ...prev,
      ]);
    } catch (e: unknown) {
      setDeploying(false);
      setDeployError(e instanceof Error ? e.message : "Failed to start deployment");
    }
  }

  async function handleDeployTerminate(deploymentId: string) {
    if (!token) return;
    try {
      await deployService.terminateDeployment(token, deploymentId);
      await loadDeployments();
      toast.success("Deployment terminated.");
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Failed to terminate");
    }
  }

  async function copyDeployEndpoint(url: string) {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast.success("Endpoint URL copied.");
    } catch {
      toast.error("Failed to copy");
    }
  }

  return {
    token,
    toast,
    deployRegion,
    setDeployRegion,
    deploying,
    deploymentPollId,
    deployments,
    deployError,
    copied,
    setCopied,
    handleDeployStart,
    handleDeployTerminate,
    copyDeployEndpoint,
  };
}
