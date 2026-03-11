"use client";

import { useMemo } from "react";
import {
  ReactFlow,
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { Architecture, ArchitectureLayer } from "@/api";

const NODE_HEIGHT = 44;
const SPACING = 56;

function layerLabel(layer: ArchitectureLayer): string {
  const { type, params } = layer;
  const p = params as Record<string, number>;
  switch (type) {
    case "conv2d":
      return `Conv2d ${p.in_channels ?? "?"}→${p.out_channels ?? "?"}`;
    case "linear":
      return `Linear ${p.in_features ?? "?"}→${p.out_features ?? "?"}`;
    case "lstm":
      return `LSTM ${p.hidden_size ?? "?"}`;
    case "embedding":
      return `Embed ${p.num_embeddings ?? "?"}×${p.embedding_dim ?? "?"}`;
    case "dropout":
      return `Dropout ${p.p ?? "?"}`;
    case "batchnorm2d":
      return `BatchNorm ${p.num_features ?? "?"}`;
    case "maxpool2d":
    case "avgpool2d":
      return `${type} ${p.kernel_size ?? "?"}`;
    default:
      return type;
  }
}

function architectureToFlow(architecture: Architecture | null): {
  nodes: Node[];
  edges: Edge[];
} {
  if (!architecture?.layers?.length) {
    return { nodes: [], edges: [] };
  }
  const nodes: Node[] = architecture.layers.map((layer, i) => ({
    id: `layer-${i}`,
    type: "default",
    position: { x: 0, y: i * (NODE_HEIGHT + SPACING) },
    data: { label: layerLabel(layer) },
  }));
  const edges: Edge[] = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    edges.push({
      id: `e-${i}-${i + 1}`,
      source: `layer-${i}`,
      target: `layer-${i + 1}`,
    });
  }
  return { nodes, edges };
}

interface Props {
  architecture: Architecture | null;
}

export default function ArchitectureGraph({ architecture }: Props) {
  const { nodes: initialNodes, edges: initialEdges } = useMemo(
    () => architectureToFlow(architecture),
    [architecture]
  );
  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  if (!architecture?.layers?.length) return null;

  const graphKey = `${architecture.name}-${architecture.layers.length}-${architecture.layers.map((l) => l.type).join(",")}`;

  return (
    <div className="architecture-graph" style={{ height: 320, minHeight: 320 }}>
      <ReactFlow
        key={graphKey}
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.3}
        maxZoom={1.5}
        nodesDraggable
        nodesConnectable={false}
        elementsSelectable={false}
      >
        <Background />
        <Controls showInteractive={false} />
        <MiniMap nodeColor="#444" maskColor="rgba(0,0,0,0.6)" />
      </ReactFlow>
    </div>
  );
}
