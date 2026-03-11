"use client";

import { useMemo } from "react";
import {
  ReactFlow,
  Node,
  Edge,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { useTheme } from "@mui/material/styles";
import type { Architecture, ArchitectureLayer } from "@/api";
import { themeTokens } from "@/theme";

const NODE_HEIGHT = 52;
const SPACING = 56;

function layerShortParams(layer: ArchitectureLayer): string {
  const { type, params } = layer;
  const p = params as Record<string, number>;
  switch (type) {
    case "conv2d":
      return `${p.in_channels ?? "?"}→${p.out_channels ?? "?"}, k=${p.kernel_size ?? "?"}, s=${p.stride ?? "?"}, p=${p.padding ?? "?"}`;
    case "linear":
      return `${p.in_features ?? "?"}→${p.out_features ?? "?"}`;
    case "lstm":
      return `hidden=${p.hidden_size ?? "?"}`;
    case "embedding":
      return `${p.num_embeddings ?? "?"}×${p.embedding_dim ?? "?"}`;
    case "dropout":
      return `p=${p.p ?? "?"}`;
    case "batchnorm2d":
      return `features=${p.num_features ?? "?"}`;
    case "maxpool2d":
    case "avgpool2d":
      return `k=${p.kernel_size ?? "?"}`;
    default:
      return Object.entries(p)
        .map(([k, v]) => `${k}=${v}`)
        .slice(0, 3)
        .join(", ") || "";
  }
}

type LayerNodeData = {
  layerType: string;
  paramsText: string;
};

function LayerNode({ data }: NodeProps<Node<LayerNodeData>>) {
  const theme = useTheme();
  return (
    <Box
      sx={{
        minWidth: 140,
        maxWidth: 200,
        py: 1,
        px: 1.25,
        borderRadius: 1,
        borderLeft: "3px solid",
        borderLeftColor: themeTokens.accent,
        bgcolor: theme.palette.background.paper,
        transition: "opacity 0.25s ease-out",
      }}
    >
      <Typography variant="body2" fontWeight={600} sx={{ display: "block", mb: 0.25 }}>
        {data.layerType}
      </Typography>
      <Typography
        variant="caption"
        sx={{
          color: "text.secondary",
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: "0.6875rem",
        }}
      >
        {data.paramsText}
      </Typography>
    </Box>
  );
}

const nodeTypes = { layer: LayerNode };

function architectureToFlow(architecture: Architecture | null): {
  nodes: Node<LayerNodeData>[];
  edges: Edge[];
} {
  if (!architecture?.layers?.length) {
    return { nodes: [], edges: [] };
  }
  const nodes: Node<LayerNodeData>[] = architecture.layers.map((layer, i) => ({
    id: `layer-${i}`,
    type: "layer",
    position: { x: 0, y: i * (NODE_HEIGHT + SPACING) },
    data: {
      layerType: layer.type,
      paramsText: layerShortParams(layer),
    },
  }));
  const edges: Edge[] = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    edges.push({
      id: `e-${i}-${i + 1}`,
      source: `layer-${i}`,
      target: `layer-${i + 1}`,
      style: { stroke: "rgba(126, 184, 255, 0.3)", strokeWidth: 1.5 },
      type: "smoothstep",
    });
  }
  return { nodes, edges };
}

interface Props {
  architecture: Architecture | null;
}

export default function ArchitectureGraph({ architecture }: Props) {
  const theme = useTheme();
  const { nodes: initialNodes, edges: initialEdges } = useMemo(
    () => architectureToFlow(architecture),
    [architecture]
  );
  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  if (!architecture?.layers?.length) return null;

  const graphKey = `${architecture.name}-${architecture.layers.length}-${architecture.layers.map((l) => l.type).join(",")}`;

  return (
    <Box
      sx={{
        height: 320,
        minHeight: 320,
        border: "1px solid",
        borderColor: "divider",
        borderRadius: 1,
        bgcolor: "background.default",
        overflow: "hidden",
        "& .react-flow__controls": {
          button: {
            bgcolor: "background.paper",
            color: "text.secondary",
            borderColor: "divider",
            "&:hover": { bgcolor: "action.hover", color: "text.primary" },
          },
        },
        "& .react-flow__background": {
          patternColor: theme.palette.divider,
        },
      }}
    >
      <ReactFlow
        key={graphKey}
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.3}
        maxZoom={1.5}
        nodesDraggable
        nodesConnectable={false}
        elementsSelectable={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background color={theme.palette.divider} gap={16} size={0.5} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </Box>
  );
}
