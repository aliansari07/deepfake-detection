"use client";

import React, { useCallback, useEffect, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  Handle,
  Position,
  MarkerType,
  useNodesState,
  useEdgesState,
} from "reactflow";
import type { Edge, Node, NodeProps, NodeTypes } from "reactflow";
import "reactflow/dist/style.css";
import stats from "@/lib/modelStats.json";

// Site palette. These mirror the custom properties in app/globals.css; they are
// repeated as literals because React Flow paints the SVG arrow markers inside a
// <defs> block, where a var() reference is not reliably resolved. Tailwind
// classes below use the var() form directly. Keep the two in step.
const ACCENT = "#35e0c7"; // --real, the site accent (also the focus-visible ring)
const EDGE_IDLE = "rgba(255, 255, 255, 0.16)"; // --line-strong
const DOT_GRID = "rgba(255, 255, 255, 0.09)"; // --line

type DiagramNodeData = {
  label: string;
  description?: string;
  stepNumber?: number;
  isActive?: boolean;
};

type GroupNodeData = {
  label: string;
};

// Custom Child Node (Fixed width 220px) — relative flex container pins the
// handles to the exact left/right border midpoints, so dots never drift over
// the label text.
function DiagramNode({ data, selected }: NodeProps<DiagramNodeData>) {
  return (
    <div
      className={`relative flex flex-col w-[260px] overflow-visible pl-6 pr-5 py-4 rounded-xl border transition-all duration-300 text-left shadow-lg ${
        data.isActive
          ? "bg-[var(--real-dim)] border-[var(--real)] text-[var(--text-bright)] shadow-[0_0_28px_rgba(53,224,199,0.18)] scale-105"
          : selected
            ? "bg-[var(--surface-2)] border-[var(--real)]/80 text-[var(--text-bright)]"
            : "bg-[var(--bg-raised)]/90 border-[var(--line-strong)] text-[var(--text)] hover:border-[var(--real)]/50"
      }`}
    >
      <Handle
        type="target"
        position={Position.Left}
        id="left"
        className="!absolute !left-0 !top-1/2 !-translate-x-full !-translate-y-1/2 !w-2 !h-2 !bg-[var(--real)] !border !border-[var(--bg)] !rounded-full"
      />

      <div className="text-[12px] font-semibold uppercase tracking-wider text-[var(--real)]/90 mb-1">
        {data.stepNumber ? `Step ${data.stepNumber}` : "Module"}
      </div>
      <div className="font-bold text-[17px] leading-tight mb-1.5 text-[var(--text-bright)]">
        {data.label}
      </div>
      {data.description && (
        <div className="text-[13px] text-[var(--text-dim)] font-mono leading-snug">
          {data.description}
        </div>
      )}

      <Handle
        type="source"
        position={Position.Right}
        id="right"
        className="!absolute !right-0 !top-1/2 !translate-x-full !-translate-y-1/2 !w-2 !h-2 !bg-[var(--real)] !border !border-[var(--bg)] !rounded-full"
      />
    </div>
  );
}

// Parent Group Node (Large container with visible header tag)
function GroupNode({ data }: NodeProps<GroupNodeData>) {
  return (
    <div className="w-full h-full rounded-2xl border border-[var(--real)]/25 bg-[var(--real)]/[0.02] p-4 text-[var(--real)]/80 text-[13px] font-mono uppercase tracking-widest backdrop-blur-xs select-none">
      <span className="bg-[var(--bg)]/90 px-3 py-1 rounded-md border border-[var(--real)]/20 font-semibold shadow-sm inline-block">
        {data.label}
      </span>
    </div>
  );
}

const nodeTypes: NodeTypes = {
  customNode: DiagramNode,
  groupNode: GroupNode,
};

// --- ARCHITECTURE GRAPH ---
// The graph is derived from `stats.architecture` — the arch block of the checkpoint the backend
// actually serves — rather than hand-drawn. The full model fuses a frequency stream and reasons
// across frames with a Transformer; the promoted demo is spatial-only. Drawing the gate or the
// Transformer unconditionally showed modules the site does not run, which is the one thing a
// diagram of "what's under the verdict" must not do.

const ARCH = stats.architecture;

type NodeSpec = {
  id: string;
  label: string;
  description: string;
  group?: string; // omitted = standalone root node
  pos: [number, number]; // absolute when standalone, relative to the group otherwise
};

// --- CALCULATED POSITIONS & SIZES ---
// Child Node Width = 260px, Approx Height = 110px
// Parent Padding: Top = 60px (to clear header), Bottom = 28px, Left/Right = 28px
// Horizontal Gap between children = 60px | Vertical Gap = 32px

const PREP: NodeSpec[] = [
  // The backend accepts exactly these three (website/backend/main.py). AVI is rejected, so
  // advertising it here was a claim the API contradicts.
  { id: "1", label: "Video Input", description: "MP4, MOV or WEBM upload", pos: [20, 65] },
  {
    id: "2",
    label: "Frame Sampling",
    description: `N=${ARCH.frames_per_clip} uniformly sampled frames`,
    group: "g-prep",
    pos: [28, 60],
  },
  {
    id: "3",
    label: "Face Detection & Crop",
    description: "Bounding box -> 224x224 crop",
    group: "g-prep",
    pos: [348, 60],
  },
];

const aug = (pos: [number, number]): NodeSpec => ({
  id: "4",
  label: "Augmentation",
  description: "Training only: JPEG / blur / noise",
  group: "g-feat",
  pos,
});

const SPATIAL: NodeSpec = {
  id: "5",
  label: "CNN Backbone (Spatial)",
  description: "EfficientNet-B0 embeddings",
  group: "g-feat",
  pos: [348, 60],
};

// An R5 checkpoint feeds this branch a spatial residual and takes no Fourier transform, so both
// the label and the description come from the arch block rather than being written in.
const FREQ_IS_RESIDUAL =
  (ARCH as { frequency_input?: string }).frequency_input === "residual";

const FREQ: NodeSpec = {
  id: "6",
  label: FREQ_IS_RESIDUAL ? "High-pass Residual" : "Frequency Transform",
  description: FREQ_IS_RESIDUAL
    ? "Frame minus its Gaussian blur"
    : "Log-FFT magnitude spectrum",
  group: "g-feat",
  pos: [348, 202],
};

const GATE: NodeSpec = {
  id: "7",
  label: "Gated Fusion",
  description: "Sigmoid gate blends both streams",
  group: "g-fusion",
  pos: [28, 60],
};

// Slot 8 is temporal aggregation. Which module fills it is the whole point of the two layouts:
// the Transformer when the served checkpoint has one, masked mean pooling when it does not.
const TRANSFORMER: NodeSpec = {
  id: "8",
  label: "Transformer Encoder",
  description: "2 layers, 8 heads attention",
  group: "g-fusion",
  pos: [348, 60],
};

const POOL: NodeSpec = {
  id: "8",
  label: "Frame Pooling",
  description: "Masked mean over the sampled frames",
  group: "g-fusion",
  pos: [348, 60],
};

const head = (pos: [number, number]): NodeSpec => ({
  id: "9",
  label: "Classification Head",
  description: "MLP (512 -> 256 -> 1)",
  group: "g-fusion",
  pos,
});

const output = (pos: [number, number]): NodeSpec => ({
  id: "10",
  label: "Output Prediction",
  description: "REAL / FAKE probability",
  group: "g-fusion",
  pos,
});

const ARROW_ACTIVE = {
  type: MarkerType.ArrowClosed, color: ACCENT, width: 13, height: 13,
};
const ARROW_IDLE = {
  type: MarkerType.ArrowClosed, color: EDGE_IDLE, width: 13, height: 13,
};

function buildGraph() {
  const freq = ARCH.frequency_branch;
  const transformer = ARCH.transformer;

  // The frequency branch needs a second row in the feature group; without it the two
  // surviving children (augmentation, backbone) sit side by side.
  const featureNodes = freq ? [aug([64, 115]), SPATIAL, FREQ] : [aug([28, 60]), SPATIAL];

  const temporal = transformer ? TRANSFORMER : POOL;
  // Four children when there is a gate to feed, three when the backbone feeds the head directly.
  const fusionNodes = freq
    ? [GATE, temporal, head([28, 202]), output([348, 202])]
    : [temporal, head([348, 60]), output([28, 202])];

  const ordered = [...PREP, ...featureNodes, ...fusionNodes];
  // Step numbers follow the order above, so they stay contiguous when modules are removed.
  const nodes: Node[] = ordered.map((spec, i) => ({
    id: spec.id,
    type: "customNode",
    position: { x: spec.pos[0], y: spec.pos[1] },
    ...(spec.group ? { parentId: spec.group, extent: "parent" as const } : {}),
    data: { label: spec.label, description: spec.description, stepNumber: i + 1 },
    zIndex: 10,
  }));

  const groups: Node[] = [
    {
      id: "g-prep",
      type: "groupNode",
      data: { label: "Preprocessing" },
      position: { x: 340, y: 20 },
      style: { width: 636, height: 198 },
      zIndex: -1,
    },
    {
      id: "g-feat",
      type: "groupNode",
      data: { label: "Feature Extraction" },
      position: { x: 20, y: 245 },
      style: { width: 636, height: freq ? 340 : 198 },
      zIndex: -1,
    },
    {
      id: "g-fusion",
      type: "groupNode",
      data: { label: freq ? "Fusion & Temporal Processing" : "Aggregation & Output" },
      position: { x: 716, y: 245 },
      style: { width: 636, height: 340 },
      zIndex: -1,
    },
  ];

  const links: [string, string][] = [
    ["1", "2"],
    ["2", "3"],
    ["3", "4"],
    ["4", "5"],
    // With no frequency branch the backbone feeds the temporal slot directly.
    ...(freq
      ? ([["4", "6"], ["5", "7"], ["6", "7"], ["7", "8"]] as [string, string][])
      : ([["5", "8"]] as [string, string][])),
    ["8", "9"],
    ["9", "10"],
  ];

  const edges: Edge[] = links.map(([source, target]) => ({
    id: `e${source}-${target}`,
    source,
    target,
    type: "smoothstep",
    animated: true,
    markerEnd: ARROW_IDLE,
    pathOptions: { borderRadius: 12 },
    style: { stroke: EDGE_IDLE },
  }));

  const ids = ordered.map((n) => n.id);
  // The pipeline as STAGES, not as one node per step. Where there is a frequency branch,
  // steps 5 and 6 are its two halves: the spatial CNN and the frequency transform consume the
  // same augmented frames in parallel, so lighting them one after the other would misrepresent
  // the architecture.
  const stages: string[][] = freq
    ? [
        ...ids.slice(0, 4).map((id) => [id]),
        [ids[4], ids[5]],
        ...ids.slice(6).map((id) => [id]),
      ]
    : ids.map((id) => [id]);

  const archLabel = freq
    ? transformer
      ? "spatial + frequency streams, fused, transformer over frames"
      : "spatial + frequency streams, fused, frames pooled"
    : transformer
      ? "spatial stream, transformer over frames"
      : "spatial stream, frames pooled";

  return {
    nodes: [...groups, ...nodes],
    edges,
    stages,
    archLabel,
  };
}

const graph = buildGraph();

export default function ArchitectureDiagram() {
  const [nodes, setNodes, onNodesChange] = useNodesState(graph.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(graph.edges);
  const [activeStage, setActiveStage] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  const stageOf = useCallback(
    (nodeId: string) => graph.stages.findIndex((ids) => ids.includes(nodeId)),
    []
  );

  useEffect(() => {
    if (!isPlaying) return;
    const timer: ReturnType<typeof setInterval> = setInterval(() => {
      setActiveStage((prev) => (prev + 1) % graph.stages.length);
    }, 1500);
    return () => clearInterval(timer);
  }, [isPlaying]);

  useEffect(() => {
    const current = graph.stages[activeStage] ?? [];

    setNodes((nds: Node[]) =>
      nds.map((node) => {
        if (node.type === "groupNode") return node;
        return {
          ...node,
          data: {
            ...node.data,
            isActive: current.includes(node.id),
          },
        };
      })
    );

    setEdges((eds: Edge[]) =>
      eds.map((edge) => {
        // An edge is on the current path when it feeds a node that is lighting
        // up now. Where there is a frequency branch, both fan-out edges
        // (4->5 and 4->6) light together, and both merge edges (5->7 and 6->7)
        // light together on the next stage.
        const isCurrentPath =
          current.includes(edge.target) &&
          stageOf(edge.source) === activeStage - 1;
        return {
          ...edge,
          animated: isCurrentPath || isPlaying,
          markerEnd: isCurrentPath ? ARROW_ACTIVE : ARROW_IDLE,
          style: {
            stroke: isCurrentPath ? ACCENT : EDGE_IDLE,
            strokeWidth: isCurrentPath ? 2.5 : 1.5,
          },
        };
      })
    );
  }, [activeStage, isPlaying, setNodes, setEdges, stageOf]);

  return (
    <div className="w-full h-[520px] bg-[var(--bg)] flex flex-col font-sans border border-[var(--line)] rounded-xl overflow-hidden shadow-2xl">
      <div className="p-4 bg-[var(--bg-raised)]/80 border-b border-[var(--line)] flex items-center justify-between gap-3 flex-wrap">
        <div>
          <h2 className="text-[var(--text-bright)] font-bold text-base flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-[var(--real)] inline-block animate-pulse"></span>
            DeepfakeDetector v7.0 — {stats.model.run}
          </h2>
          <p className="text-xs text-[var(--text-dim)]">
            Serving {stats.model.alias} — {graph.archLabel}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setIsPlaying((v) => !v)}
            className={`px-4 py-1.5 rounded-md font-medium text-xs transition-colors cursor-pointer ${
              isPlaying
                ? "bg-[var(--real-dim)] text-[var(--real)] border border-[var(--real)]/40 hover:bg-[var(--real)]/20"
                : "bg-[var(--surface-2)] text-[var(--text)] border border-[var(--line-strong)] hover:bg-[var(--bg-raised)]"
            }`}
          >
            {isPlaying ? "Pause Stream" : "Simulate Flow"}
          </button>

          <div className="flex items-center bg-[var(--bg)] px-2.5 py-1 rounded border border-[var(--line)] gap-1 text-xs text-[var(--text-dim)]">
            <span>Stage:</span>
            <span className="font-mono text-[var(--real)] font-bold">
              {activeStage + 1} / {graph.stages.length}
            </span>
          </div>
        </div>
      </div>

      <div className="flex-1 w-full h-full relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.08 }}
          minZoom={0.2}
          defaultEdgeOptions={{ type: "smoothstep", markerEnd: ARROW_IDLE }}
          proOptions={{ hideAttribution: true }}
          className="bg-[var(--bg)]"
        >
          <Background color={DOT_GRID} gap={20} size={1} />
          <Controls className="!bg-[var(--bg-raised)] !border-[var(--line)] !text-[var(--text)] fill-[var(--text)]" />
        </ReactFlow>
      </div>
    </div>
  );
}