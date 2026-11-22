import React, { useMemo } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Maximize2, Play, RotateCcw } from "lucide-react";
import {
  useWorkflowStore,
  type StageFlowNode,
} from "../../store/useWorkflowStore";
import { StageNode } from "./StageNode";

const PipelineGraphInner: React.FC = () => {
  const nodes = useWorkflowStore((s) => s.nodes);
  const edges = useWorkflowStore((s) => s.edges);
  const onNodesChange = useWorkflowStore((s) => s.onNodesChange);
  const onEdgesChange = useWorkflowStore((s) => s.onEdgesChange);
  const selectNode = useWorkflowStore((s) => s.selectNode);
  const resetGraphLayout = useWorkflowStore((s) => s.resetGraphLayout);
  const isPipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);
  const runPipeline = useWorkflowStore((s) => s.runPipeline);

  const { fitView } = useReactFlow();

  const nodeTypes = useMemo<NodeTypes>(
    () => ({
      stageNode: StageNode,
    }),
    []
  );

  const handleReset = () => {
    resetGraphLayout();
    window.setTimeout(() => {
      void fitView({ padding: 0.08, duration: 250 });
    }, 30);
  };

  return (
    <div className="relative h-full w-full overflow-hidden bg-[#161616]">
      {/* 紧凑左上控制条（去线框 #222222 明度块） */}
      <div className="absolute left-3 top-3 z-10 flex items-center gap-1 rounded-[4px] border border-white/[0.06] bg-[#222222] p-1 text-[12px]">
        <button
          type="button"
          onClick={() => void runPipeline({ parseOnly: true })}
          disabled={isPipelineRunning}
          title="执行真实 4 阶段解析（Demucs + WhisperX + pyannote + LLM），生成剧本供审校"
          className="flex h-7 items-center gap-1.5 rounded-[4px] bg-[#3A6EA5] px-2.5 py-1 font-medium text-[#FFFFFF] transition-colors hover:bg-[#467DB8] disabled:opacity-50"
        >
          <Play className="h-3.5 w-3.5" strokeWidth={1.5} />
          <span>{isPipelineRunning ? "运行中..." : "仅解析剧本"}</span>
        </button>

        <button
          type="button"
          onClick={() => void runPipeline({ parseOnly: false })}
          disabled={isPipelineRunning}
          title="依次执行真实解析 -> 批量语音合成 -> FFmpeg 成片导出"
          className="flex h-7 items-center gap-1.5 rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-2.5 py-1 font-medium text-primary transition-colors hover:bg-[#363636] disabled:opacity-50"
        >
          <Play className="h-3.5 w-3.5 text-secondary" strokeWidth={1.5} />
          <span>运行全管线</span>
        </button>

        <button
          type="button"
          onClick={() => void fitView({ padding: 0.08, duration: 250 })}
          className="flex h-7 items-center gap-1.5 rounded-[4px] px-2 text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
        >
          <Maximize2 className="h-3.5 w-3.5" strokeWidth={1.5} />
          <span>适配画布</span>
        </button>

        <button
          type="button"
          onClick={handleReset}
          className="flex h-7 items-center gap-1.5 rounded-[4px] px-2 text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
        >
          <RotateCcw className="h-3.5 w-3.5" strokeWidth={1.5} />
          <span>重置</span>
        </button>
      </div>

      <ReactFlow<StageFlowNode>
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={(_, node) => selectNode(node.id)}
        fitView
        fitViewOptions={{ padding: 0.08 }}
        minZoom={0.4}
        maxZoom={1.8}
        proOptions={{ hideAttribution: true }}
      >
        <Background
          variant={BackgroundVariant.Lines}
          gap={24}
          lineWidth={1}
          color="rgba(255, 255, 255, 0.03)"
        />
        <Controls position="bottom-left" />
        <MiniMap
          position="bottom-right"
          pannable
          zoomable
          nodeColor={(node) => {
            const cat = node.data?.category;
            if (cat === "input") return "#6E524B";
            if (cat === "parse") return "#5E4F66";
            if (cat === "hub") return "#6E5F42";
            if (cat === "synthesis") return "#446B64";
            return "#486178";
          }}
          maskColor="rgba(20, 20, 20, 0.72)"
        />
      </ReactFlow>
    </div>
  );
};

export const PipelineGraph: React.FC = () => {
  return (
    <ReactFlowProvider>
      <PipelineGraphInner />
    </ReactFlowProvider>
  );
};
