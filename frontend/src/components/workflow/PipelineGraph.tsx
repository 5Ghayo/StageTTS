import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  type NodeTypes,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import type React from "react";
import { useMemo } from "react";
import "@xyflow/react/dist/style.css";
import { Maximize2, Play, RotateCcw } from "lucide-react";
import { type StageFlowNode, useWorkflowStore } from "../../store/useWorkflowStore";
import { pipelineStartForNode, pipelineNodeHint } from "../../types/workflow";
import { PipelineRunControls } from "./PipelineRunControls";
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
  const config = useWorkflowStore((s) => s.pipelineConfig);
  const updateConfig = useWorkflowStore((s) => s.updatePipelineConfig);
  const selectedNodeId = useWorkflowStore((s) => s.selectedNodeId);
  const startStep = pipelineStartForNode(selectedNodeId);
  const selectedTitle = nodes.find((node) => node.id === selectedNodeId)?.data.title;
  const canGenerateScript = startStep === "parse" || startStep === "translate";

  const { fitView } = useReactFlow();

  const nodeTypes = useMemo<NodeTypes>(
    () => ({
      stageNode: StageNode,
    }),
    [],
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
      <div className="absolute left-3 right-3 top-3 z-10 flex w-fit max-w-[calc(100%-1.5rem)] flex-wrap items-center gap-1 rounded-[4px] border border-white/[0.06] bg-[#222222] p-1 text-[12px]">
        <span className="px-2 text-secondary" title={startStep ? "运行时从当前选中节点开始" : pipelineNodeHint(selectedNodeId)}>
          {selectedTitle ? `已选：${selectedTitle}` : "点击节点选择运行起点"}
        </span>
        {!startStep && selectedNodeId && <span className="px-1 text-[11px] text-status-warning">{pipelineNodeHint(selectedNodeId)}</span>}
        {canGenerateScript && (
          <label className="flex items-center gap-1.5 px-1 text-secondary">
            目标语言
            <select
              aria-label="管线目标语言"
              value={config.targetLang}
              disabled={isPipelineRunning}
              onChange={(event) => updateConfig({ targetLang: event.target.value as "zh" | "en" | "ja" })}
              className="h-7 rounded-[4px] border border-white/[0.08] bg-[#181818] px-2 text-primary disabled:opacity-50"
            >
              <option value="zh">中文</option>
              <option value="en">英文</option>
              <option value="ja">日语</option>
            </select>
          </label>
        )}
        <button
          type="button"
          onClick={() => void runPipeline({ parseOnly: true })}
          disabled={isPipelineRunning || !canGenerateScript}
          title={startStep === "translate" ? "保存当前台词与设置，翻译已有对白并保存剧本" : "执行 Banquet、MOSS 和翻译，生成剧本供审校"}
          className="flex h-7 items-center gap-1.5 rounded-[4px] bg-[#3A6EA5] px-2.5 py-1 font-medium text-[#FFFFFF] transition-colors hover:bg-[#467DB8] disabled:opacity-50"
        >
          <Play className="h-3.5 w-3.5" strokeWidth={1.5} />
          <span>{isPipelineRunning ? "运行中..." : startStep === "translate" ? "翻译并保存剧本" : "仅解析剧本"}</span>
        </button>

        <button
          type="button"
          onClick={() => void runPipeline({ parseOnly: false })}
          disabled={isPipelineRunning || !startStep}
          title={startStep ? `从「${selectedTitle}」开始运行后续管线` : pipelineNodeHint(selectedNodeId)}
          className="flex h-7 items-center gap-1.5 rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-2.5 py-1 font-medium text-primary transition-colors hover:bg-[#363636] disabled:opacity-50"
        >
          <Play className="h-3.5 w-3.5 text-secondary" strokeWidth={1.5} />
          <span>{selectedTitle && startStep ? `从${selectedTitle}运行` : "请先选择节点"}</span>
        </button>

        <PipelineRunControls />
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
          disabled={isPipelineRunning}
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
        onPaneClick={() => selectNode(null)}
        multiSelectionKeyCode={null}
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
