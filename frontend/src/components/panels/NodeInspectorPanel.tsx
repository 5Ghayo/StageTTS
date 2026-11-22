import React from "react";
import { AlertTriangle, Cpu, Sliders, Sparkles, Terminal } from "lucide-react";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";

/**
 * 扁平化节点参数检查器（#222222 面板 + #181818 凹槽控件 + 极淡虚线分组）。
 */
export const NodeInspectorPanel: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const setMode = useProjectStore((s) => s.setMode);
  const updateBgmVolume = useProjectStore((s) => s.updateBgmVolume);

  const selectedNodeId = useWorkflowStore((s) => s.selectedNodeId);
  const nodes = useWorkflowStore((s) => s.nodes);
  const pipelineConfig = useWorkflowStore((s) => s.pipelineConfig);
  const updatePipelineConfig = useWorkflowStore((s) => s.updatePipelineConfig);
  const setWorkspacePreset = useWorkflowStore((s) => s.setWorkspacePreset);

  const selectedNode =
    nodes.find((n) => n.id === selectedNodeId) ?? nodes[0] ?? null;

  if (!selectedNode) {
    return (
      <div className="flex h-full items-center justify-center bg-[#222222] p-4 text-[12px] text-muted">
        点击画布中的任意节点查看参数
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto bg-[#222222] py-3 pl-3.5 pr-4 text-[12px]">
      {/* 节点概览 */}
      <div className="border-b border-dashed border-white/[0.08] pb-3">
        <div className="flex items-center justify-between gap-2">
          <span className="flex items-center gap-1.5 text-[13px] font-medium text-primary">
            <Sliders
              className="h-3.5 w-3.5 shrink-0 text-secondary"
              strokeWidth={1.5}
            />
            <span>{selectedNode.data.title}</span>
          </span>
          <span className="shrink-0 whitespace-nowrap rounded-[4px] bg-white/[0.04] px-2 py-0.5 font-mono tabular-nums text-[11px] text-secondary">
            {selectedNode.data.vramGb > 0
              ? `${selectedNode.data.vramGb.toFixed(1)} GB`
              : "纯 CPU"}
          </span>
        </div>
        <p className="mt-1.5 text-secondary">{selectedNode.data.summary}</p>
        <div className="mt-2 flex items-center gap-1.5 text-[11px] text-muted">
          <Cpu className="h-3.5 w-3.5 text-secondary" strokeWidth={1.5} />
          <span>当前状态：{selectedNode.data.status}</span>
        </div>
      </div>

      {selectedNode.id === "demucs" && (
        <label className="flex items-center gap-2 border-b border-dashed border-white/[0.08] pb-3 text-secondary">
          <input
            type="checkbox"
            checked={pipelineConfig.allowDemucsCpu}
            onChange={(e) => updatePipelineConfig({ allowDemucsCpu: e.target.checked })}
          />
          允许 CUDA 不可用或失败时用 CPU 分离（较慢）
        </label>
      )}

      {/* 识别精度控制 */}
      {selectedNode.id === "whisperx" && (
        <div className="space-y-2 border-b border-dashed border-white/[0.08] pb-3">
          <div className="whitespace-nowrap text-[11px] text-secondary">
            识别精度
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            {(
              [
                { id: "float16", title: "标准 FP16", desc: "峰值约 5.2 GB" },
                { id: "int8", title: "省显存 INT8", desc: "峰值约 3.4 GB" },
              ] as const
            ).map((opt) => {
              const active = pipelineConfig.computeType === opt.id;
              return (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => updatePipelineConfig({ computeType: opt.id })}
                  className={`rounded-[4px] px-2.5 py-2 text-left transition-colors ${
                    active
                      ? "bg-[#333333] text-[#FFFFFF]"
                      : "bg-[#2A2A2A] text-secondary hover:bg-[#363636] hover:text-primary"
                  }`}
                >
                  <div className="font-medium">{opt.title}</div>
                  <div className="mt-0.5 font-mono tabular-nums text-[10px] text-muted">
                    {opt.desc}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* 目标翻译语种 */}
      {selectedNode.id === "llm_tagger" && (
        <div className="flex items-center gap-2 border-b border-dashed border-white/[0.08] pb-3">
          <label className="w-20 shrink-0 whitespace-nowrap text-[11px] text-secondary">
            目标配音语言
          </label>
          <select
            value={pipelineConfig.targetLang}
            onChange={(e) =>
              updatePipelineConfig({
                targetLang: e.target.value as "zh" | "en" | "ja",
              })
            }
            className="h-7 flex-1 rounded-[4px] border border-white/[0.07] bg-[#181818] px-2 text-[12px] text-primary transition-colors focus:border-white/25 focus:outline-none"
          >
            <option value="zh">中文配音</option>
            <option value="en">英文配音</option>
            <option value="ja">日语配音</option>
          </select>
        </div>
      )}

      {/* 混音与合规水印 */}
      {selectedNode.id === "ffmpeg_mixer" && (
        <div className="space-y-3 border-b border-dashed border-white/[0.08] pb-3">
          <div className="space-y-1">
            <div className="flex justify-between text-[11px]">
              <span className="whitespace-nowrap text-secondary">伴奏音量</span>
              <span className="font-mono tabular-nums text-primary">
                {Math.round((script?.media.bgm_volume ?? 0.75) * 100)}%
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={script?.media.bgm_volume ?? 0.75}
              onChange={(e) => updateBgmVolume(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <label className="flex cursor-pointer items-start gap-2 pt-1 text-primary">
            <input
              type="checkbox"
              checked={pipelineConfig.burnWatermark}
              onChange={(e) =>
                updatePipelineConfig({ burnWatermark: e.target.checked })
              }
              className="mt-0.5 h-3.5 w-3.5"
            />
            <div>
              <div className="font-medium">导出时烧录“AI 合成语音”水印</div>
              <p className="mt-1 flex items-center gap-1 text-[11px] text-status-warning">
                <AlertTriangle
                  className="h-3.5 w-3.5 shrink-0"
                  strokeWidth={1.5}
                />
                <span>开启后将重编码画面，关闭则秒级无损导出</span>
              </p>
            </div>
          </label>
        </div>
      )}

      {/* 虚拟舞台快捷进入 */}
      {selectedNode.id === "virtual_stage_out" && (
        <div className="space-y-2.5">
          <p className="text-secondary">
            读取剧本与合成音频，实时驱动虚拟形象口型与表情演出。
          </p>
          <button
            type="button"
            onClick={() => {
              setMode("virtual_stage");
              setWorkspacePreset("stage");
            }}
            className="flex h-7 w-full items-center justify-center gap-1.5 rounded-[4px] bg-[#3A6EA5] px-2.5 py-1 text-[12px] font-medium text-[#FFFFFF] transition-colors hover:bg-[#467DB8]"
          >
            <Sparkles className="h-3.5 w-3.5" strokeWidth={1.5} />
            <span>进入虚拟舞台</span>
          </button>
        </div>
      )}
    </div>
  );
};

/**
 * 独立的管线运行日志面板（置于节点流模式右侧“运行日志”标签页）。
 */
export const PipelineLogsPanel: React.FC = () => {
  const pipelineLogs = useWorkflowStore((s) => s.pipelineLogs);

  return (
    <div className="flex h-full flex-col bg-[#222222] py-3 pl-3.5 pr-4 text-[12px]">
      <div className="mb-2 flex items-center gap-1.5 text-[11px] font-medium text-secondary">
        <Terminal className="h-3.5 w-3.5 text-secondary" strokeWidth={1.5} />
        <span>显存与任务调度记录</span>
      </div>
      <div className="flex-1 space-y-1.5 overflow-y-auto rounded-[4px] bg-[#161616] p-2.5 font-mono tabular-nums text-[11px]">
        {pipelineLogs.map((log) => (
          <div key={log.id} className="leading-relaxed">
            <span className="text-muted">[{log.timestamp}]</span>{" "}
            <span
              className={
                log.level === "success"
                  ? "text-status-ready"
                  : log.level === "warn"
                    ? "text-status-warning"
                    : log.level === "error"
                      ? "text-status-error"
                      : "text-primary"
              }
            >
              {log.message}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};
