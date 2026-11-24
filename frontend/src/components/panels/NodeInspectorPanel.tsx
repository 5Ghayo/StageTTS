import { AlertTriangle, Cpu, Sliders, Sparkles, Terminal } from "lucide-react";
import type React from "react";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import type { StemVolumes, SubtitleBurnMode } from "../../types/script";

/**
 * 扁平化节点参数检查器（#222222 面板 + #181818 凹槽控件 + 极淡虚线分组）。
 */
export const NodeInspectorPanel: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const setMode = useProjectStore((s) => s.setMode);
  const updateBgmVolume = useProjectStore((s) => s.updateBgmVolume);
  const updateStemVolume = useProjectStore((s) => s.updateStemVolume);
  const updateAutoDucking = useProjectStore((s) => s.updateAutoDucking);

  const selectedNodeId = useWorkflowStore((s) => s.selectedNodeId);
  const nodes = useWorkflowStore((s) => s.nodes);
  const pipelineConfig = useWorkflowStore((s) => s.pipelineConfig);
  const setWorkspacePreset = useWorkflowStore((s) => s.setWorkspacePreset);

  const selectedNode = nodes.find((n) => n.id === selectedNodeId) ?? nodes[0] ?? null;

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
            <Sliders className="h-3.5 w-3.5 shrink-0 text-secondary" strokeWidth={1.5} />
            <span>{selectedNode.data.title}</span>
          </span>
          <span className="shrink-0 whitespace-nowrap rounded-[4px] bg-white/[0.04] px-2 py-0.5 font-mono tabular-nums text-[11px] text-secondary">
            {selectedNode.data.vramGb === null
              ? "—"
              : selectedNode.data.vramGb > 0
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

      {selectedNode.id === "llm_tagger" && (
        <label className="flex items-center gap-2 border-b border-dashed border-white/[0.08] pb-3 text-[11px] text-secondary">
          目标配音语言
          <select value={pipelineConfig.targetLang} onChange={(event) => useWorkflowStore.getState().updatePipelineConfig({ targetLang: event.target.value as "zh" | "en" | "ja" })} className="h-7 flex-1 rounded border border-white/[0.07] bg-[#181818] px-2 text-primary">
            <option value="zh">中文配音</option><option value="en">英文配音</option><option value="ja">日语配音</option>
          </select>
        </label>
      )}

      {/* MOSS runtime information */}
      {selectedNode.id === "whisperx" && (
        <div className="space-y-1 border-b border-dashed border-white/[0.08] pb-3 text-[11px]">
          <div className="text-secondary">主识别模型</div>
          <div className="text-primary">MOSS-Transcribe-Diarize · BF16 / SDPA</div>
          <p className="text-secondary">8GB 显卡已测 10–60 秒样例，峰值约 2 GB。当前应用时长上限为 60 秒；MOSS 输出句段时间，不提供词级强制对齐。</p>
        </div>
      )}

      {/* 混音与合规水印 */}
      {selectedNode.id === "ffmpeg_mixer" && (
        <div className="space-y-3 border-b border-dashed border-white/[0.08] pb-3">
          <div className="space-y-1">
            <div className="flex justify-between text-[11px]">
              <span className="whitespace-nowrap text-secondary">伴奏主音量</span>
              <span className="font-mono tabular-nums text-primary">
                {Math.round((script?.media.bgm_volume ?? 0.75) * 100)}%
              </span>
            </div>
            <input
              aria-label="伴奏主音量"
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={script?.media.bgm_volume ?? 0.75}
              onChange={(e) => updateBgmVolume(Number(e.target.value))}
              className="w-full"
            />
          </div>

          {script?.media.cinematic_stems && (
            <div className="space-y-1.5 border-t border-white/[0.05] pt-2">
              {(
                [
                  { key: "singing", label: "歌声轨" },
                  { key: "music", label: "音乐轨" },
                  { key: "effects", label: "音效轨" },
                ] as Array<{ key: keyof StemVolumes; label: string }>
              ).map((item) => {
                const value = script.media.stem_volumes?.[item.key] ?? 1;
                return (
                  <div key={item.key} className="flex items-center gap-2 text-[11px]">
                    <span className="w-11 shrink-0 text-secondary">{item.label}</span>
                    <input
                      aria-label={item.label}
                      type="range"
                      min={0}
                      max={2}
                      step={0.05}
                      value={value}
                      onChange={(e) => updateStemVolume(item.key, Number(e.target.value))}
                      className="min-w-0 flex-1"
                    />
                    <span className="w-9 shrink-0 text-right font-mono tabular-nums text-secondary">
                      {Math.round(value * 100)}%
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          <label className="flex cursor-pointer items-center justify-between gap-2 text-[11px] text-primary">
            <span>人声自动闪避背景音乐</span>
            <input
              type="checkbox"
              checked={script?.media.auto_ducking ?? true}
              onChange={(e) => updateAutoDucking(e.target.checked)}
              className="h-3.5 w-3.5"
            />
          </label>

          <div className="flex items-center gap-2">
            <label
              htmlFor="node-burn-subtitles"
              className="w-20 shrink-0 whitespace-nowrap text-[11px] text-secondary"
            >
              烧录 ASS 字幕
            </label>
            <select
              id="node-burn-subtitles"
              value={pipelineConfig.burnSubtitles}
              onChange={(e) =>
                updatePipelineConfig({
                  burnSubtitles: e.target.value as SubtitleBurnMode,
                })
              }
              className="h-7 flex-1 rounded-[4px] border border-white/[0.07] bg-[#181818] px-2 text-[12px] text-primary transition-colors focus:border-white/25 focus:outline-none"
            >
              <option value="none">不烧录（快速无损）</option>
              <option value="target">单语译文 ASS</option>
              <option value="bilingual">双语对照 ASS</option>
            </select>
          </div>

          <label className="flex cursor-pointer items-start gap-2 pt-1 text-primary">
            <input
              type="checkbox"
              checked={pipelineConfig.burnWatermark}
              disabled={false}
              onChange={(event) => useWorkflowStore.getState().updatePipelineConfig({ burnWatermark: event.target.checked })}
              className="mt-0.5 h-3.5 w-3.5"
            />
            <div>
              <div className="font-medium">导出时烧录“AI 合成语音”水印</div>
              <p className="mt-1 flex items-center gap-1 text-[11px] text-status-warning">
                <AlertTriangle className="h-3.5 w-3.5 shrink-0" strokeWidth={1.5} />
                <span>开启硬字幕或水印时将单次重编码，全关闭则秒级无损导出</span>
              </p>
            </div>
          </label>
        </div>
      )}

      {/* 虚拟舞台快捷进入 */}
      {selectedNode.id === "virtual_stage_out" && (
        <div className="space-y-2.5">
          <p className="text-secondary">读取剧本与合成音频，实时驱动虚拟形象口型与表情演出。</p>
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
