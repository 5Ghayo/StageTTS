import { useQuery } from "@tanstack/react-query";
import {
  Clapperboard,
  Cpu,
  Download,
  Film,
  Mic,
  Play,
  Save,
  SlidersHorizontal,
  Sparkles,
  Workflow,
} from "lucide-react";
import type React from "react";
import { useState } from "react";
import {
  ApiError,
  type GpuDevice,
  fetchGpuStatus,
  getSubtitlesUrl,
  renderVideo,
} from "../../services/api";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import type { SubtitleBurnMode } from "../../types/script";
import type { WorkspacePreset } from "../../types/workflow";

const PRESET_OPTIONS: Array<{
  id: WorkspacePreset;
  label: string;
  icon: React.ReactNode;
}> = [
  {
    id: "editing",
    label: "剪辑",
    icon: <Film className="h-3.5 w-3.5" strokeWidth={1.5} />,
  },
  {
    id: "workflow",
    label: "节点流",
    icon: <Workflow className="h-3.5 w-3.5" strokeWidth={1.5} />,
  },
  {
    id: "stage",
    label: "舞台",
    icon: <Sparkles className="h-3.5 w-3.5" strokeWidth={1.5} />,
  },
];

function formatGpuMemory(gpu: GpuDevice): string {
  return `${(gpu.used_mib / 1024).toFixed(1)}/${(gpu.total_mib / 1024).toFixed(1)} GiB`;
}

/**
 * 紧凑 36px 顶部主控栏（#222222 哑光灰底、无外框暗槽分段器、极淡无框信息标签）。
 */
export const TopMenuBar: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const backendStatus = useProjectStore((s) => s.backendStatus);
  const isDirty = useProjectStore((s) => s.isDirty);
  const isLoading = useProjectStore((s) => s.isLoading);
  const isSynthesizing = useProjectStore((s) => s.isSynthesizing);
  const saveProjectScript = useProjectStore((s) => s.saveProjectScript);
  const synthesizeSegment = useProjectStore((s) => s.synthesizeSegment);
  const updateAutoDucking = useProjectStore((s) => s.updateAutoDucking);

  const workspacePreset = useWorkflowStore((s) => s.workspacePreset);
  const setWorkspacePreset = useWorkflowStore((s) => s.setWorkspacePreset);
  const pipelineConfig = useWorkflowStore((s) => s.pipelineConfig);
  const updatePipelineConfig = useWorkflowStore((s) => s.updatePipelineConfig);
  const isPipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);
  const runPipeline = useWorkflowStore((s) => s.runPipeline);
  const appendLog = useWorkflowStore((s) => s.appendLog);

  const [actionToast, setActionToast] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const { data: gpuData } = useQuery({
    queryKey: ["gpu-status"],
    queryFn: ({ signal }) => fetchGpuStatus(signal),
    enabled: backendStatus === "online",
    refetchInterval: 3000,
    retry: false,
  });
  const gpus = gpuData?.devices ?? [];

  const primaryGpu = gpus[0];
  const gpuTitle = gpus.length
    ? gpus.map((gpu) => `${gpu.name} · ${formatGpuMemory(gpu)}`).join("\n")
    : "无法读取后端机器的 NVIDIA 显存数据";

  const statusColor =
    backendStatus === "online"
      ? "bg-status-ready"
      : backendStatus === "connecting"
        ? "bg-secondary animate-pulse"
        : "bg-status-warning";

  const statusTitle =
    backendStatus === "online"
      ? "FastAPI 后端已连接"
      : backendStatus === "connecting"
        ? "正在检测后端连接..."
        : "离线演示模式（未连接后端）";

  const pendingCount = script
    ? script.segments.filter((seg) => !seg.keep_original && seg.status !== "ready").length
    : 0;

  const handleBatchSynthesize = async () => {
    if (!script || isSynthesizing) {
      return;
    }
    setActionToast("正在按需启动 GPT-SoVITS 并配音...");
    const ids =
      pendingCount === 0
        ? script.segments.filter((seg) => !seg.keep_original).map((seg) => seg.id)
        : undefined;
    const res = await synthesizeSegment(ids);
    setActionToast(res.message);
    if (res.issues?.length) {
      res.issues.forEach((issue) => appendLog("env_tts", issue, "warn"));
    } else {
      appendLog("env_tts", res.message, res.simulated ? "warn" : "success");
    }
    window.setTimeout(() => setActionToast(null), 4000);
  };

  const handleExport = async () => {
    if (!script || isExporting) {
      return;
    }
    if (backendStatus === "offline") {
      const msg = "离线模式：已记录导出配置";
      setActionToast(msg);
      appendLog("env_backend", msg, "warn");
      window.setTimeout(() => setActionToast(null), 3000);
      return;
    }
    setIsExporting(true);
    setActionToast("导出中...");
    try {
      if (useProjectStore.getState().isDirty) {
        await saveProjectScript();
        const afterSave = useProjectStore.getState();
        if (afterSave.backendStatus === "offline") {
          const msg = "离线模式：已记录导出配置";
          setActionToast(msg);
          appendLog("env_backend", msg, "warn");
          window.setTimeout(() => setActionToast(null), 3000);
          return;
        }
        if (afterSave.error || afterSave.isDirty) {
          const msg = afterSave.error ?? "请先保存当前修改后再导出";
          setActionToast(msg);
          appendLog("env_backend", msg, "error");
          window.setTimeout(() => setActionToast(null), 3000);
          return;
        }
      }
      const res = await renderVideo(
        script.project_id,
        pipelineConfig.burnWatermark,
        pipelineConfig.burnSubtitles,
        script.media.auto_ducking ?? true,
      );
      useProjectStore.setState((state) =>
        state.script?.project_id === res.project_id
          ? {
              script: {
                ...state.script,
                media: { ...state.script.media, output_video: res.output_video },
              },
            }
          : state,
      );
      setActionToast("已导出成片");
      appendLog("env_backend", `导出成功: ${res.output_video}`, "success");
    } catch (err) {
      const isStub =
        err instanceof ApiError &&
        (err.status === 503 ||
          err.status === 502 ||
          err.status === 504 ||
          err.isNotImplemented ||
          err.isNetworkError);
      const msg = isStub
        ? "离线模式：已记录导出配置"
        : err instanceof Error
          ? err.message
          : "导出失败";
      setActionToast(msg);
      appendLog("env_backend", msg, isStub ? "warn" : "error");
    } finally {
      setIsExporting(false);
    }
    window.setTimeout(() => setActionToast(null), 3000);
  };

  return (
    <header className="flex h-9 shrink-0 items-center justify-between bg-[#222222] px-3.5 text-[12px]">
      {/* 左侧：Logo · 状态点 · 无边框项目标签 · 柔和次操作保存按钮 */}
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1.5 text-[13px] font-medium text-primary">
          <Clapperboard className="h-4 w-4 text-secondary" strokeWidth={1.5} />
          <span>StageTTS</span>
        </div>

        <span title={statusTitle} className={`h-1.5 w-1.5 shrink-0 rounded-full ${statusColor}`} />

        <span className="flex h-6 items-center rounded-[4px] bg-white/[0.04] px-2 font-mono tabular-nums text-[11px] text-secondary">
          {script ? script.project_id : "未打开项目"}
        </span>

        {script && (
          <button
            type="button"
            disabled={isLoading}
            onClick={() => void saveProjectScript()}
            title="保存剧本修改"
            className="flex h-7 items-center gap-1.5 rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-2.5 py-1 text-[12px] text-primary transition-colors hover:bg-[#363636] disabled:opacity-50"
          >
            {isDirty ? (
              <span className="h-1.5 w-1.5 rounded-full bg-status-warning" />
            ) : (
              <Save className="h-3.5 w-3.5 text-secondary" strokeWidth={1.5} />
            )}
            <span>{isDirty ? "保存*" : "保存"}</span>
          </button>
        )}
      </div>

      {/* 中部：无外框暗槽分段控制器（槽底 #181818，选中项 #333333 白字） */}
      <nav
        aria-label="工作区模式"
        className="flex items-center gap-0.5 rounded-[4px] bg-[#181818] p-0.5"
      >
        {PRESET_OPTIONS.map((preset) => {
          const active = workspacePreset === preset.id;
          return (
            <button
              key={preset.id}
              type="button"
              onClick={() => setWorkspacePreset(preset.id)}
              className={`flex h-6 items-center gap-1.5 rounded-[3px] px-3 text-[12px] font-medium transition-colors ${
                active ? "bg-[#333333] text-[#FFFFFF]" : "text-secondary hover:text-primary"
              }`}
            >
              {preset.icon}
              <span>{preset.label}</span>
            </button>
          );
        })}
      </nav>

      {/* 右侧：无外框极淡标签（bg-white/[0.04]） · 实心钢蓝主按钮 */}
      <div className="relative flex items-center gap-2">
        {actionToast && (
          <span className="flex h-6 items-center rounded-[4px] bg-white/[0.06] px-2 text-[11px] text-primary">
            {actionToast}
          </span>
        )}

        {script && (
          <span className="hidden h-6 items-center whitespace-nowrap rounded-[4px] bg-white/[0.04] px-2 text-[11px] text-secondary md:flex">
            AI 合成语音
          </span>
        )}

        <div
          title={gpuTitle}
          className="hidden h-7 items-center gap-1.5 whitespace-nowrap rounded-[4px] bg-white/[0.04] px-2 font-mono tabular-nums text-[11px] text-secondary sm:flex"
        >
          <Cpu className="h-3.5 w-3.5 text-secondary" strokeWidth={1.5} />
          <span className="max-w-64 truncate">
            {primaryGpu
              ? `${primaryGpu.name.replace(/^NVIDIA GeForce /, "")} · ${formatGpuMemory(primaryGpu)}${gpus.length > 1 ? ` +${gpus.length - 1}` : ""}`
              : "显存不可用"}
          </span>
        </div>

        {workspacePreset === "workflow" ? (
          <button
            type="button"
            disabled={isPipelineRunning}
            onClick={() => void runPipeline({ parseOnly: false })}
            className="flex h-7 items-center gap-1.5 whitespace-nowrap rounded-[4px] bg-[#3A6EA5] px-3 py-1 text-[12px] font-medium text-[#FFFFFF] transition-colors hover:bg-[#467DB8] disabled:opacity-50"
          >
            <Play className="h-3.5 w-3.5" strokeWidth={1.5} />
            <span>{isPipelineRunning ? "运行中..." : "运行全管线"}</span>
          </button>
        ) : (
          script && (
            <>
              {script.segments.length > 0 && (
                <button
                  type="button"
                  disabled={isSynthesizing || isLoading}
                  onClick={() => void handleBatchSynthesize()}
                  title="自动按需启动 GPT-SoVITS 并批量合成所有待配音片段"
                  className="flex h-7 items-center gap-1.5 whitespace-nowrap rounded-[4px] border border-white/[0.1] bg-[#2C2C2C] px-3 py-1 text-[12px] font-medium text-primary transition-colors hover:bg-[#363636] disabled:opacity-50"
                >
                  <Mic className="h-3.5 w-3.5 text-secondary" strokeWidth={1.5} />
                  <span>
                    {isSynthesizing
                      ? "配音中..."
                      : pendingCount > 0
                        ? `全部配音 (${pendingCount})`
                        : "重新配音"}
                  </span>
                </button>
              )}
              <button
                type="button"
                title="导出设置与 ASS 字幕下载"
                onClick={() => setShowExportMenu((prev) => !prev)}
                className={`flex h-7 items-center gap-1 whitespace-nowrap rounded-[4px] border px-2 py-1 text-[11px] transition-colors ${
                  showExportMenu ||
                  pipelineConfig.burnSubtitles !== "none" ||
                  pipelineConfig.burnWatermark
                    ? "border-white/20 bg-[#333333] text-primary"
                    : "border-white/[0.08] bg-[#2C2C2C] text-secondary hover:bg-[#363636] hover:text-primary"
                }`}
              >
                <SlidersHorizontal className="h-3.5 w-3.5" strokeWidth={1.5} />
                <span>导出选项</span>
              </button>
              <button
                type="button"
                disabled={isExporting || isSynthesizing || isLoading || isPipelineRunning}
                onClick={() => {
                  setShowExportMenu(false);
                  void handleExport();
                }}
                className="flex h-7 items-center gap-1.5 whitespace-nowrap rounded-[4px] bg-[#3A6EA5] px-3 py-1 text-[12px] font-medium text-[#FFFFFF] transition-colors hover:bg-[#467DB8] disabled:opacity-50"
              >
                <Download className="h-3.5 w-3.5" strokeWidth={1.5} />
                <span>{isExporting ? "导出中..." : "导出成片"}</span>
              </button>
              {showExportMenu && (
                <div className="absolute right-0 top-9 z-40 w-64 space-y-2.5 rounded-[4px] border border-white/[0.08] bg-[#222222] p-3 text-[12px]">
                  <div className="flex items-center justify-between gap-2">
                    <label htmlFor="topbar-burn-subtitles" className="text-[11px] text-secondary">
                      ASS 硬字幕
                    </label>
                    <select
                      id="topbar-burn-subtitles"
                      value={pipelineConfig.burnSubtitles}
                      onChange={(e) =>
                        updatePipelineConfig({
                          burnSubtitles: e.target.value as SubtitleBurnMode,
                        })
                      }
                      className="h-7 rounded-[4px] border border-white/[0.07] bg-[#181818] px-2 text-[11px] text-primary focus:outline-none"
                    >
                      <option value="none">不烧录</option>
                      <option value="target">单语译文</option>
                      <option value="bilingual">双语对照</option>
                    </select>
                  </div>
                  <label className="flex cursor-pointer items-center justify-between gap-2 text-[11px] text-primary">
                    <span>烧录“AI 合成语音”水印</span>
                    <input
                      type="checkbox"
                      checked={pipelineConfig.burnWatermark}
                      onChange={(e) => updatePipelineConfig({ burnWatermark: e.target.checked })}
                      className="h-3.5 w-3.5"
                    />
                  </label>
                  <label className="flex cursor-pointer items-center justify-between gap-2 text-[11px] text-primary">
                    <span>人声自动闪避 BGM</span>
                    <input
                      type="checkbox"
                      checked={script.media.auto_ducking ?? true}
                      onChange={(e) => updateAutoDucking(e.target.checked)}
                      className="h-3.5 w-3.5"
                    />
                  </label>
                  <div className="flex items-center justify-between border-t border-dashed border-white/[0.08] pt-2 text-[11px]">
                    <span className="text-secondary">下载 ASS 字幕</span>
                    <div className="flex gap-1.5">
                      <a
                        href={getSubtitlesUrl(script.project_id, "target")}
                        download={`${script.project_id}_target.ass`}
                        className="rounded-[3px] border border-white/[0.08] bg-[#2C2C2C] px-2 py-0.5 text-primary hover:bg-[#363636]"
                      >
                        单语
                      </a>
                      <a
                        href={getSubtitlesUrl(script.project_id, "bilingual")}
                        download={`${script.project_id}_bilingual.ass`}
                        className="rounded-[3px] border border-white/[0.08] bg-[#2C2C2C] px-2 py-0.5 text-primary hover:bg-[#363636]"
                      >
                        双语
                      </a>
                    </div>
                  </div>
                </div>
              )}
            </>
          )
        )}
      </div>
    </header>
  );
};
