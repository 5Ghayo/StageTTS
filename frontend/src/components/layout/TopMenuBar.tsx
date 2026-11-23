import { useQuery } from "@tanstack/react-query";
import {
  Clapperboard,
  Cpu,
  Download,
  Film,
  History,
  Mic,
  Play,
  Redo2,
  RotateCcw,
  Save,
  Sparkles,
  Undo2,
  Workflow,
} from "lucide-react";
import type React from "react";
import { useEffect, useRef, useState } from "react";
import { type GpuDevice, fetchGpuStatus, fetchProjectHistory } from "../../services/api";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import type { ScriptSnapshotMeta } from "../../types/script";
import type { WorkspacePreset } from "../../types/workflow";
import { ExportModal } from "../modals/ExportModal";

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
  const pastCount = useProjectStore((s) => s.pastScripts.length);
  const futureCount = useProjectStore((s) => s.futureScripts.length);
  const saveProjectScript = useProjectStore((s) => s.saveProjectScript);
  const synthesizeSegment = useProjectStore((s) => s.synthesizeSegment);
  const undo = useProjectStore((s) => s.undo);
  const redo = useProjectStore((s) => s.redo);
  const restoreSnapshot = useProjectStore((s) => s.restoreSnapshot);

  const workspacePreset = useWorkflowStore((s) => s.workspacePreset);
  const setWorkspacePreset = useWorkflowStore((s) => s.setWorkspacePreset);
  const isPipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);
  const runPipeline = useWorkflowStore((s) => s.runPipeline);
  const appendLog = useWorkflowStore((s) => s.appendLog);

  const [actionToast, setActionToast] = useState<string | null>(null);
  const [exportModalOpen, setExportModalOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [snapshots, setSnapshots] = useState<ScriptSnapshotMeta[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const historyPopoverRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!historyOpen) {
      return;
    }
    const handleOutside = (e: MouseEvent) => {
      if (historyPopoverRef.current && !historyPopoverRef.current.contains(e.target as Node)) {
        setHistoryOpen(false);
      }
    };
    window.addEventListener("mousedown", handleOutside);
    return () => window.removeEventListener("mousedown", handleOutside);
  }, [historyOpen]);

  const toggleHistoryMenu = async () => {
    const nextOpen = !historyOpen;
    setHistoryOpen(nextOpen);
    if (!nextOpen || !script) {
      return;
    }
    if (backendStatus !== "online") {
      setSnapshots([]);
      setHistoryError("离线模式暂不支持读取后端历史快照");
      return;
    }
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const res = await fetchProjectHistory(script.project_id);
      setSnapshots(res.snapshots);
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : "读取历史快照失败");
    } finally {
      setHistoryLoading(false);
    }
  };

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

  return (
    <>
      <header className="flex h-9 shrink-0 items-center justify-between bg-[#222222] px-3.5 text-[12px]">
        {/* 左侧：Logo · 状态点 · 无边框项目标签 · 柔和次操作保存按钮 */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 text-[13px] font-medium text-primary">
            <Clapperboard className="h-4 w-4 text-secondary" strokeWidth={1.5} />
            <span>StageTTS</span>
          </div>

          <span
            title={statusTitle}
            className={`h-1.5 w-1.5 shrink-0 rounded-full ${statusColor}`}
          />

          <span className="flex h-6 items-center rounded-[4px] bg-white/[0.04] px-2 font-mono tabular-nums text-[11px] text-secondary">
            {script ? script.project_id : "未打开项目"}
          </span>

          {script && (
            <>
              <button
                type="button"
                disabled={isLoading}
                onClick={() => void saveProjectScript()}
                title="保存剧本修改 (Ctrl/Cmd+S)"
                className="flex h-7 items-center gap-1.5 rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-2.5 py-1 text-[12px] text-primary transition-colors hover:bg-[#363636] disabled:opacity-50"
              >
                {isDirty ? (
                  <span className="h-1.5 w-1.5 rounded-full bg-status-warning" />
                ) : (
                  <Save className="h-3.5 w-3.5 text-secondary" strokeWidth={1.5} />
                )}
                <span>{isDirty ? "保存*" : "保存"}</span>
              </button>

              <div className="flex items-center gap-0.5 rounded-[4px] bg-[#181818] p-0.5">
                <button
                  type="button"
                  disabled={pastCount === 0 || isLoading}
                  onClick={() => undo()}
                  title="撤销 (Ctrl/Cmd+Z)"
                  className="flex h-6 w-6 items-center justify-center rounded-[3px] text-secondary transition-colors hover:bg-[#333333] hover:text-primary disabled:opacity-35"
                >
                  <Undo2 className="h-3.5 w-3.5" strokeWidth={1.5} />
                </button>
                <button
                  type="button"
                  disabled={futureCount === 0 || isLoading}
                  onClick={() => redo()}
                  title="重做 (Ctrl/Cmd+Shift+Z)"
                  className="flex h-6 w-6 items-center justify-center rounded-[3px] text-secondary transition-colors hover:bg-[#333333] hover:text-primary disabled:opacity-35"
                >
                  <Redo2 className="h-3.5 w-3.5" strokeWidth={1.5} />
                </button>
              </div>

              <div className="relative" ref={historyPopoverRef}>
                <button
                  type="button"
                  disabled={isLoading || isPipelineRunning}
                  onClick={() => void toggleHistoryMenu()}
                  title="查看并恢复服务端历史快照"
                  className="flex h-7 items-center gap-1 rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-2 py-1 text-[11px] text-secondary transition-colors hover:bg-[#363636] hover:text-primary disabled:opacity-50"
                >
                  <History className="h-3.5 w-3.5" strokeWidth={1.5} />
                  <span>快照</span>
                </button>

                {historyOpen && (
                  <div className="absolute left-0 top-8 z-50 w-72 rounded-[4px] border border-white/[0.10] bg-[#222222] p-2 text-[11px]">
                    <div className="mb-1.5 flex items-center justify-between border-b border-white/[0.06] pb-1.5">
                      <span className="font-medium text-primary">历史快照（最多保留 20 条）</span>
                      <span className="font-mono tabular-nums text-[10px] text-muted">
                        {snapshots.length} 条
                      </span>
                    </div>
                    {historyLoading ? (
                      <div className="py-3 text-center text-muted">加载中...</div>
                    ) : historyError ? (
                      <div className="py-2 text-status-warning">{historyError}</div>
                    ) : snapshots.length === 0 ? (
                      <div className="py-3 text-center text-muted">
                        暂无快照，点击“保存”后将自动记录
                      </div>
                    ) : (
                      <ul className="max-h-64 space-y-1 overflow-y-auto">
                        {snapshots.map((snap) => (
                          <li
                            key={snap.snapshot_id}
                            className="flex items-center justify-between gap-2 rounded-[3px] bg-[#181818] px-2 py-1.5"
                          >
                            <div className="min-w-0 flex-1">
                              <div className="truncate font-medium text-primary">{snap.label}</div>
                              <div className="truncate font-mono tabular-nums text-[10px] text-muted">
                                {new Date(snap.created_at).toLocaleString()} · {snap.segment_count}{" "}
                                句
                              </div>
                            </div>
                            <button
                              type="button"
                              disabled={isLoading || isSynthesizing || isPipelineRunning}
                              onClick={() => {
                                void restoreSnapshot(snap.snapshot_id).then(() => {
                                  setHistoryOpen(false);
                                  setActionToast(`已恢复快照：${snap.label}`);
                                  window.setTimeout(() => setActionToast(null), 3500);
                                });
                              }}
                              className="flex shrink-0 items-center gap-1 rounded-[3px] border border-white/[0.08] bg-[#2C2C2C] px-1.5 py-0.5 text-[10px] text-primary transition-colors hover:bg-[#363636] disabled:opacity-40"
                            >
                              <RotateCcw className="h-3 w-3" strokeWidth={1.5} />
                              <span>恢复</span>
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
              </div>
            </>
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
        <div className="flex items-center gap-2">
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
                  disabled={isSynthesizing || isLoading || isPipelineRunning}
                  onClick={() => setExportModalOpen(true)}
                  className="flex h-7 items-center gap-1.5 whitespace-nowrap rounded-[4px] bg-[#3A6EA5] px-3 py-1 text-[12px] font-medium text-[#FFFFFF] transition-colors hover:bg-[#467DB8] disabled:opacity-50"
                >
                  <Download className="h-3.5 w-3.5" strokeWidth={1.5} />
                  <span>导出成片</span>
                </button>
              </>
            )
          )}
        </div>
      </header>

      <ExportModal
        open={exportModalOpen}
        onClose={() => setExportModalOpen(false)}
        onExportSuccess={(msg) => {
          setActionToast(msg);
          window.setTimeout(() => setActionToast(null), 3500);
        }}
      />
    </>
  );
};
