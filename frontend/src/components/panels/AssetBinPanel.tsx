import { X } from "lucide-react";
import type React from "react";
import { useState } from "react";
import { getSubtitlesUrl } from "../../services/api";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import type { StemVolumes, SubtitleLayout } from "../../types/script";
import { isSegmentOverflow, isSegmentTooShort } from "../../utils/synthesisIssues";
import { CharacterPanel } from "./CharacterPanel";
import { TranslationSettings } from "./TranslationSettings";
import { ModelParameters } from "./ModelParameters";

type DrawerSegmentFilter = "all" | "review" | "pending" | "issue";

/**
 * 240px 按需展开的左侧停靠抽屉（#222222 面板，极淡虚线分隔）。
 */
export const AssetBinPanel: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const backendStatus = useProjectStore((s) => s.backendStatus);
  const isDirty = useProjectStore((s) => s.isDirty);
  const isLoading = useProjectStore((s) => s.isLoading);
  const isSynthesizing = useProjectStore((s) => s.isSynthesizing);
  const selectSegment = useProjectStore((s) => s.selectSegment);
  const setCurrentTime = useProjectStore((s) => s.setCurrentTime);
  const saveProjectScript = useProjectStore((s) => s.saveProjectScript);
  const updateStemVolume = useProjectStore((s) => s.updateStemVolume);
  const updateAutoDucking = useProjectStore((s) => s.updateAutoDucking);
  const confirmAllPendingReviews = useProjectStore((s) => s.confirmAllPendingReviews);

  const leftDrawer = useWorkflowStore((s) => s.leftDrawer);
  const isPipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);
  const toggleLeftDrawer = useWorkflowStore((s) => s.toggleLeftDrawer);
  const [segmentFilter, setSegmentFilter] = useState<DrawerSegmentFilter>("all");
  const [subNote, setSubNote] = useState<string | null>(null);

  if (!script || leftDrawer === "none") {
    return null;
  }

  const editsDisabled = isLoading || isSynthesizing || isPipelineRunning;

  const handleDownloadSubtitles = async (layout: SubtitleLayout) => {
    if (backendStatus !== "online") {
      setSubNote("离线模式暂不支持导出后端 ASS 字幕");
      return;
    }
    setSubNote(null);
    if (isDirty) {
      await saveProjectScript();
      const afterSave = useProjectStore.getState();
      if (afterSave.error || afterSave.isDirty) {
        setSubNote(afterSave.error ?? "请先保存修改后再导出字幕");
        return;
      }
    }
    const link = document.createElement("a");
    link.href = getSubtitlesUrl(script.project_id, layout);
    link.download = `${script.project_id}_${layout}.ass`;
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const reviewPendingCount = script.segments.filter(
    (seg) => seg.alignment_needs_review || seg.speaker_needs_review,
  ).length;
  const pendingSynthCount = script.segments.filter(
    (seg) => !seg.keep_original && seg.status !== "ready",
  ).length;
  const issueCount = script.segments.filter(
    (seg) => seg.status === "error" || isSegmentOverflow(seg) || isSegmentTooShort(seg),
  ).length;
  const filteredSegments = script.segments.filter((seg) => {
    if (segmentFilter === "all") return true;
    if (segmentFilter === "review") {
      return Boolean(seg.alignment_needs_review || seg.speaker_needs_review);
    }
    if (segmentFilter === "pending") {
      return !seg.keep_original && seg.status !== "ready";
    }
    return seg.status === "error" || isSegmentOverflow(seg) || isSegmentTooShort(seg);
  });

  const stems = script.media.cinematic_stems;
  const stemVolumes = script.media.stem_volumes ?? { singing: 1, music: 1, effects: 1 };
  const autoDucking = script.media.auto_ducking ?? true;
  const stemControls: Array<{ key: keyof StemVolumes; label: string }> = [
    { key: "singing", label: "歌声" },
    { key: "music", label: "音乐" },
    { key: "effects", label: "音效" },
  ];
  const mediaRows = [
    { label: "原视频流", ready: Boolean(script.media.source_video) },
    ...(stems
      ? [
          { label: "对白轨", ready: Boolean(stems.dialogue), url: stems.dialogue },
          { label: "歌声轨", ready: Boolean(stems.singing), url: stems.singing },
          { label: "音乐轨", ready: Boolean(stems.music), url: stems.music },
          { label: "音效轨", ready: Boolean(stems.effects), url: stems.effects },
          {
            label: "背景轨",
            ready: Boolean(script.media.bgm_wav),
            url: script.media.bgm_wav ?? undefined,
          },
        ]
      : [
          { label: "人声轨", ready: Boolean(script.media.vocals_wav) },
          { label: "背景轨", ready: Boolean(script.media.bgm_wav) },
        ]),
    { label: "成片输出", ready: Boolean(script.media.output_video) },
  ];

  return (
    <div className="flex h-full w-60 shrink-0 flex-col bg-[#222222] text-[12px]">
      <div className="flex h-8 shrink-0 items-center justify-between border-b border-white/[0.06] px-3.5">
        <span className="text-[13px] font-medium text-primary">
          {leftDrawer === "characters" ? "角色与情绪音色" : "项目媒体素材"}
        </span>
        <button
          type="button"
          title="收起侧栏"
          onClick={() => toggleLeftDrawer(leftDrawer)}
          className="flex h-6 w-6 items-center justify-center rounded-[4px] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
        >
          <X className="h-3.5 w-3.5" strokeWidth={1.5} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {leftDrawer === "characters" ? (
          <CharacterPanel key={script.project_id} script={script} />
        ) : (
          <div>
            <TranslationSettings key={script.project_id} script={script} />
            <details className="space-y-3 border-b border-white/[0.08] px-3.5 py-2">
              <summary className="cursor-pointer text-primary">模型参数</summary>
              {(["separation", "recognition", "synthesis"] as const).map((group) => (
                <ModelParameters key={`${script.project_id}-${group}`} group={group} script={script} />
              ))}
            </details>
            <div className="divide-y divide-white/[0.05] border-b border-dashed border-white/[0.08]">
              {mediaRows.map((row) => (
                <div key={row.label} className="flex h-8 items-center justify-between px-3.5">
                  <span className="text-primary">{row.label}</span>
                  <span className="flex items-center gap-1.5 text-[11px] text-secondary">
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${
                        row.ready ? "bg-status-ready" : "bg-muted"
                      }`}
                    />
                    {"url" in row && row.url ? (
                      <a
                        href={row.url}
                        target="_blank"
                        rel="noreferrer"
                        aria-label={`试听${row.label}`}
                        className="underline hover:text-primary"
                      >
                        试听
                      </a>
                    ) : (
                      <span>{row.ready ? "就绪" : "未生成"}</span>
                    )}
                  </span>
                </div>
              ))}
            </div>

            <div className="space-y-2.5 border-b border-dashed border-white/[0.08] p-3.5">
              <div className="text-[11px] font-medium text-secondary">混音与四轨平衡</div>
              {stems && (
                <div className="space-y-1.5">
                  {stemControls.map((item) => {
                    const value = stemVolumes[item.key] ?? 1;
                    return (
                      <div key={item.key} className="flex items-center gap-2 text-[11px]">
                        <span className="w-8 shrink-0 text-secondary">{item.label}</span>
                        <input
                          aria-label={`${item.label}轨音量`}
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
                <span>人声自动闪避 BGM</span>
                <input
                  type="checkbox"
                  checked={autoDucking}
                  onChange={(e) => updateAutoDucking(e.target.checked)}
                  className="h-3.5 w-3.5"
                />
              </label>

              <div className="flex items-center justify-between gap-2 pt-1 text-[11px]">
                <span className="text-secondary">ASS 字幕导出</span>
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    disabled={editsDisabled || backendStatus !== "online"}
                    onClick={() => void handleDownloadSubtitles("target")}
                    className="rounded-[3px] border border-white/[0.08] bg-[#2C2C2C] px-2 py-0.5 text-primary transition-colors hover:bg-[#363636] disabled:opacity-40"
                  >
                    单语
                  </button>
                  <button
                    type="button"
                    disabled={editsDisabled || backendStatus !== "online"}
                    onClick={() => void handleDownloadSubtitles("bilingual")}
                    className="rounded-[3px] border border-white/[0.08] bg-[#2C2C2C] px-2 py-0.5 text-primary transition-colors hover:bg-[#363636] disabled:opacity-40"
                  >
                    双语
                  </button>
                </div>
              </div>
              {subNote && <div className="text-[11px] text-status-warning">{subNote}</div>}
            </div>

            <div className="p-3.5">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="text-[11px] font-medium text-secondary">台词跳转</div>
                {reviewPendingCount > 0 && (
                  <button
                    type="button"
                    disabled={editsDisabled}
                    onClick={() => confirmAllPendingReviews()}
                    className="rounded-[3px] border border-white/[0.08] bg-[#2C2C2C] px-2 py-0.5 font-mono tabular-nums text-[10px] text-primary transition-colors hover:bg-[#363636] disabled:opacity-40"
                  >
                    确认全部待校对 ({reviewPendingCount})
                  </button>
                )}
              </div>

              <div className="mb-2 grid grid-cols-4 gap-0.5 rounded-[4px] bg-[#181818] p-0.5 text-[10px]">
                {(
                  [
                    { id: "all", label: "全部", count: script.segments.length },
                    { id: "review", label: "待校对", count: reviewPendingCount },
                    { id: "pending", label: "待合成", count: pendingSynthCount },
                    { id: "issue", label: "异常", count: issueCount },
                  ] as const
                ).map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setSegmentFilter(tab.id)}
                    className={`rounded-[3px] px-1 py-1 font-mono tabular-nums transition-colors ${
                      segmentFilter === tab.id
                        ? "bg-[#333333] text-[#FFFFFF]"
                        : "text-secondary hover:text-primary"
                    }`}
                  >
                    {tab.label}({tab.count})
                  </button>
                ))}
              </div>

              <div className="space-y-0.5">
                {filteredSegments.length === 0 ? (
                  <div className="py-3 text-center text-[11px] text-muted">当前筛选下无片段</div>
                ) : (
                  filteredSegments.map((seg) => {
                    const needsReview = Boolean(
                      seg.alignment_needs_review || seg.speaker_needs_review,
                    );
                    const overflow = isSegmentOverflow(seg);
                    const tooShort = isSegmentTooShort(seg);
                    return (
                      <button
                        key={seg.id}
                        type="button"
                        onClick={() => {
                          selectSegment(seg.id);
                          setCurrentTime(seg.start);
                        }}
                        className="flex h-7 w-full items-center justify-between gap-1.5 rounded-[4px] px-2 text-left text-[11px] text-secondary transition-colors hover:bg-[#2C2C2C] hover:text-primary"
                      >
                        <span className="min-w-0 flex-1 truncate">
                          <span className="font-mono tabular-nums text-muted">#{seg.id}</span>{" "}
                          {seg.target_text}
                        </span>
                        <span className="flex shrink-0 items-center gap-1">
                          {needsReview && (
                            <span className="rounded-[2px] bg-[#2C2C2C] px-1 text-[9px] text-status-warning">
                              校对
                            </span>
                          )}
                          {overflow && (
                            <span className="rounded-[2px] bg-[#2C2C2C] px-1 text-[9px] text-status-error">
                              超时
                            </span>
                          )}
                          {!overflow && tooShort && (
                            <span className="rounded-[2px] bg-[#2C2C2C] px-1 text-[9px] text-status-warning">
                              过短
                            </span>
                          )}
                          <span className="font-mono tabular-nums text-[10px] text-muted">
                            {seg.start.toFixed(1)}s
                          </span>
                        </span>
                      </button>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
