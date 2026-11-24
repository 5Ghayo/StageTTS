import {
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  GitMerge,
  Pause,
  Play,
  Save,
  Scissors,
  Sparkles,
  Trash2,
} from "lucide-react";
import type React from "react";
import { useEffect, useRef, useState } from "react";
import { getCacheBustedUrl, useProjectStore } from "../store/useProjectStore";
import { useWorkflowStore } from "../store/useWorkflowStore";
import type { EmotionType } from "../types/script";
import { SourceEvidencePanel } from "./SourceEvidencePanel";
import { isSegmentOverflow, isSegmentTooShort } from "../utils/synthesisIssues";

const EMOTIONS: Array<{ id: EmotionType; label: string; dotColor: string }> = [
  { id: "neutral", label: "平淡", dotColor: "#9C9C9C" },
  { id: "angry", label: "愤怒", dotColor: "#C95252" },
  { id: "happy", label: "喜悦", dotColor: "#4FA882" },
  { id: "sad", label: "悲伤", dotColor: "#5B8DEF" },
  { id: "whisper", label: "低语", dotColor: "#9A86B8" },
];

const SPEAKER_FALLBACK_LABELS: Record<string, string> = {
  diarization_skipped: "已跳过说话人识别",
  hf_token_missing: "未配置说话人识别凭据",
  diarization_import_failed: "说话人识别组件加载失败",
  diarization_failed: "说话人识别失败",
  speaker_assignment_missing: "没有可靠的角色归属",
  word_speaker_missing: "部分字词的角色归属缺失",
};

function formatMiniTime(sec: number): string {
  const safe = Math.max(0, Number.isFinite(sec) ? sec : 0);
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

/**
 * 扁平化 NLE 单句精修面板（#222222 面板 + #181818 哑光凹槽输入 + 极淡虚线分组）。
 */
export const Inspector: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const selectedSegmentId = useProjectStore((s) => s.selectedSegmentId);
  const isSynthesizing = useProjectStore((s) => s.isSynthesizing);
  const isLoading = useProjectStore((s) => s.isLoading);
  const pipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);
  const audioVersionMap = useProjectStore((s) => s.audioVersionMap);

  const stepSegment = useProjectStore((s) => s.stepSegment);
  const updateSegmentLocal = useProjectStore((s) => s.updateSegmentLocal);
  const splitSegment = useProjectStore((s) => s.splitSegment);
  const mergeSegmentWithNext = useProjectStore((s) => s.mergeSegmentWithNext);
  const deleteSegment = useProjectStore((s) => s.deleteSegment);
  const saveProjectScript = useProjectStore((s) => s.saveProjectScript);
  const synthesizeSegment = useProjectStore((s) => s.synthesizeSegment);
  const currentTime = useProjectStore((s) => s.currentTime);

  const [synthNote, setSynthNote] = useState<string | null>(null);
  const [startDraft, setStartDraft] = useState("");
  const [endDraft, setEndDraft] = useState("");

  // 自定义极简试听条状态
  const audioRef = useRef<HTMLAudioElement>(null);
  const sourceAudioRef = useRef<HTMLAudioElement>(null);
  const [isAuditionPlaying, setIsAuditionPlaying] = useState(false);
  const [auditionTime, setAuditionTime] = useState(0);
  const [auditionDuration, setAuditionDuration] = useState(0);

  const segment = script?.segments.find((seg) => seg.id === selectedSegmentId) ?? null;

  const auditionUrl = segment
    ? getCacheBustedUrl(segment.tts_wav, audioVersionMap[segment.id])
    : null;

  const fallbackDuration = segment
    ? Math.max(0.1, segment.actual_duration ?? segment.end - segment.start)
    : 0;
  const activeSegmentId = segment?.id ?? null;

  useEffect(() => {
    if (activeSegmentId === null && auditionUrl === null) {
      return;
    }
    const el = audioRef.current;
    if (el) {
      el.pause();
      el.currentTime = 0;
    }
    setIsAuditionPlaying(false);
    setAuditionTime(0);
    setAuditionDuration(fallbackDuration);
  }, [activeSegmentId, auditionUrl, fallbackDuration]);

  useEffect(() => {
    if (activeSegmentId !== null) {
      sourceAudioRef.current?.pause();
    }
  }, [activeSegmentId]);

  const segStart = segment?.start;
  const segEnd = segment?.end;
  useEffect(() => {
    if (activeSegmentId !== null && segStart !== undefined && segEnd !== undefined) {
      setStartDraft(segStart.toFixed(2));
      setEndDraft(segEnd.toFixed(2));
    }
  }, [activeSegmentId, segStart, segEnd]);

  if (!script || !segment) {
    return (
      <div className="flex h-full items-center justify-center bg-[#222222] p-4 text-[12px] text-muted">
        请在时间轴中点击一个片段进行编辑
      </div>
    );
  }

  const speakers = Object.keys(script.characters);
  const editsDisabled = isLoading || isSynthesizing || pipelineRunning;
  const slotDuration = Math.max(0.1, segment.end - segment.start);
  const targetLang = segment.target_lang.replace(/^all_/, "").split("-")[0];
  const isCjk = ["zh", "ja", "ko", "yue"].includes(targetLang);
  const unit = isCjk ? "字" : "词";
  const usedUnits = isCjk
    ? Array.from(segment.target_text.replace(/\s+/g, "")).length
    : segment.target_text.trim().split(/\s+/).filter(Boolean).length;
  const recommendedUnits = Math.max(1, Math.floor(slotDuration * (isCjk ? 5 : 2.8)));
  const actualDuration = segment.actual_duration;
  const fillRatio = actualDuration ? actualDuration / slotDuration : 0;
  const isOverflow = isSegmentOverflow(segment);
  const isTooShort = isSegmentTooShort(segment);
  const limitRatio = segment.applied_speed_factor == null ? 1.15 : 1.0;

  const effectiveDuration = auditionDuration > 0 ? auditionDuration : fallbackDuration;
  const auditionProgress =
    effectiveDuration > 0
      ? Math.min(100, Math.max(0, (auditionTime / effectiveDuration) * 100))
      : 0;

  const toggleAuditionPlay = async () => {
    const el = audioRef.current;
    if (!el || !auditionUrl) return;
    if (!el.paused) {
      el.pause();
      setIsAuditionPlaying(false);
      return;
    }

    try {
      if (el.ended) {
        el.currentTime = 0;
      }
      await el.play();
      if (audioRef.current === el && !el.paused) setIsAuditionPlaying(true);
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "AbortError")) {
        setSynthNote("无法播放当前句，请检查配音文件或重试合成");
      }
    }
  };

  const handleAuditionSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0) {
      return;
    }
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const nextSec = ratio * effectiveDuration;
    setAuditionTime(nextSec);
    if (audioRef.current && Number.isFinite(audioRef.current.duration)) {
      audioRef.current.currentTime = nextSec;
    }
  };

  const handleSingleSynthesize = async () => {
    const res = await synthesizeSegment([segment.id]);
    setSynthNote(res.message);
    window.setTimeout(() => setSynthNote(null), 3500);
  };

  const commitStartDraft = () => {
    const trimmed = startDraft.trim();
    const val = Number(trimmed);
    if (!trimmed || !Number.isFinite(val) || val < 0 || segment.end - val < 0.2) {
      setStartDraft(segment.start.toFixed(2));
      return;
    }
    const rounded = Number(val.toFixed(2));
    setStartDraft(rounded.toFixed(2));
    if (rounded !== Number(segment.start.toFixed(2))) {
      updateSegmentLocal({ id: segment.id, start: rounded });
    }
  };

  const commitEndDraft = () => {
    const trimmed = endDraft.trim();
    const val = Number(trimmed);
    if (!trimmed || !Number.isFinite(val) || val - segment.start < 0.2) {
      setEndDraft(segment.end.toFixed(2));
      return;
    }
    const rounded = Number(val.toFixed(2));
    setEndDraft(rounded.toFixed(2));
    if (rounded !== Number(segment.end.toFixed(2))) {
      updateSegmentLocal({ id: segment.id, end: rounded });
    }
  };

  return (
    <div className="flex h-full flex-col justify-between overflow-hidden bg-[#222222] text-[12px]">
      <div className="flex-1 space-y-3 overflow-y-auto py-3 pl-3.5 pr-4">
        {/* 片段导航头与结构精修 */}
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <div className="flex min-w-0 items-baseline gap-2 truncate">
              <span className="shrink-0 text-[13px] font-medium text-primary">
                片段 #{segment.id}
              </span>
              <span className="truncate font-mono tabular-nums text-[11px] text-secondary">
                {slotDuration.toFixed(2)}s
              </span>
            </div>
            <div className="flex shrink-0 items-center gap-1">
              <button
                type="button"
                disabled={editsDisabled || slotDuration < 0.4}
                onClick={() => splitSegment(segment.id, currentTime)}
                className="flex h-6 w-6 items-center justify-center rounded-[4px] bg-[#2C2C2C] text-secondary transition-colors hover:bg-[#363636] hover:text-primary disabled:opacity-40"
                title="分割当前片段"
              >
                <Scissors className="h-3.5 w-3.5" strokeWidth={1.5} />
              </button>
              <button
                type="button"
                disabled={editsDisabled}
                onClick={() => mergeSegmentWithNext(segment.id)}
                className="flex h-6 w-6 items-center justify-center rounded-[4px] bg-[#2C2C2C] text-secondary transition-colors hover:bg-[#363636] hover:text-primary disabled:opacity-40"
                title="与下一句合并"
              >
                <GitMerge className="h-3.5 w-3.5" strokeWidth={1.5} />
              </button>
              <button
                type="button"
                disabled={editsDisabled}
                onClick={() => deleteSegment(segment.id)}
                className="flex h-6 w-6 items-center justify-center rounded-[4px] bg-[#2C2C2C] text-secondary transition-colors hover:bg-[#363636] hover:text-status-error disabled:opacity-40"
                title="删除片段"
              >
                <Trash2 className="h-3.5 w-3.5" strokeWidth={1.5} />
              </button>
              <button
                type="button"
                onClick={() => stepSegment(-1)}
                className="flex h-6 w-6 items-center justify-center rounded-[4px] bg-[#2C2C2C] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
                title="上一句"
              >
                <ChevronLeft className="h-3.5 w-3.5" strokeWidth={1.5} />
              </button>
              <button
                type="button"
                onClick={() => stepSegment(1)}
                className="flex h-6 w-6 items-center justify-center rounded-[4px] bg-[#2C2C2C] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
                title="下一句"
              >
                <ChevronRight className="h-3.5 w-3.5" strokeWidth={1.5} />
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <label className="flex items-center justify-between gap-1.5 rounded-[4px] border border-white/[0.06] bg-[#181818] px-2 py-1 text-[11px] text-secondary">
              <span>开始(s)</span>
              <input
                aria-label="片段开始时间"
                type="number"
                step={0.05}
                min={0}
                disabled={editsDisabled}
                value={startDraft}
                onChange={(e) => setStartDraft(e.target.value)}
                onBlur={commitStartDraft}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    commitStartDraft();
                    e.currentTarget.blur();
                  } else if (e.key === "Escape") {
                    setStartDraft(segment.start.toFixed(2));
                    e.currentTarget.blur();
                  }
                }}
                className="w-16 bg-transparent text-right font-mono tabular-nums text-primary outline-none"
              />
            </label>
            <label className="flex items-center justify-between gap-1.5 rounded-[4px] border border-white/[0.06] bg-[#181818] px-2 py-1 text-[11px] text-secondary">
              <span>结束(s)</span>
              <input
                aria-label="片段结束时间"
                type="number"
                step={0.05}
                min={Number((segment.start + 0.2).toFixed(2))}
                disabled={editsDisabled}
                value={endDraft}
                onChange={(e) => setEndDraft(e.target.value)}
                onBlur={commitEndDraft}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    commitEndDraft();
                    e.currentTarget.blur();
                  } else if (e.key === "Escape") {
                    setEndDraft(segment.end.toFixed(2));
                    e.currentTarget.blur();
                  }
                }}
                className="w-16 bg-transparent text-right font-mono tabular-nums text-primary outline-none"
              />
            </label>
          </div>
        </div>

        <div className="space-y-2 border-t border-dashed border-white/[0.08] pt-3">
          <div className="flex items-center justify-between gap-2 text-[11px] text-secondary">
            <label htmlFor="source-transcript">原文</label>
            {(script.media.vocals_wav || script.media.source_video) && (
              <>
                <audio
                  ref={sourceAudioRef}
                  src={script.media.vocals_wav || script.media.source_video || undefined}
                  preload="none"
                  onTimeUpdate={(e) => {
                    if (e.currentTarget.currentTime >= segment.end) e.currentTarget.pause();
                  }}
                />
                <button
                  type="button"
                  className="hover:text-primary"
                  onClick={async () => {
                    const audio = sourceAudioRef.current;
                    if (!audio) return;
                    if (!audio.paused) {
                      audio.pause();
                      return;
                    }
                    try {
                      audio.currentTime = segment.start;
                      await audio.play();
                    } catch {
                      setSynthNote("原声试听失败");
                    }
                  }}
                >
                  播放 / 暂停原声
                </button>
              </>
            )}
          </div>
          <textarea
            id="source-transcript"
            rows={2}
            disabled={editsDisabled}
            className="w-full rounded-[4px] border border-white/[0.07] bg-[#181818] px-2.5 py-1.5 text-primary"
            value={segment.orig_text}
            onChange={(e) => updateSegmentLocal({ id: segment.id, orig_text: e.target.value })}
          />
          <p className="text-[11px] text-secondary">
            时间粒度：{segment.alignment_granularity === "word" ? "词级" : segment.alignment_granularity === "segment" ? "句段级" : "未声明"}
          </p>
          {segment.segment_timing_needs_review !== false && (
            <div className="flex flex-wrap items-center justify-between gap-1 text-[11px]">
              <span className="text-status-warning">整段原文与音频边界待确认</span>
              <button
                type="button"
                className="text-secondary hover:text-primary"
                disabled={editsDisabled || !segment.orig_text.trim()}
                onClick={() => updateSegmentLocal({ id: segment.id, segment_timing_needs_review: false })}
              >
                确认句段时间与原文
              </button>
            </div>
          )}
          {segment.alignment_needs_review !== false && (
            <div className="flex flex-wrap items-center justify-between gap-1 text-[11px]">
              <span className="text-status-warning">词级对齐待校对{!segment.words?.length && "（无词级时间）"}</span>
              <button
                type="button"
                className="text-secondary hover:text-primary"
                disabled={editsDisabled || !segment.orig_text.trim() || !segment.words?.length}
                onClick={() =>
                  updateSegmentLocal({ id: segment.id, alignment_needs_review: false })
                }
              >
                确认已有词级对齐
              </button>
            </div>
          )}
          {!!segment.words?.length && (
            <details className="text-[11px] text-secondary">
              <summary className="cursor-pointer">词级对齐</summary>
              <ul className="mt-1 max-h-28 overflow-y-auto">
                {segment.words.map((word, index) => (
                  <li key={`${segment.id}-${index}-${word.word}`}>
                    {word.word} ·{" "}
                    {word.start == null || word.end == null
                      ? "未对齐"
                      : `${word.start.toFixed(3)}–${word.end.toFixed(3)}s`}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>

        <SourceEvidencePanel key={`${script.project_id}-${segment.id}`} script={script} segment={segment} />

        {/* 配音台词输入（#181818 哑光凹槽，无亮框） */}
        <div className="space-y-1.5 border-t border-dashed border-white/[0.08] pt-3">
          <div className="flex items-center justify-between gap-2 text-[11px]">
            <span className="whitespace-nowrap text-secondary">配音台词</span>
            <span
              className={`shrink-0 whitespace-nowrap font-mono tabular-nums ${
                usedUnits > recommendedUnits ? "text-status-warning" : "text-muted"
              }`}
            >
              {usedUnits} {unit} · 建议 ≤ {recommendedUnits} {unit}
            </span>
          </div>
          <textarea
            rows={2}
            disabled={editsDisabled}
            value={segment.target_text}
            onChange={(e) =>
              updateSegmentLocal({
                id: segment.id,
                target_text: e.target.value,
              })
            }
            className="w-full rounded-[4px] border border-white/[0.07] bg-[#181818] px-2.5 py-1.5 text-[12px] leading-relaxed text-primary transition-colors focus:border-white/25 focus:outline-none"
          />
        </div>

        {/* 说话人 & 舞台动作（#181818 哑光凹槽 + 严格单行标签） */}
        <div className="space-y-2 border-t border-dashed border-white/[0.08] pt-3">
          <div className="flex items-center gap-2">
            <label className="w-16 shrink-0 whitespace-nowrap text-[11px] text-secondary">
              说话人
            </label>
            <select
              value={segment.speaker}
              disabled={editsDisabled}
              onChange={(e) => updateSegmentLocal({ id: segment.id, speaker: e.target.value })}
              className="h-7 min-w-0 flex-1 truncate rounded-[4px] border border-white/[0.07] bg-[#181818] px-2 text-[12px] text-primary transition-colors focus:border-white/25 focus:outline-none"
            >
              {speakers.map((spk) => (
                <option key={spk} value={spk}>
                  {script.characters[spk]?.display_name ?? spk}
                </option>
              ))}
            </select>
          </div>
          {segment.speaker_needs_review !== false && script.characters[segment.speaker] && (
            <div className="flex flex-wrap items-center justify-between gap-1 text-[11px]">
              <span className="text-status-warning">
                {segment.speaker === "UNKNOWN" ? "说话人识别未完成，请指定角色" : "说话人待校对"}
              </span>
              <button
                type="button"
                className="text-secondary hover:text-primary"
                disabled={editsDisabled || segment.speaker === "UNKNOWN"}
                onClick={() => updateSegmentLocal({ id: segment.id, speaker: segment.speaker })}
              >
                确认当前说话人
              </button>
            </div>
          )}
          {segment.speaker_fallback_reason && segment.speaker_needs_review !== false && (
            <p className="text-[11px] text-secondary">
              识别状态：{SPEAKER_FALLBACK_LABELS[segment.speaker_fallback_reason] ?? "需要人工确认角色"}
            </p>
          )}

          <div className="flex items-center gap-2">
            <label className="w-16 shrink-0 whitespace-nowrap text-[11px] text-secondary">
              舞台动作
            </label>
            <input
              type="text"
              disabled={editsDisabled}
              value={segment.action ?? ""}
              onChange={(e) =>
                updateSegmentLocal({
                  id: segment.id,
                  action: e.target.value || null,
                })
              }
              placeholder="如点头 / 摇头"
              className="h-7 min-w-0 flex-1 rounded-[4px] border border-white/[0.07] bg-[#181818] px-2 text-[12px] text-primary placeholder:text-muted transition-colors focus:border-white/25 focus:outline-none disabled:opacity-50"
            />
          </div>
        </div>

        {/* 情绪与语速组（虚线分隔，12px 组间距） */}
        <div className="space-y-2.5 border-t border-dashed border-white/[0.08] pt-3">
          <div className="space-y-1.5">
            <span className="block whitespace-nowrap text-[11px] text-secondary">配音情绪</span>
            <div className="grid h-[26px] grid-cols-5 gap-0.5 rounded-[4px] bg-[#181818] p-0.5">
              {EMOTIONS.map((emo) => {
                const active = segment.emotion === emo.id;
                return (
                  <button
                    key={emo.id}
                    type="button"
                    disabled={editsDisabled}
                    onClick={() => updateSegmentLocal({ id: segment.id, emotion: emo.id })}
                    className={`flex items-center justify-center gap-1 whitespace-nowrap rounded-[3px] text-[11px] transition-colors disabled:opacity-50 ${
                      active
                        ? "bg-[#333333] font-medium text-[#FFFFFF]"
                        : "text-secondary hover:text-primary"
                    }`}
                  >
                    <span
                      className="h-1.5 w-1.5 shrink-0 rounded-full"
                      style={{
                        backgroundColor: emo.dotColor,
                        opacity: active ? 1 : 0.45,
                      }}
                    />
                    <span>{emo.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-16 shrink-0 whitespace-nowrap text-[11px] text-secondary">
              语速倍率
            </span>
            <input
              aria-label="语速倍率"
              type="range"
              disabled={editsDisabled}
              min={0.8}
              max={1.4}
              step={0.05}
              value={segment.speed_factor}
              onChange={(e) =>
                updateSegmentLocal({
                  id: segment.id,
                  speed_factor: Number(e.target.value),
                })
              }
              className="min-w-0 flex-1 disabled:opacity-50"
            />
            <span
              className={`w-10 shrink-0 whitespace-nowrap text-right font-mono tabular-nums text-[11px] ${
                segment.speed_factor > 1.15 ? "text-status-warning" : "text-primary"
              }`}
            >
              {segment.speed_factor.toFixed(2)}x
            </span>
          </div>

          {segment.speed_factor > 1.15 && (
            <p className="flex items-center gap-1 text-[11px] text-status-warning">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0" strokeWidth={1.5} />
              <span>加速过高易失真，超长句建议改用 AI 助手精简</span>
            </p>
          )}

          <label className="flex cursor-pointer items-center justify-between gap-2 text-[11px]">
            <span className="truncate text-secondary">保留原声（笑声、叹息不替换）</span>
            <input
              type="checkbox"
              disabled={editsDisabled}
              checked={segment.keep_original}
              onChange={(e) =>
                updateSegmentLocal({
                  id: segment.id,
                  keep_original: e.target.checked,
                })
              }
              className="h-3.5 w-3.5 shrink-0 rounded-[2px] disabled:opacity-50"
            />
          </label>
        </div>

        {/* 时长匹配与当前句试听组（虚线分隔） */}
        <div className="space-y-2.5 border-t border-dashed border-white/[0.08] pt-3">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between gap-2 text-[11px]">
              <span className="whitespace-nowrap text-secondary">时长匹配</span>
              <span
                className={`shrink-0 whitespace-nowrap font-mono tabular-nums ${
                  isOverflow || isTooShort
                    ? "text-status-error"
                    : actualDuration != null
                      ? "text-primary"
                      : "text-status-warning"
                }`}
              >
                {actualDuration != null
                  ? `${actualDuration.toFixed(1)}s / ${slotDuration.toFixed(1)}s`
                  : `待合成 (${slotDuration.toFixed(1)}s)`}
              </span>
            </div>
            <div className="relative h-1 w-full overflow-hidden rounded-[2px] bg-[#181818]">
              <div
                className={`h-full transition-all ${
                  isOverflow || isTooShort ? "bg-status-error" : "bg-[#A8A8A8]"
                }`}
                style={{
                  width: `${Math.min(100, Math.round((fillRatio / 1.3) * 100))}%`,
                }}
              />
              <div
                title={segment.applied_speed_factor == null ? "115% 超时预警线" : "100% 时间槽上限"}
                style={{ left: `${Math.round((limitRatio / 1.3) * 100)}%` }}
                className="absolute bottom-0 top-0 w-px bg-status-warning"
              />
            </div>
          </div>

          {/* 极简自定义 28px 试听条 */}
          {auditionUrl && (
            <div className="space-y-1">
              <span className="block whitespace-nowrap text-[11px] text-secondary">当前句试听</span>
              <audio
                key={auditionUrl}
                ref={audioRef}
                src={auditionUrl}
                preload="metadata"
                className="hidden"
                onLoadedMetadata={(e) => {
                  const dur = e.currentTarget.duration;
                  if (Number.isFinite(dur) && dur > 0) {
                    setAuditionDuration(dur);
                  }
                }}
                onTimeUpdate={(e) => setAuditionTime(e.currentTarget.currentTime)}
                onEnded={() => {
                  setIsAuditionPlaying(false);
                  setAuditionTime(0);
                }}
              />
              <div className="flex h-7 items-center gap-2.5 rounded-[4px] bg-[#2A2A2A] px-2.5">
                <button
                  type="button"
                  onClick={() => void toggleAuditionPlay()}
                  title={isAuditionPlaying ? "暂停试听" : "播放当前句"}
                  className="flex h-5 w-5 shrink-0 items-center justify-center rounded-[3px] text-primary transition-colors hover:bg-[#363636]"
                >
                  {isAuditionPlaying ? (
                    <Pause className="h-3.5 w-3.5 fill-current" strokeWidth={1.5} />
                  ) : (
                    <Play className="ml-0.5 h-3.5 w-3.5 fill-current" strokeWidth={1.5} />
                  )}
                </button>

                <div
                  role="slider"
                  aria-label="当前句试听进度"
                  aria-valuemin={0}
                  aria-valuemax={Math.round(effectiveDuration)}
                  aria-valuenow={Math.round(auditionTime)}
                  onClick={handleAuditionSeek}
                  className="flex h-4 flex-1 cursor-pointer items-center"
                >
                  <div className="h-[3px] w-full overflow-hidden rounded-[2px] bg-[#3E3E3E]">
                    <div
                      className="h-full bg-[#B5B5B5] transition-[width] duration-75"
                      style={{ width: `${auditionProgress}%` }}
                    />
                  </div>
                </div>

                <span className="shrink-0 whitespace-nowrap font-mono tabular-nums text-[11px] text-secondary">
                  {formatMiniTime(auditionTime)} / {formatMiniTime(effectiveDuration)}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 底部操作栏：柔和次按钮（保存） + 实心钢蓝主按钮（重合成试听） */}
      <div className="shrink-0 space-y-1.5 border-t border-white/[0.06] bg-[#222222] py-2 pl-3.5 pr-4">
        {synthNote && (
          <p className="rounded-[4px] bg-white/[0.05] px-2.5 py-1 text-[11px] text-secondary">
            {synthNote}
          </p>
        )}
        <div className="flex items-center gap-2">
          <button
            type="button"
            disabled={editsDisabled}
            onClick={() => void saveProjectScript()}
            className="flex h-7 flex-1 items-center justify-center gap-1.5 rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-2.5 py-1 text-[12px] font-medium text-primary transition-colors hover:bg-[#363636] disabled:opacity-50"
          >
            <Save className="h-3.5 w-3.5 text-secondary" strokeWidth={1.5} />
            <span>保存</span>
          </button>
          <button
            type="button"
            onClick={() => void handleSingleSynthesize()}
            disabled={editsDisabled || segment.keep_original}
            className="flex h-7 flex-1 items-center justify-center gap-1.5 rounded-[4px] bg-[#3A6EA5] px-2.5 py-1 text-[12px] font-medium text-[#FFFFFF] transition-colors hover:bg-[#467DB8] disabled:opacity-50"
          >
            <Sparkles className="h-3.5 w-3.5" strokeWidth={1.5} />
            <span>{isSynthesizing ? "合成中..." : "重合成试听"}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
