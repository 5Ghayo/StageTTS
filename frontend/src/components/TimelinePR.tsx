import React, { useEffect, useMemo, useRef } from "react";
import WaveSurfer from "wavesurfer.js";
import { AlertCircle, ZoomIn, ZoomOut } from "lucide-react";
import { useProjectStore } from "../store/useProjectStore";
import type { EmotionType } from "../types/script";
import { normalizeSpeakerColor } from "../utils/speakerColor";

function formatShortTime(sec: number): string {
  const mins = Math.floor(sec / 60);
  const secs = (sec % 60).toFixed(1).padStart(4, "0");
  return `${String(mins).padStart(2, "0")}:${secs}`;
}

const EMOTION_LABELS: Record<EmotionType, string> = {
  neutral: "平淡",
  angry: "愤怒",
  happy: "喜悦",
  sad: "悲伤",
  whisper: "低语",
};

/**
 * 3 轨 Premiere 风格单色时间轴（5px 短刻度 + 极淡垂直虚线网格 + 纯色块片段）。
 */
export const TimelinePR: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const selectedSegmentId = useProjectStore((s) => s.selectedSegmentId);
  const currentTime = useProjectStore((s) => s.currentTime);
  const timelineZoom = useProjectStore((s) => s.timelineZoom);
  const trackStates = useProjectStore((s) => s.trackStates);
  const isSynthesizing = useProjectStore((s) => s.isSynthesizing);

  const selectSegment = useProjectStore((s) => s.selectSegment);
  const setCurrentTime = useProjectStore((s) => s.setCurrentTime);
  const setTimelineZoom = useProjectStore((s) => s.setTimelineZoom);
  const toggleTrackState = useProjectStore((s) => s.toggleTrackState);
  const synthesizeSegment = useProjectStore((s) => s.synthesizeSegment);

  const waveContainerRef = useRef<HTMLDivElement>(null);

  const duration = Math.max(10, script?.media.duration ?? 18.5);
  const trackWidthPx = Math.max(760, Math.ceil(duration * timelineZoom) + 80);

  useEffect(() => {
    const container = waveContainerRef.current;
    const vocalsUrl = script?.media.vocals_wav;
    if (!container || !vocalsUrl) {
      return;
    }
    let destroyed = false;
    let ws: WaveSurfer | null = null;
    try {
      ws = WaveSurfer.create({
        container,
        height: 44,
        waveColor: "#6E7681",
        progressColor: "#B0B8C1",
        cursorWidth: 0,
        interact: false,
        normalize: true,
      });
      ws.on("error", () => {});
      void ws.load(vocalsUrl).catch(() => {});
    } catch {
      // ignore offline audio fallback
    }
    return () => {
      if (!destroyed && ws) {
        destroyed = true;
        ws.destroy();
      }
    };
  }, [script?.media.vocals_wav]);

  const tickMarks = useMemo(() => {
    const step = timelineZoom >= 90 ? 1 : 2;
    const marks: number[] = [];
    for (let t = 0; t <= Math.ceil(duration); t += step) {
      marks.push(t);
    }
    return marks;
  }, [duration, timelineZoom]);

  const fallbackWaveBars = useMemo(() => {
    const barCount = 120;
    return Array.from({ length: barCount }).map((_, idx) => {
      const t = (idx / barCount) * duration;
      const inSeg = script?.segments.some(
        (seg) => t >= seg.start && t <= seg.end
      );
      const amp = inSeg
        ? 0.35 + 0.6 * Math.abs(Math.sin(idx * 1.37) * Math.cos(idx * 0.43))
        : 0.08 + 0.08 * Math.abs(Math.sin(idx * 0.7));
      return {
        height: Math.round(amp * 34),
        active: Boolean(inSeg),
      };
    });
  }, [duration, script?.segments]);

  if (!script) {
    return null;
  }

  const pendingCount = script.segments.filter(
    (seg) => !seg.keep_original && seg.status !== "ready"
  ).length;

  const handleRulerSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const offsetX = Math.max(0, e.clientX - rect.left);
    setCurrentTime(offsetX / timelineZoom);
  };

  return (
    <div className="flex h-full select-none flex-col bg-[#222222] text-[12px]">
      {/* 紧凑工具条（右侧留足 pr-4，避免 + 按钮贴边裁切） */}
      <div className="flex h-9 shrink-0 items-center justify-between gap-2 border-b border-white/[0.06] bg-[#222222] pl-3.5 pr-4">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="shrink-0 text-[13px] font-medium text-primary">
            时间轴
          </span>
          <span className="truncate font-mono tabular-nums text-[11px] text-secondary">
            <span className="text-primary">{formatShortTime(currentTime)}</span>{" "}
            / {formatShortTime(duration)}
          </span>
        </div>

        <div className="flex shrink-0 items-center gap-3">
          {/* 次操作按钮：#2C2C2C 灰底 + 极淡边框 + 6px #D4933D 琥珀色圆点 */}
          <button
            type="button"
            onClick={() => void synthesizeSegment()}
            disabled={isSynthesizing}
            className="flex h-7 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-2.5 py-1 text-[11px] text-primary transition-colors hover:bg-[#363636] disabled:opacity-50"
          >
            <span
              className="h-1.5 w-1.5 shrink-0 rounded-full"
              style={{
                backgroundColor: pendingCount > 0 ? "#D4933D" : "#4FA882",
              }}
            />
            <span className="font-mono tabular-nums">
              {isSynthesizing
                ? "合成中..."
                : pendingCount > 0
                  ? `合成待处理 (${pendingCount})`
                  : "重合成待处理"}
            </span>
          </button>

          <div className="flex shrink-0 items-center gap-1">
            <button
              type="button"
              onClick={() => setTimelineZoom(timelineZoom - 15)}
              className="flex h-6 w-6 shrink-0 items-center justify-center rounded-[4px] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
              title="缩小"
            >
              <ZoomOut className="h-3.5 w-3.5" strokeWidth={1.5} />
            </button>
            <input
              aria-label="时间轴缩放"
              type="range"
              min={30}
              max={180}
              step={6}
              value={timelineZoom}
              onChange={(e) => setTimelineZoom(Number(e.target.value))}
              className="w-20 shrink-0"
            />
            <button
              type="button"
              onClick={() => setTimelineZoom(timelineZoom + 15)}
              className="flex h-6 w-6 shrink-0 items-center justify-center rounded-[4px] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
              title="放大"
            >
              <ZoomIn className="h-3.5 w-3.5" strokeWidth={1.5} />
            </button>
          </div>
        </div>
      </div>

      {/* 3 轨主体 */}
      <div className="flex min-h-0 flex-1 overflow-y-auto overflow-x-hidden bg-[#161616]">
        {/* 左侧固定轨道头 (104px)，#222222 哑光灰，轨道间极弱 5% 分隔线 */}
        <div className="w-[104px] shrink-0 border-r border-white/[0.06] bg-[#222222]">
          {/* 标尺占位 (24px) */}
          <div className="h-6 border-b border-white/[0.05]" />

          {/* 视频 V1 (32px) */}
          <div className="flex h-8 items-center justify-between border-b border-white/[0.05] px-2.5">
            <span className="whitespace-nowrap text-[11px] font-medium text-secondary">
              视频 V1
            </span>
            <button
              type="button"
              onClick={() => toggleTrackState("videoMuted")}
              className={`rounded-[3px] px-1 font-mono text-[10px] font-medium transition-colors ${
                trackStates.videoMuted
                  ? "bg-[#2C2C2C] text-status-error"
                  : "text-muted hover:text-primary"
              }`}
            >
              M
            </button>
          </div>

          {/* 配音 A1 (72px) */}
          <div className="flex h-[72px] items-center justify-between border-b border-white/[0.05] px-2.5">
            <span className="whitespace-nowrap text-[11px] font-medium text-secondary">
              配音 A1
            </span>
            <div className="flex gap-1">
              <button
                type="button"
                onClick={() => toggleTrackState("dubMuted")}
                className={`rounded-[3px] px-1 font-mono text-[10px] font-medium transition-colors ${
                  trackStates.dubMuted
                    ? "bg-[#2C2C2C] text-status-error"
                    : "text-muted hover:text-primary"
                }`}
              >
                M
              </button>
              <button
                type="button"
                onClick={() => toggleTrackState("dubSolo")}
                className={`rounded-[3px] px-1 font-mono text-[10px] font-medium transition-colors ${
                  trackStates.dubSolo
                    ? "bg-[#2C2C2C] text-status-warning"
                    : "text-muted hover:text-primary"
                }`}
              >
                S
              </button>
            </div>
          </div>

          {/* 波形 / BGM A2 (56px) */}
          <div className="flex h-14 items-center justify-between px-2.5">
            <span className="whitespace-nowrap text-[11px] font-medium text-secondary">
              波形 / BGM
            </span>
            <button
              type="button"
              onClick={() => toggleTrackState("vocalsMuted")}
              className={`rounded-[3px] px-1 font-mono text-[10px] font-medium transition-colors ${
                trackStates.vocalsMuted
                  ? "bg-[#2C2C2C] text-status-error"
                  : "text-muted hover:text-primary"
              }`}
            >
              M
            </button>
          </div>
        </div>

        {/* 右侧可滚动轨道画布（#161616 深灰井） */}
        <div className="relative flex-1 overflow-x-auto overflow-y-hidden bg-[#161616] pl-2.5">
          <div
            className="relative min-h-full"
            style={{ width: `${trackWidthPx}px` }}
          >
            {/* 贯穿所有轨道的极淡垂直虚线时间网格（rgba(255,255,255,0.05)） */}
            <div className="pointer-events-none absolute inset-x-0 bottom-0 top-6 z-0">
              {tickMarks.map((sec) => (
                <div
                  key={`grid-${sec}`}
                  style={{ left: `${sec * timelineZoom}px` }}
                  className="absolute bottom-0 top-0 border-l border-dashed border-white/[0.05]"
                />
              ))}
            </div>

            {/* 时间标尺 (24px)：底部 5px 小短刻度，彻底去掉亮色方框 */}
            <div
              onClick={handleRulerSeek}
              className="relative z-10 h-6 cursor-pointer border-b border-white/[0.05] bg-[#1C1C1C]"
            >
              {tickMarks.map((sec) => (
                <div
                  key={sec}
                  style={{ left: `${sec * timelineZoom}px` }}
                  className="absolute bottom-0 top-0"
                >
                  <span className="block pl-1 pt-0.5 font-mono tabular-nums text-[10px] leading-none text-muted">
                    {sec}s
                  </span>
                  <span className="absolute bottom-0 left-0 h-[5px] w-px bg-white/[0.15]" />
                </div>
              ))}
            </div>

            {/* V1 (32px)：纯色块 #2A2A2A，无外框 */}
            <div className="relative z-10 h-8 border-b border-white/[0.05] py-1">
              <div
                style={{ width: `${duration * timelineZoom}px` }}
                className={`flex h-full items-center rounded-[4px] bg-[#2A2A2A] px-2.5 text-[11px] text-secondary ${
                  trackStates.videoMuted ? "opacity-40" : ""
                }`}
              >
                <span className="truncate font-mono tabular-nums">
                  源画面轨 · {duration.toFixed(1)}s
                </span>
              </div>
            </div>

            {/* A1 配音块轨 (72px)：纯色块 #2A2A2A + 仅左侧 3px 角色色条；选中才显示 1px 低饱和角色色外框 */}
            <div
              className={`relative z-10 h-[72px] border-b border-white/[0.05] ${
                trackStates.dubMuted ? "opacity-40" : ""
              }`}
            >
              {script.segments.map((seg) => {
                const character = script.characters[seg.speaker];
                const speakerAccent = normalizeSpeakerColor(character?.color);
                const slotDuration = Math.max(0.2, seg.end - seg.start);
                const left = seg.start * timelineZoom;
                const width = Math.max(80, slotDuration * timelineZoom);
                const isSelected = seg.id === selectedSegmentId;
                const isOverflow =
                  seg.actual_duration !== null &&
                  seg.actual_duration > slotDuration * 1.15;

                return (
                  <button
                    key={seg.id}
                    type="button"
                    onClick={() => {
                      selectSegment(seg.id);
                      setCurrentTime(seg.start);
                    }}
                    style={{
                      left: `${left}px`,
                      width: `${width}px`,
                      borderColor: isSelected ? speakerAccent : "transparent",
                      borderLeftColor: speakerAccent,
                    }}
                    className={`absolute inset-y-1.5 flex flex-col justify-between overflow-hidden rounded-[4px] border border-l-[3px] px-2 py-1 text-left transition-colors ${
                      isSelected
                        ? "bg-[#333333]"
                        : "bg-[#2A2A2A] hover:bg-[#303030]"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-1 text-[10px] leading-tight">
                      <span className="truncate font-medium text-secondary">
                        <span className="font-mono tabular-nums">
                          #{seg.id}
                        </span>{" "}
                        {character?.display_name ?? seg.speaker}
                      </span>
                      <span
                        className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                          seg.status === "ready"
                            ? "bg-status-ready"
                            : seg.status === "pending"
                              ? "bg-status-warning"
                              : "bg-status-error"
                        }`}
                      />
                    </div>

                    <p className="truncate text-[11px] leading-tight text-primary">
                      {seg.target_text}
                    </p>

                    <div className="flex items-center gap-1 text-[10px] leading-none">
                      <span className="rounded-[3px] bg-black/25 px-1 py-0.5 text-secondary">
                        {EMOTION_LABELS[seg.emotion]}
                      </span>
                      {seg.keep_original && (
                        <span className="rounded-[3px] bg-black/25 px-1 py-0.5 text-status-ready">
                          原声
                        </span>
                      )}
                      {isOverflow && (
                        <span className="flex items-center gap-0.5 rounded-[3px] bg-black/25 px-1 py-0.5 text-status-error">
                          <AlertCircle
                            className="h-2.5 w-2.5"
                            strokeWidth={1.5}
                          />
                          超时
                        </span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>

            {/* A2 中性波形与 BGM 包络轨 (56px) */}
            <div
              className={`relative z-10 flex h-14 items-center ${
                trackStates.vocalsMuted ? "opacity-35" : ""
              }`}
            >
              <div
                ref={waveContainerRef}
                className="pointer-events-none absolute inset-0 opacity-85"
                style={{ width: `${duration * timelineZoom}px` }}
              />
              <div
                style={{ width: `${duration * timelineZoom}px` }}
                className="pointer-events-none flex h-full items-center justify-between px-1"
              >
                {fallbackWaveBars.map((bar, i) => (
                  <div
                    key={i}
                    className={`w-[2px] rounded-[1px] ${
                      bar.active ? "bg-[#B0B8C1]/65" : "bg-[#6E7681]/50"
                    }`}
                    style={{ height: `${bar.height}px` }}
                  />
                ))}
              </div>
              {/* 极淡 BGM 增益参考虚线 */}
              <div
                className="pointer-events-none absolute left-0 border-t border-dashed border-white/[0.14]"
                style={{
                  width: `${duration * timelineZoom}px`,
                  bottom: `${Math.min(88, Math.max(12, script.media.bgm_volume * 45))}%`,
                }}
              />
              <span className="pointer-events-none absolute right-2 top-1 rounded-[4px] bg-black/60 px-1.5 py-0.5 font-mono tabular-nums text-[10px] text-secondary">
                伴奏 {Math.round(script.media.bgm_volume * 100)}%
              </span>
            </div>

            {/* 1.5px 珊瑚红播放头竖线（贯穿时间标尺与所有轨道） */}
            <div
              style={{ left: `${currentTime * timelineZoom}px` }}
              className="pointer-events-none absolute bottom-0 top-0 z-20 w-[1.5px] bg-[#D94A4A]"
            >
              <div className="-ml-[3.75px] h-2 w-[9px] rounded-b-[1px] bg-[#D94A4A]" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
