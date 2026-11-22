import React, { useEffect, useMemo, useRef } from "react";
import { Pause, Play } from "lucide-react";
import { useProjectStore } from "../store/useProjectStore";
import type { EmotionType } from "../types/script";
import { normalizeSpeakerColor } from "../utils/speakerColor";
import { TimelineAudio, useDubClips } from "./TimelineAudio";

const EMOTION_LABELS: Record<EmotionType, string> = {
  neutral: "平淡",
  angry: "愤怒",
  happy: "喜悦",
  sad: "悲伤",
  whisper: "低语",
};

/**
 * 虚拟演播舞台（模式 B，#222222 面板头尾 + #161616 深灰舞台视口）。
 */
export const VirtualStage: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const currentTime = useProjectStore((s) => s.currentTime);
  const isPlaying = useProjectStore((s) => s.isPlaying);
  const selectedSegmentId = useProjectStore((s) => s.selectedSegmentId);
  const trackStates = useProjectStore((s) => s.trackStates);
  const setIsPlaying = useProjectStore((s) => s.setIsPlaying);
  const stepSegment = useProjectStore((s) => s.stepSegment);

  const pixiContainerRef = useRef<HTMLDivElement>(null);

  const activeSegment = useMemo(() => {
    if (!script) {
      return null;
    }
    return (
      script.segments.find(
        (seg) => currentTime >= seg.start && currentTime <= seg.end
      ) ??
      script.segments.find((seg) => seg.id === selectedSegmentId) ??
      script.segments[0] ??
      null
    );
  }, [currentTime, script, selectedSegmentId]);

  const activeCharacter = activeSegment
    ? script?.characters[activeSegment.speaker]
    : undefined;

  const { upcomingClips } = useDubClips(currentTime);

  const rmsEnergy = useMemo(() => {
    if (!isPlaying || !activeSegment) {
      return 0.08;
    }
    const wave =
      Math.abs(Math.sin(currentTime * 14)) * 0.55 +
      Math.abs(Math.cos(currentTime * 23)) * 0.35;
    return Math.min(1, Math.max(0.12, wave));
  }, [activeSegment, currentTime, isPlaying]);

  useEffect(() => {
    void pixiContainerRef.current;
  }, [activeCharacter?.avatar?.model_url]);

  const emotionTint =
    activeSegment?.emotion === "angry"
      ? "#C95252"
      : activeSegment?.emotion === "happy"
        ? "#4FA882"
        : activeSegment?.emotion === "sad"
          ? "#5B8DEF"
          : activeSegment?.emotion === "whisper"
            ? "#9A86B8"
            : normalizeSpeakerColor(activeCharacter?.color);

  const mouthHeight = 4 + rmsEnergy * 18;
  const headTiltDeg =
    activeSegment?.action === "shake_head" && isPlaying
      ? Math.sin(currentTime * 10) * 6
      : activeSegment?.action === "nod" && isPlaying
        ? Math.sin(currentTime * 7) * 4
        : 0;

  return (
    <div className="flex h-full flex-col bg-[#222222] text-[12px]">
      {/* 轻量顶条 */}
      <div className="flex h-8 shrink-0 items-center justify-between border-b border-white/[0.06] bg-[#222222] px-3.5">
        <span className="text-[13px] font-medium text-primary">
          虚拟演播舞台
        </span>
        {activeSegment && (
          <span className="flex items-center gap-1.5 rounded-[4px] bg-white/[0.04] px-2 py-0.5 text-[11px] text-secondary">
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ backgroundColor: emotionTint }}
            />
            <span className="text-primary">
              {activeCharacter?.display_name ?? activeSegment.speaker}
            </span>
            <span>·</span>
            <span>{EMOTION_LABELS[activeSegment.emotion]}</span>
            {activeSegment.action && (
              <>
                <span>·</span>
                <span className="font-mono text-secondary">
                  {activeSegment.action}
                </span>
              </>
            )}
          </span>
        )}
      </div>

      {/* 舞台主画布 */}
      <div
        ref={pixiContainerRef}
        className="relative flex min-h-0 flex-1 flex-col items-center justify-center overflow-hidden bg-[#161616] p-6"
      >
        <TimelineAudio clips={upcomingClips} time={currentTime} playing={isPlaying} dubMuted={trackStates.dubMuted} />

        {/* 虚拟角色口型实时演出 */}
        <div
          className="flex flex-col items-center transition-transform duration-150"
          style={{ transform: `rotate(${headTiltDeg}deg)` }}
        >
          <svg width="152" height="152" viewBox="0 0 160 160">
            <circle
              cx="80"
              cy="80"
              r="66"
              fill="#222222"
              stroke={emotionTint}
              strokeWidth="2.5"
            />
            <line
              x1="46"
              y1={activeSegment?.emotion === "angry" ? "58" : "54"}
              x2="66"
              y2={activeSegment?.emotion === "angry" ? "63" : "54"}
              stroke="#E6E6E6"
              strokeWidth="2.5"
              strokeLinecap="round"
            />
            <line
              x1="94"
              y1={activeSegment?.emotion === "angry" ? "63" : "54"}
              x2="114"
              y2={activeSegment?.emotion === "angry" ? "58" : "54"}
              stroke="#E6E6E6"
              strokeWidth="2.5"
              strokeLinecap="round"
            />
            <circle cx="56" cy="72" r="5.5" fill="#E6E6E6" />
            <circle cx="104" cy="72" r="5.5" fill="#E6E6E6" />
            <ellipse
              cx="80"
              cy="104"
              rx="14"
              ry={mouthHeight / 2}
              fill={emotionTint}
              stroke="#E6E6E6"
              strokeWidth="1.75"
            />
          </svg>

          <div className="mt-3 flex h-5 items-end gap-1">
            {Array.from({ length: 10 }).map((_, idx) => {
              const barScale =
                0.25 +
                rmsEnergy *
                  (0.4 + 0.6 * Math.abs(Math.sin(idx * 0.9 + currentTime * 12)));
              return (
                <div
                  key={idx}
                  className="w-1 rounded-[1px] transition-all duration-75"
                  style={{
                    height: `${Math.round(barScale * 18)}px`,
                    backgroundColor: isPlaying ? "#B0B8C1" : "#6E7681",
                  }}
                />
              );
            })}
          </div>
        </div>

        {/* 无边框提词字幕 */}
        <div className="mt-6 max-w-lg rounded-[4px] bg-black/60 px-4 py-2 text-center text-[13px] text-primary">
          {activeSegment?.target_text ?? "选择台词片段开始演播"}
        </div>
      </div>

      {/* 演播控制栏 */}
      <div className="flex h-9 shrink-0 items-center justify-center gap-2 border-t border-white/[0.06] bg-[#222222] px-3.5">
        <button
          type="button"
          onClick={() => stepSegment(-1)}
          className="h-7 rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-2.5 py-1 text-[12px] text-primary transition-colors hover:bg-[#363636]"
        >
          上一句
        </button>
        <button
          type="button"
          onClick={() => setIsPlaying(!isPlaying)}
          className="flex h-7 items-center gap-1.5 rounded-[4px] bg-[#3A6EA5] px-3 py-1 text-[12px] font-medium text-[#FFFFFF] transition-colors hover:bg-[#467DB8]"
        >
          {isPlaying ? (
            <>
              <Pause className="h-3.5 w-3.5 fill-current" strokeWidth={1.5} />
              <span>暂停</span>
            </>
          ) : (
            <>
              <Play className="h-3.5 w-3.5 fill-current" strokeWidth={1.5} />
              <span>演播预览</span>
            </>
          )}
        </button>
        <button
          type="button"
          onClick={() => stepSegment(1)}
          className="h-7 rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-2.5 py-1 text-[12px] text-primary transition-colors hover:bg-[#363636]"
        >
          下一句
        </button>
      </div>
    </div>
  );
};
