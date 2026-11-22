import { Pause, Play, SkipBack, SkipForward, Volume2, VolumeX } from "lucide-react";
import type React from "react";
import { useMemo, useState } from "react";
import { useProjectStore } from "../store/useProjectStore";
import type { EmotionType } from "../types/script";
import { normalizeSpeakerColor } from "../utils/speakerColor";
import { TimelineAudio, useDubClips, useSyncedMedia } from "./TimelineAudio";

function formatSmpte(seconds: number, fps = 30): string {
  const safe = Math.max(0, seconds);
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  const frames = Math.floor((safe % 1) * fps);
  return [mins, secs, frames].map((n) => String(n).padStart(2, "0")).join(":");
}

const EMOTION_LABELS: Record<EmotionType, string> = {
  neutral: "平淡",
  angry: "愤怒",
  happy: "喜悦",
  sad: "悲伤",
  whisper: "低语",
};

/**
 * NLE 风格视频监视器（模式 A，#222222 面板头尾 + #161616 深灰视口）。
 */
export const VideoPreview: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const currentTime = useProjectStore((s) => s.currentTime);
  const isPlaying = useProjectStore((s) => s.isPlaying);
  const selectedSegmentId = useProjectStore((s) => s.selectedSegmentId);
  const trackStates = useProjectStore((s) => s.trackStates);
  const setCurrentTime = useProjectStore((s) => s.setCurrentTime);
  const setIsPlaying = useProjectStore((s) => s.setIsPlaying);
  const stepSegment = useProjectStore((s) => s.stepSegment);
  const toggleTrackState = useProjectStore((s) => s.toggleTrackState);
  const updateBgmVolume = useProjectStore((s) => s.updateBgmVolume);

  const [failedState, setFailedState] = useState<{ key: string; urls: string[] }>({
    key: "",
    urls: [],
  });

  const duration = script?.media.duration ?? 18.5;
  const activeSegment = useMemo(() => {
    if (!script) {
      return null;
    }
    return (
      script.segments.find((seg) => currentTime >= seg.start && currentTime <= seg.end) ??
      script.segments.find((seg) => seg.id === selectedSegmentId) ??
      null
    );
  }, [currentTime, script, selectedSegmentId]);

  const activeSpeaker = activeSegment ? script?.characters[activeSegment.speaker] : undefined;
  const speakerColor = normalizeSpeakerColor(activeSpeaker?.color);

  const source = script?.media.source_video ?? null;
  const output = script?.media.output_video ?? null;
  const mediaKey = `${script?.project_id ?? ""}\0${source ?? ""}\0${output ?? ""}`;
  const failedUrls = failedState.key === mediaKey ? failedState.urls : [];
  const videoSrc = [source, output].find((url) => url && !failedUrls.includes(url)) ?? null;
  const video = useSyncedMedia<HTMLVideoElement>(videoSrc, currentTime, isPlaying);
  const { replacesSource, upcomingClips } = useDubClips(currentTime);
  const bgm = script?.media.bgm_wav;
  const vocals = bgm || !videoSrc || videoSrc !== source ? script?.media.vocals_wav : null;
  const muteSource = trackStates.dubSolo || replacesSource;
  const muteVideo =
    trackStates.videoMuted || muteSource || Boolean(vocals || bgm) || trackStates.vocalsMuted;

  return (
    <div className="flex h-full flex-col bg-[#222222] text-[12px]">
      {/* 轻量标题条 */}
      <div className="flex h-8 shrink-0 items-center justify-between border-b border-white/[0.06] bg-[#222222] px-3.5">
        <div className="flex min-w-0 items-center gap-2">
          <span className="text-[13px] font-medium text-primary">视频预览</span>
          {output && (
            <a
              href={output}
              target="_blank"
              rel="noreferrer"
              className="text-secondary hover:text-primary"
            >
              查看上次导出
            </a>
          )}
        </div>
        {activeSegment && (
          <span className="flex items-center gap-1.5 rounded-[4px] bg-white/[0.04] px-2 py-0.5 text-[11px] text-primary">
            <span
              className="h-1.5 w-1.5 shrink-0 rounded-full"
              style={{ backgroundColor: speakerColor }}
            />
            <span>
              {activeSpeaker?.display_name ?? activeSegment.speaker} ·{" "}
              {EMOTION_LABELS[activeSegment.emotion]}
            </span>
          </span>
        )}
      </div>

      {/* 16:9 监视器视口（#161616 深灰槽底，无边框） */}
      <div className="flex min-h-0 flex-1 items-center justify-center bg-[#161616] p-2.5">
        <div className="relative aspect-video max-h-full w-full max-w-full overflow-hidden rounded-[2px] bg-[#111111]">
          {videoSrc && (
            <video
              {...video}
              src={videoSrc}
              muted={muteVideo}
              onError={() =>
                setFailedState((prev) => ({
                  key: mediaKey,
                  urls: prev.key === mediaKey ? [...prev.urls, videoSrc] : [videoSrc],
                }))
              }
              className="h-full w-full object-contain"
            />
          )}

          {!videoSrc && (
            <div className="flex h-full w-full flex-col items-center justify-center px-6 text-center">
              <div
                className="mb-2 h-1.5 w-1.5 rounded-full"
                style={{ backgroundColor: speakerColor }}
              />
              <p className="text-[13px] font-medium text-primary">
                {activeSpeaker?.display_name ?? "视频监视器"}
                {activeSegment
                  ? activeSegment.keep_original
                    ? " · 原声片段"
                    : ` · ${EMOTION_LABELS[activeSegment.emotion]}情绪`
                  : ""}
              </p>
              <p className="mt-1 text-[11px] text-muted">
                {script?.media.output_video ? "成片已就绪" : "实时台词与节奏预览"}
              </p>
            </div>
          )}

          {/* 无边框时间码浮标 */}
          <div className="pointer-events-none absolute left-2.5 top-2 rounded-[4px] bg-black/60 px-2 py-0.5 font-mono tabular-nums text-[11px] text-primary">
            {formatSmpte(currentTime)}
          </div>

          {/* 无边框字幕条 */}
          {activeSegment && (
            <div className="pointer-events-none absolute inset-x-8 bottom-3 flex flex-col items-center gap-1 text-center">
              <div className="rounded-[4px] bg-black/65 px-3 py-1 text-[13px] font-medium text-[#FFFFFF]">
                {activeSegment.target_text}
              </div>
              {activeSegment.orig_text && (
                <div className="rounded-[4px] bg-black/50 px-2 py-0.5 text-[11px] text-secondary">
                  {activeSegment.orig_text}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      <TimelineAudio
        clips={upcomingClips}
        time={currentTime}
        playing={isPlaying}
        dubMuted={trackStates.dubMuted}
        vocals={vocals}
        muteVocals={muteSource || trackStates.vocalsMuted}
        bgm={bgm}
        muteBgm={trackStates.dubSolo || trackStates.bgmMuted}
        bgmVolume={script?.media.bgm_volume}
      />

      {/* 紧凑单色播控条 */}
      <div className="flex h-9 shrink-0 items-center justify-between gap-2 border-t border-white/[0.06] bg-[#222222] pl-3.5 pr-4">
        <div className="shrink-0 whitespace-nowrap font-mono tabular-nums text-[11px] text-secondary">
          <span className="text-primary">{formatSmpte(currentTime)}</span> / {formatSmpte(duration)}
        </div>

        <div className="flex shrink-0 items-center gap-1">
          <button
            type="button"
            title="回到开头"
            onClick={() => setCurrentTime(0)}
            className="flex h-7 w-7 items-center justify-center rounded-[4px] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
          >
            <SkipBack className="h-3.5 w-3.5" strokeWidth={1.5} />
          </button>
          <button
            type="button"
            title="上一句 (←)"
            onClick={() => stepSegment(-1)}
            className="h-7 whitespace-nowrap rounded-[4px] px-2 text-[12px] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
          >
            上一句
          </button>
          <button
            type="button"
            title="播放 / 暂停 (Space)"
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] text-[#FFFFFF] transition-colors hover:bg-[#363636]"
          >
            {isPlaying ? (
              <Pause className="h-3.5 w-3.5 fill-current" strokeWidth={1.5} />
            ) : (
              <Play className="ml-0.5 h-3.5 w-3.5 fill-current" strokeWidth={1.5} />
            )}
          </button>
          <button
            type="button"
            title="下一句 (→)"
            onClick={() => stepSegment(1)}
            className="h-7 whitespace-nowrap rounded-[4px] px-2 text-[12px] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
          >
            下一句
          </button>
          <button
            type="button"
            title="跳到末尾"
            onClick={() => setCurrentTime(duration)}
            className="flex h-7 w-7 items-center justify-center rounded-[4px] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
          >
            <SkipForward className="h-3.5 w-3.5" strokeWidth={1.5} />
          </button>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          <button
            type="button"
            title="配音静音"
            onClick={() => toggleTrackState("dubMuted")}
            className="flex h-7 w-7 items-center justify-center rounded-[4px] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
          >
            {trackStates.dubMuted ? (
              <VolumeX className="h-3.5 w-3.5 text-status-error" strokeWidth={1.5} />
            ) : (
              <Volume2 className="h-3.5 w-3.5" strokeWidth={1.5} />
            )}
          </button>
          <span className="whitespace-nowrap text-[11px] text-secondary">伴奏</span>
          <input
            aria-label="伴奏音量"
            type="range"
            min={0}
            max={1.5}
            step={0.05}
            value={script?.media.bgm_volume ?? 0.75}
            onChange={(e) => updateBgmVolume(Number(e.target.value))}
            className="w-14"
          />
          <span className="w-8 whitespace-nowrap text-right font-mono tabular-nums text-[11px] text-secondary">
            {Math.round((script?.media.bgm_volume ?? 0.75) * 100)}%
          </span>
        </div>
      </div>
    </div>
  );
};
