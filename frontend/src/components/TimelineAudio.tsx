import { useEffect, useMemo, useRef } from "react";
import { getCacheBustedUrl, useProjectStore } from "../store/useProjectStore";
import type { Segment } from "../types/script";

const EMPTY_SEGMENTS: Segment[] = [];

function syncMediaElement(
  media: HTMLMediaElement | null,
  src: string | null,
  time: number,
  playing: boolean,
  rate: number,
) {
  if (!media || !src) return;
  if (media.playbackRate !== rate) media.playbackRate = rate;
  const position = Math.max(0, time);
  const inRange = time >= 0 && (!Number.isFinite(media.duration) || time < media.duration);
  if (
    media.readyState >= HTMLMediaElement.HAVE_METADATA &&
    !media.seeking &&
    inRange &&
    (media.ended || Math.abs(media.currentTime - position) > (playing ? 0.12 : 0.01))
  ) {
    media.currentTime = position;
  }
  if (playing && inRange && media.paused) {
    void media.play().catch(() => {});
  } else if ((!playing || !inRange) && !media.paused) {
    media.pause();
  }
}

export function useSyncedMedia<T extends HTMLMediaElement>(
  src: string | null,
  time: number,
  playing: boolean,
  rate = 1,
) {
  const ref = useRef<T>(null);

  useEffect(() => {
    syncMediaElement(ref.current, src, time, playing, rate);
  }, [src, time, playing, rate]);
  return {
    ref,
    onLoadedMetadata: () => syncMediaElement(ref.current, src, time, playing, rate),
  };
}

let sharedAudioContext: AudioContext | null = null;
const mediaGainNodes = new WeakMap<HTMLAudioElement, GainNode>();

function getOrCreateGainNode(el: HTMLAudioElement): { ctx: AudioContext; gain: GainNode } | null {
  if (typeof AudioContext === "undefined") return null;
  if (!sharedAudioContext || sharedAudioContext.state === "closed") {
    sharedAudioContext = new AudioContext();
  }
  let gain = mediaGainNodes.get(el);
  if (!gain) {
    const source = sharedAudioContext.createMediaElementSource(el);
    gain = sharedAudioContext.createGain();
    source.connect(gain);
    gain.connect(sharedAudioContext.destination);
    mediaGainNodes.set(el, gain);
  }
  return { ctx: sharedAudioContext, gain };
}

function PreviewAudio({
  src,
  time,
  playing,
  muted,
  volume = 1,
  rate = 1,
}: {
  src: string;
  time: number;
  playing: boolean;
  muted: boolean;
  volume?: number;
  rate?: number;
}) {
  const media = useSyncedMedia<HTMLAudioElement>(src, time, playing, rate);

  useEffect(() => {
    const el = media.ref.current;
    if (!el) return;
    const clamped = Math.max(0, volume);
    let audioGraph: { ctx: AudioContext; gain: GainNode } | null = null;
    if (clamped > 1 || mediaGainNodes.has(el)) {
      try {
        audioGraph = getOrCreateGainNode(el);
      } catch {
        // Fallback to HTMLMediaElement.volume when Web Audio is unavailable
      }
    }
    if (audioGraph) {
      el.volume = 1;
      audioGraph.gain.gain.value = clamped;
      if (playing && audioGraph.ctx.state === "suspended") {
        void audioGraph.ctx.resume().catch(() => {});
      }
    } else {
      el.volume = Math.min(1, clamped);
    }
  }, [media.ref, playing, volume]);

  return <audio {...media} src={src} preload="auto" muted={muted} />;
}

export function useDubClips(time: number) {
  const segments = useProjectStore((s) => s.script?.segments ?? EMPTY_SEGMENTS);
  const versions = useProjectStore((s) => s.audioVersionMap);
  const clips = useMemo(
    () =>
      segments.flatMap((segment) => {
        if (segment.keep_original || segment.status !== "ready" || !segment.tts_wav) return [];
        const src = getCacheBustedUrl(segment.tts_wav, versions[segment.id]) ?? segment.tts_wav;
        const slot = segment.end - segment.start;
        const rate = segment.applied_speed_factor == null ? segment.speed_factor : 1;
        const audibleEnd = Math.min(
          segment.end,
          segment.start + (segment.actual_duration ?? slot) / rate,
        );
        return [{ segment, src, rate, end: audibleEnd }];
      }),
    [segments, versions],
  );
  return {
    replacesSource: clips.some(({ segment, end }) => time >= segment.start && time < end),
    upcomingClips: clips.filter(({ segment, end }) => time >= segment.start - 0.5 && time < end),
  };
}

export function TimelineAudio({
  clips,
  time,
  playing,
  dubMuted,
  vocals,
  bgm,
  muteVocals,
  muteBgm,
  bgmVolume,
}: {
  clips: ReturnType<typeof useDubClips>["upcomingClips"];
  time: number;
  playing: boolean;
  dubMuted: boolean;
  vocals?: string | null;
  bgm?: string | null;
  muteVocals?: boolean;
  muteBgm?: boolean;
  bgmVolume?: number;
}) {
  return (
    <>
      {vocals && (
        <PreviewAudio src={vocals} time={time} playing={playing} muted={Boolean(muteVocals)} />
      )}
      {bgm && (
        <PreviewAudio
          src={bgm}
          time={time}
          playing={playing}
          muted={Boolean(muteBgm)}
          volume={bgmVolume}
        />
      )}
      {clips.map(({ segment, src, rate }) => (
        <PreviewAudio
          key={`${segment.id}:${src}`}
          src={src}
          time={(time - segment.start) * rate}
          playing={playing}
          muted={dubMuted}
          rate={rate}
        />
      ))}
    </>
  );
}
