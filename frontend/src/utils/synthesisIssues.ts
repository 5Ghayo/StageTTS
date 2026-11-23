import type { Segment, SynthesizeResponse } from "../types/script";

const TOO_LONG_RATIO = 1.15;
const TOO_SHORT_RATIO = 0.8;
const MIN_SECONDS = 1;

export type DurationState = "too_short" | "normal" | "too_long";

export function segmentDurationState(
  seg: Pick<Segment, "start" | "end" | "actual_duration">,
): DurationState {
  if (seg.actual_duration == null) return "normal";
  const slot = Math.max(0.05, seg.end - seg.start);
  const ratio = seg.actual_duration / slot;
  if (ratio > TOO_LONG_RATIO) return "too_long";
  if (ratio < TOO_SHORT_RATIO && slot >= MIN_SECONDS) return "too_short";
  return "normal";
}

export function isSegmentOverflow(
  seg: Pick<Segment, "start" | "end" | "actual_duration">,
): boolean {
  return segmentDurationState(seg) === "too_long";
}

export function isSegmentTooShort(
  seg: Pick<Segment, "start" | "end" | "actual_duration">,
): boolean {
  return segmentDurationState(seg) === "too_short";
}

export function synthesisIssues(result: SynthesizeResponse): string[] {
  return [
    ...result.overflow_segments.map((item) => {
      if (item.duration_state === "too_short") {
        return `片段 #${item.segment_id} 明显短于时间槽 ${Math.abs(item.overflow_seconds).toFixed(3)} 秒`;
      }
      return `片段 #${item.segment_id} 仍超出时间槽 ${item.overflow_seconds.toFixed(3)} 秒`;
    }),
    ...result.errors,
  ];
}
