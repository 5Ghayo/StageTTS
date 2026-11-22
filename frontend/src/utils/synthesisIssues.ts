import type { Segment, SynthesizeResponse } from "../types/script";

export function isSegmentOverflow(
  seg: Pick<Segment, "start" | "end" | "actual_duration" | "applied_speed_factor">,
): boolean {
  if (seg.actual_duration == null) return false;
  const slot = Math.max(0.05, seg.end - seg.start);
  return seg.applied_speed_factor == null
    ? seg.actual_duration > slot * 1.15
    : seg.actual_duration - slot > 0.02;
}

export function synthesisIssues(result: SynthesizeResponse): string[] {
  return [
    ...result.overflow_segments.map(
      (item) => `片段 #${item.segment_id} 仍超出时间槽 ${item.overflow_seconds.toFixed(3)} 秒`,
    ),
    ...result.errors,
  ];
}
