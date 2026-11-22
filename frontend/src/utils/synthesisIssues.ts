import type { SynthesizeResponse } from "../types/script";

export function synthesisIssues(result: SynthesizeResponse): string[] {
  return [
    ...result.overflow_segments.map(
      (item) => `片段 #${item.segment_id} 仍超出时间槽 ${item.overflow_seconds.toFixed(3)} 秒`,
    ),
    ...result.errors,
  ];
}
