import { useMemo, useState } from "react";
import { segmentsInDubbingRange, useProjectStore } from "../store/useProjectStore";
import { useWorkflowStore } from "../store/useWorkflowStore";
import type { ProjectScript } from "../types/script";
import "./dubbingRange.css";

export function formatDubbingTime(time: number): string {
  const seconds = Math.floor(Math.max(0, time));
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, "0")}.${String(Math.floor((Math.max(0, time) % 1) * 100)).padStart(2, "0")}`;
}

export function DubbingRange({ script }: { script: ProjectScript }) {
  const range = useProjectStore((s) => s.dubbingRange);
  const available = useProjectStore(
    (s) => s.backendStatus === "online" && !s.isLoading && !s.isSynthesizing && !s.gpuBusy,
  );
  const pipeline = useWorkflowStore((s) => s.isPipelineRunning);
  const [message, setMessage] = useState("");
  const duration = script.media.duration;
  const active = range?.projectId === script.project_id ? range : null;
  const start = Math.min(active?.start ?? 0, duration);
  const end = Math.min(active?.end ?? duration, duration);
  const ids = useMemo(
    () =>
      active
        ? segmentsInDubbingRange(script, active)
            .filter((seg) => !seg.keep_original)
            .map((seg) => seg.id)
        : [],
    [script, active],
  );

  if (duration <= 0) return null;
  const update = (which: "start" | "end", value: number) => {
    const gap = Math.min(0.01, duration);
    const next =
      which === "start"
        ? { start: Math.max(0, Math.min(value, end - gap)), end }
        : { start, end: Math.min(duration, Math.max(value, start + gap)) };
    const { setDubbingRange, setIsPlaying, setCurrentTime } = useProjectStore.getState();
    setDubbingRange({ projectId: script.project_id, ...next });
    setIsPlaying(false);
    setCurrentTime(next[which]);
    setMessage("");
  };

  return (
    <section className="shrink-0 border-t border-white/[0.06] px-3 py-1.5" aria-label="配音范围">
      <div className="flex flex-wrap items-center justify-between gap-1">
        <span className="font-mono text-secondary">
          {formatDubbingTime(start)} – {formatDubbingTime(end)}
        </span>
        <div className="flex items-center gap-2">
          {active && (
            <button
              type="button"
              onClick={() => {
                useProjectStore.getState().setDubbingRange(null);
                setMessage("");
              }}
              className="text-secondary hover:text-primary"
            >
              清除
            </button>
          )}
          <button
            type="button"
            disabled={!available || pipeline || ids.length === 0}
            className="rounded bg-[#39475e] px-2 py-0.5 text-primary disabled:opacity-40"
            onClick={async () => {
              const { setIsPlaying, synthesizeSegment } = useProjectStore.getState();
              setIsPlaying(false);
              const result = await synthesizeSegment(ids);
              setMessage(result.message);
              if (
                useProjectStore.getState().script?.project_id === script.project_id &&
                !result.simulated
              ) {
                useWorkflowStore.getState().setRightTab("speakers");
              }
            }}
          >
            配音选区{active ? `（${ids.length} 句）` : ""}
          </button>
        </div>
      </div>
      <div
        className="dubbing-range relative mt-1 h-5"
        onPointerDown={(event) => {
          if (event.target !== event.currentTarget) return;
          const bounds = event.currentTarget.getBoundingClientRect();
          const value = (duration * (event.clientX - bounds.left)) / bounds.width;
          update(Math.abs(value - start) <= Math.abs(value - end) ? "start" : "end", value);
        }}
      >
        <div className="pointer-events-none absolute inset-x-0 top-2 h-1 rounded bg-white/15" />
        <div
          className="pointer-events-none absolute top-2 h-1 rounded bg-blue-400"
          style={{
            left: `${(start / duration) * 100}%`,
            width: `${((end - start) / duration) * 100}%`,
          }}
        />
        {(["start", "end"] as const).map((edge) => {
          const value = edge === "start" ? start : end;
          return (
            <input
              key={edge}
              type="range"
              aria-label={edge === "start" ? "配音起点" : "配音终点"}
              aria-valuetext={formatDubbingTime(value)}
              min={0}
              max={duration}
              step="any"
              value={value}
              onChange={(e) => update(edge, Number(e.target.value))}
            />
          );
        })}
      </div>
      <div className="min-h-4 text-[11px] text-secondary" role="status">
        {message || (active ? "边界碰到的台词按整句合成" : "拖动两端选取配音范围")}
      </div>
    </section>
  );
}
