import { useMemo, useState } from "react";
import { formatDubbingTime } from "../DubbingRange";
import { segmentsInDubbingRange, useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import type { ProjectScript } from "../../types/script";

export function SpeakerReviewPanel({ script }: { script: ProjectScript }) {
  const { project_id: projectId } = script;
  const range = useProjectStore((s) => s.dubbingRange);
  const selected = useProjectStore((s) => s.selectedSegmentId);
  const busy = useProjectStore((s) => s.isLoading || s.isSynthesizing);
  const online = useProjectStore((s) => s.backendStatus === "online");
  const pipeline = useWorkflowStore((s) => s.isPipelineRunning);
  const [scope, setScope] = useState("all");
  const [message, setMessage] = useState("");
  const activeRange = range?.projectId === projectId ? range : null;
  const rows = useMemo(() => {
    const segments = segmentsInDubbingRange(script, scope === "range" ? activeRange : null);
    return [...segments].sort((a, b) => a.start - b.start || a.id - b.id);
  }, [script, scope, activeRange]);

  const characters = Object.entries(script.characters).filter(([key]) => key !== "UNKNOWN");
  const disabled = busy || pipeline || !online;
  const locate = (id: number, time: number) => {
    const { setIsPlaying, selectSegment, setCurrentTime } = useProjectStore.getState();
    setIsPlaying(false);
    selectSegment(id);
    setCurrentTime(time);
  };
  const confirm = async (id: number, speaker: string) => {
    if (!speaker || speaker === "UNKNOWN") return;
    const store = useProjectStore.getState();
    if (store.script?.project_id !== projectId || store.isLoading || store.isSynthesizing) return;
    store.updateSegmentLocal({ id, speaker, speaker_reviewed_by_user: true });
    const saved = await store.saveProjectScript();
    if (useProjectStore.getState().script?.project_id === projectId) {
      setMessage(
        saved ? `第 ${id + 1} 句的角色已保存` : "角色修改尚未写入项目，请恢复连接后再次保存。",
      );
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col text-[12px]">
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-white/10 p-2">
        <label className="flex items-center gap-1 text-secondary">
          <span className="sr-only">复核范围</span>
          <select
            value={scope}
            onChange={(e) => setScope(e.target.value)}
            className="rounded bg-[#303030] px-1 py-0.5 text-primary"
          >
            <option value="all">全部台词</option>
            {activeRange && <option value="range">仅配音选区</option>}
          </select>
        </label>
        <span className="text-secondary">
          已复核 {rows.filter((row) => row.speaker_reviewed_by_user).length}/{rows.length}
        </span>
      </div>
      {(message || !online) && (
        <p role="status" className="shrink-0 border-b border-white/10 px-2 py-1 text-secondary">
          {message || "连接后端后可保存角色修正。"}
        </p>
      )}
      <ul className="min-h-0 flex-1 overflow-y-auto">
        {rows.map((row) => (
          <li
            key={row.id}
            className={`space-y-1.5 border-b border-white/10 px-2 py-2 ${row.id === selected ? "bg-white/5" : ""}`}
          >
            <div className="flex items-baseline justify-between gap-2">
              <button
                type="button"
                className="shrink-0 font-mono text-blue-200 hover:underline"
                onClick={() => locate(row.id, row.speech_cue?.speech_start ?? row.start)}
              >
                {formatDubbingTime(row.speech_cue?.speech_start ?? row.start)}
              </button>
              <span className="text-[11px] text-secondary">
                {row.keep_original
                  ? "保留原声"
                  : row.status === "ready"
                    ? "已配音"
                    : row.status === "error"
                      ? "合成失败"
                      : "待配音"}
              </span>
            </div>
            <p className="break-words text-primary">{row.target_text || row.orig_text}</p>
            <div className="flex flex-wrap items-center gap-1.5">
              <select
                aria-label={`第 ${row.id + 1} 句角色`}
                value={row.speaker}
                disabled={disabled}
                onChange={(e) => void confirm(row.id, e.target.value)}
                className="min-w-20 max-w-full flex-1 rounded bg-[#303030] px-1 py-0.5 text-primary disabled:opacity-60"
              >
                {!characters.some(([key]) => key === row.speaker) && (
                  <option value={row.speaker} disabled>
                    {script.characters[row.speaker]?.display_name ?? "待分配角色"}
                  </option>
                )}
                {characters.map(([key, actor]) => (
                  <option key={key} value={key}>
                    {actor.display_name}
                  </option>
                ))}
              </select>
              <button
                type="button"
                disabled={disabled || row.speaker === "UNKNOWN" || !script.characters[row.speaker]}
                className="rounded px-1.5 py-0.5 text-blue-200 hover:bg-white/10 disabled:opacity-40"
                onClick={() => void confirm(row.id, row.speaker)}
              >
                {row.speaker_reviewed_by_user ? "已复核" : "确认角色"}
              </button>
              {!row.keep_original && row.status !== "ready" && (
                <button
                  type="button"
                  disabled={disabled || !row.target_text.trim()}
                  className="rounded px-1.5 py-0.5 text-secondary hover:bg-white/10 disabled:opacity-40"
                  onClick={async () =>
                    setMessage(
                      (await useProjectStore.getState().synthesizeSegment([row.id])).message,
                    )
                  }
                >
                  重配
                </button>
              )}
            </div>
          </li>
        ))}
        {rows.length === 0 && <li className="p-3 text-secondary">暂无可复核台词。</li>}
      </ul>
    </div>
  );
}
