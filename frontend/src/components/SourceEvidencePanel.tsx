import { useState } from "react";
import { analyzePerformance } from "../services/api";
import { useProjectStore } from "../store/useProjectStore";
import { useWorkflowStore } from "../store/useWorkflowStore";
import type { ProjectScript, Segment } from "../types/script";

const STATUS_LABELS: Record<string, string> = {
  ok: "已分析",
  partial: "部分完成，仍需复核",
  not_run: "未运行",
  not_configured: "未配置",
  blocked: "证据不可靠，未分析",
  error: "分析失败",
};

export function SourceEvidencePanel({ script, segment }: { script: ProjectScript; segment: Segment }) {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const dirty = useProjectStore((state) => state.isDirty);
  const online = useProjectStore((state) => state.backendStatus === "online");
  const loading = useProjectStore((state) => state.isLoading || state.isSynthesizing);
  const pipelineRunning = useWorkflowStore((state) => state.isPipelineRunning);
  const setScript = useProjectStore((state) => state.setScript);
  const evidence = script.evidence;
  const annotations = evidence?.annotations?.filter((item) =>
    (segment.source_turn_ids ?? []).some((id) => (item.source_turn_ids ?? []).includes(id))
    || (segment.source_token_ids ?? []).some((id) => (item.source_token_ids ?? []).includes(id))
  ) ?? [];

  async function analyze(emotion: boolean) {
    setBusy(true);
    setNote(null);
    try {
      const result = await analyzePerformance(script.project_id, [segment.id], emotion);
      // Do not overwrite local edits or a newly selected project while analysis ran.
      const current = useProjectStore.getState();
      if (current.script?.project_id === script.project_id && !current.isDirty) setScript(result.script);
      setNote([STATUS_LABELS[result.status] ?? result.status, ...(result.errors ?? []), ...(result.review_reasons ?? [])].join(" · "));
    } catch (error) {
      setNote(error instanceof Error ? error.message : "源声分析失败");
    } finally {
      setBusy(false);
    }
  }

  const disabled = busy || dirty || !online || loading || pipelineRunning;
  return (
    <div className="space-y-2 border-t border-dashed border-white/[0.08] pt-3 text-[11px]">
      <div className="font-medium text-primary">源声证据与表演分析</div>
      {!script.source_lang && <p className="text-status-warning">原声语言待设置：请在翻译设置中填写真实语种，再生成自动参考音。</p>}
      {evidence ? (
        <p className="text-secondary">
          源词 {(segment.source_token_ids ?? []).length} · 声学轮次 {(segment.source_turn_ids ?? []).length}
          {(evidence.review_reasons ?? []).length > 0 && " · 证据待复核"}
        </p>
      ) : (
        <p className="text-secondary">旧项目暂无独立源证据。未运行分析不代表中性情绪。</p>
      )}
      {annotations.map((annotation) => (
        <div key={annotation.id} className="space-y-1 rounded border border-white/[0.07] p-2 text-secondary">
          <p>{annotation.start.toFixed(2)}–{annotation.end.toFixed(2)}s · {annotation.status ? STATUS_LABELS[annotation.status] ?? annotation.status : "未记录分析状态"} · {annotation.review?.needs_review !== false ? "待复核" : "已记录结果"}</p>
          {(annotation.emotions ?? []).length > 0 && (
            <p>情绪：{(annotation.emotions ?? []).map((item) => `${item.label}${item.score == null ? "" : ` ${item.score.toFixed(3)}`}`).join(" / ")}（模型原始分数）</p>
          )}
          {(annotation.prosody ?? []).filter((item) => item.value != null).map((item) => (
            <span key={item.name} className="mr-2 inline-block">{item.name}: {item.value?.toFixed(3)} {item.unit}</span>
          ))}
          {(annotation.prosody ?? []).filter((item) => item.value == null).map((item) => (
            <p key={item.name} className="text-muted">{item.name}：未测得{item.reason ? `（${item.reason}）` : ""}</p>
          ))}
          {(annotation.review?.reasons ?? []).length > 0 && <p className="text-status-warning">{(annotation.review?.reasons ?? []).join(" / ")}</p>}
        </div>
      ))}
      <div className="flex flex-wrap gap-2">
        <button type="button" disabled={disabled} className="rounded border border-white/[0.1] px-2 py-1 disabled:opacity-40" onClick={() => void analyze(false)}>分析源声韵律</button>
        <button type="button" disabled={disabled} className="rounded border border-white/[0.1] px-2 py-1 disabled:opacity-40" onClick={() => void analyze(true)}>分析情绪与韵律</button>
      </div>
      {dirty && <p className="text-secondary">先保存编辑，再分析对应源声。</p>}
      <p className="text-muted">仅使用已配置的本地分析环境与权重；缺少配置时显示未运行，不自动下载。</p>
      {note && <p role="status" className="text-status-warning">{note}</p>}
    </div>
  );
}
