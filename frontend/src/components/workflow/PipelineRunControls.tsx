import { Pause, Play, Square } from "lucide-react";
import { useWorkflowStore } from "../../store/useWorkflowStore";

/** Shared controls stay available even when the user leaves the graph view. */
export function PipelineRunControls() {
  const running = useWorkflowStore((s) => s.isPipelineRunning);
  const requested = useWorkflowStore((s) => s.isPauseRequested);
  const paused = useWorkflowStore((s) => s.isPipelinePaused);
  const stopping = useWorkflowStore((s) => s.isStopRequested);
  const pause = useWorkflowStore((s) => s.pausePipeline);
  const resume = useWorkflowStore((s) => s.resumePipeline);
  const stop = useWorkflowStore((s) => s.stopPipeline);
  if (!running) return null;

  const buttonClass = "flex h-7 items-center gap-1.5 whitespace-nowrap rounded-[4px] border border-white/[0.1] bg-[#2C2C2C] px-2.5 text-[12px] text-primary hover:bg-[#363636] disabled:opacity-50";
  return (
    <div className="flex flex-wrap items-center gap-1" role="group" aria-label="管线运行控制">
      <span role="status" className="px-1 text-[11px] text-secondary">
        {stopping ? "等待当前步骤结束后停止" : paused ? "已暂停" : requested ? "等待当前步骤完成后暂停" : "运行中"}
      </span>
      <button
        type="button"
        onClick={requested ? resume : pause}
        disabled={stopping}
        title={requested ? "继续剩余步骤；等待期间可取消暂停请求" : "当前步骤完成并保存后暂停，后续步骤等待继续"}
        className={buttonClass}
      >
        {requested ? <Play className="h-3.5 w-3.5" /> : <Pause className="h-3.5 w-3.5" />}
        {paused ? "继续" : requested ? "取消暂停" : "暂停"}
      </button>
      <button
        type="button"
        onClick={stop}
        disabled={stopping}
        title="保留已完成结果；当前请求结束后停止后续调度"
        className={buttonClass}
      >
        <Square className="h-3.5 w-3.5" />
        结束流程
      </button>
    </div>
  );
}
