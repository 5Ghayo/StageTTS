import { X } from "lucide-react";
import type React from "react";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import { CharacterPanel } from "./CharacterPanel";
import { TranslationSettings } from "./TranslationSettings";

/**
 * 240px 按需展开的左侧停靠抽屉（#222222 面板，极淡虚线分隔）。
 */
export const AssetBinPanel: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const selectSegment = useProjectStore((s) => s.selectSegment);
  const setCurrentTime = useProjectStore((s) => s.setCurrentTime);

  const leftDrawer = useWorkflowStore((s) => s.leftDrawer);
  const toggleLeftDrawer = useWorkflowStore((s) => s.toggleLeftDrawer);

  if (!script || leftDrawer === "none") {
    return null;
  }

  const stems = script.media.cinematic_stems;
  const mediaRows = [
    { label: "原视频流", ready: Boolean(script.media.source_video) },
    ...(stems
      ? [
          { label: "对白轨", ready: Boolean(stems.dialogue), url: stems.dialogue },
          { label: "歌声轨", ready: Boolean(stems.singing), url: stems.singing },
          { label: "音乐轨", ready: Boolean(stems.music), url: stems.music },
          { label: "音效轨", ready: Boolean(stems.effects), url: stems.effects },
          {
            label: "背景轨",
            ready: Boolean(script.media.bgm_wav),
            url: script.media.bgm_wav ?? undefined,
          },
        ]
      : [
          { label: "人声轨", ready: Boolean(script.media.vocals_wav) },
          { label: "背景轨", ready: Boolean(script.media.bgm_wav) },
        ]),
    { label: "成片输出", ready: Boolean(script.media.output_video) },
  ];

  return (
    <div className="flex h-full w-60 shrink-0 flex-col bg-[#222222] text-[12px]">
      <div className="flex h-8 shrink-0 items-center justify-between border-b border-white/[0.06] px-3.5">
        <span className="text-[13px] font-medium text-primary">
          {leftDrawer === "characters" ? "角色与情绪音色" : "项目媒体素材"}
        </span>
        <button
          type="button"
          title="收起侧栏"
          onClick={() => toggleLeftDrawer(leftDrawer)}
          className="flex h-6 w-6 items-center justify-center rounded-[4px] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
        >
          <X className="h-3.5 w-3.5" strokeWidth={1.5} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {leftDrawer === "characters" ? (
          <CharacterPanel key={script.project_id} script={script} />
        ) : (
          <div>
            <TranslationSettings key={script.project_id} script={script} />
            <div className="divide-y divide-white/[0.05] border-b border-dashed border-white/[0.08]">
              {mediaRows.map((row) => (
                <div key={row.label} className="flex h-8 items-center justify-between px-3.5">
                  <span className="text-primary">{row.label}</span>
                  <span className="flex items-center gap-1.5 text-[11px] text-secondary">
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${
                        row.ready ? "bg-status-ready" : "bg-muted"
                      }`}
                    />
                    {"url" in row && row.url ? (
                      <a
                        href={row.url}
                        target="_blank"
                        rel="noreferrer"
                        aria-label={`试听${row.label}`}
                        className="underline hover:text-primary"
                      >
                        试听
                      </a>
                    ) : (
                      <span>{row.ready ? "就绪" : "未生成"}</span>
                    )}
                  </span>
                </div>
              ))}
            </div>

            <div className="p-3.5">
              <div className="mb-1.5 text-[11px] font-medium text-secondary">台词跳转</div>
              <div className="space-y-0.5">
                {script.segments.map((seg) => (
                  <button
                    key={seg.id}
                    type="button"
                    onClick={() => {
                      selectSegment(seg.id);
                      setCurrentTime(seg.start);
                    }}
                    className="flex h-7 w-full items-center justify-between rounded-[4px] px-2 text-left text-[11px] text-secondary transition-colors hover:bg-[#2C2C2C] hover:text-primary"
                  >
                    <span className="truncate">
                      <span className="font-mono tabular-nums text-muted">#{seg.id}</span>{" "}
                      {seg.target_text}
                    </span>
                    <span className="ml-2 shrink-0 font-mono tabular-nums text-[10px] text-muted">
                      {seg.start.toFixed(1)}s
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
