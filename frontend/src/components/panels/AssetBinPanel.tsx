import React from "react";
import { X } from "lucide-react";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import type { EmotionType } from "../../types/script";
import { normalizeSpeakerColor } from "../../utils/speakerColor";

const EMOTION_LABELS: Record<EmotionType, string> = {
  neutral: "平淡",
  angry: "愤怒",
  happy: "喜悦",
  sad: "悲伤",
  whisper: "低语",
};

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

  const mediaRows = [
    { label: "原视频流", ready: Boolean(script.media.source_video) },
    { label: "纯人声轨", ready: Boolean(script.media.vocals_wav) },
    { label: "背景伴奏", ready: Boolean(script.media.bgm_wav) },
    { label: "成片输出", ready: Boolean(script.media.output_video) },
  ];

  const characterEntries = Object.entries(script.characters);

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
          characterEntries.length > 0 ? (
            <div className="divide-y divide-dashed divide-white/[0.08]">
              {characterEntries.map(([speakerKey, char]) => {
                const hasWeights = Boolean(
                  char.gpt_weights && char.sovits_weights
                );
                const emotionKeys = Object.keys(
                  char.emotion_refs
                ) as EmotionType[];
                const segCount = script.segments.filter(
                  (seg) => seg.speaker === speakerKey
                ).length;

                return (
                  <div key={speakerKey} className="px-3.5 py-3">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex min-w-0 items-center gap-2">
                        <span
                          className="h-1.5 w-1.5 shrink-0 rounded-full"
                          style={{
                            backgroundColor: normalizeSpeakerColor(char.color),
                          }}
                        />
                        <span className="truncate font-medium text-primary">
                          {char.display_name}
                        </span>
                      </div>
                      <span className="shrink-0 font-mono tabular-nums text-[11px] text-secondary">
                        {segCount} 句
                      </span>
                    </div>

                    <div className="mt-1.5 flex items-center gap-1.5 text-[11px]">
                      <span
                        className={`h-1.5 w-1.5 rounded-full ${
                          hasWeights ? "bg-status-ready" : "bg-status-warning"
                        }`}
                      />
                      <span
                        className={
                          hasWeights ? "text-secondary" : "text-muted"
                        }
                      >
                        {hasWeights ? "已绑定微调模型" : "零样本兜底"}
                      </span>
                    </div>

                    {emotionKeys.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {emotionKeys.map((emo) => (
                          <span
                            key={emo}
                            className="rounded-[4px] bg-white/[0.05] px-1.5 py-0.5 text-[10px] text-secondary"
                          >
                            {EMOTION_LABELS[emo]}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="p-3.5 text-muted">当前剧本暂无角色配置</p>
          )
        ) : (
          <div>
            <div className="divide-y divide-white/[0.05] border-b border-dashed border-white/[0.08]">
              {mediaRows.map((row) => (
                <div
                  key={row.label}
                  className="flex h-8 items-center justify-between px-3.5"
                >
                  <span className="text-primary">{row.label}</span>
                  <span className="flex items-center gap-1.5 text-[11px] text-secondary">
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${
                        row.ready ? "bg-status-ready" : "bg-muted"
                      }`}
                    />
                    <span>{row.ready ? "就绪" : "未生成"}</span>
                  </span>
                </div>
              ))}
            </div>

            <div className="p-3.5">
              <div className="mb-1.5 text-[11px] font-medium text-secondary">
                台词跳转
              </div>
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
                      <span className="font-mono tabular-nums text-muted">
                        #{seg.id}
                      </span>{" "}
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
