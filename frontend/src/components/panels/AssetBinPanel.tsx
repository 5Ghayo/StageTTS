import { X } from "lucide-react";
import type React from "react";
import { getSubtitlesUrl } from "../../services/api";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import type { StemVolumes } from "../../types/script";
import { CharacterPanel } from "./CharacterPanel";
import { TranslationSettings } from "./TranslationSettings";

/**
 * 240px 按需展开的左侧停靠抽屉（#222222 面板，极淡虚线分隔）。
 */
export const AssetBinPanel: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const selectSegment = useProjectStore((s) => s.selectSegment);
  const setCurrentTime = useProjectStore((s) => s.setCurrentTime);
  const updateStemVolume = useProjectStore((s) => s.updateStemVolume);
  const updateAutoDucking = useProjectStore((s) => s.updateAutoDucking);

  const leftDrawer = useWorkflowStore((s) => s.leftDrawer);
  const toggleLeftDrawer = useWorkflowStore((s) => s.toggleLeftDrawer);

  if (!script || leftDrawer === "none") {
    return null;
  }

  const stems = script.media.cinematic_stems;
  const stemVolumes = script.media.stem_volumes ?? { singing: 1, music: 1, effects: 1 };
  const autoDucking = script.media.auto_ducking ?? true;
  const stemControls: Array<{ key: keyof StemVolumes; label: string }> = [
    { key: "singing", label: "歌声" },
    { key: "music", label: "音乐" },
    { key: "effects", label: "音效" },
  ];
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

            <div className="space-y-2.5 border-b border-dashed border-white/[0.08] p-3.5">
              <div className="text-[11px] font-medium text-secondary">混音与四轨平衡</div>
              {stems && (
                <div className="space-y-1.5">
                  {stemControls.map((item) => {
                    const value = stemVolumes[item.key] ?? 1;
                    return (
                      <div key={item.key} className="flex items-center gap-2 text-[11px]">
                        <span className="w-8 shrink-0 text-secondary">{item.label}</span>
                        <input
                          aria-label={`${item.label}轨音量`}
                          type="range"
                          min={0}
                          max={2}
                          step={0.05}
                          value={value}
                          onChange={(e) => updateStemVolume(item.key, Number(e.target.value))}
                          className="min-w-0 flex-1"
                        />
                        <span className="w-9 shrink-0 text-right font-mono tabular-nums text-secondary">
                          {Math.round(value * 100)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
              <label className="flex cursor-pointer items-center justify-between gap-2 text-[11px] text-primary">
                <span>人声自动闪避 BGM</span>
                <input
                  type="checkbox"
                  checked={autoDucking}
                  onChange={(e) => updateAutoDucking(e.target.checked)}
                  className="h-3.5 w-3.5"
                />
              </label>

              <div className="flex items-center justify-between gap-2 pt-1 text-[11px]">
                <span className="text-secondary">ASS 字幕导出</span>
                <div className="flex items-center gap-1.5">
                  <a
                    href={getSubtitlesUrl(script.project_id, "target")}
                    download={`${script.project_id}_target.ass`}
                    className="rounded-[3px] border border-white/[0.08] bg-[#2C2C2C] px-2 py-0.5 text-primary transition-colors hover:bg-[#363636]"
                  >
                    单语
                  </a>
                  <a
                    href={getSubtitlesUrl(script.project_id, "bilingual")}
                    download={`${script.project_id}_bilingual.ass`}
                    className="rounded-[3px] border border-white/[0.08] bg-[#2C2C2C] px-2 py-0.5 text-primary transition-colors hover:bg-[#363636]"
                  >
                    双语
                  </a>
                </div>
              </div>
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
