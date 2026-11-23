import { useState } from "react";
import { previewAutomaticReference, uploadReference } from "../../services/api";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import type { CharacterConfig, EmotionRef, EmotionType, ProjectScript } from "../../types/script";

const emotions: Record<EmotionType, string> = {
  neutral: "平淡",
  angry: "愤怒",
  happy: "喜悦",
  sad: "悲伤",
  whisper: "低语",
};
const field = "min-w-0 w-full rounded border border-white/10 bg-[#181818] px-2 py-1.5 text-primary";
const button =
  "rounded bg-[#333333] px-2 py-1.5 text-primary hover:bg-[#404040] disabled:opacity-40";

function referenceUrl(wav: string): string | undefined {
  if (wav.startsWith("/assets/") || wav.startsWith("/workspace/")) return wav;
  if (wav.startsWith("assets/") || wav.startsWith("workspace/")) return `/${wav}`;
  if (wav.startsWith("refs/")) return `/assets/${wav}`;
}

function ReferenceEditor({
  script,
  speaker,
  emotion,
  character,
  busy,
}: {
  script: ProjectScript;
  speaker: string;
  emotion: EmotionType;
  character: CharacterConfig;
  busy: boolean;
}) {
  const [draft, setDraft] = useState<EmotionRef>(
    character.emotion_refs[emotion] ?? { wav: "", text: "", lang: script.source_lang ?? "" },
  );
  const [inFlight, setInFlight] = useState(false);
  const [message, setMessage] = useState("");
  const updateCharacter = useProjectStore((s) => s.updateCharacter);
  const audioUrl = referenceUrl(draft.wav);

  async function loadAudio(file?: File) {
    setInFlight(true);
    setMessage("");
    try {
      if (file) {
        const { wav } = await uploadReference(script.project_id, file);
        setDraft((old) => ({ ...old, wav, text: "" }));
        setMessage("已上传，请填写与音频一致的逐字稿。");
        return;
      }
      const state = useProjectStore.getState();
      if (state.isDirty) await state.saveProjectScript();
      if (useProjectStore.getState().script?.project_id !== script.project_id) return;
      const saved = useProjectStore.getState();
      if (saved.error || saved.isDirty || saved.backendStatus !== "online") {
        throw new Error(saved.error ?? "请先保存当前校对结果");
      }
      const { wav, text, lang, segment_ids } = await previewAutomaticReference(
        script.project_id,
        speaker,
      );
      setDraft({ wav, text, lang });
      setMessage(`来自片段 ${segment_ids.map((id) => `#${id}`).join("、")}。可试听并修订逐字稿。`);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "无法加载参考音");
    } finally {
      setInFlight(false);
    }
  }

  const refs = character.emotion_refs;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1">
        <label
          className={`${button} ${busy || inFlight ? "pointer-events-none opacity-40" : "cursor-pointer"}`}
        >
          上传参考音
          <input
            className="sr-only"
            type="file"
            accept="audio/*,.wav,.flac,.m4a,.mp3,.opus"
            disabled={busy || inFlight}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void loadAudio(file);
              e.target.value = "";
            }}
          />
        </label>
        <button
          className={button}
          type="button"
          disabled={busy || inFlight}
          onClick={() => void loadAudio()}
        >
          生成自动参考试听
        </button>
      </div>
      {audioUrl && (
        <audio
          className="h-9 w-full"
          src={audioUrl}
          controls
          preload="none"
          onError={() => setMessage("参考音不可播放，请重新上传")}
        />
      )}
      <label className="block space-y-1">
        <span>参考音逐字稿</span>
        <textarea
          className={field}
          rows={3}
          value={draft.text}
          onChange={(e) => setDraft({ ...draft, text: e.target.value })}
        />
      </label>
      <label className="block space-y-1">
        <span>参考音语种</span>
        <input
          className={field}
          placeholder="如 zh、en、ja"
          value={draft.lang}
          onChange={(e) => setDraft({ ...draft, lang: e.target.value })}
        />
      </label>
      <div className="flex flex-wrap gap-1">
        <button
          className={button}
          type="button"
          disabled={inFlight || !draft.wav || !draft.text.trim() || !draft.lang.trim()}
          onClick={() => {
            updateCharacter(speaker, {
              emotion_refs: {
                ...refs,
                [emotion]: {
                  ...draft,
                  text: draft.text.trim(),
                  lang: draft.lang.trim().toLowerCase(),
                },
              },
            });
            setMessage("已应用，保存项目后保留。");
          }}
        >
          应用参考音
        </button>
        {refs[emotion] && (
          <button
            className={button}
            type="button"
            disabled={inFlight}
            onClick={() => {
              const { [emotion]: _removed, ...remaining } = refs;
              updateCharacter(speaker, { emotion_refs: remaining });
              setDraft({ wav: "", text: "", lang: script.source_lang ?? "" });
              setMessage("已移除当前情绪的参考音。");
            }}
          >
            移除
          </button>
        )}
      </div>
      {(inFlight || message) && <p role="status">{inFlight ? "正在处理参考音…" : message}</p>}
    </div>
  );
}

function SpeakerEditor({
  script,
  speaker,
  busy,
}: { script: ProjectScript; speaker: string; busy: boolean }) {
  const character = script.characters[speaker];
  const updateCharacter = useProjectStore((s) => s.updateCharacter);
  const mergeCharacters = useProjectStore((s) => s.mergeCharacters);
  const batchUpdateSegments = useProjectStore((s) => s.batchUpdateSegmentsLocal);
  const selectSegment = useProjectStore((s) => s.selectSegment);
  const setCurrentTime = useProjectStore((s) => s.setCurrentTime);
  const [emotion, setEmotion] = useState<EmotionType>("neutral");
  const [destination, setDestination] = useState("");
  const [selected, setSelected] = useState<number[]>([]);
  const segments = script.segments.filter((seg) => seg.speaker === speaker);
  const ids = segments.filter((seg) => selected.includes(seg.id)).map((seg) => seg.id);

  return (
    <div className="space-y-2">
      <label className="block space-y-1">
        <span>角色名</span>
        <input
          className={field}
          value={character.display_name}
          onChange={(e) => updateCharacter(speaker, { display_name: e.target.value })}
        />
      </label>
      <label className="block space-y-1">
        <span>说话语气</span>
        <textarea
          className={field}
          rows={2}
          value={character.speaking_style ?? ""}
          onChange={(e) => updateCharacter(speaker, { speaking_style: e.target.value })}
        />
      </label>
      <details className="space-y-2 border-t border-white/10 pt-2" open>
        <summary className="cursor-pointer text-primary">参考音与权重</summary>
        {!Object.values(character.emotion_refs).some((ref) => ref?.wav && ref.text?.trim() && ref.lang?.trim()) && (
          <p className="text-[11px] leading-relaxed text-status-warning">
            尚未配置手动参考音，合成时会自动选取。MOSS 句段将自动裁剪并重新识别参考音逐字稿，首次生成可能较慢，成功后会缓存。
            {segments.some((seg) => seg.timing_source === "moss_joint_generated")
              ? " 自动选取失败时，可在此上传参考音与对应原文逐字稿。"
              : " 其他来源需要已确认原文、时间和说话人的 3–10 秒句段，或上传手动参考音。"}
          </p>
        )}
        <select
          className={field}
          aria-label="参考音情绪"
          value={emotion}
          onChange={(e) => setEmotion(e.target.value as EmotionType)}
        >
          {Object.entries(emotions).map(([key, label]) => (
            <option key={key} value={key}>
              {label}
              {character.emotion_refs[key as EmotionType] ? " · 已配置" : ""}
            </option>
          ))}
        </select>
        <ReferenceEditor
          key={emotion}
          script={script}
          speaker={speaker}
          emotion={emotion}
          character={character}
          busy={busy}
        />
        <label className="block space-y-1">
          <span>GPT 权重路径（可选）</span>
          <input
            className={field}
            value={character.gpt_weights ?? ""}
            onChange={(e) => updateCharacter(speaker, { gpt_weights: e.target.value || null })}
          />
        </label>
        <label className="block space-y-1">
          <span>SoVITS 权重路径（可选）</span>
          <input
            className={field}
            value={character.sovits_weights ?? ""}
            onChange={(e) => updateCharacter(speaker, { sovits_weights: e.target.value || null })}
          />
        </label>
      </details>
      <details className="space-y-2 border-t border-white/10 pt-2">
        <summary className="cursor-pointer text-primary">说话人校对 · {segments.length} 句</summary>
        {segments.length > 0 && (
          <>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={ids.length === segments.length}
                onChange={(e) => setSelected(e.target.checked ? segments.map((seg) => seg.id) : [])}
              />
              选择全部台词
            </label>
            <ul className="max-h-40 space-y-1 overflow-y-auto">
              {segments.map((seg) => (
                <li key={seg.id} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    aria-label={`选择片段 ${seg.id}`}
                    checked={selected.includes(seg.id)}
                    onChange={(e) =>
                      setSelected(
                        e.target.checked
                          ? [...selected, seg.id]
                          : selected.filter((id) => id !== seg.id),
                      )
                    }
                  />
                  <button
                    type="button"
                    className="min-w-0 flex-1 truncate text-left hover:text-primary"
                    onClick={() => {
                      selectSegment(seg.id);
                      setCurrentTime(seg.start);
                    }}
                  >
                    #{seg.id} {seg.speaker_needs_review !== false ? "待校对 · " : ""}
                    {seg.orig_text}
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}
        <label className="block space-y-1">
          <span>目标角色</span>
          <select
            className={field}
            value={destination || speaker}
            onChange={(e) => setDestination(e.target.value)}
          >
            {Object.entries(script.characters).map(([id, item]) => (
              <option key={id} value={id}>
                {item.display_name || id}
              </option>
            ))}
          </select>
        </label>
        <div className="flex flex-wrap gap-1">
          <button
            className={button}
            type="button"
            disabled={!ids.length}
            onClick={() => {
              batchUpdateSegments(ids.map((id) => ({ id, speaker: destination || speaker })));
              setSelected([]);
            }}
          >
            {!destination || destination === speaker ? "确认选中说话人" : "批量改派选中"}
          </button>
          <button
            className={button}
            type="button"
            disabled={!destination || destination === speaker}
            onClick={() => mergeCharacters(speaker, destination)}
          >
            合并到目标角色
          </button>
        </div>
      </details>
    </div>
  );
}

export function CharacterPanel({ script }: { script: ProjectScript }) {
  const [speakerKey, setSpeakerKey] = useState("");
  const [name, setName] = useState("");
  const addCharacter = useProjectStore((s) => s.addCharacter);
  const offline = useProjectStore((s) => s.backendStatus !== "online");
  const isLoading = useProjectStore((s) => s.isLoading || s.isSynthesizing);
  const pipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);
  const speaker = script.characters[speakerKey] ? speakerKey : Object.keys(script.characters)[0];

  return (
    <fieldset
      className="min-w-0 space-y-3 p-3.5 text-secondary"
      disabled={isLoading || pipelineRunning}
    >
      <form
        className="flex gap-1"
        onSubmit={(e) => {
          e.preventDefault();
          const key = addCharacter(name);
          if (key) {
            setSpeakerKey(key);
            setName("");
          }
        }}
      >
        <input
          className={field}
          aria-label="新角色名称"
          placeholder="新角色名称"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <button className={`${button} shrink-0`} type="submit" disabled={!name.trim()}>
          新建
        </button>
      </form>
      {speaker && (
        <>
          <select
            className={field}
            aria-label="选择要编辑的角色"
            value={speaker}
            onChange={(e) => setSpeakerKey(e.target.value)}
          >
            {Object.entries(script.characters).map(([id, character]) => (
              <option key={id} value={id}>
                {character.display_name || id}
              </option>
            ))}
          </select>
          <SpeakerEditor
            key={speaker}
            script={script}
            speaker={speaker}
            busy={offline || isLoading || pipelineRunning}
          />
        </>
      )}
    </fieldset>
  );
}
