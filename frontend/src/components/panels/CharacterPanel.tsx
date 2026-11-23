import { useCallback, useEffect, useState } from "react";
import {
  createVoicePreset,
  deleteVoicePreset,
  fetchVoicePresets,
  previewAutomaticReference,
  uploadReference,
} from "../../services/api";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import type {
  CharacterConfig,
  EmotionRef,
  EmotionType,
  ProjectScript,
  VoicePreset,
} from "../../types/script";

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

  const boundRef = character.emotion_refs[emotion];
  const boundWav = boundRef?.wav ?? "";
  const boundText = boundRef?.text ?? "";
  const boundLang = boundRef?.lang ?? script.source_lang ?? "";
  useEffect(() => {
    setDraft({ wav: boundWav, text: boundText, lang: boundLang });
  }, [boundWav, boundText, boundLang]);

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
      <VoicePresetLibrary speaker={speaker} character={character} busy={busy} />
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

function VoicePresetLibrary({
  speaker,
  character,
  busy,
}: {
  speaker: string;
  character: CharacterConfig;
  busy: boolean;
}) {
  const backendStatus = useProjectStore((s) => s.backendStatus);
  const script = useProjectStore((s) => s.script);
  const isDirty = useProjectStore((s) => s.isDirty);
  const applyVoicePresetToCharacter = useProjectStore((s) => s.applyVoicePresetToCharacter);
  const [presets, setPresets] = useState<VoicePreset[]>([]);
  const [presetName, setPresetName] = useState("");
  const [loading, setLoading] = useState(false);
  const [note, setNote] = useState("");

  const loadPresets = useCallback(async () => {
    if (backendStatus !== "online") {
      return;
    }
    setLoading(true);
    try {
      const res = await fetchVoicePresets();
      setPresets(res.presets);
    } catch (err) {
      setNote(err instanceof Error ? err.message : "加载音色预设失败");
    } finally {
      setLoading(false);
    }
  }, [backendStatus]);

  useEffect(() => {
    void loadPresets();
  }, [loadPresets]);

  const handleSavePreset = async () => {
    const cleanName = presetName.trim() || character.display_name || speaker;
    setLoading(true);
    setNote("");
    try {
      const created = await createVoicePreset({
        name: cleanName,
        speaking_style: character.speaking_style ?? "",
        gpt_weights: character.gpt_weights ?? null,
        sovits_weights: character.sovits_weights ?? null,
        emotion_refs: character.emotion_refs,
      });
      setPresets((prev) => [created, ...prev]);
      setPresetName("");
      setNote(`已保存跨项目预设“${created.name}”`);
    } catch (err) {
      setNote(err instanceof Error ? err.message : "保存音色预设失败");
    } finally {
      setLoading(false);
    }
  };

  const handleDeletePreset = async (preset: VoicePreset) => {
    const prefix = `/assets/refs/presets/${preset.preset_id}/`;
    const referencedLocally = Object.values(script?.characters ?? {}).some((char) =>
      Object.values(char.emotion_refs).some((ref) => ref?.wav.startsWith(prefix)),
    );
    const confirmMsg = referencedLocally
      ? `确定删除预设“${preset.name}”吗？${
          isDirty
            ? "当前项目有未保存修改正在引用该预设音频，建议先保存项目再删除。"
            : "删除后预设列表中将移除，但已保存项目引用的音频会自动保留。"
        }`
      : `确定删除跨项目预设“${preset.name}”吗？`;
    if (!window.confirm(confirmMsg)) {
      return;
    }
    setLoading(true);
    setNote("");
    try {
      await deleteVoicePreset(preset.preset_id);
      setPresets((prev) => prev.filter((item) => item.preset_id !== preset.preset_id));
    } catch (err) {
      setNote(err instanceof Error ? err.message : "删除预设失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <details className="space-y-2 border-t border-white/10 pt-2">
      <summary className="cursor-pointer text-primary">跨项目音色预设库 ({presets.length})</summary>
      <div className="flex gap-1">
        <input
          className={field}
          aria-label="预设名称"
          placeholder={`预设名（默认 ${character.display_name || speaker}）`}
          value={presetName}
          onChange={(e) => setPresetName(e.target.value)}
        />
        <button
          className={`${button} shrink-0`}
          type="button"
          disabled={busy || loading || backendStatus !== "online"}
          onClick={() => void handleSavePreset()}
        >
          存为预设
        </button>
      </div>
      {note && <div className="text-[11px] text-secondary">{note}</div>}
      {presets.length === 0 ? (
        <div className="text-[11px] text-muted">
          {backendStatus === "online" ? "暂无全局音色预设" : "连接后端后可使用跨项目预设库"}
        </div>
      ) : (
        <ul className="max-h-40 space-y-1 overflow-y-auto">
          {presets.map((preset) => (
            <li
              key={preset.preset_id}
              className="flex items-center justify-between gap-1 rounded bg-[#181818] px-2 py-1 text-[11px]"
            >
              <div className="min-w-0 flex-1">
                <div className="truncate font-medium text-primary">{preset.name}</div>
                <div className="truncate font-mono tabular-nums text-[10px] text-muted">
                  参考音 {Object.keys(preset.emotion_refs).length} 项
                  {preset.gpt_weights || preset.sovits_weights ? " · 含权重" : ""}
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-1">
                <button
                  type="button"
                  disabled={busy || loading}
                  onClick={() => {
                    applyVoicePresetToCharacter(speaker, preset);
                    setNote(`已套用预设“${preset.name}”`);
                  }}
                  className={button}
                >
                  套用
                </button>
                <button
                  type="button"
                  disabled={busy || loading}
                  onClick={() => void handleDeletePreset(preset)}
                  className="rounded bg-[#2C2C2C] px-2 py-1.5 text-status-error hover:bg-[#3A2828] disabled:opacity-40"
                >
                  删除
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </details>
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
