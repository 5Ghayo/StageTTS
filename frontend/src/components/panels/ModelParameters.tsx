import { useId, useState } from "react";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import type { ProjectScript } from "../../types/script";

type Parameters = NonNullable<ProjectScript["model_parameters"]>;
type Parameter = {
  key: keyof Parameters;
  label: string;
  min: number;
  max: number;
  step: number;
  defaultValue: string;
  description: string;
};

const groups = {
  separation: {
    title: "Banquet 分离参数",
    hint: "修改后重新解析生效。长音频仍按约 5 分钟切块，模型按 8 秒窗口推理。",
    fields: [
      {
        key: "banquet_hop_seconds",
        label: "窗口步长（秒）",
        min: 0.25,
        max: 4,
        step: 0.05,
        defaultValue: "继承服务器（通常 1）",
        description: "步长越小，预测窗口重叠越多，分离耗时越长。",
      },
    ],
  },
  recognition: {
    title: "WhisperX · DiariZen 参数",
    hint: "修改后重新解析生效。DiariZen 只聚类，改参数后请复核角色分组。",
    fields: [
      {
        key: "asr_vad_threshold",
        label: "WhisperX 初切 VAD",
        min: 0.01,
        max: 0.99,
        step: 0.01,
        defaultValue: "0.65",
        description: "控制初次语音检测的进入和退出阈值。升高后更容易忽略轻声。",
      },
      {
        key: "asr_recheck_vad_threshold",
        label: "WhisperX 复查 VAD",
        min: 0.01,
        max: 0.99,
        step: 0.01,
        defaultValue: "0.80",
        description: "仅在 SaT 提议继续拆分时使用，须大于或等于初切阈值。",
      },
      {
        key: "diarizen_ahc_threshold",
        label: "DiariZen AHC 阈值",
        min: 0.01,
        max: 1,
        step: 0.01,
        defaultValue: "0.35",
        description: "控制初始声纹簇的合并距离。增大通常会让初始分组更粗。",
      },
      {
        key: "diarizen_fa",
        label: "DiariZen VBx Fa",
        min: 0.01,
        max: 1,
        step: 0.01,
        defaultValue: "0.15",
        description: "缩放 VBx 对声纹观测证据的权重，影响如何细化初始分组。",
      },
      {
        key: "diarizen_fb",
        label: "DiariZen VBx Fb",
        min: 0.01,
        max: 1,
        step: 0.01,
        defaultValue: "0.37",
        description: "缩放 VBx 的说话人先验权重。与 Fa 共同影响最终分组。",
      },
    ],
  },
  synthesis: {
    title: "GPT-SoVITS 采样参数",
    hint: "应用于之后的整段合成、选区配音、单句重配和时长重配。",
    fields: [
      {
        key: "tts_temperature",
        label: "温度 Temperature",
        min: 0.01,
        max: 2,
        step: 0.01,
        defaultValue: "推理服务默认",
        description: "越低越偏向概率高的候选，越高则语气和发音的变化可能增多。",
      },
      {
        key: "tts_top_k",
        label: "Top-k",
        min: 1,
        max: 100,
        step: 1,
        defaultValue: "推理服务默认",
        description: "每步最多保留多少个候选。设为 1 时只保留概率最高的候选。",
      },
      {
        key: "tts_top_p",
        label: "Top-p",
        min: 0.01,
        max: 1,
        step: 0.01,
        defaultValue: "推理服务默认",
        description: "按累计概率限制候选。越低候选越集中，1 表示不按累计概率截断。",
      },
      {
        key: "tts_repetition_penalty",
        label: "重复惩罚",
        min: 1,
        max: 2,
        step: 0.01,
        defaultValue: "1.35",
        description: "提高后会更强地抑制已生成的语义单元重复，过高可能影响正常叠词。",
      },
    ],
  },
} satisfies Record<string, { title: string; hint: string; fields: Parameter[] }>;

export type ModelParameterGroup = keyof typeof groups;

export function ModelParameters({
  group,
  script,
}: { group: ModelParameterGroup; script: ProjectScript }) {
  const { title, hint, fields } = groups[group];
  const id = useId();
  const initial = script.model_parameters;
  const [draft, setDraft] = useState<Partial<Record<keyof Parameters, string>>>({});
  const [message, setMessage] = useState("");
  const available = useProjectStore(
    (s) => s.backendStatus === "online" && !s.isLoading && !s.isSynthesizing,
  );
  const pipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);
  const disabled = !available || pipelineRunning;

  async function save(useDefaults = false) {
    const store = useProjectStore.getState();
    if (disabled || store.script?.project_id !== script.project_id) return;
    const next = { ...store.script.model_parameters };
    for (const { key } of fields) {
      if (!useDefaults && draft[key] === undefined) continue;
      const value = useDefaults ? "" : draft[key];
      next[key] = value === "" ? null : Number(value);
    }
    if ((next.asr_recheck_vad_threshold ?? 0.8) < (next.asr_vad_threshold ?? 0.65)) {
      setMessage("复查 VAD 阈值须大于或等于初切 VAD 阈值。");
      return;
    }
    store.setModelParameters(next);
    const saved = await store.saveProjectScript();
    if (useProjectStore.getState().script?.project_id !== script.project_id) return;
    if (saved) setDraft({});
    setMessage(
      saved ? "参数已保存到项目。" : (useProjectStore.getState().error ?? "参数尚未保存到项目。"),
    );
  }

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        void save();
      }}
      className="space-y-2 border-b border-dashed border-white/[0.08] pb-3 text-[11px]"
    >
      <fieldset disabled={disabled} className="space-y-2 disabled:opacity-60">
        <legend className="text-[12px] font-medium text-primary">{title}</legend>
        <p className="text-secondary">{hint}</p>
        {fields.map(({ key, label, min, max, step, defaultValue, description }) => (
          <label key={key} className="block space-y-1">
            <span className="flex flex-wrap justify-between gap-x-2 text-primary">
              {label}
              <span className="text-secondary">留空使用 {defaultValue}</span>
            </span>
            <input
              type="number"
              min={min}
              max={max}
              step={step}
              aria-describedby={`${id}-${key}`}
              value={draft[key] ?? initial?.[key] ?? ""}
              placeholder={defaultValue}
              onChange={(event) => setDraft({ ...draft, [key]: event.target.value })}
              className="h-7 w-full rounded border border-white/10 bg-[#181818] px-2 text-primary"
            />
            <span id={`${id}-${key}`} className="block text-secondary">
              {description}
            </span>
          </label>
        ))}
        <div className="flex gap-2">
          <button type="submit" className="rounded bg-[#39475e] px-2 py-1 text-primary">
            保存参数
          </button>
          <button
            type="button"
            onClick={() => void save(true)}
            className="rounded px-2 py-1 text-secondary hover:bg-white/10"
          >
            恢复默认
          </button>
        </div>
      </fieldset>
      {message && (
        <p role="status" className="text-secondary">
          {message}
        </p>
      )}
    </form>
  );
}
