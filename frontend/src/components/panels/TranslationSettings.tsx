import { useEffect, useState } from "react";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import type { ProjectScript } from "../../types/script";

const field = "w-full rounded border border-white/10 bg-[#181818] px-2 py-1.5 text-primary";
const button =
  "rounded bg-[#333333] px-2 py-1.5 text-primary hover:bg-[#404040] disabled:opacity-40";

export function TranslationSettings({ script }: { script: ProjectScript }) {
  const updateSettings = useProjectStore((s) => s.updateTranslationSettings);
  const retranslate = useProjectStore((s) => s.retranslate);
  const offline = useProjectStore((s) => s.backendStatus !== "online");
  const busy = useProjectStore((s) => s.isLoading || s.isSynthesizing);
  const pipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);
  const [sourceLang, setSourceLang] = useState("");
  const [terms, setTerms] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    setSourceLang(script.source_lang ?? "");
    setTerms(
      Object.entries(script.glossary ?? {})
        .map(([source, target]) => `${source} = ${target}`)
        .join("\n"),
    );
  }, [script.source_lang, script.glossary]);

  function applySettings() {
    const entries: Array<[string, string]> = [];
    for (const [index, line] of terms.split("\n").entries()) {
      if (!line.trim()) continue;
      const separator = line.indexOf("=");
      const source = line.slice(0, separator).trim();
      const target = line.slice(separator + 1).trim();
      if (separator < 0 || !source || !target) {
        setMessage(`术语表第 ${index + 1} 行请填写「原词 = 译词」。`);
        return false;
      }
      entries.push([source, target]);
    }
    updateSettings(sourceLang.trim().toLowerCase() || null, Object.fromEntries(entries));
    return true;
  }

  return (
    <details className="space-y-2 border-b border-dashed border-white/10 p-3.5 text-secondary">
      <summary className="cursor-pointer text-primary">翻译设置</summary>
      <label className="block space-y-1">
        <span>原声语言</span>
        <select
          className={field}
          value={sourceLang}
          onChange={(e) => setSourceLang(e.target.value)}
          disabled={busy || pipelineRunning}
        >
          <option value="">待确认（MOSS 不自动检测）</option>
          <option value="ja">日语</option>
          <option value="en">英语</option>
          <option value="zh">普通话</option>
          <option value="ko">韩语</option>
          <option value="yue">粤语</option>
          {sourceLang && !["ja", "en", "zh", "ko", "yue"].includes(sourceLang) && (
            <option value={sourceLang}>{sourceLang}（旧值，自动参考不支持）</option>
          )}
        </select>
      </label>
      <label className="block space-y-1">
        <span>术语表</span>
        <textarea
          className={field}
          rows={4}
          placeholder="每行填写 原词 = 译词"
          value={terms}
          onChange={(e) => setTerms(e.target.value)}
        />
      </label>
      <div className="flex flex-wrap gap-1">
        <button
          className={button}
          type="button"
          disabled={busy || pipelineRunning}
          onClick={() => {
            if (applySettings()) setMessage("已应用，保存项目后保留。");
          }}
        >
          应用设置
        </button>
        <button
          className={button}
          type="button"
          disabled={
            offline ||
            busy ||
            pipelineRunning ||
            !script.segments.some((seg) => seg.orig_text.trim())
          }
          onClick={async () => {
            if (!applySettings()) return;
            setMessage("正在重译台词…");
            setMessage(await retranslate());
          }}
        >
          重译全部台词
        </button>
      </div>
      {message && <p role="status">{message}</p>}
    </details>
  );
}
