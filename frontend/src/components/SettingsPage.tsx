import { Settings, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { fetchLLMSettings, saveLLMSettings, type LLMSettings } from "../services/api";

export function SettingsPage({ onClose }: { onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [settings, setSettings] = useState<LLMSettings | null>(null);
  const [baseUrl, setBaseUrl] = useState("");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [clearKey, setClearKey] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [reload, setReload] = useState(0);

  useEffect(() => {
    dialog.current?.showModal();
    return () => dialog.current?.close();
  }, []);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    void fetchLLMSettings().then((result) => {
      if (!active) return;
      setSettings(result);
      setBaseUrl(result.base_url);
      setModel(result.model);
    }).catch(() => {
      if (active) setError("无法读取设置，请确认后端已启动并更新到支持设置的版本。");
    }).finally(() => {
      if (active) setLoading(false);
    });
    return () => { active = false; };
  }, [reload]);

  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (saving || loading || !settings) return;
    setSaving(true);
    setError("");
    setMessage("");
    try {
      const result = await saveLLMSettings({
        base_url: baseUrl.trim(), model: model.trim(),
        ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}), clear_api_key: clearKey,
      });
      setSettings(result);
      setBaseUrl(result.base_url);
      setModel(result.model);
      setApiKey("");
      setClearKey(false);
      setMessage(result.configured ? "设置已保存，新的 LLM 请求将使用此配置。" : "设置已保存，请配置有效的 API Key 后再运行翻译。");
    } catch {
      setError("保存失败。请检查接口地址、模型名称和后端连接；原设置未被替换。");
    } finally {
      setSaving(false);
    }
  }

  const inputClass = "h-9 w-full rounded border border-white/10 bg-[#181818] px-3 text-[13px] text-primary focus:border-[#6094c6] focus:outline-none disabled:opacity-50";
  return createPortal(
    <dialog
      ref={dialog}
      onCancel={(event) => { if (saving) event.preventDefault(); else onClose(); }}
      aria-labelledby="settings-title"
      className="m-auto w-[min(640px,calc(100vw-32px))] max-h-[calc(100vh-32px)] overflow-y-auto rounded-lg border border-white/10 bg-[#222222] p-0 text-primary shadow-2xl backdrop:bg-black/60"
    >
      <div className="flex items-center justify-between border-b border-white/10 px-6 py-4">
        <h1 id="settings-title" className="flex items-center gap-2 text-[16px] font-medium"><Settings className="h-4 w-4" />设置</h1>
        <button type="button" aria-label="关闭设置" disabled={saving} onClick={onClose} className="rounded p-1 text-secondary hover:bg-white/10 disabled:opacity-50"><X className="h-4 w-4" /></button>
      </div>
      <form onSubmit={(event) => void save(event)} className="space-y-5 p-6 text-[13px]">
        <div>
          <h2 className="font-medium">翻译与 LLM</h2>
          <p className="mt-2 leading-relaxed text-secondary">配置 OpenAI 兼容的 Chat Completions 服务。解析中的翻译、已有对白重译、配音时长改写和 AI 助手共用此配置。</p>
        </div>
        {loading ? <p role="status" className="text-secondary">正在读取设置…</p> : settings ? (
          <fieldset disabled={saving} className="space-y-4 disabled:opacity-60">
            <label className="block space-y-2">
              <span>接口地址（Base URL）</span>
              <input type="url" required maxLength={2048} value={baseUrl} onChange={(event) => { setBaseUrl(event.target.value); setMessage(""); }} placeholder="https://api.example.com/v1" autoComplete="off" spellCheck={false} className={inputClass} />
              <span className="block text-[11px] text-secondary">填写服务商提供的 API 根地址，通常以 /v1 结尾，不需要填写 /chat/completions。</span>
            </label>
            <label className="block space-y-2">
              <span>模型名称</span>
              <input required maxLength={256} value={model} onChange={(event) => { setModel(event.target.value); setMessage(""); }} placeholder="服务商提供的模型 ID" autoComplete="off" spellCheck={false} className={inputClass} />
            </label>
            <label className="block space-y-2">
              <span>API Key <span className="text-[11px] text-secondary">{settings.api_key_configured ? "（已有 Key，留空保留）" : "（尚未配置）"}</span></span>
              <input type="password" maxLength={8192} value={apiKey} disabled={clearKey} onChange={(event) => { setApiKey(event.target.value); setMessage(""); }} placeholder="输入新 Key 可替换，留空沿用现有 Key" autoComplete="new-password" spellCheck={false} className={inputClass} />
            </label>
            <label className="flex items-center gap-2 text-secondary">
              <input type="checkbox" checked={clearKey} onChange={(event) => { setClearKey(event.target.checked); setApiKey(""); setMessage(""); }} />清除已保存的 API Key
            </label>
            <p className="text-[11px] leading-relaxed text-secondary">设置保存在本机后端，Key 不会回显或保存在浏览器。保存后无需重启；已发出的请求继续使用原配置。保存不会发起付费模型请求。</p>
          </fieldset>
        ) : <button type="button" onClick={() => setReload((value) => value + 1)} className="text-[#8bb5de]">重新读取</button>}
        {error && <p role="alert" className="text-status-error">{error}</p>}
        {message && <p role="status" className="text-status-ready">{message}</p>}
        <div className="flex justify-end gap-2 border-t border-white/10 pt-4">
          <button type="button" onClick={onClose} disabled={saving} className="h-8 rounded bg-white/5 px-4 hover:bg-white/10 disabled:opacity-50">关闭</button>
          <button type="submit" disabled={loading || saving || !settings || !model.trim() || !baseUrl.trim()} className="h-8 rounded bg-[#3A6EA5] px-4 text-white hover:bg-[#467DB8] disabled:opacity-50">{saving ? "保存中…" : "保存设置"}</button>
        </div>
      </form>
    </dialog>, document.body,
  );
}
