import { ChevronDown, ChevronRight, Send, Wrench } from "lucide-react";
import type React from "react";
import { useState } from "react";
import { ApiError, sendAgentChat } from "../services/api";
import { useProjectStore } from "../store/useProjectStore";
import type { SegmentPatch, ToolCallLog } from "../types/script";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolLogs?: ToolCallLog[];
}

const QUICK_CHIPS = ["0-5秒改愤怒", "压缩超长句", "BGM 降到 40%"];

/**
 * AI 导演助手面板（#222222 面板 + #181818 凹槽输入 + 无边框色块气泡）。
 */
export const AgentChat: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const backendStatus = useProjectStore((s) => s.backendStatus);
  const setScript = useProjectStore((s) => s.setScript);
  const bumpAudioVersion = useProjectStore((s) => s.bumpAudioVersion);
  const batchUpdateSegmentsLocal = useProjectStore((s) => s.batchUpdateSegmentsLocal);
  const updateBgmVolume = useProjectStore((s) => s.updateBgmVolume);
  const synthesizeSegment = useProjectStore((s) => s.synthesizeSegment);

  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [expandedMessages, setExpandedMessages] = useState<Record<string, boolean>>({});
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "告诉我需要批量修改的情绪、台词或伴奏音量，我来帮您完成。",
    },
  ]);

  const toggleExpanded = (msgId: string) =>
    setExpandedMessages((prev) => ({ ...prev, [msgId]: !prev[msgId] }));

  const runSimulatedAgent = async (
    instruction: string,
  ): Promise<{ reply: string; toolLogs: ToolCallLog[] }> => {
    if (!script) {
      return { reply: "请先打开一个项目。", toolLogs: [] };
    }

    const logs: ToolCallLog[] = [];

    if (instruction.includes("BGM") || instruction.includes("伴奏")) {
      const percentMatch = instruction.match(/(\d+)\s*%/);
      const targetVol = percentMatch
        ? Math.min(2, Math.max(0, Number(percentMatch[1]) / 100))
        : 0.4;
      updateBgmVolume(targetVol);
      logs.push({
        tool_name: "update_mix_settings",
        arguments: { project_id: script.project_id, bgm_volume: targetVol },
        result_summary: `伴奏音量调至 ${Math.round(targetVol * 100)}%`,
      });
      return {
        reply: `已将背景伴奏音量调至 ${Math.round(targetVol * 100)}%。`,
        toolLogs: logs,
      };
    }

    if (instruction.includes("愤怒") || instruction.includes("0-5")) {
      const matched = script.segments.filter((seg) => seg.start <= 5.0 && !seg.keep_original);
      logs.push({
        tool_name: "query_script",
        arguments: { project_id: script.project_id, start_sec: 0, end_sec: 5 },
        result_summary: `筛选出 ${matched.length} 条片段`,
      });
      const patches: SegmentPatch[] = matched.map((seg) => ({
        id: seg.id,
        emotion: "angry",
      }));
      const updatedIds = batchUpdateSegmentsLocal(patches);
      logs.push({
        tool_name: "batch_update_segments",
        arguments: { project_id: script.project_id, updates: patches },
        result_summary: `已更新片段 [${updatedIds.join(", ")}] 为愤怒语气`,
      });
      if (updatedIds.length > 0) {
        await synthesizeSegment(updatedIds);
        logs.push({
          tool_name: "synthesize_segments",
          arguments: { project_id: script.project_id, segment_ids: updatedIds },
          result_summary: "完成分组重合成",
        });
      }
      return {
        reply: `已将 0–5 秒内 ${matched.length} 条片段切换为愤怒语气并重合成。`,
        toolLogs: logs,
      };
    }

    const candidates = script.segments.filter((seg) => !seg.keep_original);
    logs.push({
      tool_name: "query_script",
      arguments: { project_id: script.project_id },
      result_summary: `扫描 ${candidates.length} 条片段`,
    });

    const longSegs = candidates.filter((seg) => {
      const slot = Math.max(0.2, seg.end - seg.start);
      return (
        (seg.actual_duration != null && seg.actual_duration > slot * 1.15) ||
        seg.target_text.length > Math.floor(slot * 5) + 4
      );
    });

    if (longSegs.length > 0) {
      const patches: SegmentPatch[] = longSegs.map((seg) => ({
        id: seg.id,
        target_text:
          seg.target_text.length > 16 ? `${seg.target_text.slice(0, 14)}！` : seg.target_text,
      }));
      batchUpdateSegmentsLocal(patches);
      logs.push({
        tool_name: "batch_update_segments",
        arguments: { project_id: script.project_id, updates: patches },
        result_summary: `精简 ${patches.length} 条偏长台词`,
      });
    }

    const pendingIds = (longSegs.length > 0 ? longSegs : candidates.slice(-1)).map((s) => s.id);
    await synthesizeSegment(pendingIds);
    logs.push({
      tool_name: "synthesize_segments",
      arguments: { project_id: script.project_id, segment_ids: pendingIds },
      result_summary: `重合成片段 [${pendingIds.join(", ")}]`,
    });

    return {
      reply: `已完成超长句反思精简，并重合成片段 [${pendingIds.join(", ")}]。`,
      toolLogs: logs,
    };
  };

  const handleSend = async (rawText: string) => {
    const trimmed = rawText.trim();
    if (!trimmed || !script || isSending) {
      return;
    }

    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: trimmed,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsSending(true);

    if (backendStatus === "offline") {
      const sim = await runSimulatedAgent(trimmed);
      setMessages((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}`,
          role: "assistant",
          content: sim.reply,
          toolLogs: sim.toolLogs,
        },
      ]);
      setIsSending(false);
      return;
    }

    try {
      const res = await sendAgentChat(script.project_id, trimmed, true);
      setScript(res.script);
      bumpAudioVersion(res.synthesized_ids ?? []);
      setMessages((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}`,
          role: "assistant",
          content: res.reply,
          toolLogs: res.tool_logs,
        },
      ]);
    } catch (err) {
      const isUnavailableOrOffline =
        err instanceof ApiError &&
        (err.status === 502 || err.status === 504 || err.isNotImplemented || err.isNetworkError);
      if (isUnavailableOrOffline) {
        const sim = await runSimulatedAgent(trimmed);
        setMessages((prev) => [
          ...prev,
          {
            id: `a-${Date.now()}`,
            role: "assistant",
            content: sim.reply,
            toolLogs: sim.toolLogs,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            id: `a-${Date.now()}`,
            role: "assistant",
            content: err instanceof Error ? err.message : "请求失败",
          },
        ]);
      }
    } finally {
      setIsSending(false);
    }
  };

  return (
    <div className="flex h-full flex-col justify-between overflow-hidden bg-[#222222] text-[12px]">
      {/* 快捷指令条（无边框极淡灰底） */}
      <div className="flex flex-wrap gap-1.5 border-b border-dashed border-white/[0.08] py-2 pl-3.5 pr-4">
        {QUICK_CHIPS.map((chip) => (
          <button
            key={chip}
            type="button"
            disabled={isSending || !script}
            onClick={() => void handleSend(chip)}
            className="h-6 rounded-[4px] bg-[#2C2C2C] px-2 text-[11px] text-secondary transition-colors hover:bg-[#363636] hover:text-primary disabled:opacity-40"
          >
            {chip}
          </button>
        ))}
      </div>

      {/* 消息列表 */}
      <div className="flex-1 space-y-2.5 overflow-y-auto py-3 pl-3.5 pr-4">
        {messages.map((msg) => {
          const isExpanded = Boolean(expandedMessages[msg.id]);
          return (
            <div
              key={msg.id}
              className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}
            >
              <div
                className={`max-w-[92%] rounded-[4px] px-2.5 py-2 leading-relaxed ${
                  msg.role === "user" ? "bg-[#333333] text-[#FFFFFF]" : "bg-[#2A2A2A] text-primary"
                }`}
              >
                <p>{msg.content}</p>

                {msg.toolLogs && msg.toolLogs.length > 0 && (
                  <div className="mt-2">
                    <button
                      type="button"
                      onClick={() => toggleExpanded(msg.id)}
                      className="inline-flex items-center gap-1.5 rounded-[4px] bg-[#222222] px-2 py-0.5 text-[11px] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
                    >
                      <span className="h-1.5 w-1.5 rounded-full bg-status-ready" />
                      <Wrench className="h-3 w-3" strokeWidth={1.5} />
                      <span className="font-mono tabular-nums">
                        已执行 {msg.toolLogs.length} 项工具操作
                      </span>
                      {isExpanded ? (
                        <ChevronDown className="h-3 w-3" strokeWidth={1.5} />
                      ) : (
                        <ChevronRight className="h-3 w-3" strokeWidth={1.5} />
                      )}
                    </button>

                    {isExpanded && (
                      <div className="mt-2 space-y-1.5 border-t border-dashed border-white/[0.08] pt-2">
                        {msg.toolLogs.map((log, idx) => (
                          <div
                            key={idx}
                            className="rounded-[4px] bg-[#161616] px-2 py-1.5 text-[11px]"
                          >
                            <div className="font-mono text-primary">{log.tool_name}</div>
                            <div className="mt-0.5 text-secondary">{log.result_summary}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* 底部输入框（#181818 哑光凹槽 + 实心钢蓝发送） */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void handleSend(input);
        }}
        className="flex gap-2 border-t border-white/[0.06] bg-[#222222] py-2.5 pl-3.5 pr-4"
      >
        <input
          type="text"
          value={input}
          disabled={!script}
          onChange={(e) => setInput(e.target.value)}
          placeholder={script ? "输入剪辑或调音指令..." : "请先打开项目"}
          className="h-7 min-w-0 flex-1 rounded-[4px] border border-white/[0.07] bg-[#181818] px-2.5 text-[12px] text-primary placeholder:text-muted transition-colors focus:border-white/25 focus:outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={isSending || !input.trim() || !script}
          className="flex h-7 shrink-0 items-center gap-1 whitespace-nowrap rounded-[4px] bg-[#3A6EA5] px-2.5 py-1 text-[12px] font-medium text-[#FFFFFF] transition-colors hover:bg-[#467DB8] disabled:opacity-40"
        >
          <Send className="h-3.5 w-3.5" strokeWidth={1.5} />
          <span>发送</span>
        </button>
      </form>
    </div>
  );
};
