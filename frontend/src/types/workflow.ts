/**
 * ComfyUI 节点流管线与 Premiere 工作区相关类型定义。
 */

import type { SubtitleBurnMode } from "./script";

export type WorkspacePreset = "editing" | "workflow" | "stage";

export type CenterViewTab = "video" | "stage" | "graph";

export type RightPanelTab = "inspector" | "speakers" | "agent" | "node" | "logs";

export type LeftDrawerTab = "none" | "characters" | "media";

export type EnvBadge = "env_parse" | "env_tts" | "env_backend" | "frontend";

export type NodeCategory = "input" | "parse" | "hub" | "synthesis" | "output";

export type NodeRunStatus = "idle" | "queued" | "running" | "done" | "error";

export interface PortSpec {
  id: string;
  label: string;
  type: string;
}

export interface StageNodeData extends Record<string, unknown> {
  title: string;
  category: NodeCategory;
  env: EnvBadge;
  vramGb: number | null;
  status: NodeRunStatus;
  inputs: PortSpec[];
  outputs: PortSpec[];
  summary: string;
  params: Record<string, string | number | boolean>;
}

export type PipelineStartStep = "parse" | "translate" | "synthesize" | "render";

export function pipelineStartForNode(nodeId: string | null): PipelineStartStep | null {
  switch (nodeId) {
    case "media_input":
    case "banquet": return "parse";
    case "llm_tagger": return "translate";
    case "script_hub":
    case "agent_tts": return "synthesize";
    case "ffmpeg_mixer": return "render";
    default: return null;
  }
}

export function pipelineNodeHint(nodeId: string | null): string {
  if (nodeId === "whisperx") return "WhisperX 识别节点不能单独启动；请从音轨分离节点运行完整解析，或选择翻译节点处理已有对白。";
  if (nodeId === "virtual_stage_out") return "舞台节点请使用右侧「进入虚拟舞台」。";
  return "请先在画布中选中一个可运行节点。";
}

export interface PipelineConfig {
  targetLang: "zh" | "en" | "ja";
  burnWatermark: boolean;
  burnSubtitles: SubtitleBurnMode;
}

export interface PipelineLogEntry {
  id: string;
  timestamp: string;
  env: EnvBadge | "system";
  message: string;
  level: "info" | "success" | "warn" | "error";
}
