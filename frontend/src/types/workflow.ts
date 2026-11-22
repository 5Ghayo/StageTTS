/**
 * ComfyUI 节点流管线与 Premiere 工作区相关类型定义。
 */

export type WorkspacePreset = "editing" | "workflow" | "stage";

export type CenterViewTab = "video" | "stage" | "graph";

export type RightPanelTab = "inspector" | "agent" | "node" | "logs";

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
  vramGb: number;
  status: NodeRunStatus;
  inputs: PortSpec[];
  outputs: PortSpec[];
  summary: string;
  params: Record<string, string | number | boolean>;
}

export interface PipelineConfig {
  computeType: "float16" | "int8";
  targetLang: "zh" | "en" | "ja";
  allowDemucsCpu: boolean;
  burnWatermark: boolean;
}

export interface VramState {
  usedGb: number;
  limitGb: number;
  activeEnv: EnvBadge | "idle";
}

export interface PipelineLogEntry {
  id: string;
  timestamp: string;
  env: EnvBadge | "system";
  message: string;
  level: "info" | "success" | "warn" | "error";
}
