/**
 * StageTTS 核心 TypeScript 数据契约。
 * 必须与 ARCHITECTURE.md 及 backend/schemas.py 保持 1:1 严格一致。
 */

export type ProjectMode = "video_dubbing" | "virtual_stage";

export type EmotionType = "neutral" | "angry" | "happy" | "sad" | "whisper";

export type SegmentStatus = "pending" | "synthesizing" | "ready" | "error";

export type AvatarType = "live2d" | "vrm";

export interface MediaInfo {
  source_video: string | null;
  vocals_wav: string | null;
  bgm_wav: string | null;
  output_video: string | null;
  duration: number;
  bgm_volume: number;
}

export interface EmotionRef {
  wav: string;
  text: string;
  lang: string;
}

export interface AvatarConfig {
  type: AvatarType;
  model_url: string;
}

export interface CharacterConfig {
  display_name: string;
  speaking_style?: string;
  color: string;
  gpt_weights: string | null;
  sovits_weights: string | null;
  emotion_refs: Partial<Record<EmotionType, EmotionRef>>;
  avatar: AvatarConfig | null;
}

export interface AlignedWord {
  word: string;
  start: number | null;
  end: number | null;
  speaker: string | null;
}

export interface Segment {
  id: number;
  start: number;
  end: number;
  speaker: string;
  orig_text: string;
  words?: AlignedWord[];
  alignment_needs_review?: boolean;
  speaker_needs_review?: boolean;
  target_text: string;
  target_lang: string;
  emotion: EmotionType;
  action: string | null;
  keep_original: boolean;
  speed_factor: number;
  tts_wav: string | null;
  applied_speed_factor?: number | null;
  actual_duration: number | null;
  status: SegmentStatus;
}

export interface ProjectScript {
  project_id: string;
  mode: ProjectMode;
  media: MediaInfo;
  source_lang?: string | null;
  glossary?: Record<string, string>;
  characters: Record<string, CharacterConfig>;
  segments: Segment[];
}

export interface SegmentPatch {
  id: number;
  speaker?: string;
  orig_text?: string;
  alignment_needs_review?: boolean;
  speaker_needs_review?: boolean;
  target_text?: string;
  target_lang?: string;
  emotion?: EmotionType;
  action?: string | null;
  keep_original?: boolean;
  speed_factor?: number;
}

export interface ReferencePreview extends EmotionRef {
  segment_ids: number[];
}

export interface TranslationResponse {
  script: ProjectScript;
  translated_ids: number[];
  skipped_count: number;
}

export interface SegmentDurationCheck {
  segment_id: number;
  slot_duration: number;
  actual_duration: number;
  overflow_ratio: number;
  overflow_seconds: number;
  needs_reflection: boolean;
}

export interface SynthesizeResponse {
  project_id: string;
  synthesized_ids: number[];
  overflow_segments: SegmentDurationCheck[];
  errors: string[];
  script: ProjectScript;
}

export interface ToolCallLog {
  tool_name: string;
  arguments: Record<string, unknown>;
  result_summary: string;
}

export interface AgentChatResponse {
  project_id: string;
  reply: string;
  tool_logs: ToolCallLog[];
  synthesized_ids: number[];
  script: ProjectScript;
}

export interface RenderResponse {
  project_id: string;
  output_video: string;
  duration: number;
  burn_watermark: boolean;
}
