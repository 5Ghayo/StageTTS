/**
 * StageTTS 核心 TypeScript 数据契约。
 * 由 OpenAPI（backend/schemas.py -> api.generated.ts）自动生成并统一导出，避免手写双份类型漂移。
 */

import type { components } from "./api.generated";

export type CinematicStems = components["schemas"]["CinematicStems"];
export type MediaInfo = components["schemas"]["MediaInfo"];
export type EmotionRef = components["schemas"]["EmotionRef"];
export type AvatarConfig = components["schemas"]["AvatarConfig"];
export type CharacterConfig = components["schemas"]["CharacterConfig"];
export type AlignedWord = components["schemas"]["AlignedWord"];
export type Segment = components["schemas"]["Segment"];
export type ProjectScript = components["schemas"]["ProjectScript"];

export type ProjectMode = ProjectScript["mode"];
export type EmotionType = Segment["emotion"];
export type SegmentStatus = Segment["status"];
export type AvatarType = AvatarConfig["type"];

export type SegmentPatch = { id: Segment["id"] } & Partial<
  Pick<
    Segment,
    | "speaker"
    | "orig_text"
    | "alignment_needs_review"
    | "speaker_needs_review"
    | "target_text"
    | "target_lang"
    | "emotion"
    | "action"
    | "keep_original"
    | "speed_factor"
  >
>;

export type ReadinessInfo = components["schemas"]["ReadinessInfo"];
export type HealthResponse = components["schemas"]["HealthResponse"];
export type GpuDevice = components["schemas"]["GpuDevice"];
export type GpuStatusResponse = components["schemas"]["GpuStatusResponse"];
export type ProjectListResponse = components["schemas"]["ProjectListResponse"];
export type ReferenceUploadResponse = components["schemas"]["ReferenceUploadResponse"];
export type UploadResponse = components["schemas"]["UploadResponse"];
export type ParseStatusResponse = components["schemas"]["ParseStatusResponse"];
export type ParseRequest = components["schemas"]["ParseRequest"];
export type ReferencePreview = components["schemas"]["ReferencePreview"];
export type TranslationResponse = components["schemas"]["TranslationResponse"];
export type SegmentDurationCheck = components["schemas"]["SegmentDurationCheck"];
export type SynthesizeResponse = components["schemas"]["SynthesizeResponse"];
export type ToolCallLog = components["schemas"]["ToolCallLog"];
export type AgentChatResponse = components["schemas"]["AgentChatResponse"];
export type RenderResponse = components["schemas"]["RenderResponse"];
