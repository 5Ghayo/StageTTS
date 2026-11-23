"""StageTTS 核心数据契约（Pydantic v2）。

必须与 ARCHITECTURE.md 及 frontend/src/types/script.ts 保持 1:1 严格一致。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

ProjectMode = Literal["video_dubbing", "virtual_stage"]
EmotionType = Literal["neutral", "angry", "happy", "sad", "whisper"]
SegmentStatus = Literal["pending", "synthesizing", "ready", "error"]
AvatarType = Literal["live2d", "vrm"]
SubtitleBurnMode = Literal["none", "target", "bilingual"]
SubtitleLayout = Literal["target", "bilingual"]


class CinematicStems(BaseModel):
    dialogue: str
    singing: str
    music: str
    effects: str


class StemVolumes(BaseModel):
    """Banquet 四轨分离后各背景子轨独立音量系数（0.0 ~ 2.0）。"""

    singing: float = Field(default=1.0, ge=0.0, le=2.0, description="歌声轨音量系数")
    music: float = Field(default=1.0, ge=0.0, le=2.0, description="伴奏音乐轨音量系数")
    effects: float = Field(default=1.0, ge=0.0, le=2.0, description="环境与音效轨音量系数")


class MediaInfo(BaseModel):
    """项目关联音视频资源与全局混音参数。"""

    source_video: Optional[str] = Field(default=None, description="源视频或源音频 URL")
    vocals_wav: Optional[str] = Field(default=None, description="识别和参考音使用的对白轨 URL")
    bgm_wav: Optional[str] = Field(default=None, description="歌声、伴奏和音效合成的背景轨 URL")
    cinematic_stems: Optional[CinematicStems] = Field(default=None, description="影视四轨分离结果")
    stem_volumes: Optional[StemVolumes] = Field(default=None, description="四轨分离背景子轨独立音量系数")
    output_video: Optional[str] = Field(default=None, description="最终混音成片 URL")
    duration: float = Field(ge=0.0, description="媒体总时长（秒）")
    bgm_volume: float = Field(default=0.75, ge=0.0, le=2.0, description="BGM 混音音量系数")
    auto_ducking: bool = Field(default=True, description="人声出现时是否自动闪避背景音乐")


class EmotionRef(BaseModel):
    """单种情绪对应的 GPT-SoVITS 参考音频元数据。"""

    wav: str = Field(description="参考音频路径或 URL")
    text: str = Field(description="参考音频对应台词文本")
    lang: str = Field(default="zh", description="参考音频语种代码")


class AvatarConfig(BaseModel):
    """虚拟舞台角色形象配置（模式 B）。"""

    type: AvatarType = Field(description="虚拟形象渲染类型：live2d 或 vrm")
    model_url: str = Field(description="模型入口文件 URL（如 .model3.json 或 .vrm）")


class CharacterConfig(BaseModel):
    """说话人音色权重、情绪参考库与虚拟形象配置。"""

    display_name: str = Field(description="前端展示名称")
    speaking_style: str = ""
    color: str = Field(default="#3b82f6", description="时间轴轨道色块 Hex 颜色")
    gpt_weights: Optional[str] = Field(default=None, description="GPT-SoVITS GPT 模型路径，留空走零样本兜底")
    sovits_weights: Optional[str] = Field(default=None, description="GPT-SoVITS SoVITS 模型路径，留空走零样本兜底")
    emotion_refs: Dict[EmotionType, EmotionRef] = Field(
        default_factory=dict,
        json_schema_extra={"default": {}},
        description="按情绪标签映射的参考音频字典",
    )
    avatar: Optional[AvatarConfig] = Field(default=None, description="虚拟演播室形象配置")


class AlignedWord(BaseModel):
    word: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None


class Segment(BaseModel):
    """时间轴单句台词片段。"""

    id: int = Field(ge=0, description="片段唯一整数 ID")
    start: float = Field(ge=0.0, description="起始时间（秒）")
    end: float = Field(gt=0.0, description="结束时间（秒）")
    speaker: str = Field(description="关联 characters 的说话人 Key")
    orig_text: str = Field(default="", description="原片识别文本")
    words: List[AlignedWord] = Field(default_factory=list, json_schema_extra={"default": []})
    alignment_needs_review: bool = True
    speaker_needs_review: bool = True
    target_text: str = Field(default="", description="目标配音文本")
    target_lang: str = Field(default="zh", description="目标语种代码")
    emotion: EmotionType = Field(default="neutral", description="情绪标签")
    action: Optional[str] = Field(default=None, description="虚拟舞台触发动作标签")
    keep_original: bool = Field(default=False, description="为 True 时保留原声不走 TTS 替换")
    speed_factor: float = Field(default=1.0, ge=0.5, le=2.0, description="温和变速系数（反思闭环后建议 <= 1.15）")
    tts_wav: Optional[str] = Field(default=None, description="合成后单句音频 URL")
    applied_speed_factor: Optional[float] = Field(default=None, ge=0.5, le=2.0, description="已写入 WAV 的倍率，旧音频为空")
    actual_duration: Optional[float] = Field(default=None, ge=0.0, description="实际合成音频时长（秒）")
    status: SegmentStatus = Field(default="pending", description="片段当前合成状态")

    @model_validator(mode="after")
    def _validate_time_range(self) -> "Segment":
        if self.end <= self.start:
            raise ValueError(f"Segment {self.id}: end ({self.end}) must be greater than start ({self.start})")
        return self

    def invalidate_audio(self) -> None:
        self.tts_wav = None
        self.applied_speed_factor = None
        self.actual_duration = round(self.end - self.start, 3) if self.keep_original else None
        self.status = "ready" if self.keep_original else "pending"


class ProjectScript(BaseModel):
    """全链路唯一真相源：script.json 顶层契约模型。"""

    project_id: str = Field(min_length=1, description="项目唯一标识")
    mode: ProjectMode = Field(default="video_dubbing", description="工作模式")
    media: MediaInfo
    source_lang: str | None = None
    glossary: Dict[str, str] = Field(default_factory=dict, json_schema_extra={"default": {}})
    characters: Dict[str, CharacterConfig] = Field(
        default_factory=dict, json_schema_extra={"default": {}},
    )
    segments: List[Segment] = Field(default_factory=list, json_schema_extra={"default": []})


# ---------------------------------------------------------------------------
# REST API 与 Agent 工具请求/响应模型
# ---------------------------------------------------------------------------


class ReadinessInfo(BaseModel):
    """后端外部依赖与环境就绪状态。"""

    ffmpeg: bool
    tts_service: bool
    tts_auto_manage: bool
    tts_url: str
    llm_configured: bool
    hf_token_configured: bool
    banquet_configured: bool
    env_parse_python: str


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str
    busy: bool
    workspace_dir: str
    assets_dir: str
    readiness: ReadinessInfo


class GpuDevice(BaseModel):
    """单张 NVIDIA GPU 显存状态（MiB）。"""

    name: str
    used_mib: int
    total_mib: int


class GpuStatusResponse(BaseModel):
    """GPU 显存状态列表响应。"""

    devices: List[GpuDevice] = Field(default_factory=list, json_schema_extra={"default": []})


class ProjectListResponse(BaseModel):
    """项目列表响应。"""

    projects: List[str] = Field(default_factory=list, json_schema_extra={"default": []})


class ReferenceUploadResponse(BaseModel):
    """参考音上传响应。"""

    wav: str


class UploadResponse(BaseModel):
    """项目媒体上传响应。"""

    project_id: str
    media_url: str
    duration: float


class ParseStatusResponse(BaseModel):
    """离线解析进度与日志响应。"""

    active: bool
    project_id: Optional[str] = None
    stage: str
    logs: List[str] = Field(default_factory=list, json_schema_extra={"default": []})
    log_count: int = 0
    error: Optional[str] = None
    updated_at: float


class ParseRequest(BaseModel):
    """触发上游 parse_worker.py 解析请求参数。"""

    target_lang: str = Field(default="zh", description="目标翻译语种")
    compute_type: Literal["float16", "int8"] = Field(default="float16", description="WhisperX 计算精度")


class SegmentPatch(BaseModel):
    """单句或批量更新字段补丁（仅提供需修改的字段）。"""

    id: int = Field(ge=0)
    speaker: Optional[str] = None
    orig_text: Optional[str] = None
    alignment_needs_review: Optional[bool] = None
    speaker_needs_review: Optional[bool] = None
    target_text: Optional[str] = None
    target_lang: Optional[str] = None
    emotion: Optional[EmotionType] = None
    action: Optional[str] = None
    keep_original: Optional[bool] = None
    speed_factor: Optional[float] = Field(default=None, ge=0.5, le=2.0)


class SynthesizeRequest(BaseModel):
    """批量或单句语音合成请求。"""

    segment_ids: Optional[List[int]] = Field(
        default=None,
        description="待合成片段 ID 列表；为空时合成全部 status != ready 且非 keep_original 的片段",
    )


class TranslationResponse(BaseModel):
    script: ProjectScript
    translated_ids: list[int]
    skipped_count: int
    errors: List[str] = Field(default_factory=list, json_schema_extra={"default": []})


class ReferencePreview(EmotionRef):
    segment_ids: list[int]


class SegmentDurationCheck(BaseModel):
    """单句时长质检报告（用于双向时长约束反思闭环）。"""

    segment_id: int
    slot_duration: float
    actual_duration: float
    fill_ratio: float
    duration_state: Literal["too_short", "normal", "too_long"]
    overflow_ratio: float
    overflow_seconds: float
    needs_reflection: bool


class SynthesizeResponse(BaseModel):
    """语音合成接口返回结果。"""

    project_id: str
    synthesized_ids: List[int] = Field(default_factory=list, json_schema_extra={"default": []})
    overflow_segments: List[SegmentDurationCheck] = Field(
        default_factory=list, json_schema_extra={"default": []},
    )
    errors: List[str] = Field(default_factory=list, json_schema_extra={"default": []})
    script: ProjectScript


class AgentChatRequest(BaseModel):
    """自然语言剪辑 Agent 对话请求。"""

    message: str = Field(min_length=1, description="用户自然语言编辑指令")
    auto_synthesize: bool = Field(default=True, description="修改后是否允许 Agent 自动触发合成与时长反思闭环")


class ToolCallLog(BaseModel):
    """供前端 AgentChat 展示的工具调用审计日志。"""

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict, json_schema_extra={"default": {}})
    result_summary: str = ""


class AgentChatResponse(BaseModel):
    """Agent 对话接口返回结果。"""

    project_id: str
    reply: str
    tool_logs: List[ToolCallLog] = Field(default_factory=list, json_schema_extra={"default": []})
    synthesized_ids: List[int] = Field(default_factory=list, json_schema_extra={"default": []})
    script: ProjectScript


class ExportDirOption(BaseModel):
    """前端导出弹窗可选本机目录。"""

    label: str
    path: str


class ExportDirsResponse(BaseModel):
    """工作区与本机推荐导出目录列表。"""

    workspace_dir: str
    suggested_dirs: List[ExportDirOption] = Field(
        default_factory=list, json_schema_extra={"default": []},
    )


class RenderRequest(BaseModel):
    """成片导出混音请求。"""

    burn_watermark: bool = Field(
        default=False,
        description="为 True 时通过重编码烧录“AI 合成语音”硬水印；False 时使用 -c:v copy 快速无损导出",
    )
    burn_subtitles: SubtitleBurnMode = Field(
        default="none",
        description="是否将 ASS 字幕烧录进成片画面：none / target / bilingual",
    )
    export_ass_sidecar: Optional[SubtitleLayout] = Field(
        default=None,
        description="可选：同步导出同名外挂 ASS 字幕（target / bilingual）",
    )
    auto_ducking: Optional[bool] = Field(
        default=None,
        description="是否启用 BGM 自动闪避；为空时沿用 script.media.auto_ducking",
    )
    output_filename: Optional[str] = Field(
        default=None,
        description="可选自定义输出文件名（如 my_dub.mp4）",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="可选本机额外保存目录绝对路径（支持 ~ 展开）",
    )


class RenderResponse(BaseModel):
    """成片导出返回结果。"""

    project_id: str
    output_video: str
    duration: float
    burn_watermark: bool
    burn_subtitles: SubtitleBurnMode = "none"
    auto_ducking: bool = True
    saved_path: str = ""
    subtitle_url: Optional[str] = None
    subtitle_saved_path: Optional[str] = None
