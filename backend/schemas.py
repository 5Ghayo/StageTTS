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


class MediaInfo(BaseModel):
    """项目关联音视频资源与全局混音参数。"""

    source_video: Optional[str] = Field(default=None, description="源视频或源音频 URL")
    vocals_wav: Optional[str] = Field(default=None, description="Demucs 分离纯人声 URL")
    bgm_wav: Optional[str] = Field(default=None, description="Demucs 分离背景伴奏 URL")
    output_video: Optional[str] = Field(default=None, description="最终混音成片 URL")
    duration: float = Field(ge=0.0, description="媒体总时长（秒）")
    bgm_volume: float = Field(default=0.75, ge=0.0, le=2.0, description="BGM 混音音量系数")


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
    color: str = Field(default="#3b82f6", description="时间轴轨道色块 Hex 颜色")
    gpt_weights: Optional[str] = Field(default=None, description="GPT-SoVITS GPT 模型路径，留空走零样本兜底")
    sovits_weights: Optional[str] = Field(default=None, description="GPT-SoVITS SoVITS 模型路径，留空走零样本兜底")
    emotion_refs: Dict[EmotionType, EmotionRef] = Field(
        default_factory=dict,
        description="按情绪标签映射的参考音频字典",
    )
    avatar: Optional[AvatarConfig] = Field(default=None, description="虚拟演播室形象配置")


class Segment(BaseModel):
    """时间轴单句台词片段。"""

    id: int = Field(ge=0, description="片段唯一整数 ID")
    start: float = Field(ge=0.0, description="起始时间（秒）")
    end: float = Field(gt=0.0, description="结束时间（秒）")
    speaker: str = Field(description="关联 characters 的说话人 Key")
    orig_text: str = Field(default="", description="原片识别文本")
    target_text: str = Field(default="", description="目标配音文本")
    target_lang: str = Field(default="zh", description="目标语种代码")
    emotion: EmotionType = Field(default="neutral", description="情绪标签")
    action: Optional[str] = Field(default=None, description="虚拟舞台触发动作标签")
    keep_original: bool = Field(default=False, description="为 True 时保留原声不走 TTS 替换")
    speed_factor: float = Field(default=1.0, ge=0.5, le=2.0, description="温和变速系数（反思闭环后建议 <= 1.15）")
    tts_wav: Optional[str] = Field(default=None, description="合成后单句音频 URL")
    actual_duration: Optional[float] = Field(default=None, ge=0.0, description="实际合成音频时长（秒）")
    status: SegmentStatus = Field(default="pending", description="片段当前合成状态")

    @model_validator(mode="after")
    def _validate_time_range(self) -> "Segment":
        if self.end <= self.start:
            raise ValueError(f"Segment {self.id}: end ({self.end}) must be greater than start ({self.start})")
        return self


class ProjectScript(BaseModel):
    """全链路唯一真相源：script.json 顶层契约模型。"""

    project_id: str = Field(min_length=1, description="项目唯一标识")
    mode: ProjectMode = Field(default="video_dubbing", description="工作模式")
    media: MediaInfo
    characters: Dict[str, CharacterConfig] = Field(default_factory=dict)
    segments: List[Segment] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# REST API 与 Agent 工具请求/响应模型
# ---------------------------------------------------------------------------


class ParseRequest(BaseModel):
    """触发上游 parse_worker.py 解析请求参数。"""

    target_lang: str = Field(default="zh", description="目标翻译语种")
    compute_type: Literal["float16", "int8"] = Field(default="float16", description="WhisperX 计算精度")


class SegmentPatch(BaseModel):
    """单句或批量更新字段补丁（仅提供需修改的字段）。"""

    id: int = Field(ge=0)
    speaker: Optional[str] = None
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


class SegmentDurationCheck(BaseModel):
    """单句时长质检报告（用于时长约束反思闭环）。"""

    segment_id: int
    slot_duration: float
    actual_duration: float
    overflow_ratio: float
    overflow_seconds: float
    needs_reflection: bool


class SynthesizeResponse(BaseModel):
    """语音合成接口返回结果。"""

    project_id: str
    synthesized_ids: List[int] = Field(default_factory=list)
    overflow_segments: List[SegmentDurationCheck] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    script: ProjectScript


class AgentChatRequest(BaseModel):
    """自然语言剪辑 Agent 对话请求。"""

    message: str = Field(min_length=1, description="用户自然语言编辑指令")
    auto_synthesize: bool = Field(default=True, description="修改后是否允许 Agent 自动触发合成与时长反思闭环")


class ToolCallLog(BaseModel):
    """供前端 AgentChat 展示的工具调用审计日志。"""

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""


class AgentChatResponse(BaseModel):
    """Agent 对话接口返回结果。"""

    project_id: str
    reply: str
    tool_logs: List[ToolCallLog] = Field(default_factory=list)
    script: ProjectScript


class RenderRequest(BaseModel):
    """成片导出混音请求。"""

    burn_watermark: bool = Field(
        default=False,
        description="为 True 时通过重编码烧录“AI 合成语音”硬水印；False 时使用 -c:v copy 快速无损导出",
    )


class RenderResponse(BaseModel):
    """成片导出返回结果。"""

    project_id: str
    output_video: str
    duration: float
    burn_watermark: bool
