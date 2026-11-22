"""StageTTS 主后端 FastAPI 入口（运行在 env_backend，纯 CPU 0 显存占用）。"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.agent.orchestrator import run_agent_chat
from backend.agent.tools import (
    load_project_script,
    project_task_guard,
    render_final_video,
    save_project_script,
    synthesize_project,
)
from backend.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    MediaInfo,
    ParseRequest,
    ProjectMode,
    ProjectScript,
    ReferencePreview,
    RenderRequest,
    RenderResponse,
    SynthesizeRequest,
    SynthesizeResponse,
    TranslationResponse,
)
from backend.services.media_mixer import check_ffmpeg_available
from backend.services.references import automatic_reference, store_reference_audio
from backend.services.translation import LLMNotConfiguredError, create_openai_client, translate_script
from backend.services.task_runner import (
    GpuBusyError,
    ParseWorkerError,
    get_parse_progress,
    is_gpu_busy,
    probe_media_duration,
    read_gpu_memory,
    run_parse_worker,
)
from backend.services.tts_client import (
    TTSServiceUnavailableError,
    check_tts_service,
)

# 确保挂载目录存在，避免 StaticFiles 初始化报错
config.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)

_PROJECT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

app = FastAPI(
    title="StageTTS Backend API",
    version="0.2.0",
    description="基于剧本中间表示 script.json 与 Agent 协同的多角色配音及虚拟演播平台后端服务",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件挂载：供前端加载视频、纯人声/BGM、单句 TTS 音频以及 Live2D/VRM 模型
app.mount("/workspace", StaticFiles(directory=str(config.WORKSPACE_DIR)), name="workspace")
app.mount("/assets", StaticFiles(directory=str(config.ASSETS_DIR)), name="assets")


# ---------------------------------------------------------------------------
# 统一业务异常映射（保持接口错误码语义清晰）
# ---------------------------------------------------------------------------


def _err_detail(exc: Exception) -> str:
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


@app.exception_handler(GpuBusyError)
def _gpu_busy_handler(_: Request, exc: GpuBusyError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": _err_detail(exc)})


@app.exception_handler(ParseWorkerError)
def _parse_worker_handler(_: Request, exc: ParseWorkerError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": _err_detail(exc)})


@app.exception_handler(TTSServiceUnavailableError)
def _tts_unavailable_handler(_: Request, exc: TTSServiceUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": _err_detail(exc)})


@app.exception_handler(LLMNotConfiguredError)
def _llm_unconfigured_handler(_: Request, exc: LLMNotConfiguredError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": _err_detail(exc)})


@app.exception_handler(FileNotFoundError)
def _not_found_handler(_: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": _err_detail(exc)})


@app.exception_handler(ValueError)
def _value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": _err_detail(exc)})


@app.exception_handler(KeyError)
def _key_error_handler(_: Request, exc: KeyError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": _err_detail(exc)})


@app.exception_handler(NotImplementedError)
def _not_implemented_handler(_: Request, exc: NotImplementedError) -> JSONResponse:
    return JSONResponse(status_code=501, content={"detail": _err_detail(exc)})


def _validate_project_id(raw_id: str) -> str:
    cleaned = raw_id.strip()
    if not _PROJECT_ID_RE.match(cleaned):
        raise ValueError("project_id 仅允许字母、数字、下划线或连字符（1~64 位）")
    return cleaned


def _find_input_media(project_id: str, script: ProjectScript | None) -> Path:
    if script and script.media.source_video:
        candidate = config.resolve_media_path(script.media.source_video)
        if candidate.is_file():
            return candidate

    project_dir = config.WORKSPACE_DIR / project_id
    if project_dir.is_dir():
        for match in sorted(project_dir.glob("input.*")):
            if match.is_file():
                return match
    raise FileNotFoundError(f"项目 {project_id} 尚未上传源视频或音频文件")


# ---------------------------------------------------------------------------
# 健康检查与项目剧本读写接口
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health_check() -> Dict[str, object]:
    """服务健康检查、GPU 单任务锁状态与外部环境就绪度探测。"""
    return {
        "status": "ok",
        "busy": is_gpu_busy(),
        "workspace_dir": str(config.WORKSPACE_DIR),
        "assets_dir": str(config.ASSETS_DIR),
        "readiness": {
            "ffmpeg": check_ffmpeg_available(),
            "tts_service": check_tts_service(),
            "tts_auto_manage": config.TTS_AUTO_MANAGE,
            "tts_url": config.TTS_API_URL,
            "llm_configured": config.is_llm_configured(),
            "hf_token_configured": config.is_hf_token_configured(),
            "env_parse_python": config.ENV_PARSE_PYTHON,
        },
    }


@app.get("/api/gpu")
def gpu_status() -> Dict[str, object]:
    return {"devices": read_gpu_memory()}


@app.get("/api/projects")
def list_projects() -> Dict[str, List[str]]:
    """列出 workspace/ 下所有包含 script.json 的有效项目 ID。"""
    projects: List[str] = []
    if config.WORKSPACE_DIR.exists():
        for item in sorted(config.WORKSPACE_DIR.iterdir()):
            if item.is_dir() and (item / "script.json").is_file():
                projects.append(item.name)
    return {"projects": projects}


@app.get("/api/projects/{project_id}/script", response_model=ProjectScript)
def get_project_script(project_id: str) -> ProjectScript:
    """读取指定项目的唯一真相源 script.json。"""
    safe_id = _validate_project_id(project_id)
    return load_project_script(safe_id)


@app.put("/api/projects/{project_id}/script", response_model=ProjectScript)
def put_project_script(project_id: str, payload: ProjectScript) -> ProjectScript:
    """校验并保存前端修改后的完整 ProjectScript。"""
    safe_id = _validate_project_id(project_id)
    if payload.project_id != safe_id:
        raise ValueError(
            f"路径 project_id ({safe_id}) 与请求体 project_id ({payload.project_id}) 不一致"
        )
    save_project_script(payload)
    return payload


@app.post("/api/projects/{project_id}/references")
def upload_project_reference(project_id: str, file: UploadFile = File(...)) -> dict[str, str]:
    safe_id = _validate_project_id(project_id)
    load_project_script(safe_id)
    if file.size is not None and file.size > 32 * 1024 * 1024:
        raise ValueError("参考音文件不能超过 32 MB")
    return {"wav": store_reference_audio(safe_id, file.file)}


@app.post("/api/projects/{project_id}/characters/{speaker}/reference-preview", response_model=ReferencePreview)
def preview_automatic_reference(project_id: str, speaker: str) -> ReferencePreview:
    safe_id = _validate_project_id(project_id)
    with project_task_guard(safe_id):
        script = load_project_script(safe_id)
        if speaker not in script.characters:
            raise FileNotFoundError("角色不存在")
        ref = automatic_reference(script, speaker)
        if ref is None:
            raise ValueError("找不到 3–10 秒的同角色参考音。请确认台词原文、时间范围、说话人和源语种，或上传参考音")
        return ref


@app.post("/api/projects/{project_id}/translate", response_model=TranslationResponse)
def retranslate_project(project_id: str) -> TranslationResponse:
    safe_id = _validate_project_id(project_id)
    with project_task_guard(safe_id):
        script = load_project_script(safe_id)
        translated = translate_script(script, create_openai_client())
        if translated:
            save_project_script(script)
        return TranslationResponse(
            script=script, translated_ids=translated, skipped_count=len(script.segments) - len(translated),
        )


# ---------------------------------------------------------------------------
# 上传、解析、分组 TTS 合成、Agent 协同与后期导出接口
# ---------------------------------------------------------------------------


@app.post("/api/projects/upload")
async def upload_project_media(
    file: UploadFile = File(...),
    mode: ProjectMode = Form("video_dubbing"),
    project_id: str = Form(""),
) -> Dict[str, object]:
    """接收用户上传视频/音频，初始化 workspace/{project_id}/ 与初始 script.json。"""
    safe_id = (
        _validate_project_id(project_id)
        if project_id.strip()
        else f"proj_{int(time.time())}"
    )

    project_dir = config.WORKSPACE_DIR / safe_id
    project_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename or "input.mp4").suffix.lower() or ".mp4"
    input_name = f"input{ext}"
    input_path = project_dir / input_name

    with input_path.open("wb") as out_fp:
        shutil.copyfileobj(file.file, out_fp)

    duration = probe_media_duration(input_path)
    media_url = f"/workspace/{safe_id}/{input_name}"

    initial_script = ProjectScript(
        project_id=safe_id,
        mode=mode,
        media=MediaInfo(
            source_video=media_url,
            duration=duration,
            bgm_volume=0.75,
        ),
        characters={},
        segments=[],
    )
    save_project_script(initial_script)

    return {
        "project_id": safe_id,
        "media_url": media_url,
        "duration": duration,
    }


@app.get("/api/projects/{project_id}/parse/status")
def get_project_parse_status(project_id: str) -> Dict[str, object]:
    """查询指定项目的离线解析实时阶段与日志（供前端节点流轮询）。"""
    safe_id = _validate_project_id(project_id)
    return get_parse_progress(safe_id)


@app.post("/api/projects/{project_id}/parse", response_model=ProjectScript)
def parse_project_media(project_id: str, req: ParseRequest) -> ProjectScript:
    """启动 env_parse 子进程执行离线解析并生成 script.json（受 GPU 单任务锁保护）。"""
    safe_id = _validate_project_id(project_id)
    with project_task_guard(safe_id):
        existing_script: ProjectScript | None = None
        try:
            existing_script = load_project_script(safe_id)
        except FileNotFoundError:
            existing_script = None

        input_path = _find_input_media(safe_id, existing_script)
        mode = existing_script.mode if existing_script else "video_dubbing"

        run_parse_worker(
            project_id=safe_id,
            input_path=input_path,
            target_lang=req.target_lang,
            compute_type=req.compute_type,
            mode=mode,
            allow_demucs_cpu=req.allow_demucs_cpu,
        )
        return load_project_script(safe_id)


@app.post("/api/projects/{project_id}/synthesize", response_model=SynthesizeResponse)
def synthesize_project_segments(project_id: str, req: SynthesizeRequest) -> SynthesizeResponse:
    """分组配音并执行时长闭环，返回最终剧本和剩余超长。"""
    return synthesize_project(_validate_project_id(project_id), req.segment_ids)


@app.post("/api/projects/{project_id}/agent/chat", response_model=AgentChatResponse)
def agent_chat(project_id: str, req: AgentChatRequest) -> AgentChatResponse:
    """自然语言批量剪辑对话（5 大确定性原子工具调度 + 时长超标反思改写闭环）。"""
    safe_id = _validate_project_id(project_id)
    return run_agent_chat(
        project_id=safe_id,
        user_message=req.message,
        auto_synthesize=req.auto_synthesize,
    )


@app.post("/api/projects/{project_id}/render", response_model=RenderResponse)
def render_project_video(project_id: str, req: RenderRequest) -> RenderResponse:
    """执行 pydub 毫秒拼轨与 FFmpeg 混音封装，导出 output.mp4（或虚拟舞台音频）。"""
    safe_id = _validate_project_id(project_id)
    return RenderResponse(
        project_id=safe_id,
        **render_final_video(safe_id, burn_watermark=req.burn_watermark),
    )
