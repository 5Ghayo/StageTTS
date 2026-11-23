"""StageTTS 主后端 FastAPI 入口（运行在 env_backend，纯 CPU 0 显存占用）。"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import AsyncIterator, Dict, List

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.agent.orchestrator import run_agent_chat
from backend.agent.tools import (
    load_project_script,
    project_task_guard,
    render_final_video,
    restore_project_snapshot,
    save_project_script,
    synthesize_project,
)
from backend.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    ExportDirOption,
    ExportDirsResponse,
    GpuStatusResponse,
    HealthResponse,
    MediaInfo,
    ParseRequest,
    ParseStatusResponse,
    ProjectListResponse,
    ProjectMode,
    ProjectScript,
    ReferencePreview,
    ReferenceUploadResponse,
    RenderRequest,
    RenderResponse,
    ScriptSnapshotListResponse,
    SubtitleLayout,
    SynthesizeRequest,
    SynthesizeResponse,
    TranslationResponse,
    UploadResponse,
    VoicePreset,
    VoicePresetCreateRequest,
    VoicePresetDeleteResponse,
    VoicePresetListResponse,
)
from backend.services.history import create_snapshot, list_snapshots
from backend.services.media_mixer import check_ffmpeg_available
from backend.services.references import automatic_reference, store_reference_audio
from backend.services.subtitles import generate_ass_subtitles
from backend.services.task_runner import (
    GpuBusyError,
    ParseWorkerError,
    get_parse_progress,
    is_gpu_busy,
    probe_media_duration,
    read_gpu_memory,
    run_parse_worker,
)
from backend.services.translation import (
    LLMNotConfiguredError,
    LLMServiceError,
    create_openai_client,
    translate_script,
)
from backend.services.tts_client import (
    TTSServiceUnavailableError,
    check_tts_service,
)
from backend.services.voice_presets import (
    create_voice_preset,
    delete_voice_preset,
    list_voice_presets,
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


@app.exception_handler(LLMServiceError)
def _llm_service_error_handler(_: Request, exc: LLMServiceError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": _err_detail(exc)})


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


@app.get("/api/health", response_model=HealthResponse)
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
            "banquet_configured": config.is_banquet_configured(),
            "env_parse_python": config.ENV_PARSE_PYTHON,
        },
    }


@app.get("/api/gpu", response_model=GpuStatusResponse)
def gpu_status() -> Dict[str, object]:
    return {"devices": read_gpu_memory()}


@app.get("/api/projects", response_model=ProjectListResponse)
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
    """校验并保存前端修改后的完整 ProjectScript，并记录去重历史快照。"""
    safe_id = _validate_project_id(project_id)
    if payload.project_id != safe_id:
        raise ValueError(
            f"路径 project_id ({safe_id}) 与请求体 project_id ({payload.project_id}) 不一致"
        )
    save_project_script(payload)
    create_snapshot(payload, label="手动保存")
    return payload


@app.get("/api/projects/{project_id}/history", response_model=ScriptSnapshotListResponse)
def get_project_history(project_id: str) -> ScriptSnapshotListResponse:
    """列出项目的历史快照元数据列表。"""
    safe_id = _validate_project_id(project_id)
    return ScriptSnapshotListResponse(
        project_id=safe_id,
        snapshots=list_snapshots(safe_id),
    )


@app.post("/api/projects/{project_id}/history/{snapshot_id}/restore", response_model=ProjectScript)
def post_restore_project_history(project_id: str, snapshot_id: str) -> ProjectScript:
    """回滚恢复项目到指定历史快照。"""
    safe_id = _validate_project_id(project_id)
    return restore_project_snapshot(safe_id, snapshot_id)


@app.post("/api/projects/{project_id}/references", response_model=ReferenceUploadResponse)
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
        pre_translate_snapshot = script.model_copy(deep=True)
        batch_errors: list[str] = []
        translated = translate_script(script, create_openai_client(), errors=batch_errors)
        if not translated and batch_errors:
            raise LLMServiceError(f"台词重译失败：{batch_errors[0]}")
        if translated:
            create_snapshot(pre_translate_snapshot, label="重译前备份")
            save_project_script(script)
        return TranslationResponse(
            script=script,
            translated_ids=translated,
            skipped_count=len(script.segments) - len(translated),
            errors=batch_errors,
        )


# ---------------------------------------------------------------------------
# 上传、解析、分组 TTS 合成、Agent 协同与后期导出接口
# ---------------------------------------------------------------------------


@app.post("/api/projects/upload", response_model=UploadResponse)
def upload_project_media(
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

    with project_task_guard(safe_id):
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


@app.get("/api/projects/{project_id}/parse/status", response_model=ParseStatusResponse)
def get_project_parse_status(project_id: str) -> Dict[str, object]:
    """查询指定项目的离线解析实时阶段与日志（供前端节点流轮询）。"""
    safe_id = _validate_project_id(project_id)
    return get_parse_progress(safe_id)


@app.get("/api/projects/{project_id}/parse/stream")
async def stream_project_parse_status(project_id: str, request: Request) -> StreamingResponse:
    """SSE 实时推送指定项目的离线解析阶段与日志。"""
    safe_id = _validate_project_id(project_id)

    async def event_generator() -> AsyncIterator[str]:
        last_key: tuple[object, ...] | None = None
        saw_active = False
        idle_ticks = 0
        while not await request.is_disconnected():
            status = get_parse_progress(safe_id)
            active = bool(status.get("active"))
            if active:
                saw_active = True
            key = (active, status.get("stage"), status.get("log_count"), status.get("error"))
            if key != last_key:
                last_key = key
                yield f"data: {json.dumps(status, ensure_ascii=False)}\n\n"
            if not active and (saw_active or idle_ticks >= 15):
                break
            idle_ticks += 1
            await asyncio.sleep(0.4)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
        if existing_script and existing_script.segments:
            create_snapshot(existing_script, label="重新解析前")

        run_parse_worker(
            project_id=safe_id,
            input_path=input_path,
            target_lang=req.target_lang,
            compute_type=req.compute_type,
            mode=mode,
        )
        parsed = load_project_script(safe_id)
        create_snapshot(parsed, label="解析完成")
        return parsed


@app.post("/api/projects/{project_id}/synthesize", response_model=SynthesizeResponse)
def synthesize_project_segments(project_id: str, req: SynthesizeRequest) -> SynthesizeResponse:
    """分组配音并执行双向时长闭环，返回最终剧本和仍偏离时间槽的片段。"""
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


@app.get("/api/projects/{project_id}/subtitles")
def download_project_subtitles(
    project_id: str,
    layout: SubtitleLayout = "target",
) -> Response:
    """导出指定项目的 ASS 字幕文件（单语译文或双语对照）。"""
    safe_id = _validate_project_id(project_id)
    script = load_project_script(safe_id)
    ass_content = generate_ass_subtitles(script, layout=layout)
    return Response(
        content=ass_content.encode("utf-8-sig"),
        media_type="text/x-ssa; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_id}_{layout}.ass"',
            "Cache-Control": "no-store",
        },
    )


def _xdg_user_dir(kind: str, home: Path) -> Path | None:
    cmd = shutil.which("xdg-user-dir")
    if not cmd:
        return None
    try:
        proc = subprocess.run(
            [cmd, kind],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    raw = proc.stdout.strip() if proc.returncode == 0 else ""
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        return None
    try:
        resolved = candidate.resolve()
        if resolved == home.resolve():
            return None
    except OSError:
        return None
    return resolved


def _discover_user_export_dirs() -> List[ExportDirOption]:
    """按操作系统与本地化习惯探测可用的常用导出目录。"""
    home = Path.home()
    candidates: List[tuple[str, Path]] = []

    if sys.platform == "darwin":
        candidates.extend([
            ("影片目录", home / "Movies"),
            ("下载目录", home / "Downloads"),
            ("桌面", home / "Desktop"),
        ])
    elif os.name == "nt":
        candidates.extend([
            ("视频目录", home / "Videos"),
            ("下载目录", home / "Downloads"),
            ("桌面", home / "Desktop"),
        ])
    else:
        for label, xdg_key, fallbacks in (
            ("视频目录", "VIDEOS", ("Videos", "视频")),
            ("下载目录", "DOWNLOAD", ("Downloads", "下载")),
            ("桌面", "DESKTOP", ("Desktop", "桌面")),
        ):
            xdg_dir = _xdg_user_dir(xdg_key, home)
            if xdg_dir is not None:
                candidates.append((label, xdg_dir))
            for rel in fallbacks:
                candidates.append((label, home / rel))

    suggested: List[ExportDirOption] = []
    seen: set[Path] = set()
    for label, path in candidates:
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        suggested.append(ExportDirOption(label=label, path=str(resolved)))
    return suggested


@app.get("/api/export/dirs", response_model=ExportDirsResponse)
def get_export_dirs() -> ExportDirsResponse:
    """返回项目工作区及本机常用推荐导出目录。"""
    return ExportDirsResponse(
        workspace_dir=str(config.WORKSPACE_DIR.resolve()),
        suggested_dirs=_discover_user_export_dirs(),
    )


@app.post("/api/projects/{project_id}/render", response_model=RenderResponse)
def render_project_video(project_id: str, req: RenderRequest) -> RenderResponse:
    """执行 pydub 毫秒拼轨与 FFmpeg 混音封装，导出 output.mp4（或虚拟舞台音频）。"""
    safe_id = _validate_project_id(project_id)
    return RenderResponse(
        project_id=safe_id,
        **render_final_video(
            safe_id,
            burn_watermark=req.burn_watermark,
            burn_subtitles=req.burn_subtitles,
            export_ass_sidecar=req.export_ass_sidecar,
            auto_ducking=req.auto_ducking,
            output_filename=req.output_filename,
            output_dir=req.output_dir,
        ),
    )


@app.get("/api/voice-presets", response_model=VoicePresetListResponse)
def get_voice_presets() -> VoicePresetListResponse:
    """列出跨项目全局音色预设库。"""
    return VoicePresetListResponse(presets=list_voice_presets())


@app.post("/api/voice-presets", response_model=VoicePreset)
def post_voice_preset(req: VoicePresetCreateRequest) -> VoicePreset:
    """将当前角色音色配置及参考音频保存为跨项目预设。"""
    return create_voice_preset(req)


@app.delete("/api/voice-presets/{preset_id}", response_model=VoicePresetDeleteResponse)
def remove_voice_preset(preset_id: str) -> VoicePresetDeleteResponse:
    """删除全局音色预设并清理其隔离参考音频文件。"""
    return VoicePresetDeleteResponse(deleted_id=delete_voice_preset(preset_id))
