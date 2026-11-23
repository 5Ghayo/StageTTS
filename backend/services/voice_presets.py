"""跨项目全局音色预设库（存储于 assets/refs/presets.json 与 assets/refs/presets/）。"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from backend import config
from backend.schemas import EmotionRef, EmotionType, VoicePreset, VoicePresetCreateRequest

_PRESET_LOCK = threading.Lock()
_PRESET_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")
_ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus"}
_MAX_PRESET_REF_BYTES = 32 * 1024 * 1024


def _presets_root() -> Path:
    return config.ASSETS_DIR / "refs"


def _presets_index_path() -> Path:
    return _presets_root() / "presets.json"


def _presets_audio_dir() -> Path:
    return _presets_root() / "presets"


def _validate_preset_id(preset_id: str) -> str:
    if not _PRESET_ID_RE.match(preset_id):
        raise ValueError(f"非法音色预设 ID: {preset_id}")
    return preset_id


def _load_presets_locked() -> List[VoicePreset]:
    index_path = _presets_index_path()
    if not index_path.is_file():
        return []
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"音色预设索引文件损坏: {index_path}") from exc
    if not isinstance(raw, list):
        return []
    presets: List[VoicePreset] = []
    for item in raw:
        try:
            presets.append(VoicePreset.model_validate(item))
        except ValueError:
            continue
    return presets


def _save_presets_locked(presets: List[VoicePreset]) -> None:
    index_path = _presets_index_path()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.with_name(f".presets.{uuid.uuid4().hex[:6]}.tmp")
    payload = [preset.model_dump(mode="json") for preset in presets]
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, index_path)


def _resolve_safe_ref_path(raw_wav: str) -> Path:
    src_path = config.resolve_media_path(raw_wav)
    ws_root = config.WORKSPACE_DIR.resolve()
    assets_root = config.ASSETS_DIR.resolve()
    if ws_root not in src_path.parents and assets_root not in src_path.parents:
        raise ValueError("参考音频路径越界，仅允许引用项目工作区或资源目录内的音频")
    if not src_path.is_file():
        raise ValueError(f"参考音频文件不存在: {raw_wav}")
    if src_path.suffix.lower() not in _ALLOWED_AUDIO_EXTS:
        raise ValueError(f"不支持的参考音频格式: {src_path.suffix or '无后缀'}")
    if src_path.stat().st_size > _MAX_PRESET_REF_BYTES:
        raise ValueError("参考音频文件大小超过 32 MB 上限")
    return src_path


def _is_preset_referenced_by_projects(preset_id: str) -> bool:
    ws_dir = config.WORKSPACE_DIR
    if not ws_dir.is_dir():
        return False
    prefixes = (
        f"/assets/refs/presets/{preset_id}/",
        f"assets/refs/presets/{preset_id}/",
        f"refs/presets/{preset_id}/",
    )
    for script_path in ws_dir.glob("*/script.json"):
        try:
            raw = json.loads(script_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        characters = raw.get("characters")
        if not isinstance(characters, dict):
            continue
        for char in characters.values():
            if not isinstance(char, dict):
                continue
            refs = char.get("emotion_refs")
            if not isinstance(refs, dict):
                continue
            for ref in refs.values():
                if not isinstance(ref, dict):
                    continue
                wav_url = str(ref.get("wav") or "").replace("\\", "/")
                if any(wav_url.startswith(prefix) for prefix in prefixes):
                    return True
    return False


def list_voice_presets() -> List[VoicePreset]:
    """返回按创建时间倒序排列的全局音色预设。"""
    with _PRESET_LOCK:
        presets = _load_presets_locked()
    return sorted(presets, key=lambda item: item.created_at, reverse=True)


def create_voice_preset(req: VoicePresetCreateRequest) -> VoicePreset:
    """保存新音色预设，并将情绪参考音频复制到跨项目隔离目录。"""
    clean_name = req.name.strip()
    if not clean_name:
        raise ValueError("音色预设名称不能为空")

    preset_id = f"preset_{uuid.uuid4().hex[:10]}"
    preset_audio_dir = _presets_audio_dir() / preset_id
    copied_refs: Dict[EmotionType, EmotionRef] = {}

    try:
        if req.emotion_refs:
            preset_audio_dir.mkdir(parents=True, exist_ok=True)
            for emotion, ref in req.emotion_refs.items():
                try:
                    src_path = _resolve_safe_ref_path(ref.wav)
                except ValueError as exc:
                    raise ValueError(f"情绪 {emotion} 的参考音频无效: {exc}") from exc
                suffix = src_path.suffix.lower() or ".wav"
                dest_filename = f"{emotion}{suffix}"
                dest_path = preset_audio_dir / dest_filename
                shutil.copy2(src_path, dest_path)
                copied_refs[emotion] = EmotionRef(
                    wav=f"/assets/refs/presets/{preset_id}/{dest_filename}",
                    text=ref.text,
                    lang=ref.lang,
                )

        preset = VoicePreset(
            preset_id=preset_id,
            name=clean_name,
            speaking_style=req.speaking_style.strip(),
            gpt_weights=req.gpt_weights.strip() if req.gpt_weights else None,
            sovits_weights=req.sovits_weights.strip() if req.sovits_weights else None,
            emotion_refs=copied_refs,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with _PRESET_LOCK:
            presets = _load_presets_locked()
            presets.append(preset)
            _save_presets_locked(presets)
        return preset
    except Exception:
        shutil.rmtree(preset_audio_dir, ignore_errors=True)
        raise


def delete_voice_preset(preset_id: str) -> str:
    """删除指定全局音色预设；若无项目引用则清理其隔离音频目录。"""
    safe_id = _validate_preset_id(preset_id)
    with _PRESET_LOCK:
        presets = _load_presets_locked()
        remaining = [item for item in presets if item.preset_id != safe_id]
        if len(remaining) == len(presets):
            raise FileNotFoundError(f"音色预设 {safe_id} 不存在")
        _save_presets_locked(remaining)

    if not _is_preset_referenced_by_projects(safe_id):
        shutil.rmtree(_presets_audio_dir() / safe_id, ignore_errors=True)
    return safe_id
