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
    raw = json.loads(index_path.read_text(encoding="utf-8"))
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
                src_path = config.resolve_media_path(ref.wav)
                if not src_path.is_file():
                    raise ValueError(f"情绪 {emotion} 的参考音频文件不存在: {ref.wav}")
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
    """删除指定全局音色预设并清理其隔离音频目录。"""
    safe_id = _validate_preset_id(preset_id)
    with _PRESET_LOCK:
        presets = _load_presets_locked()
        remaining = [item for item in presets if item.preset_id != safe_id]
        if len(remaining) == len(presets):
            raise FileNotFoundError(f"音色预设 {safe_id} 不存在")
        _save_presets_locked(remaining)

    shutil.rmtree(_presets_audio_dir() / safe_id, ignore_errors=True)
    return safe_id
