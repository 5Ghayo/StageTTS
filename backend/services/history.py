"""项目剧本历史快照持久化与恢复服务（存储于 workspace/{project_id}/.history/）。"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from backend import config
from backend.schemas import ProjectScript, ScriptSnapshotMeta

MAX_SNAPSHOTS = 20
_SNAPSHOT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")


def _history_dir(project_id: str) -> Path:
    return config.WORKSPACE_DIR / project_id / ".history"


def _validate_snapshot_id(snapshot_id: str) -> str:
    if not _SNAPSHOT_ID_RE.match(snapshot_id):
        raise ValueError(f"非法快照 ID: {snapshot_id}")
    return snapshot_id


def _atomic_write_json(target: Path, payload: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{uuid.uuid4().hex[:6]}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, target)


def _prune_old_snapshots(history_dir: Path, max_keep: int = MAX_SNAPSHOTS) -> None:
    files = sorted(history_dir.glob("*.json"))
    excess = len(files) - max_keep
    if excess <= 0:
        return
    for stale in files[:excess]:
        try:
            stale.unlink()
        except OSError:
            pass


def create_snapshot(
    script: ProjectScript,
    label: str = "手动保存",
) -> Optional[ScriptSnapshotMeta]:
    """为当前 ProjectScript 创建去重历史快照，并保留最近 MAX_SNAPSHOTS 条记录。"""
    history_dir = _history_dir(script.project_id)
    history_dir.mkdir(parents=True, exist_ok=True)

    script_data = script.model_dump(mode="json")
    existing_files = sorted(history_dir.glob("*.json"))
    if existing_files:
        latest_file = existing_files[-1]
        try:
            latest_payload = json.loads(latest_file.read_text(encoding="utf-8"))
            if latest_payload.get("script") == script_data:
                return None
        except (OSError, json.JSONDecodeError):
            pass

    snapshot_id = f"{int(time.time() * 1000):013d}_{uuid.uuid4().hex[:6]}"
    created_at = datetime.now(timezone.utc).isoformat()
    clean_label = (label or "手动保存").strip()[:80] or "手动保存"
    meta = ScriptSnapshotMeta(
        snapshot_id=snapshot_id,
        created_at=created_at,
        label=clean_label,
        segment_count=len(script.segments),
    )
    payload = {
        **meta.model_dump(mode="json"),
        "script": script_data,
    }
    _atomic_write_json(history_dir / f"{snapshot_id}.json", payload)
    _prune_old_snapshots(history_dir)
    return meta


def list_snapshots(project_id: str) -> List[ScriptSnapshotMeta]:
    """按时间倒序列出项目的历史快照元数据。"""
    history_dir = _history_dir(project_id)
    if not history_dir.is_dir():
        return []

    items: List[ScriptSnapshotMeta] = []
    for path in sorted(history_dir.glob("*.json"), reverse=True):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            items.append(
                ScriptSnapshotMeta(
                    snapshot_id=raw["snapshot_id"],
                    created_at=raw["created_at"],
                    label=raw.get("label", "历史快照"),
                    segment_count=int(raw.get("segment_count", 0)),
                )
            )
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return items


def load_snapshot_script(project_id: str, snapshot_id: str) -> ProjectScript:
    """读取并校验指定历史快照的完整 ProjectScript。"""
    safe_snap_id = _validate_snapshot_id(snapshot_id)
    snap_path = _history_dir(project_id) / f"{safe_snap_id}.json"
    if not snap_path.is_file():
        raise FileNotFoundError(f"历史快照 {safe_snap_id} 不存在")
    raw = json.loads(snap_path.read_text(encoding="utf-8"))
    script = ProjectScript.model_validate(raw["script"])
    if script.project_id != project_id:
        raise ValueError(f"快照 {safe_snap_id} 不属于项目 {project_id}")
    return script
