"""Private, atomic application LLM preferences; never served as static assets."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr, field_validator
from backend import config

SETTINGS_PATH = config.ROOT_DIR / "backend" / "llm_settings.local.json"
_lock = RLock()


class LLMSettingsUpdate(BaseModel):
    base_url: str = Field(min_length=1, max_length=2048)
    model: str = Field(min_length=1, max_length=256)
    api_key: SecretStr | None = None
    clear_api_key: bool = False

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("接口地址必须是 HTTP(S) URL，且不能包含用户名、密码、查询参数或片段")
        return value

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(char) < 32 for char in value):
            raise ValueError("请填写有效的模型名称")
        return value


class LLMSettingsResponse(BaseModel):
    base_url: str
    model: str
    api_key_configured: bool
    configured: bool


@dataclass(frozen=True)
class LLMSnapshot:
    base_url: str
    model: str
    api_key: str = field(repr=False)

    def configured(self) -> bool:
        key = self.api_key
        if not key or key.startswith("your_"):
            return False
        if key in {"sk-xxx", "changeme", "local-gateway-no-key"}:
            return urlsplit(self.base_url).hostname in {"localhost", "127.0.0.1", "::1"}
        return True

    def public(self) -> LLMSettingsResponse:
        return LLMSettingsResponse(base_url=self.base_url, model=self.model,
                                   api_key_configured=bool(self.api_key), configured=self.configured())


def get_llm_snapshot() -> LLMSnapshot:
    with _lock:
        defaults = dict(base_url=config.LLM_BASE_URL, model=config.LLM_MODEL, api_key=config.LLM_API_KEY)
        # Parse subprocess receives a frozen snapshot from its parent process.
        if os.environ.get("STAGETTS_LLM_SNAPSHOT") != "1" and SETTINGS_PATH.exists():
            try:
                saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                if not isinstance(saved, dict) or not all(isinstance(saved.get(key), str) for key in defaults):
                    raise ValueError()
                defaults.update({key: saved[key] for key in defaults})
            except (OSError, ValueError):
                raise ValueError("无法读取本地 LLM 设置，请检查设置文件格式") from None
        return LLMSnapshot(**defaults)


def save_llm_settings(request: LLMSettingsUpdate) -> LLMSettingsResponse:
    with _lock:
        previous = get_llm_snapshot()
        key = request.api_key.get_secret_value().strip() if request.api_key is not None else ""
        if request.clear_api_key and key:
            raise ValueError("清除 API Key 时不能同时填写新 Key")
        if len(key) > 8192 or any(ord(char) < 32 for char in key):
            raise ValueError("API Key 格式无效")
        current = LLMSnapshot(request.base_url, request.model, "" if request.clear_api_key else key or previous.api_key)
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=SETTINGS_PATH.parent,
                                             prefix=".llm-settings-", suffix=".tmp", delete=False) as stream:
                temp_path = Path(stream.name)
                json.dump(dict(base_url=current.base_url, model=current.model, api_key=current.api_key), stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, SETTINGS_PATH)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return current.public()
