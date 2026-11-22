"""StageTTS 主后端全局配置（纯 CPU 环境 env_backend）。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # 允许未安装 python-dotenv 时仅使用系统环境变量
    load_dotenv = None

# 仓库根目录：StageTTS/
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
BACKEND_DIR: Path = ROOT_DIR / "backend"
WORKERS_DIR: Path = ROOT_DIR / "workers"

if load_dotenv is not None:
    env_file = BACKEND_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def _resolve_dir(env_key: str, default_path: Path) -> Path:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return default_path
    p = Path(raw)
    return p if p.is_absolute() else (ROOT_DIR / p).resolve()


def _resolve_optional_dir(env_key: str) -> Optional[Path]:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else (ROOT_DIR / p).resolve()


def _safe_float_env(env_key: str, default: float) -> float:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


WORKSPACE_DIR: Path = _resolve_dir("WORKSPACE_DIR", ROOT_DIR / "workspace")
ASSETS_DIR: Path = _resolve_dir("ASSETS_DIR", ROOT_DIR / "assets")
PARSE_WORKER_SCRIPT: Path = WORKERS_DIR / "parse_worker.py"
GPT_SOVITS_ROOT: Optional[Path] = _resolve_optional_dir("GPT_SOVITS_ROOT")
GPT_SOVITS_PYTHON: str = (
    os.getenv("GPT_SOVITS_PYTHON", "").strip()
    or (str(GPT_SOVITS_ROOT / "runtime" / "python.exe") if GPT_SOVITS_ROOT else "")
)
GPT_SOVITS_CONFIG: str = (
    os.getenv("GPT_SOVITS_CONFIG", "").strip()
    or (str(BACKEND_DIR / "tts_v2.yaml") if (BACKEND_DIR / "tts_v2.yaml").is_file() else "")
)
TTS_AUTO_MANAGE: bool = os.getenv("TTS_AUTO_MANAGE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
TTS_STARTUP_TIMEOUT_SEC: float = _safe_float_env("TTS_STARTUP_TIMEOUT_SEC", 90.0)

# 外部算力服务与子进程 Python 路径
ENV_PARSE_PYTHON: str = os.getenv("ENV_PARSE_PYTHON", "python").strip() or "python"
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3").strip() or "large-v3"
WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16").strip() or "float16"
TTS_API_URL: str = os.getenv("TTS_API_URL", "http://127.0.0.1:9880").rstrip("/")

# 可选模型缓存路径（用于重定向至非 C 盘目录）
HF_HOME: str = os.getenv("HF_HOME", "").strip()
HUGGINGFACE_HUB_CACHE: str = os.getenv("HUGGINGFACE_HUB_CACHE", "").strip()
TORCH_HOME: str = os.getenv("TORCH_HOME", "").strip()

# 超时控制（秒）
TTS_TIMEOUT_SEC: float = _safe_float_env("TTS_TIMEOUT_SEC", 120.0)
PARSE_TIMEOUT_SEC: float = _safe_float_env("PARSE_TIMEOUT_SEC", 1800.0)
LLM_TIMEOUT_SEC: float = _safe_float_env("LLM_TIMEOUT_SEC", 60.0)

# 大模型 API（兼容 OpenAI SDK）
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

# HuggingFace pyannote 鉴权 Token
HF_TOKEN: str = os.getenv("HF_TOKEN", "").strip()

# 时长约束反思闭环核心阈值（见 ARCHITECTURE.md 铁律 #6）
DURATION_OVERFLOW_RATIO: float = 1.15
MAX_REFLECTION_ROUNDS: int = 2
MAX_GENTLE_SPEED: float = 1.15

# CORS 默认放行端口（Vite 开发服务器）
ALLOWED_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


def _resolve_under_root(base_dir: Path, relative_part: str) -> Path:
    base_resolved = base_dir.resolve()
    target = (base_resolved / relative_part.lstrip("/\\")).resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise ValueError(f"非法路径越界访问: {relative_part}")
    return target


def resolve_media_path(url_or_path: str) -> Path:
    """将 /workspace/*、/assets/*、refs/* 或本地路径安全解析为绝对 Path。"""
    raw = (url_or_path or "").strip()
    if not raw:
        raise ValueError("媒体路径不能为空")

    normalized = raw.replace("\\", "/")
    if normalized.startswith("/workspace/"):
        return _resolve_under_root(WORKSPACE_DIR, normalized[len("/workspace/"):])
    if normalized.startswith("workspace/"):
        return _resolve_under_root(WORKSPACE_DIR, normalized[len("workspace/"):])
    if normalized.startswith("/assets/"):
        return _resolve_under_root(ASSETS_DIR, normalized[len("/assets/"):])
    if normalized.startswith("assets/"):
        return _resolve_under_root(ASSETS_DIR, normalized[len("assets/"):])
    if normalized.startswith("refs/"):
        return _resolve_under_root(ASSETS_DIR / "refs", normalized[len("refs/"):])

    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()
    return (ROOT_DIR / candidate).resolve()


def resolve_weights_path(weights_path: str) -> str:
    """解析 GPT/SoVITS 权重路径：配置了 GPT_SOVITS_ROOT 且相对路径存在时转绝对路径，否则原样透传。"""
    raw = (weights_path or "").strip()
    if not raw:
        return raw
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    if GPT_SOVITS_ROOT is not None:
        candidate = (GPT_SOVITS_ROOT / p).resolve()
        if candidate.exists():
            return str(candidate)
    return raw


def is_llm_configured() -> bool:
    """判断大模型 API Key 是否已有效配置（非空且非示例占位值）。"""
    if not LLM_API_KEY:
        return False
    placeholders = {
        "your_llm_api_key_here",
        "sk-xxx",
        "changeme",
        "local-gateway-no-key",
    }
    if LLM_API_KEY in placeholders:
        # 本地网关常常不校验 Key；若 base_url 指向本机，则视为可用配置。
        return LLM_BASE_URL.startswith("http://127.0.0.1:") or LLM_BASE_URL.startswith(
            "http://localhost:"
        )
    return not LLM_API_KEY.startswith("your_")


def is_hf_token_configured() -> bool:
    """判断 HuggingFace Token 是否已有效配置。"""
    if not HF_TOKEN:
        return False
    return HF_TOKEN != "hf_your_token_here" and not HF_TOKEN.startswith("hf_your_")
