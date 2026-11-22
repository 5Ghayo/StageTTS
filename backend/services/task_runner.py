"""重算力任务调度与 8GB 显存防 OOM 单任务队列锁。"""

from __future__ import annotations

import atexit
import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlparse

import requests

from backend import config


class GpuBusyError(RuntimeError):
    """当已有 GPU 重任务在运行时，拒绝并发请求以防止 8GB 显存 OOM。"""


class ParseWorkerError(RuntimeError):
    """解析子进程无法完成任务。"""


class GpuTaskLock:
    """支持同线程重入、跨线程立即拒绝的单任务互斥锁。

    - 同线程内调用（如 agent_chat -> synthesize_segments -> 反思重合成）可安全重入；
    - 跨线程并发调用立即抛出 GpuBusyError，避免在 RTX 5060 8GB 上同时触发多路显存峰值。
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._owner_thread_id: Optional[int] = None
        self._depth: int = 0

    def is_locked(self) -> bool:
        with self._guard:
            return self._depth > 0

    def acquire(self) -> None:
        tid = threading.get_ident()
        with self._guard:
            if self._depth > 0 and self._owner_thread_id != tid:
                raise GpuBusyError("GPU 单任务队列忙：已有解析或语音合成任务正在执行，请稍后再试。")
            self._owner_thread_id = tid
            self._depth += 1

    def release(self) -> None:
        tid = threading.get_ident()
        with self._guard:
            if self._depth == 0 or self._owner_thread_id != tid:
                return
            self._depth -= 1
            if self._depth == 0:
                self._owner_thread_id = None


_gpu_task_lock = GpuTaskLock()

# ---------------------------------------------------------------------------
# RTX 5060 8GB 按需启停管理器：
#   - 平时不常驻 GPT-SoVITS（0 显存占用）
#   - 轮到解析 (env_parse) 前自动关闭 GPT-SoVITS，交出 100% 显存
#   - 轮到语音合成 (env_tts) 时自动按需拉起 GPT-SoVITS，连贯精修期间保持热驻留
# ---------------------------------------------------------------------------
_tts_proc_lock = threading.Lock()
_tts_proc: Optional[subprocess.Popen[Any]] = None
_tts_log_fp: Optional[Any] = None


def _probe_tts_ready(timeout: float = 1.0) -> bool:
    try:
        resp = requests.get(f"{config.TTS_API_URL}/", timeout=timeout)
        return resp.status_code < 500
    except requests.RequestException:
        return False


def ensure_tts_stopped() -> bool:
    """停止后端持有的 GPT-SoVITS 进程，等它退出后才释放句柄。"""
    global _tts_proc, _tts_log_fp
    stopped = False
    with _tts_proc_lock:
        if _tts_proc is not None:
            if _tts_proc.poll() is None:
                try:
                    _tts_proc.terminate()
                    _tts_proc.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    _tts_proc.kill()
                    _tts_proc.wait(timeout=3)
                stopped = True
            _tts_proc = None
        if _tts_log_fp is not None:
            try:
                _tts_log_fp.close()
            except Exception:
                pass
            _tts_log_fp = None

    if stopped:
        # 清理权重切换缓存，确保下次重启后按需重新加载角色权重
        try:
            from backend.services.tts_client import reset_active_weights_cache

            reset_active_weights_cache()
        except Exception:
            pass
        # 给 Windows WDDM 驱动 1 秒时间回收显存
        time.sleep(1.0)
    return stopped


def _prepare_gpu_for_parse() -> None:
    try:
        stopped = config.TTS_AUTO_MANAGE and ensure_tts_stopped()
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GpuBusyError("无法确认受管 GPT-SoVITS 已退出，请停止该服务后重试解析。") from exc

    endpoint = urlparse(config.TTS_API_URL)
    host = endpoint.hostname or "localhost"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address and address.is_unspecified:
        host = "127.0.0.1" if address.version == 4 else "::1"
        address = ipaddress.ip_address(host)
    if host == "localhost" or (address and address.is_loopback):
        port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=0.5):
                pass
        except OSError:
            pass
        else:
            raise GpuBusyError(
                f"本机 GPT-SoVITS 端口 {port} 仍在监听，可能占用同一张显卡。"
                "请停止手动启动的服务后重试，或将 TTS_API_URL 指向另一台机器。"
            )
    if stopped:
        _record_parse_line("[parse_worker][Init] 已确认受管 GPT-SoVITS 退出")


def ensure_tts_running() -> None:
    """按需启动 GPT-SoVITS api_v2.py 并等待就绪；若已在线则直接复用。"""
    global _tts_proc, _tts_log_fp

    if _probe_tts_ready(timeout=1.2):
        return

    if not config.TTS_AUTO_MANAGE:
        return

    if config.GPT_SOVITS_ROOT is None:
        return

    api_script = config.GPT_SOVITS_ROOT / "api_v2.py"
    python_bin = Path(config.GPT_SOVITS_PYTHON) if config.GPT_SOVITS_PYTHON else None
    if not api_script.is_file() or python_bin is None or not python_bin.is_file():
        return

    parsed = urlparse(config.TTS_API_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 9880

    cmd = [
        str(python_bin),
        "-u",
        str(api_script),
        "-a",
        host,
        "-p",
        str(port),
    ]
    if config.GPT_SOVITS_CONFIG and Path(config.GPT_SOVITS_CONFIG).is_file():
        cmd.extend(["-c", config.GPT_SOVITS_CONFIG])

    with _tts_proc_lock:
        if _probe_tts_ready(timeout=1.0):
            return
        if _tts_proc is not None and _tts_proc.poll() is not None:
            _tts_proc = None
        if _tts_proc is None:
            config.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
            log_path = config.WORKSPACE_DIR / "_tts_server.log"
            _tts_log_fp = log_path.open("w", encoding="utf-8", errors="replace")
            _tts_proc = subprocess.Popen(
                cmd,
                cwd=str(config.GPT_SOVITS_ROOT),
                stdout=_tts_log_fp,
                stderr=subprocess.STDOUT,
            )

    deadline = time.time() + config.TTS_STARTUP_TIMEOUT_SEC
    while time.time() < deadline:
        if _probe_tts_ready(timeout=1.0):
            return
        with _tts_proc_lock:
            if _tts_proc is not None and _tts_proc.poll() is not None:
                code = _tts_proc.returncode
                raise RuntimeError(
                    f"按需启动 GPT-SoVITS 失败 (exit={code})，请查看 workspace/_tts_server.log"
                )
        time.sleep(0.8)

    raise RuntimeError(
        f"按需启动 GPT-SoVITS 超时（>{int(config.TTS_STARTUP_TIMEOUT_SEC)}s），请查看 workspace/_tts_server.log"
    )


atexit.register(ensure_tts_stopped)

_parse_progress_lock = threading.Lock()
_parse_progress: Dict[str, Any] = {
    "active": False,
    "project_id": None,
    "stage": "idle",
    "logs": [],
    "log_count": 0,
    "error": None,
    "updated_at": 0.0,
}
_STAGE_TAG_RE = re.compile(r"^\[parse_worker\]\[([^\]]+)\]\s*(.*)$")


def _reset_parse_progress(project_id: str) -> None:
    with _parse_progress_lock:
        _parse_progress.update(
            {
                "active": True,
                "project_id": project_id,
                "stage": "Init",
                "logs": [],
                "log_count": 0,
                "error": None,
                "updated_at": time.time(),
            }
        )


def _record_parse_line(line: str) -> None:
    cleaned = line.strip()
    if not cleaned:
        return
    match = _STAGE_TAG_RE.match(cleaned)
    with _parse_progress_lock:
        if match:
            _parse_progress["stage"] = match.group(1)
            entry = f"[{match.group(1)}] {match.group(2)}"
        elif any(
            kw in cleaned
            for kw in ("Detected language", "Loading diarization", "Downloading:", "ERROR", "Traceback")
        ):
            entry = cleaned[:220]
        else:
            return
        _parse_progress["logs"] = (_parse_progress["logs"] + [entry])[-60:]
        _parse_progress["log_count"] += 1
        _parse_progress["updated_at"] = time.time()


def _finish_parse_progress(error: Optional[str] = None) -> None:
    with _parse_progress_lock:
        _parse_progress["active"] = False
        if error:
            _parse_progress["stage"] = "Error"
            _parse_progress["error"] = error
        else:
            _parse_progress["stage"] = "Done"
        _parse_progress["updated_at"] = time.time()


def get_parse_progress(project_id: Optional[str] = None) -> Dict[str, Any]:
    """返回当前或最近一次 parse_worker 实时阶段与日志（供前端轮询驱动节点状态）。"""
    with _parse_progress_lock:
        if project_id and _parse_progress["project_id"] not in (None, project_id):
            return {
                "active": False,
                "project_id": project_id,
                "stage": "idle",
                "logs": [],
                "log_count": 0,
                "error": None,
                "updated_at": time.time(),
            }
        return {
            "active": bool(_parse_progress["active"]),
            "project_id": _parse_progress["project_id"],
            "stage": _parse_progress["stage"],
            "logs": list(_parse_progress["logs"]),
            "log_count": _parse_progress["log_count"],
            "error": _parse_progress["error"],
            "updated_at": _parse_progress["updated_at"],
        }


def is_gpu_busy() -> bool:
    """检查当前是否有 GPU 重任务正在执行。"""
    return _gpu_task_lock.is_locked()


def read_gpu_memory() -> List[Dict[str, Any]]:
    """读取本机 NVIDIA GPU 的整卡显存占用，单位为 MiB。"""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return []
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    devices = []
    for line in result.stdout.splitlines():
        try:
            name, used, total = line.rsplit(",", 2)
            used_mib, total_mib = int(used), int(total)
        except ValueError:
            continue
        if name.strip() and total_mib > 0 and used_mib >= 0:
            devices.append({"name": name.strip(), "used_mib": used_mib, "total_mib": total_mib})
    return devices


@contextmanager
def gpu_task_guard() -> Iterator[None]:
    """保护 GPU 重算力任务的上下文管理器（并发冲突时抛出 GpuBusyError）。"""
    _gpu_task_lock.acquire()
    try:
        yield
    finally:
        _gpu_task_lock.release()


def probe_media_duration(media_path: Path) -> float:
    """通过 ffprobe 提取音视频文件精确时长（秒），失败时返回 0.0。"""
    if not media_path.is_file() or shutil.which("ffprobe") is None:
        return 0.0
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(media_path),
    ]
    try:
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        payload = json.loads(res.stdout or "{}")
        duration_raw = payload.get("format", {}).get("duration")
        return max(0.0, round(float(duration_raw), 3)) if duration_raw else 0.0
    except Exception:
        return 0.0


def run_parse_worker(
    project_id: str,
    input_path: Path,
    target_lang: str = "zh",
    compute_type: Optional[str] = None,
    mode: str = "video_dubbing",
    allow_demucs_cpu: bool = False,
) -> Path:
    """以独立子进程启动 workers/parse_worker.py（运行在 env_parse）。

    注意：
    - 子进程运行结束后由操作系统 100% 回收 CUDA 显存，严禁在 env_backend 中直接导入 torch/whisperx。
    - 全过程受 gpu_task_guard 保护，防止并发触发 RTX 5060 8GB OOM。
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"待解析输入媒体文件不存在: {input_path}")
    if not config.PARSE_WORKER_SCRIPT.is_file():
        raise FileNotFoundError(f"未找到解析 Worker 脚本: {config.PARSE_WORKER_SCRIPT}")

    python_bin = config.ENV_PARSE_PYTHON
    if not Path(python_bin).is_file() and shutil.which(python_bin) is None:
        raise ParseWorkerError(
            f"未找到 env_parse Python 可执行文件: {python_bin}。请检查 backend/.env 中的 ENV_PARSE_PYTHON 配置。"
        )

    output_dir = config.WORKSPACE_DIR / project_id
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_compute = compute_type or config.WHISPER_COMPUTE_TYPE

    cmd = [
        python_bin,
        str(config.PARSE_WORKER_SCRIPT),
        "--project-id",
        project_id,
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--target-lang",
        target_lang,
        "--whisper-model",
        config.WHISPER_MODEL,
        "--compute-type",
        effective_compute,
        "--mode",
        mode,
    ]
    if allow_demucs_cpu:
        cmd.append("--allow-demucs-cpu")

    sub_env = os.environ.copy()
    sub_env["PYTHONUTF8"] = "1"
    if config.HF_HOME:
        sub_env["HF_HOME"] = config.HF_HOME
    if config.HUGGINGFACE_HUB_CACHE:
        sub_env["HUGGINGFACE_HUB_CACHE"] = config.HUGGINGFACE_HUB_CACHE
    if config.TORCH_HOME:
        sub_env["TORCH_HOME"] = config.TORCH_HOME
    if config.HF_TOKEN:
        sub_env["HF_TOKEN"] = config.HF_TOKEN
    if config.LLM_API_KEY:
        sub_env["LLM_API_KEY"] = config.LLM_API_KEY
    if config.LLM_BASE_URL:
        sub_env["LLM_BASE_URL"] = config.LLM_BASE_URL
    if config.LLM_MODEL:
        sub_env["LLM_MODEL"] = config.LLM_MODEL

    with gpu_task_guard():
        _reset_parse_progress(project_id)
        try:
            _prepare_gpu_for_parse()
        except GpuBusyError as exc:
            _finish_parse_progress(error=str(exc))
            raise
        collected_lines: List[str] = []
        proc: Optional[subprocess.Popen[str]] = None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(config.ROOT_DIR),
                env=sub_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=os.name != "nt",
            )

            def _drain_stdout() -> None:
                assert proc is not None and proc.stdout is not None
                for raw_line in proc.stdout:
                    collected_lines.append(raw_line)
                    _record_parse_line(raw_line)

            reader = threading.Thread(target=_drain_stdout, daemon=True)
            reader.start()
            try:
                returncode = proc.wait(timeout=config.PARSE_TIMEOUT_SEC)
            except subprocess.TimeoutExpired as exc:
                if os.name == "nt":
                    try:
                        subprocess.run(
                            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                            capture_output=True, timeout=5,
                        )
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                    if proc.poll() is None:
                        proc.kill()
                else:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                proc.wait()
                reader.join(timeout=3)
                err_msg = f"parse_worker 执行超时（>{config.PARSE_TIMEOUT_SEC}s）"
                _finish_parse_progress(error=err_msg)
                raise ParseWorkerError(err_msg) from exc

            reader.join(timeout=5)
        except ParseWorkerError:
            raise
        except Exception as exc:
            err_msg = f"启动 parse_worker 失败: {exc}"
            _finish_parse_progress(error=err_msg)
            raise ParseWorkerError(err_msg) from exc

        if returncode != 0:
            tail = "".join(collected_lines).strip()[-800:]
            err_msg = f"parse_worker 子进程异常退出 (code={returncode}): {tail or '无日志输出'}"
            _finish_parse_progress(error=err_msg)
            raise ParseWorkerError(err_msg)

        script_path = output_dir / "script.json"
        if not script_path.is_file():
            err_msg = "parse_worker 已退出，但没有生成 script.json"
            _finish_parse_progress(error=err_msg)
            raise ParseWorkerError(err_msg)
        _finish_parse_progress()
    return script_path
