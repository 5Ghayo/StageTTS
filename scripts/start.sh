#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/backend/main.py" ]]; then
  ROOT_DIR="$SCRIPT_DIR"
elif [[ -f "$SCRIPT_DIR/../backend/main.py" ]]; then
  ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  echo "[StageTTS][ERROR] 无法定位项目根目录（未找到 backend/main.py）。" >&2
  exit 1
fi
cd "$ROOT_DIR"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"
CHECK_ONLY=0

usage() {
  cat <<'EOF'
StageTTS 一键启动脚本

用法:
  ./start.sh [选项]

选项:
  --no-open    启动后不自动打开浏览器
  --check      仅执行环境与端口预检，不启动服务
  -h, --help   显示此帮助信息

环境变量覆盖:
  BACKEND_HOST   后端监听地址（默认: 127.0.0.1）
  BACKEND_PORT   后端监听端口（默认: 8000）
  FRONTEND_HOST  前端监听地址（默认: 127.0.0.1）
  FRONTEND_PORT  前端监听端口（默认: 5173）
  OPEN_BROWSER   是否自动打开浏览器（1/0，默认: 1）
EOF
}

while (($# > 0)); do
  case "$1" in
    --no-open)
      OPEN_BROWSER=0
      shift
      ;;
    --check)
      CHECK_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[StageTTS] 未知参数: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

log_info() {
  printf '[StageTTS] %s\n' "$*"
}

log_warn() {
  printf '[StageTTS][WARN] %s\n' "$*" >&2
}

log_err() {
  printf '[StageTTS][ERROR] %s\n' "$*" >&2
}

# 1. 加载 Node / pnpm（兼容非交互式 Shell 下的 nvm 与 macOS Homebrew）
ensure_node_path() {
  local extra_dir
  for extra_dir in /opt/homebrew/bin /usr/local/bin; do
    if [[ -d "$extra_dir" ]] && [[ ":$PATH:" != *":$extra_dir:"* ]]; then
      export PATH="$extra_dir:$PATH"
    fi
  done

  if command -v pnpm >/dev/null 2>&1 && command -v node >/dev/null 2>&1; then
    return 0
  fi

  local nvm_dir="${NVM_DIR:-$HOME/.nvm}"
  if [[ -s "$nvm_dir/nvm.sh" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "$nvm_dir/nvm.sh" >/dev/null 2>&1 || true
    set -u
  fi

  if ! command -v pnpm >/dev/null 2>&1 && [[ -d "$nvm_dir/versions/node" ]]; then
    local latest_bin
    latest_bin="$(
      find "$nvm_dir/versions/node" -maxdepth 2 -type d -name bin 2>/dev/null \
        | awk -F/ '{v=$(NF-1); gsub(/^v/,"",v); split(v,a,"."); printf "%05d.%05d.%05d %s\n", a[1]+0, a[2]+0, a[3]+0, $0}' \
        | sort \
        | tail -n 1 \
        | cut -d' ' -f2- || true
    )"
    if [[ -n "$latest_bin" ]]; then
      export PATH="$latest_bin:$PATH"
    fi
  fi
}

ensure_node_path

if ! command -v node >/dev/null 2>&1 || ! command -v pnpm >/dev/null 2>&1; then
  log_err "未找到 node 或 pnpm，请先安装 Node.js 20+ 与 pnpm。"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  log_err "未找到 ffmpeg 或 ffprobe，请先安装 FFmpeg 并加入 PATH。"
  exit 1
fi

# 2. 选择后端 Python 虚拟环境
BACKEND_PYTHON=""
PYTHON_CANDIDATES=(
  "$ROOT_DIR/workspace/env_backend/bin/python"
  "$ROOT_DIR/.venv_backend/bin/python"
  "$ROOT_DIR/.venv/bin/python"
)
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_CANDIDATES+=("$VIRTUAL_ENV/bin/python")
fi
for cmd_name in python3 python; do
  if command -v "$cmd_name" >/dev/null 2>&1; then
    PYTHON_CANDIDATES+=("$(command -v "$cmd_name")")
  fi
done

for candidate in "${PYTHON_CANDIDATES[@]}"; do
  if [[ -x "$candidate" ]] && "$candidate" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
    BACKEND_PYTHON="$candidate"
    break
  fi
done

if [[ -z "$BACKEND_PYTHON" ]]; then
  log_err "未找到可用的后端 Python 环境（已检查 workspace/env_backend、.venv_backend、.venv 及当前 PATH）。"
  exit 1
fi

# 3. 检查配置文件与 Banquet 模型路径
if [[ ! -f "$ROOT_DIR/backend/.env" ]]; then
  if [[ -f "$ROOT_DIR/backend/.env.example" ]]; then
    cp "$ROOT_DIR/backend/.env.example" "$ROOT_DIR/backend/.env"
    log_warn "未检测到 backend/.env，已从 backend/.env.example 复制模板，请确认其中的路径与密钥配置。"
  else
    log_err "未找到 backend/.env。"
    exit 1
  fi
fi

# 若环境与 backend/.env 均未配置 Banquet，则自动探测 workspace/model-cache 中的本地目录
if [[ -z "${BANQUET_ROOT:-}" ]]; then
  env_banquet_root="$("$BACKEND_PYTHON" -c '
from pathlib import Path
try:
    from dotenv import dotenv_values
    vals = dotenv_values(Path("backend/.env"))
except ImportError:
    from backend.config import _read_dotenv_fallback
    vals = _read_dotenv_fallback(Path("backend/.env"))
print((vals.get("BANQUET_ROOT") or "").strip())
')"
  if [[ -z "$env_banquet_root" && -f "$ROOT_DIR/workspace/model-cache/bandit-v2/src/models/bandit/bandit.py" ]]; then
    export BANQUET_ROOT="$ROOT_DIR/workspace/model-cache/bandit-v2"
  fi
fi

if [[ -z "${BANQUET_CHECKPOINT:-}" ]]; then
  env_banquet_ckpt="$("$BACKEND_PYTHON" -c '
from pathlib import Path
try:
    from dotenv import dotenv_values
    vals = dotenv_values(Path("backend/.env"))
except ImportError:
    from backend.config import _read_dotenv_fallback
    vals = _read_dotenv_fallback(Path("backend/.env"))
print((vals.get("BANQUET_CHECKPOINT") or "").strip())
')"
  if [[ -z "$env_banquet_ckpt" && -f "$ROOT_DIR/workspace/model-cache/banquet-split.ckpt" ]]; then
    export BANQUET_CHECKPOINT="$ROOT_DIR/workspace/model-cache/banquet-split.ckpt"
  fi
fi

# 4. 检查前端依赖
if [[ ! -d "$ROOT_DIR/frontend/node_modules" ]]; then
  log_info "首次运行：正在安装前端依赖 (pnpm install)..."
  pnpm --dir "$ROOT_DIR/frontend" install
fi

# 5. 预检核心模块与端口占用
is_port_open() {
  local host="$1"
  local port="$2"
  "$BACKEND_PYTHON" - "$host" "$port" <<'PY'
import socket, sys
host = sys.argv[1]
port = int(sys.argv[2])
probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.settimeout(0.5)
    sys.exit(0 if s.connect_ex((probe_host, port)) == 0 else 1)
PY
}

preflight_summary() {
  "$BACKEND_PYTHON" - <<'PY'
import shutil
from pathlib import Path
from backend import config

parse_py = config.ENV_PARSE_PYTHON
parse_ok = Path(parse_py).is_file() or shutil.which(parse_py) is not None
tts_py = Path(config.GPT_SOVITS_PYTHON) if config.GPT_SOVITS_PYTHON else None
tts_ok = bool(
    config.GPT_SOVITS_ROOT
    and (config.GPT_SOVITS_ROOT / "api_v2.py").is_file()
    and tts_py
    and tts_py.is_file()
)
print(
    f"ffmpeg={shutil.which('ffmpeg') is not None} | "
    f"parse_env={parse_ok} | "
    f"banquet={config.is_banquet_configured()} | "
    f"gpt_sovits={tts_ok} (auto_manage={config.TTS_AUTO_MANAGE}) | "
    f"llm={config.is_llm_configured()} | "
    f"hf_token={config.is_hf_token_configured()}"
)
PY
}

if is_port_open "$BACKEND_HOST" "$BACKEND_PORT"; then
  log_err "后端端口 $BACKEND_HOST:$BACKEND_PORT 已被占用，请先停止占用进程或通过 BACKEND_PORT 指定其他端口。"
  exit 1
fi

if is_port_open "$FRONTEND_HOST" "$FRONTEND_PORT"; then
  log_err "前端端口 $FRONTEND_HOST:$FRONTEND_PORT 已被占用，请先停止占用进程或通过 FRONTEND_PORT 指定其他端口。"
  exit 1
fi

if is_port_open "127.0.0.1" "9880"; then
  log_warn "检测到本机 9880 端口已有服务在监听；若运行解析流程（Stage1-3），后端将提示先释放 9880 显存。"
fi

log_info "使用后端 Python: $BACKEND_PYTHON"
log_info "使用前端 Node: $(command -v node) ($(node -v)) / pnpm ($(pnpm -v))"
log_info "环境预检: $(preflight_summary)"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  log_info "预检完成（--check 模式，未启动服务）。"
  exit 0
fi

mkdir -p "$ROOT_DIR/workspace"
BACKEND_LOG="$ROOT_DIR/workspace/_backend.log"
FRONTEND_LOG="$ROOT_DIR/workspace/_frontend.log"

BACKEND_PID=""
FRONTEND_PID=""
CLEANED_UP=0

collect_descendants() {
  local parent="$1"
  local child
  for child in $(pgrep -P "$parent" 2>/dev/null || true); do
    collect_descendants "$child"
    echo "$child"
  done
}

stop_proc_tree() {
  local pid="$1"
  local label="$2"
  [[ -z "$pid" ]] && return 0
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    return 0
  fi

  local descendants=""
  if command -v pgrep >/dev/null 2>&1; then
    descendants="$(collect_descendants "$pid")"
  fi

  kill -TERM "$pid" >/dev/null 2>&1 || true
  for child in $descendants; do
    kill -TERM "$child" >/dev/null 2>&1 || true
  done

  local waited=0
  while kill -0 "$pid" >/dev/null 2>&1 && ((waited < 50)); do
    sleep 0.1
    ((waited += 1))
  done

  if kill -0 "$pid" >/dev/null 2>&1; then
    log_warn "$label 未在 5s 内退出，发送 SIGKILL..."
    for child in $descendants; do
      kill -KILL "$child" >/dev/null 2>&1 || true
    done
    kill -KILL "$pid" >/dev/null 2>&1 || true
  else
    for child in $descendants; do
      if kill -0 "$child" >/dev/null 2>&1; then
        kill -TERM "$child" >/dev/null 2>&1 || true
      fi
    done
  fi
  wait "$pid" >/dev/null 2>&1 || true
}

cleanup() {
  local exit_code=$?
  if [[ "$CLEANED_UP" -eq 1 ]]; then
    return
  fi
  CLEANED_UP=1
  trap - EXIT INT TERM

  if [[ -n "$FRONTEND_PID" || -n "$BACKEND_PID" ]]; then
    echo ""
    log_info "正在停止 StageTTS 服务..."
  fi
  stop_proc_tree "$FRONTEND_PID" "前端 Vite"
  stop_proc_tree "$BACKEND_PID" "后端 FastAPI"
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

wait_for_http() {
  local url="$1"
  local timeout_sec="$2"
  local pid="$3"
  local label="$4"
  local log_file="$5"

  "$BACKEND_PYTHON" - "$url" "$timeout_sec" "$pid" <<'PY'
import os, sys, time, urllib.request, urllib.error

url = sys.argv[1]
deadline = time.time() + float(sys.argv[2])
pid = int(sys.argv[3])

while time.time() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        sys.exit(2)
    try:
        with urllib.request.urlopen(url, timeout=1.0) as resp:
            if resp.status < 500:
                sys.exit(0)
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
    time.sleep(0.25)

sys.exit(1)
PY
  local status=$?
  if [[ $status -eq 0 ]]; then
    return 0
  elif [[ $status -eq 2 ]]; then
    log_err "$label 启动后提前退出，最近日志如下 ($log_file):"
  else
    log_err "$label 启动超时 (>${timeout_sec}s)，最近日志如下 ($log_file):"
  fi
  tail -n 30 "$log_file" >&2 || true
  return 1
}

# 6. 启动后端 FastAPI（GPT-SoVITS 由后端按需自动拉起/回收）
log_info "正在启动后端服务 (http://$BACKEND_HOST:$BACKEND_PORT)..."
"$BACKEND_PYTHON" -m uvicorn backend.main:app \
  --host "$BACKEND_HOST" \
  --port "$BACKEND_PORT" \
  >"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

BACKEND_PROBE_HOST="$BACKEND_HOST"
[[ "$BACKEND_PROBE_HOST" == "0.0.0.0" || "$BACKEND_PROBE_HOST" == "::" ]] && BACKEND_PROBE_HOST="127.0.0.1"
[[ "$BACKEND_PROBE_HOST" == *:* && "$BACKEND_PROBE_HOST" != \[*\*\] ]] && BACKEND_PROBE_HOST="[$BACKEND_PROBE_HOST]"
wait_for_http "http://$BACKEND_PROBE_HOST:$BACKEND_PORT/api/health" 25 "$BACKEND_PID" "后端 FastAPI" "$BACKEND_LOG"

# 7. 启动前端 Vite（同步注入动态后端代理目标）
log_info "正在启动前端界面 (http://$FRONTEND_HOST:$FRONTEND_PORT)..."
VITE_BACKEND_URL="http://$BACKEND_PROBE_HOST:$BACKEND_PORT" \
  pnpm --dir "$ROOT_DIR/frontend" exec vite \
  --host "$FRONTEND_HOST" \
  --port "$FRONTEND_PORT" \
  --strictPort \
  >"$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

FRONTEND_PROBE_HOST="$FRONTEND_HOST"
[[ "$FRONTEND_PROBE_HOST" == "0.0.0.0" || "$FRONTEND_PROBE_HOST" == "::" ]] && FRONTEND_PROBE_HOST="127.0.0.1"
[[ "$FRONTEND_PROBE_HOST" == *:* && "$FRONTEND_PROBE_HOST" != \[*\*\] ]] && FRONTEND_PROBE_HOST="[$FRONTEND_PROBE_HOST]"
wait_for_http "http://$FRONTEND_PROBE_HOST:$FRONTEND_PORT/" 25 "$FRONTEND_PID" "前端 Vite" "$FRONTEND_LOG"

# 8. 输出就绪摘要
HEALTH_SUMMARY="$("$BACKEND_PYTHON" - "http://$BACKEND_PROBE_HOST:$BACKEND_PORT/api/health" <<'PY'
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=3.0) as resp:
    data = json.loads(resp.read().decode("utf-8"))
r = data.get("readiness", {})
print(
    f"status={data.get('status')} | "
    f"ffmpeg={r.get('ffmpeg')} | "
    f"banquet={r.get('banquet_configured')} | "
    f"llm={r.get('llm_configured')} | "
    f"hf_token={r.get('hf_token_configured')} | "
    f"tts_auto_manage={r.get('tts_auto_manage')}"
)
PY
)"

FRONTEND_URL="http://$FRONTEND_PROBE_HOST:$FRONTEND_PORT/"
BACKEND_URL="http://$BACKEND_PROBE_HOST:$BACKEND_PORT"

echo ""
log_info "=================================================="
log_info "StageTTS 已就绪！"
log_info "前端地址: $FRONTEND_URL"
log_info "后端地址: $BACKEND_URL  (健康检查: $HEALTH_SUMMARY)"
log_info "TTS 推理: 由后端按需自动管理 (GPT-SoVITS api_v2.py)"
log_info "日志文件: $BACKEND_LOG | $FRONTEND_LOG"
log_info "按 Ctrl+C 可一键停止所有服务"
log_info "=================================================="
echo ""

if [[ "$OPEN_BROWSER" -eq 1 ]]; then
  if [[ "$(uname -s)" == "Darwin" ]] && command -v open >/dev/null 2>&1; then
    open "$FRONTEND_URL" </dev/null >/dev/null 2>&1 || true
  elif [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && command -v xdg-open >/dev/null 2>&1; then
    if command -v setsid >/dev/null 2>&1; then
      setsid -f xdg-open "$FRONTEND_URL" </dev/null >/dev/null 2>&1 || true
    else
      nohup xdg-open "$FRONTEND_URL" </dev/null >/dev/null 2>&1 &
    fi
  fi
fi

# 9. 守护两个前台服务，任一异常退出时输出日志并统一清理
while true; do
  if ! kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    log_err "后端服务意外退出，最近日志 ($BACKEND_LOG):"
    tail -n 30 "$BACKEND_LOG" >&2 || true
    exit 1
  fi
  if ! kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
    log_err "前端服务意外退出，最近日志 ($FRONTEND_LOG):"
    tail -n 30 "$FRONTEND_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done
