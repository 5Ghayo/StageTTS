<#
.SYNOPSIS
  StageTTS Windows / PowerShell 一键启动脚本。

.DESCRIPTION
  - 检查 Node.js / pnpm / FFmpeg / 后端 Python 虚拟环境与端口占用。
  - 自动初始化 backend/.env 与本地 Banquet 模型路径（若未配置）。
  - 首次运行自动执行前端 pnpm install。
  - 并发拉起后端 FastAPI 与前端 Vite，并在退出或 Ctrl+C 时统一清理进程树。

.PARAMETER NoOpen
  启动后不自动打开浏览器。

.PARAMETER Check
  仅执行环境与端口预检，不启动服务。

.PARAMETER Help
  显示此帮助信息。

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1

.EXAMPLE
  .\scripts\start.ps1 -Check

.EXAMPLE
  .\scripts\start.ps1 -NoOpen
#>

[CmdletBinding()]
param(
    [switch]$NoOpen,
    [switch]$Check,
    [Alias("h")]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
    # 非交互式或受限宿主下忽略编码设置失败
}

function Show-Usage {
    param([switch]$ToStderr)

    $text = @"
StageTTS 一键启动脚本 (PowerShell)

用法:
  .\scripts\start.ps1 [选项]
  powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1 [选项]

选项:
  -NoOpen, --no-open    启动后不自动打开浏览器
  -Check, --check       仅执行环境与端口预检，不启动服务
  -Help, -h, --help     显示此帮助信息

环境变量覆盖:
  BACKEND_HOST   后端监听地址（默认: 127.0.0.1）
  BACKEND_PORT   后端监听端口（默认: 8000）
  FRONTEND_HOST  前端监听地址（默认: 127.0.0.1）
  FRONTEND_PORT  前端监听端口（默认: 5173）
  OPEN_BROWSER   是否自动打开浏览器（1/0，默认: 1）
"@

    if ($ToStderr) {
        [Console]::Error.WriteLine($text)
    } else {
        Write-Host $text
    }
}

foreach ($arg in $RemainingArgs) {
    switch ($arg) {
        "--no-open" { $NoOpen = $true }
        "--check"   { $Check = $true }
        "--help"    { $Help = $true }
        "-?"        { $Help = $true }
        default {
            [Console]::Error.WriteLine("[StageTTS] 未知参数: $arg")
            Show-Usage -ToStderr
            exit 2
        }
    }
}

if ($Help) {
    Show-Usage
    exit 0
}

function Write-Info {
    param([string]$Message)
    Write-Host "[StageTTS] $Message"
}

function Write-Warn {
    param([string]$Message)
    [Console]::Error.WriteLine("[StageTTS][WARN] $Message")
}

function Write-Err {
    param([string]$Message)
    [Console]::Error.WriteLine("[StageTTS][ERROR] $Message")
}

$script:IsWinPlatform = ($env:OS -eq "Windows_NT") -or ($PSVersionTable.PSVersion.Major -lt 6) -or ($IsWindows -eq $true)

# 定位项目根目录（兼容放在仓库根目录或 scripts/ 下）
$ScriptDir = $PSScriptRoot
if (Test-Path (Join-Path (Join-Path $ScriptDir "backend") "main.py")) {
    $RootDir = $ScriptDir
} elseif (Test-Path (Join-Path (Join-Path (Split-Path -Parent $ScriptDir) "backend") "main.py")) {
    $RootDir = Split-Path -Parent $ScriptDir
} else {
    Write-Err "无法定位项目根目录（未找到 backend/main.py）。"
    exit 1
}

function Get-EnvOrDefault {
    param(
        [string]$Name,
        [string]$DefaultValue
    )
    $val = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($val)) {
        return $DefaultValue
    }
    return $val.Trim()
}

function Parse-Port {
    param(
        [string]$RawPort,
        [string]$Name
    )
    $parsed = 0
    if (-not [int]::TryParse($RawPort, [ref]$parsed) -or $parsed -lt 1 -or $parsed -gt 65535) {
        Write-Err "$Name 端口无效: $RawPort"
        exit 2
    }
    return $parsed
}

$BackendHost = Get-EnvOrDefault "BACKEND_HOST" "127.0.0.1"
$BackendPort = Parse-Port (Get-EnvOrDefault "BACKEND_PORT" "8000") "BACKEND_PORT"
$FrontendHost = Get-EnvOrDefault "FRONTEND_HOST" "127.0.0.1"
$FrontendPort = Parse-Port (Get-EnvOrDefault "FRONTEND_PORT" "5173") "FRONTEND_PORT"

$OpenBrowserRaw = Get-EnvOrDefault "OPEN_BROWSER" "1"
$OpenBrowserFlag = -not ($OpenBrowserRaw.ToLowerInvariant() -in @("0", "false", "no", "off"))
if ($NoOpen) {
    $OpenBrowserFlag = $false
}

function Invoke-SilentCheck {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )
    $prevAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Executable @Arguments *>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $prevAction
    }
}

function Add-PathIfPresent {
    param([string]$DirPath)
    if ([string]::IsNullOrWhiteSpace($DirPath) -or -not (Test-Path $DirPath -PathType Container)) {
        return
    }
    $sep = [System.IO.Path]::PathSeparator
    $parts = @($env:PATH -split [regex]::Escape([string]$sep) | Where-Object { $_ })
    $norm = $DirPath.TrimEnd("\", "/")
    foreach ($p in $parts) {
        if ($script:IsWinPlatform) {
            if ($p.TrimEnd("\", "/") -ieq $norm) { return }
        } else {
            if ($p.TrimEnd("\", "/") -eq $norm) { return }
        }
    }
    $env:PATH = "$norm$sep$env:PATH"
}

# 1. 加载 Node / pnpm / FFmpeg
function Ensure-NodePath {
    $hasNode = $null -ne (Get-Command node -ErrorAction SilentlyContinue)
    $hasPnpm = $null -ne (Get-Command pnpm -ErrorAction SilentlyContinue)
    if ($hasNode -and $hasPnpm) {
        return
    }

    if ($script:IsWinPlatform) {
        Add-PathIfPresent $env:NVM_SYMLINK
        Add-PathIfPresent (Join-Path $env:ProgramFiles "nodejs")
        if ($env:LOCALAPPDATA) {
            Add-PathIfPresent (Join-Path $env:LOCALAPPDATA "pnpm")
            Add-PathIfPresent (Join-Path (Join-Path $env:LOCALAPPDATA "Programs") "nodejs")
        }
        if ($env:APPDATA) {
            Add-PathIfPresent (Join-Path $env:APPDATA "npm")
        }
    } else {
        $nvmDir = if ($env:NVM_DIR) { $env:NVM_DIR } else { Join-Path $HOME ".nvm" }
        $nodeVersions = Join-Path (Join-Path $nvmDir "versions") "node"
        if (Test-Path $nodeVersions -PathType Container) {
            $latestNode = Get-ChildItem -Path $nodeVersions -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name | Select-Object -Last 1
            if ($latestNode) {
                Add-PathIfPresent (Join-Path $latestNode.FullName "bin")
            }
        }
    }
}

Ensure-NodePath

$NodeCmd = if ($script:IsWinPlatform) {
    Get-Command node.exe, node -ErrorAction SilentlyContinue | Select-Object -First 1
} else {
    Get-Command node -ErrorAction SilentlyContinue | Select-Object -First 1
}

$PnpmCmd = if ($script:IsWinPlatform) {
    Get-Command pnpm.exe, pnpm.cmd, pnpm -ErrorAction SilentlyContinue | Select-Object -First 1
} else {
    Get-Command pnpm -ErrorAction SilentlyContinue | Select-Object -First 1
}

if (-not $NodeCmd -or -not $PnpmCmd) {
    Write-Err "未找到 node 或 pnpm，请先安装 Node.js 20+ 与 pnpm。"
    exit 1
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    if ($script:IsWinPlatform) {
        Write-Err "未找到 ffmpeg 或 ffprobe，请先运行 .\scripts\setup_ffmpeg_windows.ps1 或安装 FFmpeg 并加入 PATH。"
    } else {
        Write-Err "未找到 ffmpeg 或 ffprobe，请先安装 FFmpeg 并加入 PATH。"
    }
    exit 1
}

# 2. 选择后端 Python 虚拟环境
$PythonCandidates = New-Object System.Collections.Generic.List[string]

$WorkspaceEnvDir = Join-Path (Join-Path $RootDir "workspace") "env_backend"
$VenvBackendDir = Join-Path $RootDir ".venv_backend"
$DotVenvDir = Join-Path $RootDir ".venv"

foreach ($baseDir in @($WorkspaceEnvDir, $VenvBackendDir, $DotVenvDir)) {
    $PythonCandidates.Add((Join-Path (Join-Path $baseDir "Scripts") "python.exe"))
    $PythonCandidates.Add((Join-Path (Join-Path $baseDir "bin") "python"))
}

if ($env:VIRTUAL_ENV) {
    $PythonCandidates.Add((Join-Path (Join-Path $env:VIRTUAL_ENV "Scripts") "python.exe"))
    $PythonCandidates.Add((Join-Path (Join-Path $env:VIRTUAL_ENV "bin") "python"))
}

foreach ($cmdName in @("python.exe", "python", "python3")) {
    $cmd = Get-Command $cmdName -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd -and $cmd.Source) {
        $PythonCandidates.Add($cmd.Source)
    }
}

$BackendPython = ""
$CheckedPaths = @{}
foreach ($candidate in $PythonCandidates) {
    if ([string]::IsNullOrWhiteSpace($candidate) -or $CheckedPaths.ContainsKey($candidate)) {
        continue
    }
    $CheckedPaths[$candidate] = $true
    if ((Test-Path $candidate -PathType Leaf) -and (Invoke-SilentCheck $candidate @("-c", "import fastapi, uvicorn"))) {
        $BackendPython = $candidate
        break
    }
}

if ([string]::IsNullOrWhiteSpace($BackendPython)) {
    Write-Err "未找到可用的后端 Python 环境（已检查 workspace/env_backend、.venv_backend、.venv 及当前 PATH）。"
    exit 1
}

$script:BackendProc = $null
$script:FrontendProc = $null
$script:CleanedUp = $false
$script:PushedLocation = $false

function ConvertTo-ProcessArg {
    param([string]$Value)
    if ($null -eq $Value -or $Value -eq "") {
        return '""'
    }
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    $escaped = $Value -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

function Start-LoggedProcess {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$WorkingDirectory,
        [string]$LogFile
    )

    $pyRunner = "import subprocess, sys; f = open(sys.argv[1], 'wb'); sys.exit(subprocess.call(sys.argv[2:], stdout=f, stderr=subprocess.STDOUT))"
    $fullArgs = New-Object System.Collections.Generic.List[string]
    $fullArgs.Add("-c")
    $fullArgs.Add($pyRunner)
    $fullArgs.Add($LogFile)

    if ($script:IsWinPlatform -and ($Executable -match '\.(cmd|bat)$')) {
        $comspec = if ($env:ComSpec) { $env:ComSpec } else { "cmd.exe" }
        $fullArgs.Add($comspec)
        $fullArgs.Add("/c")
    }
    $fullArgs.Add($Executable)
    foreach ($a in $Arguments) {
        $fullArgs.Add($a)
    }

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $BackendPython
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1"
    $psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8"

    if ($psi.PSObject.Properties["ArgumentList"]) {
        foreach ($item in $fullArgs) {
            [void]$psi.ArgumentList.Add($item)
        }
    } else {
        $psi.Arguments = (($fullArgs | ForEach-Object { ConvertTo-ProcessArg $_ }) -join " ")
    }

    return [System.Diagnostics.Process]::Start($psi)
}

function Stop-ProcTree {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$Label
    )
    if ($null -eq $Process) { return }
    try {
        if ($Process.HasExited) { return }
    } catch {
        return
    }

    $targetPid = $Process.Id
    if ($script:IsWinPlatform) {
        $taskkill = Get-Command taskkill.exe -ErrorAction SilentlyContinue
        if ($taskkill) {
            $prevAction = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                & $taskkill.Source /PID $targetPid /T /F *>$null
            } finally {
                $ErrorActionPreference = $prevAction
            }
        }
        try {
            if (-not $Process.HasExited) {
                $Process.Kill()
            }
        } catch {
            # 进程已退出时忽略
        }
    } else {
        $descendants = @()
        if (Get-Command pgrep -ErrorAction SilentlyContinue) {
            $collect = {
                param($p)
                $children = @(& pgrep -P $p 2>$null)
                foreach ($c in $children) {
                    & $collect $c
                    $c
                }
            }
            $descendants = @(& $collect $targetPid)
        }
        & kill -TERM $targetPid 2>$null | Out-Null
        foreach ($child in $descendants) {
            & kill -TERM $child 2>$null | Out-Null
        }
        $waited = 0
        while (-not $Process.HasExited -and $waited -lt 50) {
            Start-Sleep -Milliseconds 100
            $waited++
        }
        if (-not $Process.HasExited) {
            Write-Warn "$Label 未在 5s 内退出，发送 SIGKILL..."
            foreach ($child in $descendants) {
                & kill -KILL $child 2>$null | Out-Null
            }
            & kill -KILL $targetPid 2>$null | Out-Null
        }
    }

    try {
        $Process.WaitForExit(5000) | Out-Null
    } catch {
        # 忽略等待超时
    }
}

function Invoke-Cleanup {
    if ($script:CleanedUp) { return }
    $script:CleanedUp = $true

    if ($null -ne $script:FrontendProc -or $null -ne $script:BackendProc) {
        Write-Host ""
        Write-Info "正在停止 StageTTS 服务..."
    }
    Stop-ProcTree -Process $script:FrontendProc -Label "前端 Vite"
    Stop-ProcTree -Process $script:BackendProc -Label "后端 FastAPI"
}

function Show-Tail {
    param(
        [string]$Path,
        [int]$Lines = 30
    )
    if (-not (Test-Path $Path -PathType Leaf)) { return }
    try {
        Get-Content -Path $Path -Tail $Lines -Encoding UTF8 -ErrorAction SilentlyContinue | ForEach-Object {
            [Console]::Error.WriteLine($_)
        }
    } catch {
        # 忽略日志读取错误
    }
}

function Test-PortOpen {
    param(
        [string]$HostName,
        [int]$Port
    )
    $probeHost = if ($HostName -in @("0.0.0.0", "::", "")) { "127.0.0.1" } else { $HostName }
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync($probeHost, $Port)
        if ($task.Wait(500) -and $client.Connected) {
            return $true
        }
        return $false
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Test-HttpEndpoint {
    param([string]$Url)
    try {
        $req = [System.Net.WebRequest]::Create($Url)
        $req.Timeout = 1000
        $req.Method = "GET"
        $resp = $req.GetResponse()
        try {
            $code = [int]([System.Net.HttpWebResponse]$resp).StatusCode
            return ($code -lt 500)
        } finally {
            $resp.Close()
        }
    } catch [System.Net.WebException] {
        $errResp = $_.Exception.Response
        if ($null -ne $errResp) {
            try {
                $code = [int]([System.Net.HttpWebResponse]$errResp).StatusCode
                return ($code -lt 500)
            } finally {
                $errResp.Close()
            }
        }
        return $false
    } catch {
        return $false
    }
}

function Wait-ForHttp {
    param(
        [string]$Url,
        [int]$TimeoutSec,
        [System.Diagnostics.Process]$Process,
        [string]$Label,
        [string]$LogFile
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if ($Process.HasExited) {
            Write-Err "$Label 启动后提前退出 (exit=$($Process.ExitCode))，最近日志如下 ($LogFile):"
            Show-Tail -Path $LogFile -Lines 30
            return $false
        }
        if (Test-HttpEndpoint -Url $Url) {
            return $true
        }
        Start-Sleep -Milliseconds 250
    }
    Write-Err "$Label 启动超时 (>${TimeoutSec}s)，最近日志如下 ($LogFile):"
    Show-Tail -Path $LogFile -Lines 30
    return $false
}

try {
    # 以下所有相对路径操作均基于项目根目录
    Push-Location $RootDir
    $script:PushedLocation = $true

    # 3. 检查配置文件与 Banquet 模型路径
    $BackendEnvFile = Join-Path (Join-Path $RootDir "backend") ".env"
    $BackendEnvExample = Join-Path (Join-Path $RootDir "backend") ".env.example"

    if (-not (Test-Path $BackendEnvFile -PathType Leaf)) {
        if (Test-Path $BackendEnvExample -PathType Leaf) {
            Copy-Item -Path $BackendEnvExample -Destination $BackendEnvFile
            Write-Warn "未检测到 backend/.env，已从 backend/.env.example 复制模板，请确认其中的路径与密钥配置。"
        } else {
            Write-Err "未找到 backend/.env。"
            exit 1
        }
    }

    function Get-DotEnvValue {
        param([string]$Key)
        $pyCode = @"
from pathlib import Path
try:
    from dotenv import dotenv_values
    vals = dotenv_values(Path("backend/.env"))
except ImportError:
    from backend.config import _read_dotenv_fallback
    vals = _read_dotenv_fallback(Path("backend/.env"))
print((vals.get("$Key") or "").strip())
"@
        $out = $pyCode | & $BackendPython -
        if ($LASTEXITCODE -ne 0) { return "" }
        return ($out -join "`n").Trim()
    }

    $ModelCacheDir = Join-Path (Join-Path $RootDir "workspace") "model-cache"
    if ([string]::IsNullOrEmpty($env:BANQUET_ROOT)) {
        $envBanquetRoot = Get-DotEnvValue "BANQUET_ROOT"
        $defaultBanditDir = Join-Path $ModelCacheDir "bandit-v2"
        $defaultBanditFile = Join-Path (Join-Path (Join-Path (Join-Path $defaultBanditDir "src") "models") "bandit") "bandit.py"
        if ([string]::IsNullOrEmpty($envBanquetRoot) -and (Test-Path $defaultBanditFile -PathType Leaf)) {
            $env:BANQUET_ROOT = $defaultBanditDir
        }
    }

    if ([string]::IsNullOrEmpty($env:BANQUET_CHECKPOINT)) {
        $envBanquetCkpt = Get-DotEnvValue "BANQUET_CHECKPOINT"
        $defaultCkpt = Join-Path $ModelCacheDir "banquet-split.ckpt"
        if ([string]::IsNullOrEmpty($envBanquetCkpt) -and (Test-Path $defaultCkpt -PathType Leaf)) {
            $env:BANQUET_CHECKPOINT = $defaultCkpt
        }
    }

    # 4. 检查前端依赖
    $FrontendDir = Join-Path $RootDir "frontend"
    $NodeModulesDir = Join-Path $FrontendDir "node_modules"
    if (-not (Test-Path $NodeModulesDir -PathType Container)) {
        Write-Info "首次运行：正在安装前端依赖 (pnpm install)..."
        & $PnpmCmd.Source --dir $FrontendDir install
        if ($LASTEXITCODE -ne 0) {
            Write-Err "前端依赖安装失败 (pnpm install exit=$LASTEXITCODE)。"
            exit 1
        }
    }

    # 5. 预检核心模块与端口占用
    if (Test-PortOpen -HostName $BackendHost -Port $BackendPort) {
        Write-Err "后端端口 ${BackendHost}:${BackendPort} 已被占用，请先停止占用进程或通过 BACKEND_PORT 指定其他端口。"
        exit 1
    }

    if (Test-PortOpen -HostName $FrontendHost -Port $FrontendPort) {
        Write-Err "前端端口 ${FrontendHost}:${FrontendPort} 已被占用，请先停止占用进程或通过 FRONTEND_PORT 指定其他端口。"
        exit 1
    }

    if (Test-PortOpen -HostName "127.0.0.1" -Port 9880) {
        Write-Warn "检测到本机 9880 端口已有服务在监听；若运行解析流程（Stage1-3），后端将提示先释放 9880 显存。"
    }

    $nodeVersion = (& $NodeCmd.Source -v).Trim()
    $pnpmVersion = (& $PnpmCmd.Source -v).Trim()

    $preflightPy = @'
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
'@
    $preflightSummary = (($preflightPy | & $BackendPython -) -join " ").Trim()

    Write-Info "使用后端 Python: $BackendPython"
    Write-Info "使用前端 Node: $($NodeCmd.Source) ($nodeVersion) / pnpm ($pnpmVersion)"
    Write-Info "环境预检: $preflightSummary"

    if ($Check) {
        Write-Info "预检完成（-Check 模式，未启动服务）。"
        exit 0
    }

    $WorkspaceDir = Join-Path $RootDir "workspace"
    New-Item -ItemType Directory -Force -Path $WorkspaceDir | Out-Null
    $BackendLog = Join-Path $WorkspaceDir "_backend.log"
    $FrontendLog = Join-Path $WorkspaceDir "_frontend.log"

    # 6. 启动后端 FastAPI（GPT-SoVITS 由后端按需自动拉起/回收）
    Write-Info "正在启动后端服务 (http://${BackendHost}:${BackendPort})..."
    $script:BackendProc = Start-LoggedProcess `
        -Executable $BackendPython `
        -Arguments @("-m", "uvicorn", "backend.main:app", "--host", $BackendHost, "--port", [string]$BackendPort) `
        -WorkingDirectory $RootDir `
        -LogFile $BackendLog

    $BackendProbeHost = if ($BackendHost -in @("0.0.0.0", "::", "")) { "127.0.0.1" } else { $BackendHost }
    if ($BackendProbeHost -match ":" -and $BackendProbeHost -notmatch "^\[.*\]$") {
        $BackendProbeHost = "[$BackendProbeHost]"
    }
    if (-not (Wait-ForHttp -Url "http://${BackendProbeHost}:${BackendPort}/api/health" -TimeoutSec 25 -Process $script:BackendProc -Label "后端 FastAPI" -LogFile $BackendLog)) {
        exit 1
    }

    # 7. 启动前端 Vite
    Write-Info "正在启动前端界面 (http://${FrontendHost}:${FrontendPort})..."
    $ViteBin = Join-Path (Join-Path (Join-Path $NodeModulesDir "vite") "bin") "vite.js"
    if (Test-Path $ViteBin -PathType Leaf) {
        $script:FrontendProc = Start-LoggedProcess `
            -Executable $NodeCmd.Source `
            -Arguments @($ViteBin, "--host", $FrontendHost, "--port", [string]$FrontendPort, "--strictPort") `
            -WorkingDirectory $FrontendDir `
            -LogFile $FrontendLog
    } else {
        $script:FrontendProc = Start-LoggedProcess `
            -Executable $PnpmCmd.Source `
            -Arguments @("--dir", $FrontendDir, "exec", "vite", "--host", $FrontendHost, "--port", [string]$FrontendPort, "--strictPort") `
            -WorkingDirectory $RootDir `
            -LogFile $FrontendLog
    }

    $FrontendProbeHost = if ($FrontendHost -in @("0.0.0.0", "::", "")) { "127.0.0.1" } else { $FrontendHost }
    if ($FrontendProbeHost -match ":" -and $FrontendProbeHost -notmatch "^\[.*\]$") {
        $FrontendProbeHost = "[$FrontendProbeHost]"
    }
    if (-not (Wait-ForHttp -Url "http://${FrontendProbeHost}:${FrontendPort}/" -TimeoutSec 25 -Process $script:FrontendProc -Label "前端 Vite" -LogFile $FrontendLog)) {
        exit 1
    }

    # 8. 输出就绪摘要
    $HealthSummary = ""
    try {
        $health = Invoke-RestMethod -Uri "http://${BackendProbeHost}:${BackendPort}/api/health" -TimeoutSec 3
        $r = $health.readiness
        $HealthSummary = "status=$($health.status) | ffmpeg=$($r.ffmpeg) | banquet=$($r.banquet_configured) | llm=$($r.llm_configured) | hf_token=$($r.hf_token_configured) | tts_auto_manage=$($r.tts_auto_manage)"
    } catch {
        $HealthSummary = "status=unknown"
    }

    $FrontendUrl = "http://${FrontendProbeHost}:${FrontendPort}/"
    $BackendUrl = "http://${BackendProbeHost}:${BackendPort}"

    Write-Host ""
    Write-Info "=================================================="
    Write-Info "StageTTS 已就绪！"
    Write-Info "前端地址: $FrontendUrl"
    Write-Info "后端地址: $BackendUrl  (健康检查: $HealthSummary)"
    Write-Info "TTS 推理: 由后端按需自动管理 (GPT-SoVITS api_v2.py)"
    Write-Info "日志文件: $BackendLog | $FrontendLog"
    Write-Info "按 Ctrl+C 可一键停止所有服务"
    Write-Info "=================================================="
    Write-Host ""

    if ($OpenBrowserFlag) {
        try {
            if ($script:IsWinPlatform) {
                Start-Process $FrontendUrl | Out-Null
            } elseif ($IsMacOS) {
                Start-Process "open" -ArgumentList $FrontendUrl | Out-Null
            } elseif (($env:DISPLAY -or $env:WAYLAND_DISPLAY) -and (Get-Command xdg-open -ErrorAction SilentlyContinue)) {
                Start-Process "xdg-open" -ArgumentList $FrontendUrl | Out-Null
            }
        } catch {
            Write-Warn "自动打开浏览器失败，请手动访问: $FrontendUrl"
        }
    }

    # 9. 守护两个前台服务，任一异常退出时输出日志并统一清理
    while ($true) {
        if ($script:BackendProc.HasExited) {
            Write-Err "后端服务意外退出 (exit=$($script:BackendProc.ExitCode))，最近日志 ($BackendLog):"
            Show-Tail -Path $BackendLog -Lines 30
            exit 1
        }
        if ($script:FrontendProc.HasExited) {
            Write-Err "前端服务意外退出 (exit=$($script:FrontendProc.ExitCode))，最近日志 ($FrontendLog):"
            Show-Tail -Path $FrontendLog -Lines 30
            exit 1
        }
        Start-Sleep -Seconds 1
    }
} finally {
    Invoke-Cleanup
    if ($script:PushedLocation) {
        Pop-Location
    }
}
