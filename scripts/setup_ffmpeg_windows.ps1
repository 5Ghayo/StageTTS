<#
.SYNOPSIS
  StageTTS Windows FFmpeg 一键下载与 PATH 配置脚本。

.DESCRIPTION
  - 默认下载 gyan.dev 的 FFmpeg Essentials Windows 构建（体积较小，足够 StageTTS 使用）。
  - 自动解压到本机用户目录，不要求管理员权限。
  - 自动把 ffmpeg.exe / ffprobe.exe 所在 bin 目录加入“当前用户 PATH”。
  - 不修改系统 PATH，不污染仓库目录。

.PARAMETER InstallDir
  FFmpeg 安装目录。默认：$env:LOCALAPPDATA\StageTTS\ffmpeg

.PARAMETER Full
  使用更大的 full build，包含更多编码器/滤镜。默认使用 essentials build。

.PARAMETER Force
  强制重新下载并配置，即使当前 PATH 中已有 ffmpeg / ffprobe。

.EXAMPLE
  .\scripts\setup_ffmpeg_windows.ps1

.EXAMPLE
  .\scripts\setup_ffmpeg_windows.ps1 -Full -InstallDir F:\Tools\ffmpeg

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\setup_ffmpeg_windows.ps1
#>

param(
    [string]$InstallDir = "$env:LOCALAPPDATA\StageTTS\ffmpeg",
    [switch]$Full,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# 兼容较老 Windows PowerShell 的 TLS 设置
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
} catch {
    # 忽略：较新环境默认已支持 TLS 1.2+
}

$EssentialsUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$FullUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.zip"

function Write-Step {
    param([string]$Message)
    Write-Host "[ffmpeg-setup] $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "[ffmpeg-setup] $Message" -ForegroundColor Green
}

function Download-File {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$OutFile
    )

    Write-Step "下载: $Url"
    try {
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
        return
    } catch {
        Write-Warning "[ffmpeg-setup] Invoke-WebRequest 下载失败，尝试使用 curl.exe 续试: $($_.Exception.Message)"
    }

    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        & curl.exe -L --fail --retry 3 -o $OutFile $Url
        if ($LASTEXITCODE -ne 0) {
            throw "curl.exe 下载失败，exit=$LASTEXITCODE"
        }
        return
    }

    throw "下载失败，且当前系统没有可用的 curl.exe。请手动下载: $Url"
}

function Add-To-UserPath {
    param([Parameter(Mandatory = $true)][string]$BinDir)

    $binDirNorm = $BinDir.TrimEnd('\', '/')
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $parts = @()

    if (-not [string]::IsNullOrWhiteSpace($userPath)) {
        $parts = @($userPath -split ";" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    }

    $alreadyExists = $false
    foreach ($p in $parts) {
        if ($p.TrimEnd('\', '/') -ieq $binDirNorm) {
            $alreadyExists = $true
            break
        }
    }

    if ($alreadyExists) {
        Write-Step "当前用户 PATH 已包含: $binDirNorm"
        return
    }

    $parts += $binDirNorm
    $newPath = ($parts -join ";")
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Ok "已加入当前用户 PATH: $binDirNorm"
}

# ---------------------------------------------------------------------------
# 0. 如果系统已有可用 ffmpeg / ffprobe，默认直接跳过
# ---------------------------------------------------------------------------
if (-not $Force) {
    $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    $ffprobeCmd = Get-Command ffprobe -ErrorAction SilentlyContinue
    if ($ffmpegCmd -and $ffprobeCmd) {
        Write-Ok "检测到系统已安装 FFmpeg，跳过安装。"
        & ffmpeg -version | Select-Object -First 1
        & ffprobe -version | Select-Object -First 1
        Write-Host "[ffmpeg-setup] 如需强制重装，请加 -Force 参数。" -ForegroundColor Yellow
        return
    }
}

# ---------------------------------------------------------------------------
# 1. 准备目录与下载链接
# ---------------------------------------------------------------------------
$Url = if ($Full) { $FullUrl } else { $EssentialsUrl }
$BuildName = if ($Full) { "full" } else { "essentials" }
$zipPath = Join-Path $env:TEMP "stagetts_ffmpeg_$BuildName.zip"
$extractDir = Join-Path $InstallDir "extracted"

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

Write-Step "安装目录: $InstallDir"
Write-Step "构建类型: $BuildName"

# ---------------------------------------------------------------------------
# 2. 下载 ZIP
# ---------------------------------------------------------------------------
if ((Test-Path $zipPath) -and (-not $Force)) {
    Write-Step "发现本地缓存 ZIP，直接使用: $zipPath"
} else {
    Download-File -Url $Url -OutFile $zipPath
}

# ---------------------------------------------------------------------------
# 3. 解压并定位 bin 目录
# ---------------------------------------------------------------------------
if (Test-Path $extractDir) {
    Write-Step "清理旧解压目录: $extractDir"
    Remove-Item -Recurse -Force $extractDir
}

Write-Step "解压到: $extractDir"
Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

$ffmpegExe = Get-ChildItem -Path $extractDir -Filter "ffmpeg.exe" -Recurse -File | Select-Object -First 1
$ffprobeExe = Get-ChildItem -Path $extractDir -Filter "ffprobe.exe" -Recurse -File | Select-Object -First 1

if (-not $ffmpegExe -or -not $ffprobeExe) {
    throw "解压完成但未找到 ffmpeg.exe / ffprobe.exe，请检查下载包是否完整。"
}

$binDir = Split-Path $ffmpegExe.FullName -Parent

# ---------------------------------------------------------------------------
# 4. 写入当前用户 PATH，并立即更新当前进程 PATH
# ---------------------------------------------------------------------------
Add-To-UserPath -BinDir $binDir
$env:Path = "$binDir;$env:Path"

# ---------------------------------------------------------------------------
# 5. 验证
# ---------------------------------------------------------------------------
Write-Step "验证安装结果："
& $ffmpegExe.FullName -version | Select-Object -First 1
& $ffprobeExe.FullName -version | Select-Object -First 1

Write-Ok "FFmpeg 配置完成。"
Write-Host "[ffmpeg-setup] 注意：新开终端后 PATH 才会永久生效。" -ForegroundColor Yellow
Write-Host "[ffmpeg-setup] 当前终端已立即生效，可直接继续运行 StageTTS。" -ForegroundColor Yellow
