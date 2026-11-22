# StageTTS — 多角色视频配音与虚拟演播平台

> 《人工智能专业导论》课程落地应用项目。  
> 本项目把“视频理解 / 剧本生成 / 智能剪辑 / 多角色配音 / 成片导出”串成一条本地可控的流水线，核心中间格式是 `script.json`。

---

## 0. 第一次接触请先读这里

如果你刚加入项目，建议按这个顺序理解：

1. **这个项目解决什么问题？**  
   给一段视频或音频，自动识别原文、判断说话人、翻译成目标语言、标注情绪和动作，再调用 GPT-SoVITS 生成多角色配音，最后和原视频/BGM 混音导出。

2. **项目的核心文件是什么？**  
   是 `workspace/{project_id}/script.json`。  
   它不是普通配置文件，而是整个系统的“唯一真相源”：
   - 解析结果写进去；
   - 前端时间轴读它；
   - 用户改台词也改它；
   - GPT-SoVITS 根据它合成音频；
   - 成片导出也根据它拼轨。

3. **如果想先看界面效果，最少需要什么？**  
   只需要：
   - Node.js 20+
   - Python 3.10+
   - FFmpeg

   可以只启动后端 + 前端，加载内置 demo 看 UI。  
   如果要真正跑解析和配音，还要额外配置 `env_parse` 和 GPT-SoVITS。

4. **不同电脑不用完全照抄开发机路径**  
   本 README 中的 `F:\Download\StageTTS`、`C:\Project\GPT-SoVITS...` 只是当前开发机示例。  
   你完全可以改成：
   - `D:\StageTTS\env_parse`
   - `E:\AI\GPT-SoVITS`
   - `F:\AI\hf_cache`

   只要 `backend/.env` 里的路径跟着改即可。

---

## 1. 项目一句话架构

```text
视频/音频
  ↓
Stage0 FFmpeg 抽音频
  ↓
Stage1 Demucs 分离人声 + BGM
  ↓
Stage2 WhisperX 识别 + 对齐
  ↓
Stage3 pyannote 说话人聚类
  ↓
Stage4 LLM 翻译 / 情绪 / 动作 / 是否保留原声
  ↓
script.json
  ↓
GPT-SoVITS 按句合成配音
  ↓
FFmpeg 混音导出 output.mp4
```

---

## 2. 当前开发进度与已知边界

请组员特别注意：**“代码已有”不等于“主入口已经完整接通”。**

| 模块 | 当前状态 | 说明 |
| :--- | :---: | :--- |
| `script.json` 数据契约 | 已完成 | Pydantic v2 与 TypeScript 类型 1:1 对齐 |
| 4 阶段解析流水线 | 已完成并实测 | FFmpeg → Demucs → WhisperX → pyannote → LLM 标注，已通过英文/日语真实视频测试 |
| GPT-SoVITS 分组配音 | 已完成 | 按角色切换权重、按情绪选择参考音，缺失时使用与原文对应的同角色原声片段 |
| 8GB 显存串行接力 | 已完成 | 解析前释放 GPT-SoVITS，配音时按需拉起 |
| 配音时长闭环 | 已完成 | 全部配音、全管线和智能体统一支持超出时间槽 15% 时自动改写重配（最多两轮），试听与导出使用同一份调速后的音频 |
| 参考音与角色校对 | 已完成 | 在「角色与情绪音色」中编辑/合并角色、批量改派、上传或试听 3–10 秒参考音并保存逐字稿；在「精修」中校对原文与说话人，在「项目媒体素材 → 翻译设置」中编辑术语表 |
| 评测脚本 | 骨架就绪 | 需要接入真实评测集后才能真正跑出指标 |

---

## 3. 核心概念：script.json 是什么？

### 3.1 它存什么？

`script.json` 顶层结构如下：

```json
{
  "project_id": "proj_xxx",
  "mode": "video_dubbing",
  "media": {
    "source_video": "/workspace/proj_xxx/input.mp4",
    "vocals_wav": "/workspace/proj_xxx/vocals.wav",
    "bgm_wav": "/workspace/proj_xxx/bgm.wav",
    "output_video": null,
    "duration": 60.0,
    "bgm_volume": 0.75
  },
  "characters": {
    "SPEAKER_00": {
      "display_name": "角色 A",
      "color": "#3b82f6",
      "gpt_weights": null,
      "sovits_weights": null,
      "emotion_refs": {},
      "avatar": {
        "type": "live2d",
        "model_url": "/assets/models/host_a/model3.json"
      }
    }
  },
  "segments": [
    {
      "id": 0,
      "start": 2.8,
      "end": 10.5,
      "speaker": "SPEAKER_00",
      "orig_text": "原片识别文本",
      "target_text": "目标配音文本",
      "target_lang": "zh",
      "emotion": "neutral",
      "action": "idle",
      "keep_original": false,
      "speed_factor": 1.0,
      "tts_wav": null,
      "actual_duration": null,
      "status": "pending"
    }
  ]
}
```

### 3.2 它给 GPT-SoVITS 提供什么？

真正调用 GPT-SoVITS 时，每句会组装成：

```json
{
  "text": "segment.target_text",
  "text_lang": "segment.target_lang",
  "ref_audio_path": "参考音频路径",
  "prompt_text": "参考音频对应文本",
  "prompt_lang": "参考音频语种",
  "speed_factor": 1.0,
  "media_type": "wav"
}
```

其中：

| GPT-SoVITS 参数 | 来源 |
| :--- | :--- |
| `text` | `segments[i].target_text` |
| `text_lang` | `segments[i].target_lang` |
| `speed_factor` | `segments[i].speed_factor` |
| `ref_audio_path` | `characters[speaker].emotion_refs[emotion].wav`，没有则从 `media.vocals_wav` 截取 |
| `prompt_text` | 情绪参考音文本，或零样本时使用 `orig_text` |
| `prompt_lang` | 情绪参考音语种，或自动检测 `orig_text` 语种 |

如果角色没有配置权重和参考音，当前系统会：

1. 从 `vocals.wav` 中按该 speaker 的时间段截取一段 3～8 秒原声；
2. 保存为 `workspace/{project_id}/tts/_zeroshot_SPEAKER_00.wav`；
3. 用这段原声作为音色参考；
4. 用 `orig_text` 作为参考文本；
5. 用 `target_text` 让 GPT-SoVITS 说目标语言。

---

## 4. 环境准备总览

### 4.1 你需要准备哪些东西？

| 环境 | 作用 | 是否必须 | 推荐版本 |
| :--- | :--- | :---: | :--- |
| FFmpeg | 抽音频、探测时长、最终混音 | 必须 | 6.x+，开发机为 9.1 |
| Node.js | 前端 Vite 项目 | 必须 | Node.js 20 LTS |
| Python backend | 跑 FastAPI 后端 | 必须 | Python 3.10+ |
| GPT-SoVITS | 生成配音 | 完整流程必须 | RTX 50 系列请用 CUDA 12.8 适配版 |
| env_parse | 跑 Demucs / WhisperX / pyannote | 完整解析必须 | Python 3.12 + torch cu128（开发机方案） |
| HuggingFace Token | 下载 pyannote 模型 | 说话人分离需要 | 免费，但要在网页同意协议 |
| LLM 网关 | 翻译、情绪、动作标注、Agent | 强烈建议 | OpenAI-compatible API 即可 |

---

## 4.2 按电脑情况选择配置

### 情况 A：只想看界面 / 改前端

你需要：

- Node.js 20+
- Python 3.10+
- FFmpeg

不需要配置 GPT-SoVITS 和 env_parse。  
只能使用内置 demo 看 UI 和基础交互，不能真正解析长视频和生成高质量配音。

---

### 情况 B：有 NVIDIA 显卡，8GB 左右，例如 RTX 5060 Laptop

这是当前开发机配置，推荐：

- `WHISPER_MODEL=small`
- `WHISPER_COMPUTE_TYPE=int8`
- GPT-SoVITS 使用本仓库的 `backend/tts_v2.yaml`
- 模型缓存放到非 C 盘，例如 `F:\Download\StageTTS\hf_cache`

不要使用默认 `v2ProPlus`，它在 8GB 笔记本显卡上容易 OOM。

---

### 情况 C：显存 12GB 或更多

可以尝试更高质量配置：

```env
WHISPER_MODEL=medium
WHISPER_COMPUTE_TYPE=float16
```

如果显存非常充足，例如 16GB+，也可以尝试：

```env
WHISPER_MODEL=large-v3
WHISPER_COMPUTE_TYPE=float16
```

但注意首次下载模型会很大，耗时较长。

---

### 情况 D：没有 NVIDIA GPU / 只有 CPU

可以启动后端和前端，但不建议跑完整流水线：

- Demucs 会回退 CPU，速度非常慢；
- WhisperX 也能 CPU，但长视频体验很差；
- GPT-SoVITS 实时合成基本不可用。

建议这种电脑只做前端开发、文档、接口 mock 或非实时测试。

---

### 情况 E：C 盘空间不足

把所有大文件都改到 D/E/F 盘：

```env
HF_HOME=F:/Download/StageTTS/hf_cache
HUGGINGFACE_HUB_CACHE=F:/Download/StageTTS/hf_cache/hub
TORCH_HOME=F:/Download/StageTTS/torch_cache
PIP_CACHE_DIR=F:/Download/StageTTS/pip_cache
```

`env_parse` 也可以放到非 C 盘。  
**不要把模型权重、缓存、env_parse 放进 C 盘。**

---

## 5. 详细环境安装步骤

下面以 Windows PowerShell 为主。  
如果你用 macOS / Linux，命令不同，但原则一样：**四个环境分开，路径写进 `.env`。**

---

### 5.1 安装 FFmpeg

#### Windows：一键脚本自动下载并配置（推荐）

在项目根目录打开 PowerShell，运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_ffmpeg_windows.ps1
```

脚本会自动完成：

1. 下载 FFmpeg Essentials Windows 构建；
2. 解压到：

```text
%LOCALAPPDATA%\StageTTS\ffmpeg
```

3. 把 `ffmpeg.exe` / `ffprobe.exe` 所在目录加入**当前用户 PATH**；
4. 立即在当前终端验证 `ffmpeg -version` 与 `ffprobe -version`。

如果你想装更大的 full build，或指定安装目录：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_ffmpeg_windows.ps1 -Full -InstallDir F:\Tools\ffmpeg
```

如果系统里已经有 FFmpeg，但你想强制重装：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_ffmpeg_windows.ps1 -Force
```

验证：

```powershell
ffmpeg -version
ffprobe -version
```

如果能输出版本号，说明成功。

#### Windows：手动安装

也可以手动安装：

1. 下载 FFmpeg Windows build；
2. 解压到例如：

```text
C:\Tools\ffmpeg\bin
```

3. 把该目录加入系统 PATH；
4. 新开 PowerShell 验证：

```powershell
ffmpeg -version
ffprobe -version
```

如果能输出版本号，说明成功。

#### 常见问题

- 如果 PowerShell 提示找不到 `ffmpeg`，通常是 PATH 没生效，重启终端或重新登录系统。
- 后端和 worker 都会调用系统 PATH 里的 `ffmpeg` / `ffprobe`，不是某个 Python 包里的。

---

### 5.2 安装 Node.js 与 pnpm

推荐 Node.js 20 LTS（前端使用 `pnpm` 管理依赖与 `pnpm-lock.yaml`）。

验证：

```powershell
node -v
pnpm -v
```

若尚未安装 `pnpm`，可先启用 Corepack 或全局安装：

```powershell
corepack enable
# 或 npm install -g pnpm
```

然后进入前端目录安装依赖：

```powershell
cd frontend
pnpm install
```

---

### 5.3 配置主后端 env_backend

主后端只跑 FastAPI、文件读写、任务调度，**不直接跑 GPU 模型**。

在项目根目录创建独立 Python 环境：

```powershell
python -m venv .venv_backend
.\.venv_backend\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

复制环境变量模板：

```powershell
copy backend\.env.example backend\.env
```

然后编辑 `backend\.env`。

#### backend/.env 重点字段

```env
# 项目工作区，留空默认使用仓库内 workspace/
WORKSPACE_DIR=

# 静态资源目录，留空默认使用仓库内 assets/
ASSETS_DIR=

# GPT-SoVITS 安装根目录
GPT_SOVITS_ROOT=C:/Project/GPT-SoVITS-v5-20261006nvidia50x0

# 轻量 v2 CUDA 配置
GPT_SOVITS_CONFIG=backend/tts_v2.yaml

# 自动管理 TTS 进程
TTS_AUTO_MANAGE=1

# 解析环境 Python
ENV_PARSE_PYTHON=F:/Download/StageTTS/env_parse/python.exe

# 模型缓存
HF_HOME=F:/Download/StageTTS/hf_cache
HUGGINGFACE_HUB_CACHE=F:/Download/StageTTS/hf_cache/hub
TORCH_HOME=F:/Download/StageTTS/torch_cache
PIP_CACHE_DIR=F:/Download/StageTTS/pip_cache

# WhisperX
WHISPER_MODEL=small
WHISPER_COMPUTE_TYPE=int8

# TTS 服务
TTS_API_URL=http://127.0.0.1:9880

# LLM，兼容 OpenAI SDK
LLM_BASE_URL=http://127.0.0.1:3425/v1
LLM_API_KEY=local-gateway-no-key
LLM_MODEL=workbuddy-ai/hy4flash-free

# HuggingFace Token
HF_TOKEN=hf_xxx
```

#### 注意

- **不要提交 `backend/.env`**，它已在 `.gitignore` 中。
- 如果没有本地 LLM 网关，可以填任何 OpenAI-compatible 服务，或者暂时不配置；但不配置时 Stage4 不会翻译，只会保留原文。
- `HF_TOKEN` 不要发到群里或截图里。

---

### 5.4 配置 GPT-SoVITS（env_tts）

#### 推荐方式

使用适配 RTX 50 系列 / CUDA 12.8 的 GPT-SoVITS 整合包。  
开发机路径示例：

```text
C:\Project\GPT-SoVITS-v5-20261006nvidia50x0
```

确认它能找到：

```text
runtime\python.exe
api_v2.py
```

然后在 `backend/.env` 中配置：

```env
GPT_SOVITS_ROOT=C:/Project/GPT-SoVITS-v5-20261006nvidia50x0
GPT_SOVITS_CONFIG=backend/tts_v2.yaml
TTS_AUTO_MANAGE=1
```

开启 `TTS_AUTO_MANAGE=1` 后，后端会在需要配音时自动拉起 GPT-SoVITS。

#### 如果要手动启动 GPT-SoVITS

```powershell
cd /d C:\Project\GPT-SoVITS-v5-20261006nvidia50x0
.\runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880 -c c:\Project\StageTTS\backend\tts_v2.yaml
```

#### 手动启动时的注意事项

当前 `TTS_AUTO_MANAGE` 只能管理“后端自己拉起”的 GPT-SoVITS。  
如果你手动启动了 9880，后端解析前不会自动杀掉这个外部进程。

所以：

- 如果要跑解析，建议先手动关闭 9880；
- 解析完成后，再启动 9880 配音；
- 或者不要手动启动，交给后端自动管理。

---

### 5.5 配置解析环境 env_parse

`env_parse` 专门跑：

- Demucs
- WhisperX
- pyannote
- 词级对齐
- Stage4 LLM 标注

不要把这些包装进 backend 环境。

#### 推荐方案：复制 GPT-SoVITS runtime

开发机方案是把 GPT-SoVITS 的 runtime 复制到非 C 盘：

```powershell
mkdir F:\Download\StageTTS
xcopy /E /I C:\Project\GPT-SoVITS-v5-20261006nvidia50x0\runtime F:\Download\StageTTS\env_parse
```

确认：

```powershell
F:\Download\StageTTS\env_parse\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

期望看到类似：

```text
2.7.1+cu128 True
```

千万不要把 cu128 换成 CPU torch。

#### 安装解析依赖

```powershell
F:\Download\StageTTS\env_parse\python.exe -m pip install demucs whisperx pyannote.audio==3.4.0 openai pydantic python-dotenv
```

如果下载慢，可以配置你所在网络允许的镜像或代理。

#### 设置缓存路径

在 `backend/.env` 中统一设置：

```env
HF_HOME=F:/Download/StageTTS/hf_cache
HUGGINGFACE_HUB_CACHE=F:/Download/StageTTS/hf_cache/hub
TORCH_HOME=F:/Download/StageTTS/torch_cache
PIP_CACHE_DIR=F:/Download/StageTTS/pip_cache
```

首次运行解析时会自动下载模型。  
日语词级对齐模型较大，网络不稳定时可能降级为句级时间戳，但不会阻断整个流程。

---

### 5.6 申请 HuggingFace Token

用于 pyannote 说话人分离。

步骤：

1. 注册 HuggingFace 账号；
2. 申请 Access Token；
3. 打开以下两个模型页面并同意协议：
   - `pyannote/speaker-diarization-3.1`
   - `pyannote/segmentation-3.0`
4. 把 Token 写入 `backend/.env`：

```env
HF_TOKEN=hf_xxx
```

注意：Token 是免费的，但不要公开。

---

## 6. 启动项目

### 6.1 标准启动：2 个终端

#### 终端 1：启动后端

在项目根目录：

```powershell
.\.venv_backend\Scripts\Activate.ps1
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

或者直接：

```powershell
F:\Python\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

验证：

浏览器打开：

```text
http://127.0.0.1:8000/api/health
```

应该看到：

```json
{
  "status": "ok"
}
```

#### 终端 2：启动前端

```powershell
cd frontend
pnpm dev
```

浏览器打开：

```text
http://127.0.0.1:5173/
```

---

## 7. 第一次使用流程

1. 打开 `http://127.0.0.1:5173/`
2. 如果右上角显示“后端已连接”，说明前后端连通。
3. 可以点“载入演示样例”先看内置界面。
4. 如果要处理自己的视频：
   - 点“新建 / 导入音视频”；
   - 选择 MP4 / WAV；
   - 等待创建项目。
5. 切到顶部“节点流”：
   - 点“仅解析剧本”生成 `script.json`；
   - 右侧“运行日志”会滚动显示 Stage0～Stage4。
6. 解析完成后切回“剪辑”：
   - 检查时间轴、台词、说话人；
   - 单句可点“重合成试听”；
   - 全部片段可点右上角“全部配音”。
7. 满意后点“导出成片”。

---

## 8. 常见问题排查

### Q1：前端显示“离线预览模式”

说明后端没启动或端口不对。检查：

```text
http://127.0.0.1:8000/api/health
```

如果后端改了端口，需要同步修改 `frontend/vite.config.ts` 里的代理目标。

---

### Q2：`ffmpeg` 找不到

验证：

```powershell
ffmpeg -version
```

如果失败，说明 FFmpeg 没加 PATH。

---

### Q3：解析时 GPT-SoVITS 占了显存

如果你手动启动了 9880，解析前请先关掉它。  
当前后端只能关闭自己自动拉起的 TTS 进程，不能强制关闭用户手动启动的进程。

---

### Q4：Demucs 长素材与 CUDA 显存问题

长素材使用 CUDA 分块分离（OOM 时自动缩小分块重试）。若手动启动的本机 GPT-SoVITS 仍在线，请先停止该服务再解析。CUDA 不可用或失败时默认继续使用未分离音轨，也可在「人声分离」节点中显式启用 CPU 分离。

建议：

- 使用 `small` + `int8`；
- 关闭其他占显存程序；
- 不要同时手动启动 GPT-SoVITS。

---

### Q5：日语词级对齐失败

日志可能出现：

```text
The chosen align_model ... could not be found ...
```

这是 WhisperX 把下载失败统一包装成的误导信息。  
流水线会降级保留句级时间戳，仍然可以继续生成 `script.json`。

---

### Q6：9880 根路径返回 404

GPT-SoVITS 的 `/` 返回 404 不代表不可用。  
只要 `/tts` 能正常返回音频即可。

---

### Q7：改了台词但音频没变

检查是否重新合成。  
`target_text`、`speaker`、`emotion`、`target_lang`、`keep_original` 修改后，片段应变为 `pending`，需要重新合成。

前端已有音频防缓存机制，但前提是重新合成后 `tts_wav` 被回写。

---

## 9. 仓库目录结构

```text
StageTTS/
├── ARCHITECTURE.md                     # 架构、铁律、接口契约
├── FRONTEND_DESIGN.md                  # 前端设计规范
├── README.md                           # 本文件
├── backend/
│   ├── main.py                         # FastAPI 入口
│   ├── config.py                       # 路径、环境、缓存、调度配置
│   ├── schemas.py                      # script.json 数据契约
│   ├── tts_v2.yaml                     # 8GB 显卡轻量 GPT-SoVITS 配置
│   ├── .env.example                    # 环境变量模板
│   ├── agent/                          # Agent 工具与 Prompt
│   └── services/                       # TTS、混音、GPU 调度
├── workers/
│   └── parse_worker.py                 # 解析流水线
├── frontend/
│   └── src/
│       ├── components/                 # UI 组件
│       ├── store/                      # 前端状态管理
│       ├── services/api.ts             # 后端接口封装
│       └── types/                      # TypeScript 类型
├── experiments/                        # 评测脚本骨架
├── assets/                             # 静态资源
└── workspace/                          # 运行时项目目录
```

---

## 10. 评测计划

评测脚本在 [experiments/](experiments)，目前是骨架：

- `eval_diarization.py`：Demucs 前后 WER/CER、DER 对比
- `eval_batch_speed.py`：按 speaker 分组合成 vs 逐句合成耗时/显存对比
- `eval_speaker_sim.py`：声纹相似度与 MOS 评测

后续建议补充：

- `eval_duration_reflection.py`：时长反思前后超时率
- `eval_e2e.py`：端到端成功率、阶段耗时、显存峰值

评测应该回答：

1. 解析更准了吗？
2. 剧本更自然了吗？
3. 配音更像了吗？
4. 超长台词真的变短了吗？
5. 8GB 显卡下更稳了吗？

---

## 11. AI 伦理与合规声明

- 本项目仅限使用团队成员授权音色或开源无版权许可音色。
- 严禁未经授权克隆商业声优、影视角色或公众人物声音。
- 所有导出视频与虚拟演播室界面均需保留“AI 合成语音”显式标识。
- 禁止用于欺诈、冒充、误导传播等不当用途。
