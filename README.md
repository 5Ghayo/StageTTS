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
   如果要真正跑解析和配音，还要额外配置独立的 `env_parse`（Banquet + WhisperX，并配置独立 DiariZen）和 GPT-SoVITS。

4. **不同电脑不用完全照抄开发机路径**  
   本 README 中的 `F:\Download\StageTTS`、`C:\Project\GPT-SoVITS...` 只是当前开发机示例。将 `ENV_PARSE_PYTHON` 指向本机独立 Banquet / WhisperX 环境，并修改 DiariZen、Banquet、GPT-SoVITS 和缓存路径即可。

---

## 1. 项目一句话架构

```text
视频/音频
  ↓
Stage0 FFmpeg 抽取音频
  ↓
Stage1 Banquet 分离对白、歌声、伴奏和音效
  ↓
Stage2 WhisperX 转写与词级对齐
  ↓
Stage3 DiariZen 说话人分离、证据保留与人工校对
  ↓
Stage4 LLM 翻译 / 情绪 / 动作 / 是否保留原声
  ↓
script.json
  ↓
GPT-SoVITS 按句合成配音
  ↓
FFmpeg 混音导出 output.mp4
```

应用解析固定使用 Banquet 四轨分离 → WhisperX 转写与词级对齐 → DiariZen Large v2 说话人分离 → LLM 翻译 → GPT-SoVITS 配音。DiariZen 在独立环境运行，配置见 [DiariZen 设置](docs/diarizen_setup.md)。重叠和覆盖不足的片段保留 UNKNOWN 并等待人工校对。

---

## 2. 当前开发进度与已知边界

请组员特别注意：**“代码已有”不等于“主入口已经完整接通”。**

| 模块 | 当前状态 | 说明 |
| :--- | :---: | :--- |
| WhisperX + DiariZen | 当前解析路线 | Banquet 对白轨 → WhisperX 转写和对齐 → 独立 DiariZen 说话人时间段；需要在实际动漫样本上验收质量。|

| `script.json` 数据契约 | 已完成 | Pydantic v2 与 TypeScript 类型 1:1 对齐 |
| GPT-SoVITS 分组配音 | 已完成 | 按角色切换权重、按情绪选择参考音，缺失时使用与原文对应的同角色原声片段 |
| 8GB 显存串行接力 | 已完成 | 解析前释放 GPT-SoVITS，配音时按需拉起 |
| 配音时长闭环 | 已完成 | 全部配音、全管线和智能体统一支持双向质检：>1.15x 自动精简，<0.8x 自动自然补足；最多两轮，最后才用 0.85–1.15x 温和变速兜底，试听与导出使用同一份调速后的音频 |
| 参考音与角色校对 | 已完成 | 在「角色与情绪音色」中编辑/合并角色、批量改派、上传或试听 3–10 秒参考音并保存逐字稿；自动参考音从通过校对、无说话人重叠的同角色原句中挑选，不再从任意时间点盲截；在「精修」中校对原文与说话人，在「项目媒体素材 → 翻译设置」中编辑术语表 |
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
    "vocals_wav": "/workspace/proj_xxx/cinematic-xxxx/dialogue.wav",
    "bgm_wav": "/workspace/proj_xxx/cinematic-xxxx/bgm.wav",
    "cinematic_stems": {
      "dialogue": "/workspace/proj_xxx/cinematic-xxxx/dialogue.wav",
      "singing": "/workspace/proj_xxx/cinematic-xxxx/singing.wav",
      "music": "/workspace/proj_xxx/cinematic-xxxx/music.wav",
      "effects": "/workspace/proj_xxx/cinematic-xxxx/effects.wav"
    },
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

### 3.2 Stage4 翻译怎么生成？

Stage4 使用**全文源对白上下文 + 每批最多 25 条输出**，不是只调用一次 LLM：

1. 后端把当前项目所有非空原句按时间顺序拼成完整源对白，每批请求都携带；
2. 每句保留原始 `id`、说话人、角色说话风格、术语表和软字数预算；
3. LLM 通读对白，为当前批次的 `id` 返回译文、情绪、动作和原声事件建议；
4. 后端严格校验当前批次的 ID、重复和缺失项，保留合法部分，错误显式返回；
5. 疑似漏译时单独修复译文，保留既有情绪、动作与已有原声保留标记。普通英文词被翻成中文不再被视为漏词。已有保留标记尚无人工/模型来源字段，可能包含旧模型误判，需要人工复核；M0不自动清除存量布尔值。

全文上下文帮助保持前后承接和术语一致性，语义完整性仍需校验与试听。正常短应答（Yes、Go、你好、はい等）必须翻译，不能仅因短或 LLM 返回布尔值就保留外语原声；自动保留仅允许明确非语言事件记号。

### 3.3 它给 GPT-SoVITS 提供什么？

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
| `speed_factor` | `/tts` 固定传 `1.0`；片段 `speed_factor` 在合成后由 FFmpeg `atempo` 应用 |
| `ref_audio_path` | `characters[speaker].emotion_refs[emotion].wav`，没有则从 `media.vocals_wav` 生成自动候选 |
| `prompt_text` | 情绪参考音文本，或自动候选对应的 `orig_text` |
| `prompt_lang` | 情绪参考音语种，或自动候选使用 `script.source_lang` |

当前自动参考音逻辑如下：

1. `characters[speaker].emotion_refs[emotion]` 中人工上传/绑定的参考音优先级最高；
2. 没有人工参考音时，按 `segment.emotion → neutral → 任意可用情绪` 回退；
3. 仍然没有时，才调用 `automatic_reference()` 从 `media.vocals_wav` 指向的音轨 生成自动候选；
4. 已校对路径选择**原文已校对、说话人已确认、3–10 秒、没有与其他说话人重叠**的同角色连续原句；
5. 优先使用完整 3–10 秒已确认台词；长段只有在词级文本严格对应当前 `orig_text`、时间有效、词级说话人明确且没有其他角色重叠时才能按词边界裁剪，参考文本只包含裁剪窗口中的词；
7. GPT-SoVITS 的 `/tts` 请求里 `speed_factor` 固定传 `1.0`，角色/片段的 `speed_factor` 是在后端用 FFmpeg `atempo` 对成品 WAV 做温和变速，试听和导出使用同一份处理后的文件。

### 3.4 M0 正确性边界

- 解析跳过/失败的说话人标为 `UNKNOWN`，保留失败原因并待校对，不冒充已经确认的单人；在精修面板指定真实角色后再确认。
- 长段优先自然停顿与句法边界；中日逐字对齐不能用任意 14 秒边界强切。允许受控延长，仍无可靠边界时保留完整原文并标记待审。
- 导出保留完整配音 WAV。超出原句槽但能放进后续空隙的尾部可保留；若新增串话或超过项目/源视频边界则明确阻塞导出待审，不删波形尾部、不改变源视频时长。旧项目的导出调速同样在调速后校验。
- M0 没有引入新说话人模型、声学情绪模型或新的 TTS。现有五类情绪只用于文本标记/参考音路由，不代表 GPT-SoVITS 接受任意情绪指令，也不保证源片表演或跨语种韵律逐词复现。

---

## 4. 环境准备总览

### 4.1 你需要准备哪些东西？

| 环境 | 作用 | 是否必须 | 推荐版本 |
| :--- | :--- | :---: | :--- |
| FFmpeg | 抽音频、探测时长、最终混音 | 必须 | 6.x+，开发机为 9.1 |
| Node.js | 前端 Vite 项目 | 必须 | Node.js 20 LTS |
| Python backend | 跑 FastAPI 后端 | 必须 | Python 3.10+ |
| GPT-SoVITS | 生成配音 | 完整流程必须 | RTX 50 系列请用 CUDA 12.8 适配版 |
| env_parse | Banquet 四轨分离、WhisperX 转写和对齐 | 完整解析必须 | 独立 Python 环境与匹配的 CUDA PyTorch |
| env_diarizen | DiariZen Large v2 说话人分离 | 完整解析必须 | 独立 Python 3.10；见 [配置文档](docs/diarizen_setup.md) |


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

解析使用 Banquet、WhisperX 和隔离的 DiariZen；合成使用 GPT-SoVITS。解析前关闭托管的 TTS 服务，其他手动启动的 GPU 服务需自行释放。按显卡选择匹配的 PyTorch CUDA 版本，并在目标素材上验证显存与耗时；当前流程没有固定 60 秒上限。CPU 可用于功能验证，长片处理耗时需实测。

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

# v5dev CUDA 配置
GPT_SOVITS_CONFIG=backend/tts_v5.yaml

# 自动管理 TTS 进程
TTS_AUTO_MANAGE=1

ENV_PARSE_PYTHON=/path/to/env_parse/bin/python
ENV_DIARIZATION_PYTHON=/path/to/env_diarizen/bin/python
DIARIZEN_MODEL_PATH=/path/to/diarizen-large-v2
DIARIZEN_EMBEDDING_PATH=/path/to/wespeaker-resnet34.bin
BANQUET_ROOT=F:/Download/StageTTS/banquet/bandit-v2
BANQUET_CHECKPOINT=F:/Download/StageTTS/banquet/banquet-split.ckpt
HF_HOME=F:/Download/StageTTS/hf_cache
HUGGINGFACE_HUB_CACHE=F:/Download/StageTTS/hf_cache/hub
TORCH_HOME=F:/Download/StageTTS/torch_cache
PIP_CACHE_DIR=F:/Download/StageTTS/pip_cache

PARSE_TIMEOUT_SEC=1800
LLM_TIMEOUT_SEC=60

# LLM 配置（示例值；推荐在 UI 顶部「设置」页配置）
LLM_API_KEY=local-gateway-no-key
LLM_MODEL=workbuddy-ai/hy4flash-free
```

The UI settings page persists model, endpoint and key in the ignored local backend settings file; keep secrets out of tracked documentation. Do not put access tokens in this example.

#### 在 UI 中切换翻译模型

点击顶部“设置”，在“翻译与 LLM”中修改模型名称（服务商提供的模型 ID）、OpenAI 兼容接口根地址和 API Key，然后点击“保存设置”。只换模型时 API Key 留空即可沿用；清除 Key 需要显式勾选。保存不调用模型，也不验证服务商是否支持该模型。

设置保存到后端私有的 `backend/llm_settings.local.json`（已加入 Git 忽略，含明文 Key，不在媒体静态目录中），优先于环境默认值，重启后继续生效；前端不会收到旧 Key，也不会写入浏览器存储。新发起的解析翻译、重译、时长改写和 AI 助手请求共用这套配置；已运行的请求继续使用原配置。

#### 注意

- **不要提交 `backend/.env`**，它已在 `.gitignore` 中。
- 如果没有本地 LLM 网关，可以填任何 OpenAI-compatible 服务，或者暂时不配置；但不配置时 Stage4 不会翻译，只会保留原文。
- **翻译模型设置**：顶部“设置”页或 UI；旧 `.env` 示例仅用于首次默认配置，不得提交密钥。

---

### 5.4 配置语音合成（env_tts）

在项目画布的「语音合成与反思」节点选择配音模型。选择随剧本保存，并应用于新合成和主动重配。新项目默认使用 GPT-SoVITS v5dev；已有项目的模型选择和角色权重保留。v5 重新训练见 [训练与迁移说明](docs/gpt_sovits_v5.md)。



#### GPT-SoVITS

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
GPT_SOVITS_CONFIG=backend/tts_v5.yaml
TTS_AUTO_MANAGE=1
```

开启 `TTS_AUTO_MANAGE=1` 后，后端会在需要配音时自动拉起 GPT-SoVITS。

#### 如果要手动启动 GPT-SoVITS

```powershell
cd /d C:\Project\GPT-SoVITS-v5-20261006nvidia50x0
.\runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880 -c c:\Project\StageTTS\backend\tts_v5.yaml
```

#### 手动启动时的注意事项

当前 `TTS_AUTO_MANAGE` 只能管理“后端自己拉起”的 GPT-SoVITS。  
如果你手动启动了 9880，后端解析前不会自动杀掉这个外部进程。

所以：

- 如果要跑解析，建议先手动关闭 9880；
- 解析完成后，再启动 9880 配音；
- 或者不要手动启动，交给后端自动管理。

---

### 5.5 配置解析环境 `env_parse`

Banquet 四轨分离和 WhisperX 在独立解析环境运行；DiariZen 通过独立子进程调用，后端仅负责任务协调。

#### Banquet 四轨分离

1. 在本机检出 [bandit-v2](https://github.com/kwatcharasupat/bandit-v2) 已审查提交 `d5563d9031`，并记录实际 commit/hash。
2. 单独取得 Netflix 官方 `banquet-split.ckpt`，核对 MD5 `4ac581d6792740f5b7be8e0a3f73c8ea`。
3. 配置 `BANQUET_ROOT` 和 `BANQUET_CHECKPOINT`。Bandit 依赖须与解析环境的 torch/torchaudio ABI 相容。

#### WhisperX 与 DiariZen

在 `env_parse` 安装 WhisperX 及与 Banquet 相容的依赖，配置 `ENV_PARSE_PYTHON`。DiariZen 使用 [专门的安装脚本与环境](docs/diarizen_setup.md)，通过 `ENV_DIARIZATION_PYTHON` 调用；不要把其 pyannote 框架分支安装到 WhisperX 环境。

#### 缓存目录

```env
HF_HOME=F:/Download/StageTTS/hf_cache
HUGGINGFACE_HUB_CACHE=F:/Download/StageTTS/hf_cache/hub
TORCH_HOME=F:/Download/StageTTS/torch_cache
PIP_CACHE_DIR=F:/Download/StageTTS/pip_cache
```

推理只读取经批准的本地 snapshot，不会静默下载模型。

---

---

#### Banquet / WhisperX 依赖安装注意

Banquet 使用本地 Bandit 源码和权重，安装时保留与本机 CUDA 匹配的 torch/torchaudio，避免直接用旧锁文件覆盖。DiariZen 的框架分支留在独立环境。

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
1. 在画布选中「媒体输入 / 音轨分离」开始 Banquet + WhisperX + DiariZen 完整解析；只想从已有对白开始时选择翻译节点。重叠及未对齐的对白需要人工审校。
   - 已有对白时，选中“翻译与情绪标注”节点和目标语言，点“翻译并保存剧本”，即可跳过音轨分离和识别；
   - 运行按钮会直接读取选中的节点，并从该节点继续到合成与导出；选中“剧本枢纽”或“语音合成与反思”使用现有译文，选中“视频混音导出”使用现有配音；
   - “语音识别与聚类”暂不支持单独启动，请选“音轨分离”执行完整解析；“虚拟舞台演播”通过右侧“进入虚拟舞台”打开。未选中可运行节点时运行按钮不可用；
   - 运行前会保存当前台词、术语表和角色设置。部分翻译失败时保留成功译文，并停止后续合成/导出；
   - 右侧“运行日志”显示实际执行步骤，跳过的节点保持待机；
   - 运行时可点“暂停”：当前后端步骤完成并保存后暂停，点击“继续”接着运行；等待期间可取消暂停，也可“结束流程”保留已有结果并停止后续步骤。解析（含分离/识别/翻译）、独立翻译、批量合成、导出各自是一个步骤，无法在请求内部即时暂停；最后一步完成后流程直接结束。暂停状态仅在当前页面中保留，刷新后可选中相应节点从已保存结果继续。
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

### Q4：解析失败或显存不足

检查 Banquet 权重、WhisperX 模型、DiariZen 模型和独立解释器路径。解析按阶段串行运行，但显存占用与输入有关；关闭其他 GPU 程序，使用小样本确认配置。缺少 DiariZen 配置或推理失败会终止本次解析，保留原有项目音轨和剧本。



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
│   ├── tts_v5.yaml                     # GPT-SoVITS v5dev 默认配置
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

训练语料生成与论文模型特征提取见 [训练音频工具链](workers/RESEARCH_TRAINING.md)。

评测脚本在 [experiments/](experiments)，目前是骨架：

- `eval_diarization.py`：Banquet 分离前后 WER/CER、DER 对比
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


### 3.5 M1 / M2 / M3 可选源声证据与分析

新解析可在 `script.evidence` 保存独立源证据：原始ASR整句与字词、可重叠的diarization活动区间、源turn，以及翻译/表演单位的来源链接。exclusive轨是派生方便轨，不替代真实重叠活动。旧JSON缺少evidence时仍可打开，不能从旧TTS倒造源声证据。

M2 边界候选仅来自原文字符范围和真实对齐结果。分词器可注入；缺少词典或词级时间时保守保留待审，不按字符比例生成时间。DiariZen 保留重叠活动；角色标识仅在当前录音内有效，跨录音身份需人工确认。

源声韵律/情绪分析为显式操作，不跟随每次保存自动运行。精修面板显示证据链接与分析状态；`not_run`/`not_configured`/`blocked`不是neutral。CPU能量/活动区间等测量与模型情绪分数分开保存，原始分数不冒充校准概率，不由标签猜F0。GPT-SoVITS v5 / v2 已接入；IndexTTS 保留为研究候选。

情绪分析使用独立解释器和经许可确认的本地权重：`ENV_ANALYSIS_PYTHON`、`EMOTION_MODEL_PATH`、`EMOTION_MODEL_ID`、`EMOTION_REVISION`、`EMOTION_FUNASR_VERSION`，以及默认关闭的`EMOTION_USER_AUTHORIZED`。不要更改现有 CUDA / PyTorch 运行环境；情绪分析与解析分别保留独立解释器，按各自锁文件部署。
