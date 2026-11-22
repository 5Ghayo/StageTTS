# StageTTS — 基于剧本中间表示与 Agent 协同的多角色配音及虚拟演播平台

> 《人工智能专业导论》课程落地应用项目 · 面向个人创作者与中小团队的本地化可控音视频生产平台（实测基准硬件：单卡消费级笔记本 GPU NVIDIA GeForce RTX 5060 8GB）。

---

## 当前开发进度（全链路已打通实测）

| 模块 | 状态 | 已实现与实测能力 |
| :--- | :---: | :--- |
| **全局数据契约 (`script.json`)** | 已完成 | Pydantic v2 (`backend/schemas.py`) 与 TypeScript (`frontend/src/types/script.ts`) 1:1 强类型对齐 |
| **4 阶段离线感知解析 (`workers/parse_worker.py`)** | 已完成 | FFmpeg 抽取 -> Demucs (`htdemucs`) CUDA 人声/伴奏分离 -> WhisperX ASR + 对齐 -> `pyannote/speaker-diarization-3.1` 聚类 -> 导演 LLM 口语翻译与情绪标注（已通过英/日真实视频端到端 CUDA 实测） |
| **RTX 5060 8GB 显存串行接力调度** | 已完成 | `TTS_AUTO_MANAGE=1` 按需启停：待机 0 显存占用；解析前自动停止 GPT-SoVITS 交出 100% 显卡；配音时自动拉起 `GPT-SoVITS v2` CUDA 半精度推理；内置单任务互斥锁与实时日志轮询接口 |
| **分组 TTS 合成与零样本克隆 (`tts_client.py`)** | 已完成 | 按 `speaker` 分组切换权重、动态情绪参考音频路由（缺失时自动截取 `vocals.wav` 做中/英/日零样本兜底）、`>1.15x` 超长检测 |
| **自然语言剪辑 Agent (`backend/agent/`)** | 已完成 | OpenAI SDK Function Calling 调度 5 大确定性原子工具 + 最多 2 轮时长约束反思改写闭环 |
| **毫秒拼轨与成片导出 (`media_mixer.py`)** | 已完成 | `pydub` 毫秒级时间轴拼轨 + `FFmpeg amix=normalize=0` + `-c:v copy` 无损流封装 / `burn_watermark` 合规水印重编码 |
| **Premiere × ComfyUI 融合工作台 (`frontend/`)** | 已完成 | 启动器导入/历史项目列表、视频预览 & 虚拟舞台、3 轨时间轴 (`wavesurfer.js`)、单句精修检查器、Agent 对话面板、React Flow 节点流（支持「仅解析剧本 / 运行全管线 / 全部配音」与实时阶段日志） |
| **课程评测脚本 (`experiments/`)** | 骨架就绪 | `eval_diarization.py` / `eval_batch_speed.py` / `eval_speaker_sim.py` 待接入答辩评测集 |

---

## 核心特性

1. **剧本中间表示枢纽（`script.json`）**：上游离线解析与下游交互精修彻底解耦，同时驱动 **模式 A（多角色视频重配）** 与 **模式 B（虚拟演播室）**。
2. **8 GB 单卡显存串行接力**：后端按需启停 `env_parse`（子进程退出即由 OS 100% 回收显存）与 `env_tts`（GPT-SoVITS `v2` CUDA 半精度），避免解析与合成争抢显存 OOM。
3. **动态情绪参考音频路由 & 零样本兜底**：按片段情绪（`neutral / angry / happy / sad / whisper`）动态选择参考音；未配置专属权重时自动截取分离后的纯人声做零样本克隆（支持中/英/日参考语种识别）。
4. **时长约束 Agent 反思闭环**：自然语言批量改轨；当合成音频超出原时间槽 15% 时自动反馈 LLM 精简重译并重合成（最多 2 轮），避免高倍变速失真。
5. **Premiere × ComfyUI 双态工作台**：支持「剪辑」「节点流」「舞台」三视角切换，节点流实时映射各阶段状态与解析日志。

---

## 仓库目录结构

```text
StageTTS/
├── ARCHITECTURE.md                     # 全局架构、硬性铁律与接口契约（必读）
├── FRONTEND_DESIGN.md                  # 前端专业工具化设计规范
├── README.md                           # 项目进度、环境安装与启动指南
├── backend/                            # 主后端服务 (env_backend, 纯 CPU, 0 显存)
│   ├── main.py                         # FastAPI 入口、静态挂载、REST 路由与解析状态轮询
│   ├── config.py                       # 路径、子进程环境、模型缓存与按需启停配置
│   ├── schemas.py                      # Pydantic v2 强类型数据契约
│   ├── tts_v2.yaml                     # RTX 5060 8GB 轻量 GPT-SoVITS v2 CUDA 推理配置
│   ├── requirements.txt                # 主后端依赖列表
│   ├── .env.example                    # 环境变量模板（含 F 盘缓存与按需启停示例）
│   ├── agent/                          # 大模型调度、Prompts 与 5 大原子工具
│   └── services/                       # GPU 串行锁与按需启停、分组 TTS 客户端、FFmpeg 混音
├── workers/                            # 独立重算力子进程脚本
│   └── parse_worker.py                 # 运行于 env_parse：Demucs -> WhisperX -> pyannote -> LLM 标注
├── frontend/                           # Vite + React 18 + TypeScript 融合工作台
│   └── src/
│       ├── types/                      # script.ts / workflow.ts 强类型定义
│       ├── services/api.ts             # 后端 REST 与解析状态轮询封装
│       ├── store/                      # useProjectStore.ts / useWorkflowStore.ts
│       └── components/                 # layout / panels / workflow / TimelinePR / Inspector / AgentChat / VirtualStage
├── assets/                             # 静态素材库（models/ 与 refs/）
├── experiments/                        # 课程答辩量化评测脚本
└── workspace/                          # 运行时项目目录（内置 proj_demo_01 示例剧本）
```

---

## 四环境隔离搭建指引（RTX 5060 / CUDA 12.8）

由于 RTX 5060（Blackwell 架构 `sm_120`）需要 CUDA 12.8+，且 WhisperX 与 GPT-SoVITS 存在底层包版本冲突，系统采用 **4 个物理隔离环境**：

### 0. 系统级前置条件
- 全局安装 **FFmpeg 6.x+**（本机已实测 FFmpeg 9.1）并加入系统 `PATH`。
- 申请免费 **HuggingFace Token**，并在官网同意 `pyannote/speaker-diarization-3.1` 与 `pyannote/segmentation-3.0` 的使用协议。

### 1. 语音合成环境 `env_tts`（GPT-SoVITS，CUDA 12.8）
- 部署适配 RTX 50 系列 / CUDA 12.8 的 **GPT-SoVITS** 整合包（Python 3.12 + `torch 2.7.1+cu128`）。
- 本仓库已内置轻量 `v2` 半精度配置 [backend/tts_v2.yaml](backend/tts_v2.yaml)（避开默认 `v2ProPlus` 在 8 GB 笔记本显卡上的 BAR1 OOM）。

### 2. 解析环境 `env_parse`（独立 CUDA 12.8 环境，可放非 C 盘）
- 可直接复用/克隆 GPT-SoVITS 的 Python 3.12 + `torch 2.7.1+cu128` 运行时到独立目录（如 `F:\Download\StageTTS\env_parse`），再安装解析依赖：
  ```powershell
  pip install demucs whisperx pyannote.audio==3.4.0 openai pydantic python-dotenv
  ```
- [workers/parse_worker.py](workers/parse_worker.py) 已内置 PyTorch 2.6+ `weights_only` 与 Python 3.12 `speechbrain` 兼容补丁，并自动读取 `HF_HOME` / `TORCH_HOME` 将模型缓存重定向到非 C 盘目录。

### 3. 主后端环境 `env_backend`（Python 3.10+，纯 CPU）
```powershell
pip install -r backend/requirements.txt
copy backend\.env.example backend\.env
```
编辑 `backend\.env`，配置 `GPT_SOVITS_ROOT`、`ENV_PARSE_PYTHON`、`HF_HOME` / `TORCH_HOME`、`LLM_BASE_URL` / `LLM_API_KEY` 与 `HF_TOKEN`。

### 4. 前端环境（Node.js 20 LTS）
```powershell
cd frontend
npm install
```

---

## 本地启动方式（按需显存接力，仅需开 2 个终端）

开启 `TTS_AUTO_MANAGE=1`（默认）后，**无需手动常驻启动 GPT-SoVITS**，后端会在「解析」和「配音」之间自动切换并释放 GPU 显存：

1. **启动主后端（项目根目录，`env_backend`）**
   ```powershell
   uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```
2. **启动前端工作台（`frontend/`）**
   ```powershell
   cd frontend
   npm run dev
   ```
3. 浏览器打开 `http://127.0.0.1:5173/`：
   - 点击 **「新建 / 导入音视频」** 上传本地 MP4/WAV（或打开内置 `proj_demo_01` 体验）；
   - 切到 **「节点流」** 点击 **「仅解析剧本」**（自动释放显存并实时滚动 4 阶段日志）；
   - 切回 **「剪辑」** 审校台词与角色，点击右上角 **「全部配音」**（自动按需拉起 GPT-SoVITS）或底部 **「重合成试听」**，满意后点击 **「导出成片」**。

---

## AI 伦理与合规声明

- 本项目开发、评测与公开演示仅限使用**团队成员自录授权音色**或**开源无版权许可音色**，严禁未经授权克隆商业声优、影视角色或公众人物声音。
- 所有导出视频与虚拟演播室界面均需保留“**AI 合成语音**”显式标识，禁止用于欺诈、冒充等不当用途。
