# DiariZen Large v2 说话人分离

应用固定使用 Banquet 四轨分离 → WhisperX 转写和词级对齐 → DiariZen 说话人时间段 → 人工审校 / 翻译 / GPT-SoVITS。训练语料入口使用 Emilia-Pipe，其中说话人分段同样调用独立 DiariZen。

## 固定模型与独立环境

| 资源 | 固定 revision |
| --- | --- |
| [DiariZen 源码](https://github.com/BUTSpeechFIT/DiariZen) | `844f5555b0a98acd0931511fc641a8c5b8ba92c7`，包含作者的 pyannote 框架分支 |
| [DiariZen Large v2](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2) | `027b3e221b9d81dce2be794084f1dbd3ba2da403` |
| [内部 WeSpeaker 模型](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | `837717ddb9ff5507820346191109dc79c958d614` |

用 Python 3.10 创建独立环境，不要把 DiariZen 的 pyannote 框架分支安装到 WhisperX、后端或 TTS 环境。

```sh
python scripts/setup_diarizen.py --root .cache/diarizen --environment .venv-diarizen
```

脚本下载固定源码和权重，生成本地 SHA-256 清单与 `speaker_settings.json`。CPU 默认 torch / torchaudio 2.5.1；CUDA 机器可显式选择匹配的 wheel，例如 `--torch-version 2.8.0 --torch-index https://download.pytorch.org/whl/cu128`。这不代表该机器上的完整配音质量已经验证。

在 `backend/.env` 中填写本机绝对路径。

```dotenv
ENV_PARSE_PYTHON=/absolute/path/env_parse/bin/python
ENV_DIARIZATION_PYTHON=/absolute/path/.venv-diarizen/bin/python
DIARIZEN_MODEL_PATH=/absolute/path/.cache/diarizen/diarizen-large-v2
DIARIZEN_EMBEDDING_PATH=/absolute/path/.cache/diarizen/wespeaker-resnet34.bin
DIARIZEN_BATCH_SIZE=1
```

Windows 解释器路径使用 `Scripts/python.exe`。修改后重启后端。解析的 `--device auto` 选择 CUDA 或 CPU；独立 worker 使用同一设备配置。缺少本地文件、框架不匹配或推理失败会终止新解析，原有剧本与音轨保留。`--skip-diarization` 显式跳过说话人分离并保留 UNKNOWN。

## 身份与时间证据

- 直接保留 DiariZen 的说话人分组，不再做额外音色聚类。
- `speaker_analysis.json` 保留原始活动时间段、来源音频哈希和原始到录音内角色编号的映射；编号带来源哈希，跨录音同号不自动认定为同角色。
- 转写区间必须由一个说话人至少覆盖 80%，同时不能存在其他说话人的重叠活动。重叠、多说话人或覆盖不足保留 UNKNOWN。
- WhisperX 只有真实词级时间才参与词级分配，不能按字数比例编造时间。DiariZen 的帧尾越界显式裁到音频边界，并保留原始时间段与审校原因。
- 文字与词级对齐完整匹配、时间有序且不重叠时，输出片段首尾贴合首词和末词的真实时间，避免 ASR 窗口首尾的空白或其他人物活动影响归属。原始 ASR 窗口仍保留在来源证据中。跨说话人边界的词、真实重叠或缺失对齐继续待审，不按多数覆盖强行归类。
- 本次代码检查验证接口与证据逻辑；动漫识别、分离与配音质量需用实际素材验收。

训练工具链在 Emilia 配置之外增加以下配置。

```json
{
  "diarization_python": "/absolute/path/.venv-diarizen/bin/python",
  "diarizen_model_path": "/absolute/path/models/diarizen-large-v2",
  "diarizen_embedding_path": "/absolute/path/models/wespeaker-resnet34.bin",
  "diarization_device": "cpu",
  "diarizen_batch_size": 1
}
```

Emilia 保留其 UVR、Silero VAD、Whisper 与 DNSMOS；包装器只替换旧的说话人模型及令牌门槛，训练切段只使用无重叠的说话人区间。旧 `diarization_worker.py` 仅用于隔离的基线实验。

## 音频来源与细切

识别、分段和翻译只使用音频，不读取内嵌 / 外挂字幕或 OCR 结果。原文由音频 ASR 生成，再由 LLM 翻译为中文；生成后的文字预览与字幕导出不受影响。

本次适配已在 WhisperX 3.4.5 验证。WhisperX 复用其原有 Pyannote VAD 权重和最低语音分数切点，将连续语音窗口控制在约 8 秒，保留至少 250 毫秒的停顿作为片段边界，不再跨这些停顿拼成 30 秒窗口。相邻片段最多各补 30 毫秒上下文，且不互相覆盖。短应答保留，不为凑时长重新合并。对齐后遇到句末标点、可靠停顿或说话人变化继续拆句。

这是语音片段细切，不保证每一段恰好对应完整语法句。无停顿的长句可能在 VAD 低分点分开；连读、抢话、ASR 漏字和错误的日语词级对齐仍需要审校。输入字幕不再是补写或纠错来源。

2026-11-25 在同一份 Banquet 对白轨的 3651–3771 秒做 CUDA 对照，WhisperX 3.4.5 / Whisper medium 与日语对齐模型的 revision、输入音频哈希保持一致。旧方式为 5 段（最长 27.692 秒、中位 26.493 秒），音频细切后为 24 段（最长 6.582 秒、中位 2.203 秒）。这验证分块行为生效，不代表识别准确率、人物归属或每句完整性已经人工验收；该对照没有运行字幕输入、翻译或 TTS。
