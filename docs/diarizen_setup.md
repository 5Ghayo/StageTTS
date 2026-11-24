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
