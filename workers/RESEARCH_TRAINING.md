# 研究训练数据工具链

文本与切段只依据音频；生成后的字幕导出和界面文字预览仍保留。

从仓库根目录运行 `python -m workers.research_training --toolchain /path/to/toolchain.json`。输出到独立目录，不覆盖输入项目。

## 保留的作者工具

| 模块 | 使用方式 |
| --- | --- |
| Emilia-Pipe | Amphion 原版 UVR-MDX-NET-Inst HQ 3、Silero VAD、Whisper Medium 与 DNSMOS；说话人分段替换为隔离的 DiariZen Large v2 |
| DialogueSidon | SaruLab 的 `spaces/app.py` 推理，保留默认 30 步和分块；处理已确认的双人重叠窗口 |
| emotion2vec | 官方 FunASR 接口提取 ACL 原版 base 的 768 维整句表示 |
| FACodec | Amphion 的原版 NaturalSpeech 3 编码器和解码器，提取内容、韵律、音色与声学细节 |

固定源码与模型 revision 分别见 `research_training.py` 和 `research_models.py`。特征提取不推断角色身份，也不自动给情绪特征赋训练标签。

## 环境与配置

为 Emilia、DialogueSidon、特征提取、DiariZen 使用各自独立环境。特征环境需要匹配的 PyTorch、FunASR、librosa、huggingface_hub 及 Amphion FACodec 所需包；DialogueSidon 使用匹配的 PyTorch、torchaudio、gradio、diffusers、soundfile。系统需要 FFmpeg。DiariZen 安装见 [配置说明](../docs/diarizen_setup.md)。

复制 Amphion 的 `preprocessors/Emilia/config.json`，去掉注释，填写 UVR / DNSMOS 模型的本地绝对路径。工具链 JSON 示例（相对路径以工具链文件所在目录为基准）：

```json
{
  "amphion_root": "/path/to/Amphion",
  "emilia_python": "/path/to/env_emilia/bin/python",
  "emilia_config": "/path/to/emilia_config.json",
  "sidon_root": "/path/to/Sidon",
  "sidon_python": "/path/to/env_sidon/bin/python",
  "feature_python": "/path/to/env_features/bin/python",
  "weights_dir": "/path/to/research-weights",
  "diarization_python": "/path/to/env_diarizen/bin/python",
  "diarizen_model_path": "/path/to/models/diarizen-large-v2",
  "diarizen_embedding_path": "/path/to/models/wespeaker-resnet34.bin",
  "diarization_device": "cuda",
  "diarizen_batch_size": 1
}
```

## 制作语料

```bash
python -m workers.research_training --toolchain /path/to/toolchain.json prepare \
  --input movie.wav --output /path/to/corpus

python -m workers.research_voice_corpus --corpus /path/to/corpus \
  --output /path/to/voices --min-clips 3 --min-seconds 12
```

`prepare --project /path/to/script.json` 可使用项目时间轴。两个已经确认的角色发生重叠时，DialogueSidon 优先读取项目对白轨中的对应窗口；也可重复传 `--overlap two-speaker.wav`。直接输入的音视频使用 Emilia 原版 UVR 预处理；它与应用中的 Banquet 四轨处理是不同入口。

`--emilia-min-clip-seconds` 和 `--emilia-min-dnsmos` 默认均为 3。可显式放宽；`--emilia-fine-turns` 对每个无重叠的说话人区间运行 VAD，支持 `--emilia-max-turn-seconds`、`--emilia-min-silence-ms` 和 `--emilia-tail-ms`。切段只使用对白音频、说话人活动和 VAD，不读取字幕轨、不做 OCR，也不接受字幕时间点。参数写入 manifest，便于复现。

`emilia_processed/` 保留原版片段和逐文件 JSON；`manifest.json` 保留来源、时间、文本、语种、模型和特征路径；NPZ 含 `emotion2vec`、`prosody`、`content`、`timbre`、`acoustic_detail`。角色归属需要人工审核，双人分离的通道号不代表跨窗口的固定角色。

训练清单直接按上游说话人分组，并按来源、分离通道和重叠窗口隔离作用域；不再按音色嵌入二次聚类。导出 32 kHz 单声道 WAV 和 GPT-SoVITS `labels.list`。可用 `--min-clips`、`--min-seconds`、`--min-clip-seconds`、`--max-clip-seconds` 控制导出门槛。

```bash
python -m workers.research_training --toolchain /path/to/toolchain.json separate \
  --input two-speaker.wav --output /path/to/separated

python -m workers.research_training --toolchain /path/to/toolchain.json features \
  --input reviewed.wav --output /path/to/features
```
