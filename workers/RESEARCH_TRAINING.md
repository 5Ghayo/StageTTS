# 用论文工具生成训练语音

从仓库根目录运行 `python -m workers.research_training`。入口读取原始音视频或已经解析的 StageTTS 项目，输出独立的训练数据目录。现有项目剧本和音轨可作为输入。

## 作者工具与权重

| 工作 | 运行的作者实现 | 对应输入或权重 |
| --- | --- | --- |
| DialogueSidon | SaruLab 发布的 `spaces/app.py` 推理函数，保留作者默认的 30 步及分块处理 | 第一版 DialogueSidon 的 3 个 `.pt2` 和 `metadata.json`。模型以 2 人单声道混合为输入 |
| KyaraBench / KyaraEmbed | `score.py`、`prepare/restore.py`、`embedders.kyaraembed` 及原版 10 秒载入函数 | `spellbrush/kyaraembed` 的 `kyaraembed_v1.0.pt`，核对作者发布的 SHA-256 |
| Emilia-Pipe | Amphion 的 `preprocessors/Emilia/main.py`。分离、说话人分段、VAD、识别、评分和过滤均由该程序执行 | 原版 UVR-MDX-NET-Inst HQ 3、pyannote 3.1、Silero VAD、Whisper Medium 与 DNSMOS P.835 |
| emotion2vec | 论文作者文档中的 FunASR 提取入口 | ACL 论文的 `emotion2vec_base`。输出 768 维整句表示，可供下游情绪分类训练 |
| NaturalSpeech 3 | Amphion 的原版 FACodec 编码器和解码器 | 官方 v1 encoder/decoder。输出内容、韵律、音色与声学细节。此入口集成公开的 FACodec，扩散合成属于完整 NaturalSpeech 3 的后续阶段 |

固定的 Hugging Face 权重仓库与 revision 在 [research_models.py](research_models.py) 中，官方源码提交在 [research_training.py](research_training.py) 中。权重可自动下载，也可预先放入 `weights_dir` 下对应的 `dialogue-sidon`、`kyaraembed`、`emotion2vec`、`facodec` 子目录。请使用与代码中 revision 相同的文件。

## 配置

分别获取作者仓库，并检出以下版本。

- [Amphion](https://github.com/open-mmlab/Amphion) `26f6883110181f1dbfe95c70a7c7dbaf4de5f42a`
- [Sidon](https://github.com/sarulab-speech/Sidon) `45210939881b8a38a0752fb33451a79a9d3703ee`
- [KyaraBench](https://github.com/sizigi/kyarabench) `6453727862d3bdf54a8e50a4d26a254ca3b47e14`
- [emotion2vec](https://github.com/ddlBoJack/emotion2vec) `a0e606c668227ab0c01527e196a4acc42b1efad9` 的文档提供官方 FunASR 接口

1. 为 Emilia、DialogueSidon 和特征提取准备独立 Python 环境。Emilia 按 [Amphion 的配置文档](https://github.com/open-mmlab/Amphion/tree/26f6883110181f1dbfe95c70a7c7dbaf4de5f42a/preprocessors/Emilia) 安装，WhisperX 依赖按下方版本固定。复制作者的 `config.json`，去掉 `// TODO` 注释，保留分离参数及语言列表，填写官方 UVR/DNSMOS 模型的绝对路径和 Hugging Face 访问令牌。需要先开通 [说话人模型](https://huggingface.co/pyannote/speaker-diarization-3.1) 和 [分段模型](https://huggingface.co/pyannote/segmentation-3.0) 的访问权限。
2. DialogueSidon 环境使用匹配的 PyTorch、torchaudio 2.8 和作者演示所需的 `gradio`、`diffusers`、`huggingface_hub`，另需 `soundfile` 写出 WAV。特征环境需要 `torch`、`torchaudio`、`funasr`、`anime-speaker-embedding`、`huggingface_hub`、`librosa`、`soundfile`、`einops`、`pyworld`、`setuptools<81`，以及 KyaraBench 自带 `requirements.txt`。系统需要 FFmpeg。
3. 复制下方 JSON 到本机文件，替换目录和 Python 路径。路径相对于这份工具链配置。Emilia 的配置另存于本机，令牌留在该文件中。

```json
{
  "amphion_root": "/path/to/Amphion",
  "emilia_python": "/path/to/env_emilia/bin/python",
  "emilia_config": "/path/to/private-emilia-config.json",
  "sidon_root": "/path/to/Sidon",
  "sidon_python": "/path/to/env_sidon/bin/python",
  "kyara_root": "/path/to/kyarabench",
  "feature_python": "/path/to/env_features/bin/python",
  "weights_dir": "/path/to/research-weights"
}
```

### Emilia 的依赖版本

Emilia 的识别代码使用 2024 年版 `TranscriptionOptions`。使用 [WhisperX v3.2.0](https://github.com/m-bain/whisperX/tree/v3.2.0) 及其声明的 `faster-whisper==1.0.0`、`ctranslate2==4.4.0`、`pyannote.audio==3.1.1`。验证原版 CLI 导入、识别管线构造及公开 ONNX 权重加载时使用了 Python 3.10、`torch==2.5.1`、`torchaudio==2.5.1`、`transformers==4.38.2`、`huggingface_hub==0.36.2`、`numpy<2` 和 `torchmetrics<1.5`。测试机是 macOS ARM，使用了当地可安装的 PyAV 17.1。复现作者环境时，PyAV 11 若需要从源码编译，需安装 `pkg-config` 和 FFmpeg 开发库。从源码安装 WhisperX 时，构建环境需要 `setuptools<81`。Emilia 端到端仍需上述访问权限。

## 生成数据

```bash
python -m workers.research_training --toolchain /path/to/toolchain.json prepare \
  --project workspace/your-project/script.json --output /path/to/corpus
```

- 可用 `--input recording.mp4` 直接处理音视频。Emilia-Pipe 读取原始节目音轨，按作者的 UVR 模型处理。用 StageTTS 项目时，DialogueSidon 优先从项目对白轨裁出两位已确认角色的重叠区间。也可重复传 `--overlap two-speaker.wav` 处理手工确认的双人窗口。
- `emilia_processed/` 保存原版 MP3 和逐文件 JSON。总 `manifest.json` 记录片段出处、文本、语种、时间、DNSMOS 及特征路径。切自重叠窗口的片段以该窗口起点计时，窗口在原视频中的起点见 `provenance.start`。`features/` 中每个 NPZ 包含 `kyaraembed`、`emotion2vec`、`prosody`、`content`、`timbre`、`acoustic_detail`。
- 有 StageTTS 人工确认的角色时间轴时，单角色覆盖足够的原始片段可写入 `character_id`。DialogueSidon 的两路以通道编号保存。有同角色参考片段时，KyaraEmbed 用原版池化和余弦相似度提供候选，待人工确认后才适合做有标签的角色训练。自动定位依赖剧本保留重叠时间；缺少重叠标注时可以传入人工选出的窗口。

## 单项验证和官方基准

```bash
python -m workers.research_training --toolchain /path/to/toolchain.json separate \
  --input two-speaker.wav --output /path/to/separated

python -m workers.research_training --toolchain /path/to/toolchain.json features \
  --input reviewed.wav --output /path/to/features

python -m workers.research_training --toolchain /path/to/toolchain.json kyarabench \
  --clips /path/to/restored-anim400k
```

对已获授权的原始 Anim-400K benchmark 文件，可以在最后一条命令增加 `--anim400k-source /path/to/anim400k_audio_clips`，由 KyaraBench 自带的 `prepare/restore.py` 核对源文件校验和、调用原版 Sidon 恢复，然后用冻结的试验清单评分。作者仓库提供试验清单和权重。Anim-400K 原始音频需按其访问流程单独获取。
