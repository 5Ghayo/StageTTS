# 用论文工具生成训练语音

从仓库根目录运行 `python -m workers.research_training`。入口读取原始音视频或已经解析的 StageTTS 项目，输出独立的训练数据目录。现有项目剧本和音轨可作为输入。

`prepare --backend emilia` 使用原来的 Emilia-Pipe，默认也是该模式。`prepare --backend moss --moss-language ja` 使用 MOSS 联合转写和说话人分段，再按模型时间戳导出切片，保留最多 150 毫秒尾音。两种模式共用特征提取和训练清单格式，输出清单中的 `transcription_backend` 标明来源。

MOSS 使用独立 Python 环境，按 [作者安装说明](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize#environment-setup) 安装官方包和 CUDA PyTorch。工具链 JSON 增加 `"moss_python": "/path/to/moss-env/bin/python"`。MOSS 模式需要显式提供训练语言标签，日语使用 `ja`，同时使用模型自身的切分。长音频首次运行需要下载约 1.82 GB 权重，时间和显存取决于输入长度。

```bash
python -m workers.research_training --toolchain /path/to/toolchain.json prepare \
  --backend moss --moss-language ja --input movie.wav --output /path/to/moss-corpus

python -m workers.research_voice_corpus --corpus /path/to/moss-corpus \
  --output /path/to/moss-voices --group-by speaker \
  --min-clips 1 --min-seconds 0.1 --min-clip-seconds 0.1 --max-clip-seconds 54
```

`--group-by speaker` 直接沿用上游说话人编号，并按来源文件和重叠分离窗口区分编号作用域。默认 `--group-by embedding` 仍按 KyaraEmbed 聚类，`--distance` 可调整距离。单条语音组可导出，后续模型训练仍需处理其最小样本和参考音频长度要求。

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

如需扩大上游语料，可在 `prepare` 后增加 `--emilia-min-clip-seconds 2 --emilia-min-dnsmos 2.6`。原版两项门槛都是 3。最短时长同时用于识别前的切段和最终过滤，DNSMOS 门槛用于最终过滤。输出清单会记录覆盖参数。

要保留快速轮流说话的边界，可加 `--emilia-fine-turns --emilia-min-clip-seconds 0.8`。该模式在每个单人时间区间运行 Silero VAD，保留短停顿，不跨重叠区间拼接。双人重叠可以用 DialogueSidon 处理。配置只有一种语言时，不足 2 秒的段使用该语言做识别。

可用 `--emilia-max-turn-seconds 3 --emilia-min-silence-ms 80` 进一步提高切分力度。已有同一媒体时间轴的字幕时，传入 `--emilia-turn-boundaries cuts.json`，格式为 `{"cut_seconds": [1.2, 3.4]}`。这些切点会在 VAD 和识别前强制分隔原始输入。

细切模式会先读取视频内嵌的文本字幕轨，优先用非强制轨和默认轨。若没有可用的文本时间轴，且已配置 [VideoSubFinder CLI 6.10.5](https://github.com/eritpchy/videosubfinder-cli/releases/tag/v6.10.5)，就扫描画面底部 22% 的烧录字幕，只输出时间点。工具链 JSON 可加 `"videosubfinder_cli": "/path/to/VideoSubFinderCli.run"`，也可将 `VideoSubFinderCli.run` 或 `VideoSubFinderCli` 放入 PATH。未安装工具或没有检测到字幕时，细切继续使用音频。位图字幕轨仍需先转换为切点 JSON。

若音频和原视频是同一时间轴，但输入已抽成 WAV，可用 `--emilia-subtitle-source movie.mkv` 指定字幕来源。自动提取的时刻和来源记录在输出的 `subtitle-boundaries.json`。

细切模式默认在导出时预留最多 150 毫秒尾音，允许相邻文件轻微重叠。用 `--emilia-tail-ms` 调整。清单的 `speech_end` 保留识别时的终点，`end` 是带尾音的实际文件终点。

## 按音色导出训练清单

在装有 NumPy 和 scikit-learn 的特征环境中，用 Emilia 语料及其 KyaraEmbed 特征生成每组音色的 32 kHz 单声道 WAV 和 GPT-SoVITS `labels.list`：

```bash
python -m workers.research_voice_corpus --corpus /path/to/corpus --output /path/to/voice-training
```

默认按余弦距离 0.25 做 average linkage 聚类，保留至少 3 条、合计不少于 12 秒的组，单条范围为 2–18 秒。用 `--min-clips`、`--min-seconds`、`--min-clip-seconds`、`--max-clip-seconds` 调整门槛。符合条件的组都会导出。输出里的 `voices.json` 记录每组采用的原始语料行号及相似度，聚类编号是候选音色的标识。

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
