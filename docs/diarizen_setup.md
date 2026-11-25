# DiariZen Large v2 说话人分离

应用固定使用 Banquet 四轨分离 → WhisperX 转写和对齐 → SaT 和 WhisperX VAD 复查切点 → DiariZen 声纹聚类 → 人工审校 / 翻译 / GPT-SoVITS。训练语料入口使用 Emilia-Pipe 的 Silero VAD 提供片段，再交给独立 DiariZen 聚类。

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

- DiariZen 按最终原声试听范围逐条提取 WeSpeaker 声纹（段首去空白，段尾静音最多 2 秒），再把有效声纹按原时间顺序交给一次 VBx 聚类。调用方必须提供片段，工作进程只加载声纹模型、VBx 和 PLDA 参数。默认参数为 AHC 距离阈值 `0.35`、`Fa=0.15`、`Fb=0.37`。
- 直接保留 DiariZen 的说话人分组，不再做额外音色聚类。
- `speaker_analysis.json` 保留原始活动时间段、来源音频哈希和原始到录音内角色编号的映射；编号带来源哈希，跨录音同号不自动认定为同角色。
- 转写区间必须由一个说话人至少覆盖 80%，同时不能存在其他说话人的重叠活动。重叠、多说话人或覆盖不足保留 UNKNOWN。
- WhisperX 提供词级时间。补切后落在新片段外的词时间会置空并标为待审。
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

Emilia 保留其 UVR、Silero VAD、Whisper 与 DNSMOS；包装器只替换旧的说话人模型及令牌门槛，训练切段只使用无重叠的说话人区间。旧版基线工人移到 `experiments/`，供离线对照使用。

## 音频来源与细切

### 聚类前的叠话检查

在独立环境安装 pyannote.audio 4.x，并将已下载的 Community-1 本地模型目录填入 `PYANNOTE_OVERLAP_MODEL`。将解释器设为 `PYANNOTE_OVERLAP_PYTHON`，并把 `PYANNOTE_OVERLAP_VERSION` 设为安装的精确版本。工作进程使用普通 diarization 输出，保留同一时刻多位说话人的活动区间。

已切好的片段若与叠话相交，剧本会记录 `overlap_turns` 中各位说话人的原片起止时间、`overlap_status=detected` 和 `training_excluded=true`。保留原声供核验，所有片段仍参与 DiariZen 聚类，叠话的聚类标签须审校。叠话未核验时，台词标为禁止训练。研究语料导出只发布通过重叠检查的原始音频，DialogueSidon 分离结果仅供独立分析。

聚类完成后按有效语音 RMS 筛查训练音频。低于 −45 dBFS 或无可用响度的台词标为禁止训练；保留聚类标签和试听。研究语料及 GPT-SoVITS 训练清单同样执行此阈值。

识别和分段以音频为依据。解析后按音频台词时间匹配可选的日文和中文字幕。手动上传的 SRT/VTT 优先于对应语言的内嵌文本字幕。未匹配到原声台词的字幕不进入翻译或配音。LLM 保存校对后的原文和最终译文，WhisperX 听写保存在原有字段。

WhisperX 3.4.5 首轮固定用 `vad_onset = 0.65`、`vad_offset = 0.65`，连续语音窗口约 8 秒，250 毫秒以上停顿保留为首轮片段边界。SaT 评分超过 0.25 的内部句界在前后各 1 秒内复查。复查使用同一份 WhisperX VAD 帧分数和 `0.80 / 0.80` 阈值，检测到至少 200 毫秒的停顿时切在停顿正中。SaT 不可用时会记录日志并保留首轮结果。

原声试听从 WhisperX VAD 检测到的语音起点开始，去掉段首留白。段尾先取相邻语音之间的停顿中点，再将保留的静音限制在最多 2 秒。最后一条有可用余量时保留最多 250 毫秒的源音频。每条 `speech_cue` 保存实际起音、有效语音 RMS dBFS 和配音最晚结束时间。配音可以借用最多 1 秒的后续停顿，翻译、合成检查和导出使用同一窗口。

这是语音片段细切，不保证每一段恰好对应完整语法句。无停顿的长句可能在 VAD 低分点分开；连读、抢话、ASR 漏字和错误的日语词级对齐仍需要审校。

### 本地 SaT

```sh
uv venv .venv-sat
uv pip install --python .venv-sat/bin/python 'wtpsplit[onnx-cpu]==2.2.2' 'transformers<5'
.venv-sat/bin/hf download segment-any-text/sat-3l-sm model_optimized.onnx config.json --revision 137da05 --local-dir .cache/sat/model
.venv-sat/bin/hf download FacebookAI/xlm-roberta-base tokenizer.json tokenizer_config.json config.json --revision e73636d --local-dir .cache/sat/tokenizer
```

Windows 使用 `.venv-sat/Scripts/` 下的解释器和 `hf`。可通过 `SAT_PYTHON` 和 `SAT_ASSET_DIR` 指定其他安装目录。模型下载后在本地离线处理原文。
