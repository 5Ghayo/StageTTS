# GPT-SoVITS v5 重新训练

StageTTS 新项目默认选择 `gpt_sovits_v5`，服务配置为 `backend/tts_v5.yaml`。官方实际版本名为 **`v5dev`**；`v5turbo` 使用另一套底模，本次不混用。`api_v2.py` 是服务接口文件名，v5 仍然使用它。

## 准备训练机

使用支持 v5 的官方 GPT-SoVITS 源码，保留原 v2 安装、数据和权重备份。在训练机配置匹配显卡的 PyTorch / CUDA，安装该版本官方依赖；不要修改 WhisperX、DiariZen 或 StageTTS 后端环境。

底模及公共模型放在 GPT-SoVITS 根目录下：

```text
GPT_SoVITS/pretrained_models/
  s1v3.ckpt
  gsv-v5-pretrained/s2Gv5dev.pth
  chinese-roberta-wwm-ext-large/
  chinese-hubert-base/
```

另需准备该版本官方要求的 vocoder 等配套文件，保留官方发布目录结构。权重来源：[官方模型目录](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/gsv-v5-pretrained)。版本映射见 [官方 config.py](https://github.com/bilibili/GPT-SoVITS/blob/main/config.py)。

## 重新训练

1. 沿用已校对的单角色音频和 `labels.list`，为每个角色使用新的实验名，例如 `角色名_v5dev`。日语原声的标签语种仍写 `ja`，文本必须是实际日语逐字稿；生成中文时再传入中文译文。
2. 打开官方 WebUI，选择 **v5dev**，重新执行训练数据格式化与特征提取。不要直接复用 v2 的语义特征或中间缓存。
3. GPT 训练底模选择 `s1v3.ckpt`。SoVITS 底模选择 `gsv-v5-pretrained/s2Gv5dev.pth`。
4. v5 的 SoVITS 微调走官方 **LoRA** 训练入口 `GPT_SoVITS/s2_train_v3_lora.py`，不是旧的 `s2_train.py`。使用 WebUI 生成匹配配置，避免复制旧 v2 配置。
5. 8GB 显卡可从 SoVITS batch size **1**、**2 epochs** 开始，开启梯度检查点；LoRA rank 先用官方默认值。这是起始设置，需按实际显存和试听调整。不要直接沿用旧的高 epoch 训练策略。
6. 新权重分别输出到 `GPT_weights_v5dev/` 和 `SoVITS_weights_v5dev/`。LoRA 推理仍需要对应的 v5dev 原始底模，不要删除它。

上述入口与设置依据 [官方 WebUI 实现](https://github.com/bilibili/GPT-SoVITS/blob/main/webui.py)。本仓库没有在后台启动训练。

## 接回 StageTTS

在训练机/推理机的 `backend/.env` 配置：

```dotenv
GPT_SOVITS_ROOT=/absolute/path/GPT-SoVITS
GPT_SOVITS_PYTHON=/absolute/path/tts-env/bin/python
GPT_SOVITS_CONFIG=backend/tts_v5.yaml
TTS_DEFAULT_GPT_WEIGHTS=
TTS_DEFAULT_SOVITS_WEIGHTS=
```

Windows 的 Python 路径可使用整合包的 `runtime/python.exe`。默认权重留空会使用 v5 配置里的底模；无需填写旧 v2 默认权重。

重新启动服务后，在项目的「配音模型」选择 **GPT-SoVITS v5**，在角色设置中绑定该角色新训练的 GPT 和 SoVITS LoRA 权重，再主动重配。选择模型不会自动清除旧角色权重或重做已有音频，因此必须重新绑定权重后再试听。旧版训练/试听脚本保留为历史实验，不用于本次 v5 训练。

应用解析和 Emilia / DiariZen 语料工具均可提供训练素材；训练前仍需人工确认角色、文本和参考音。

## 当前验证范围

项目默认配置、模型选项和类型契约已切换。当前电脑没有配置 `GPT_SOVITS_ROOT`，因此尚未验证实际 v5 服务启动、GPU 推理和 LoRA 训练；需在装有 GPT-SoVITS 及模型的训练机进行。
