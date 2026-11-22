"""独立感知解析 Worker（运行在 env_parse：PyTorch cu128 + Banquet + WhisperX + pyannote）。

由 backend/services/task_runner.py 通过 subprocess.run 启动，进程退出时由 OS 100% 释放显存。
严禁在 env_backend 中直接导入本文件或依赖。

四阶段流水线：
  Stage 0: FFmpeg 抽取标准 WAV 并探测媒体精确时长
  Stage 1: Banquet 分离对白、歌声、伴奏和音效
  Stage 2: WhisperX 语音识别 + 词级时间戳对齐（含 CUDA OOM 自动降级 int8 / cpu）
  Stage 3: pyannote 3.x 说话人聚类（缺失 Token 或失败时安全回退单说话人 SPEAKER_00）
  Stage 4: LLM 导演口语化翻译、情绪/动作/原声叹词标注，校验并落盘 script.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 阶段 0 前置：在导入任何 torch/whisperx 前加载 backend/.env，
# 确保 HF_HOME / HUGGINGFACE_HUB_CACHE / TORCH_HOME 严格落盘到配置的 F 盘缓存目录。
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv

    _env_file = ROOT_DIR / "backend" / ".env"
    if _env_file.is_file():
        load_dotenv(_env_file, override=False)
except ImportError:
    pass

# PyTorch 2.6+ 默认 weights_only=True 会拦截 pyannote/lightning 的 omegaconf/Task 权重反序列化。
# WhisperX / pyannote 在独立解析进程内加载可信检查点时使用兼容模式。
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from backend.agent.prompts import DIRECTOR_TRANSLATE_PROMPT  # noqa: E402
from backend.schemas import (  # noqa: E402
    AvatarConfig,
    CharacterConfig,
    CinematicStems,
    MediaInfo,
    ProjectScript,
    Segment,
)

VALID_EMOTIONS = {"neutral", "angry", "happy", "sad", "whisper"}
SPEAKER_PALETTE = [
    "#3b82f6",
    "#10b981",
    "#f59e0b",
    "#ec4899",
    "#8b5cf6",
    "#06b6d4",
    "#ef4444",
    "#84cc16",
]
SPEAKER_LABELS = ["角色 A", "角色 B", "角色 C", "角色 D", "角色 E", "角色 F"]
AUDIO_ONLY_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus"}


def _log(stage: str, message: str) -> None:
    print(f"[parse_worker][{stage}] {message}", flush=True)


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "out of memory" in text
        or "cudaerrormemoryallocation" in text
        or "cublas_status_alloc_failed" in text
        or "no kernel image" in text
    )


def _release_cuda_memory() -> None:
    """阶段间主动释放 Python 引用与 CUDA 缓存，防止 8GB 显卡阶段叠加 OOM。"""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except ImportError:
        pass


def _resolve_device(requested: str) -> str:
    if requested in {"cuda", "cpu"}:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


_PYANNOTE_COMPAT_PATCHED = False


def _enable_pyannote_torch_load_compat() -> None:
    """兼容 PyTorch 2.6+ 与 pyannote 3.x / lightning / speechbrain 在 Python 3.12 下的运行时问题。

    1. 部分 lightning 版本会显式传入 weights_only=True，绕过 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD；
       此处在独立解析 worker 内对 torch.load 做一次性降级包装。
    2. speechbrain 1.0 的 LazyModule.__getattr__ 在 Python 3.12 被 inspect.stack()/getmodule()
       访问 __file__ 时会触发 inspect.getframeinfo 无限递归；对内置元属性直接抛出 AttributeError。
    """
    global _PYANNOTE_COMPAT_PATCHED
    if _PYANNOTE_COMPAT_PATCHED:
        return
    try:
        import torch

        orig_load = torch.load

        def _compat_torch_load(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("weights_only", None) is not False:
                kwargs["weights_only"] = False
            return orig_load(*args, **kwargs)

        torch.load = _compat_torch_load  # type: ignore[assignment]
    except Exception:
        pass

    try:
        import speechbrain.utils.importutils as sb_iu

        orig_getattr = sb_iu.LazyModule.__getattr__

        def _safe_lazy_getattr(self: Any, attr: str) -> Any:
            if attr in ("__file__", "__path__", "__spec__", "__wrapped__"):
                raise AttributeError(attr)
            return orig_getattr(self, attr)

        sb_iu.LazyModule.__getattr__ = _safe_lazy_getattr  # type: ignore[method-assign]
    except Exception:
        pass

    _PYANNOTE_COMPAT_PATCHED = True


def _probe_duration(media_path: Path) -> float:
    if not media_path.is_file() or shutil.which("ffprobe") is None:
        return 0.0
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(media_path),
    ]
    try:
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        raw = json.loads(res.stdout or "{}").get("format", {}).get("duration")
        return max(0.0, round(float(raw), 3)) if raw else 0.0
    except Exception:
        return 0.0


def _extract_wav_and_probe(input_path: Path, work_wav: Path) -> float:
    """用 FFmpeg 将输入音视频统一抽取为 44.1 kHz 立体声 WAV。"""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未在系统 PATH 中找到 ffmpeg，无法抽取音频。")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(work_wav),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    if proc.returncode != 0 or not work_wav.is_file():
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        raise RuntimeError(f"FFmpeg 抽取音频失败 (code={proc.returncode}): {tail}")

    duration = _probe_duration(input_path) or _probe_duration(work_wav)
    _log("Stage0", f"音频抽取完成 ({work_wav.name})，媒体时长 {duration:.2f}s")
    return duration


def _stage_banquet(extracted_wav: Path, output_dir: Path, device: str) -> None:
    paths = {}
    for name in ("BANQUET_ROOT", "BANQUET_CHECKPOINT"):
        value = os.getenv(name, "").strip()
        if not value:
            raise RuntimeError(f"Banquet 四轨需要在 backend/.env 配置 {name}。")
        paths[name] = (ROOT_DIR / Path(value).expanduser()).resolve()
    source_root = paths["BANQUET_ROOT"]
    checkpoint = paths["BANQUET_CHECKPOINT"]
    if not (source_root / "src/models/bandit/bandit.py").is_file() or not checkpoint.is_file():
        raise RuntimeError("Banquet 源码或权重缺失，请检查 BANQUET_ROOT 和 BANQUET_CHECKPOINT。")

    env = os.environ.copy()
    env.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    devices = ("cuda", "cpu") if device == "cuda" else ("cpu",)
    for target_device in devices:
        _log("Stage1", f"Banquet ({target_device}) 直接分离对白、歌声、音乐与音效")
        proc = subprocess.run(
            [sys.executable, str(ROOT_DIR / "workers/banquet_worker.py"),
             "--source-root", str(source_root), "--checkpoint", str(checkpoint),
             "--input", str(extracted_wav), "--output-dir", str(output_dir),
             "--device", target_device],
            env=env, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=900,
        )
        if proc.returncode == 0:
            return
        detail = proc.stderr.strip() or proc.stdout.strip()
        if target_device == "cpu" or not _is_cuda_oom(RuntimeError(detail)):
            raise RuntimeError(f"Banquet ({target_device}) 分离失败：{detail[-600:]}")
        _log("Stage1", "Banquet CUDA 显存不足，改用 CPU 重试")


def _finish_four_stems(output_dir: Path, project_id: str) -> Tuple[Path, Path, CinematicStems]:
    tracks = {stem: output_dir / f"{stem}.wav" for stem in ("dialogue", "singing", "music", "effects")}
    if not all(path.is_file() and path.stat().st_size > 44 for path in tracks.values()):
        raise RuntimeError("影视四轨分离未生成完整的对白、歌声、音乐、音效 WAV 文件。")

    bgm_path = output_dir / "bgm.wav"
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y",
         "-i", str(tracks["effects"]), "-i", str(tracks["music"]), "-i", str(tracks["singing"]),
         "-filter_complex", "amix=inputs=3:duration=first:normalize=0",
         "-c:a", "pcm_f32le", str(bgm_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"影视背景轨合成失败：{proc.stderr[-600:]}")

    stems = CinematicStems(**{
        stem: _project_media_url(path, output_dir.parent, project_id) for stem, path in tracks.items()
    })
    _log("Stage1", "四轨分离完成，交接对白轨至 WhisperX")
    return tracks["dialogue"], bgm_path, stems


def _stage_whisperx_asr_align(
    vocals_wav: Path,
    compute_type: str,
    whisper_model: str,
    device: str,
) -> Tuple[Any, Dict[str, Any], str]:
    """阶段 2：WhisperX 语音识别与词级时间戳对齐（含 CUDA OOM 自动降级 int8 / cpu）。"""
    _enable_pyannote_torch_load_compat()
    import whisperx

    audio = whisperx.load_audio(str(vocals_wav))

    candidates: List[Tuple[str, str, int]] = []
    if device == "cuda":
        if compute_type == "float16":
            candidates.append(("cuda", "float16", 8))
        candidates.append(("cuda", "int8", 4))
    candidates.append(("cpu", "int8", 4))

    result: Optional[Dict[str, Any]] = None
    used_device = "cpu"
    last_exc: Optional[BaseException] = None

    for dev, ctype, batch_size in candidates:
        model = None
        try:
            _log("Stage2", f"加载 WhisperX 模型 {whisper_model} (device={dev}, compute_type={ctype})...")
            model = whisperx.load_model(
                whisper_model,
                device=dev,
                compute_type=ctype,
            )
            result = model.transcribe(audio, batch_size=batch_size)
            used_device = dev
            _log(
                "Stage2",
                f"ASR 完成：识别语言={result.get('language', 'unknown')}，初稿片段数={len(result.get('segments', []))}",
            )
            break
        except Exception as exc:
            last_exc = exc
            if dev == "cuda" and (_is_cuda_oom(exc) or "float16" in str(exc).lower()):
                _log("Stage2", f"CUDA ({ctype}) 失败，准备降级重试：{str(exc)[:180]}")
            else:
                _log("Stage2", f"WhisperX ({dev}/{ctype}) 异常，尝试下一方案：{str(exc)[:180]}")
        finally:
            if model is not None:
                del model
            _release_cuda_memory()

    if result is None:
        raise RuntimeError(f"WhisperX 语音识别全部方案失败: {last_exc}")

    detected_lang = str(result.get("language") or "en")
    if result.get("segments"):
        align_attempts = [used_device, "cpu"] if used_device == "cuda" else ["cpu"]
        for align_dev in align_attempts:
            align_model = None
            try:
                _log("Stage2", f"加载词级对齐模型 (lang={detected_lang}, device={align_dev})...")
                align_model, metadata = whisperx.load_align_model(
                    language_code=detected_lang,
                    device=align_dev,
                )
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio,
                    align_dev,
                    return_char_alignments=False,
                )
                _log("Stage2", f"词级时间戳对齐完成 ({align_dev})")
                break
            except Exception as exc:
                _log("Stage2", f"词级对齐 ({align_dev}) 跳过/降级：{str(exc)[:180]}")
            finally:
                if align_model is not None:
                    del align_model
                _release_cuda_memory()

    return audio, result, detected_lang


def _is_valid_hf_token(token: str) -> bool:
    t = (token or "").strip()
    return bool(t) and t != "hf_your_token_here" and not t.startswith("hf_your_")


def _stage_diarize(
    audio: Any,
    aligned_result: Dict[str, Any],
    device: str,
    skip_diarization: bool,
) -> List[Dict[str, Any]]:
    """阶段 3：pyannote 说话人聚类与词级说话人分配。"""
    segments = aligned_result.get("segments") or []
    if not segments:
        return []

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if skip_diarization or not _is_valid_hf_token(hf_token):
        reason = "已通过参数跳过" if skip_diarization else "未配置有效 HF_TOKEN"
        _log("Stage3", f"{reason}，全部片段默认归属 SPEAKER_00")
        return _normalize_raw_segments(segments)

    _enable_pyannote_torch_load_compat()
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    attempts = [device, "cpu"] if device == "cuda" else ["cpu"]
    for diar_dev in attempts:
        diarize_model = None
        try:
            _log("Stage3", f"启动 pyannote/speaker-diarization-3.1 (device={diar_dev})...")
            diarize_model = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
                device=diar_dev,
            )
            diarize_df = diarize_model(audio)
            assigned = assign_word_speakers(diarize_df, aligned_result, fill_nearest=True)
            _log("Stage3", f"说话人聚类完成 ({diar_dev})")
            return _normalize_raw_segments(assigned.get("segments") or segments)
        except Exception as exc:
            _log("Stage3", f"说话人聚类 ({diar_dev}) 失败：{str(exc)[:220]}")
        finally:
            if diarize_model is not None:
                del diarize_model
            _release_cuda_memory()

    _log("Stage3", "pyannote 聚类不可用，回退单说话人 SPEAKER_00")
    return _normalize_raw_segments(segments)


def _normalize_raw_segments(raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for seg in raw_segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start = max(0.0, round(float(seg.get("start", 0.0)), 3))
        end = round(float(seg.get("end", start + 0.5)), 3)
        if end <= start:
            end = round(start + 0.25, 3)
        speaker = str(seg.get("speaker") or "SPEAKER_00").strip() or "SPEAKER_00"
        cleaned.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "orig_text": text,
            }
        )
    cleaned.sort(key=lambda item: (item["start"], item["end"]))
    for idx, item in enumerate(cleaned):
        item["id"] = idx
    return cleaned


def _parse_json_array(raw_text: str) -> List[Dict[str, Any]]:
    cleaned = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    left = cleaned.find("[")
    right = cleaned.rfind("]")
    if left == -1 or right == -1 or right <= left:
        return []
    parsed = json.loads(cleaned[left : right + 1])
    return [x for x in parsed if isinstance(x, dict)] if isinstance(parsed, list) else []


def _is_llm_available() -> bool:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    if not api_key:
        return False
    placeholders = {"your_llm_api_key_here", "sk-xxx", "changeme", "local-gateway-no-key"}
    if api_key in placeholders:
        return base_url.startswith("http://127.0.0.1:") or base_url.startswith("http://localhost:")
    return not api_key.startswith("your_")


def _stage_director_annotate(
    raw_segments: List[Dict[str, Any]],
    target_lang: str,
    skip_llm: bool,
) -> List[Dict[str, Any]]:
    """阶段 4：调用大模型进行口语化翻译、情绪/动作/原声叹词标注。"""
    if not raw_segments:
        return []

    defaults: Dict[int, Dict[str, Any]] = {
        seg["id"]: {
            "target_text": seg["orig_text"],
            "emotion": "neutral",
            "action": "idle",
            "keep_original": False,
        }
        for seg in raw_segments
    }

    if skip_llm or not _is_llm_available():
        reason = "已通过参数跳过" if skip_llm else "未配置有效 LLM 网关"
        _log("Stage4", f"{reason}，保留识别原文并填充默认情绪标签")
        return [{**seg, **defaults[seg["id"]]} for seg in raw_segments]

    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY", "local-gateway-no-key").strip(),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip(),
        timeout=float(os.getenv("LLM_TIMEOUT_SEC", "60") or "60"),
    )
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

    batch_size = 25
    for start_idx in range(0, len(raw_segments), batch_size):
        chunk = raw_segments[start_idx : start_idx + batch_size]
        prompt_payload = [
            {
                "id": seg["id"],
                "speaker": seg["speaker"],
                "slot_sec": round(seg["end"] - seg["start"], 2),
                "orig_text": seg["orig_text"],
            }
            for seg in chunk
        ]
        user_prompt = (
            f"目标配音语言：{target_lang}。\n"
            "请为以下每条片段输出严格 JSON 数组，每项字段为："
            '{"id": int, "target_text": str, "emotion": "neutral|angry|happy|sad|whisper", '
            '"action": str, "keep_original": bool}。\n'
            "仅输出 JSON 数组，不要附加解释：\n"
            + json.dumps(prompt_payload, ensure_ascii=False, indent=2)
        )
        try:
            _log(
                "Stage4",
                f"调用导演 LLM ({model_name}) 标注片段 #{chunk[0]['id']}..#{chunk[-1]['id']}...",
            )
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": DIRECTOR_TRANSLATE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            content = resp.choices[0].message.content or ""
            entries = _parse_json_array(content)
            for entry in entries:
                seg_id = entry.get("id")
                if not isinstance(seg_id, int) or seg_id not in defaults:
                    continue
                target_text = str(entry.get("target_text") or "").strip()
                emotion = str(entry.get("emotion") or "neutral").strip().lower()
                if emotion not in VALID_EMOTIONS:
                    emotion = "neutral"
                action = str(entry.get("action") or "idle").strip() or "idle"
                keep_original = bool(entry.get("keep_original", False))
                defaults[seg_id] = {
                    "target_text": target_text or defaults[seg_id]["target_text"],
                    "emotion": emotion,
                    "action": action,
                    "keep_original": keep_original,
                }
        except Exception as exc:
            _log("Stage4", f"LLM 标注批次失败，回退原文兜底：{str(exc)[:200]}")

    return [{**seg, **defaults[seg["id"]]} for seg in raw_segments]


def _project_media_url(path: Path, project_dir: Path, project_id: str) -> str:
    return f"/workspace/{project_id}/{path.relative_to(project_dir).as_posix()}"


def _build_and_save_script(
    args: argparse.Namespace,
    duration: float,
    vocals_wav: Path,
    bgm_wav: Path,
    annotated_segments: List[Dict[str, Any]],
    cinematic_stems: CinematicStems,
) -> Path:
    """组装并通过 Pydantic v2 强校验后落盘 workspace/{project_id}/script.json。"""
    project_id = args.project_id
    input_ext = args.input.suffix.lower()
    is_audio_only = input_ext in AUDIO_ONLY_EXTS

    source_video_url: Optional[str] = None
    if not is_audio_only:
        # 若输入文件位于 output_dir 内，直接映射标准 /workspace/{id}/{filename}
        if args.input.resolve().parent == args.output_dir.resolve():
            source_video_url = f"/workspace/{project_id}/{args.input.name}"
        else:
            copied_input = args.output_dir / f"input{input_ext or '.mp4'}"
            if not copied_input.exists():
                shutil.copyfile(args.input, copied_input)
            source_video_url = f"/workspace/{project_id}/{copied_input.name}"

    discovered_speakers: List[str] = []
    for seg in annotated_segments:
        if seg["speaker"] not in discovered_speakers:
            discovered_speakers.append(seg["speaker"])
    if not discovered_speakers:
        discovered_speakers = ["SPEAKER_00"]

    characters: Dict[str, CharacterConfig] = {}
    for idx, spk in enumerate(discovered_speakers):
        label = SPEAKER_LABELS[idx] if idx < len(SPEAKER_LABELS) else f"角色 {idx + 1}"
        color = SPEAKER_PALETTE[idx % len(SPEAKER_PALETTE)]
        avatar_slot = "host_a" if idx % 2 == 0 else "host_b"
        characters[spk] = CharacterConfig(
            display_name=label,
            color=color,
            gpt_weights=None,
            sovits_weights=None,
            emotion_refs={},
            avatar=AvatarConfig(
                type="live2d",
                model_url=f"/assets/models/{avatar_slot}/model3.json",
            ),
        )

    segments: List[Segment] = []
    for seg in annotated_segments:
        keep_orig = bool(seg["keep_original"])
        segments.append(
            Segment(
                id=int(seg["id"]),
                start=float(seg["start"]),
                end=float(seg["end"]),
                speaker=str(seg["speaker"]),
                orig_text=str(seg["orig_text"]),
                target_text=str(seg["target_text"]),
                target_lang=args.target_lang,
                emotion=seg["emotion"],  # type: ignore[arg-type]
                action=str(seg["action"]),
                keep_original=keep_orig,
                speed_factor=1.0,
                tts_wav=None,
                actual_duration=round(seg["end"] - seg["start"], 3) if keep_orig else None,
                status="ready" if keep_orig else "pending",
            )
        )

    max_seg_end = max((s.end for s in segments), default=0.0)
    final_duration = round(max(duration, max_seg_end), 3)

    script = ProjectScript(
        project_id=project_id,
        mode=args.mode,
        media=MediaInfo(
            source_video=source_video_url,
            vocals_wav=_project_media_url(vocals_wav, args.output_dir, project_id),
            bgm_wav=_project_media_url(bgm_wav, args.output_dir, project_id),
            cinematic_stems=cinematic_stems,
            output_video=None,
            duration=final_duration,
            bgm_volume=0.75,
        ),
        characters=characters,
        segments=segments,
    )

    script_out = args.output_dir / "script.json"
    with tempfile.TemporaryDirectory(prefix=".script-", dir=args.output_dir) as tmp_name:
        next_script = Path(tmp_name) / "script.json"
        next_script.write_text(script.model_dump_json(indent=2), encoding="utf-8")
        next_script.replace(script_out)
    _log(
        "Done",
        f"script.json 生成并校验通过：{len(segments)} 条片段，{len(characters)} 个角色 -> {script_out}",
    )
    return script_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StageTTS 上游感知解析流水线 (env_parse)")
    parser.add_argument("--project-id", required=True, help="项目唯一 ID")
    parser.add_argument("--input", required=True, type=Path, help="源视频或源音频路径")
    parser.add_argument("--output-dir", required=True, type=Path, help="项目输出目录 workspace/{project_id}")
    parser.add_argument("--target-lang", default="zh", help="目标配音语种（默认 zh）")
    parser.add_argument(
        "--whisper-model",
        default=os.getenv("WHISPER_MODEL", "large-v3"),
        help="WhisperX 模型名称（默认读取 WHISPER_MODEL 或 large-v3；轻量调试可用 base/small/medium）",
    )
    parser.add_argument(
        "--compute-type",
        default=os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
        choices=["float16", "int8"],
        help="WhisperX 推理精度（RTX 5060 显存吃紧时可传 int8 压至 ~3.4GB）",
    )
    parser.add_argument(
        "--mode",
        default="video_dubbing",
        choices=["video_dubbing", "virtual_stage"],
        help="项目初始化模式",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="计算设备（默认 auto 自动探测 CUDA 并支持 OOM 降级）",
    )
    parser.add_argument("--skip-diarization", action="store_true", help="跳过 pyannote 说话人聚类")
    parser.add_argument("--skip-llm", action="store_true", help="跳过 LLM 翻译与情绪标注")
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> Path:
    if not args.input.is_file():
        raise FileNotFoundError(f"输入媒体文件不存在: {args.input}")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    _log(
        "Init",
        f"project={args.project_id}, device={device}, whisper={args.whisper_model}({args.compute_type}), "
        f"HF_HOME={os.getenv('HF_HOME', '(default)')}, TORCH_HOME={os.getenv('TORCH_HOME', '(default)')}",
    )

    # 每次分离发布独立的素材目录，解析失败时保留旧剧本指向的音轨。
    cinematic_dir = Path(tempfile.mkdtemp(prefix="cinematic-", dir=args.output_dir))
    try:
        with tempfile.TemporaryDirectory(prefix="stagetts_work_") as work_tmp:
            extracted_wav = Path(work_tmp) / "extracted_audio.wav"
            duration = _extract_wav_and_probe(args.input, extracted_wav)
            _stage_banquet(extracted_wav, cinematic_dir, device)
            vocals_wav, bgm_wav, cinematic_stems = _finish_four_stems(cinematic_dir, args.project_id)
        return _parse_dialogue(args, device, duration, vocals_wav, bgm_wav, cinematic_stems)
    except BaseException:
        shutil.rmtree(cinematic_dir, ignore_errors=True)
        raise


def _parse_dialogue(
    args: argparse.Namespace,
    device: str,
    duration: float,
    vocals_wav: Path,
    bgm_wav: Path,
    cinematic_stems: CinematicStems,
) -> Path:
    audio, aligned_result, _detected_lang = _stage_whisperx_asr_align(
        vocals_wav=vocals_wav,
        compute_type=args.compute_type,
        whisper_model=args.whisper_model,
        device=device,
    )
    raw_segments = _stage_diarize(
        audio=audio,
        aligned_result=aligned_result,
        device=device,
        skip_diarization=args.skip_diarization,
    )
    annotated_segments = _stage_director_annotate(
        raw_segments=raw_segments,
        target_lang=args.target_lang,
        skip_llm=args.skip_llm,
    )
    return _build_and_save_script(
        args=args,
        duration=duration,
        vocals_wav=vocals_wav,
        bgm_wav=bgm_wav,
        annotated_segments=annotated_segments,
        cinematic_stems=cinematic_stems,
    )


def main() -> None:
    args = parse_args()
    try:
        run_pipeline(args)
    finally:
        _release_cuda_memory()


if __name__ == "__main__":
    main()
