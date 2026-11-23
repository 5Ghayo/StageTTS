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
import shutil
import subprocess
import sys
import tempfile
import threading
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
# 本 worker 仅在受控子进程中加载官方 WhisperX/pyannote 检查点，显式开启兼容模式。
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from backend import config  # noqa: E402
from backend.schemas import (  # noqa: E402
    AvatarConfig,
    CharacterConfig,
    CinematicStems,
    MediaInfo,
    ProjectScript,
    Segment,
)
from backend.services.translation import create_openai_client, translate_script  # noqa: E402
from workers.aligned_segments import normalize_segments  # noqa: E402

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
    """用 FFmpeg 将任意音视频统一抽取为 48kHz stereo WAV（与 Banquet 采样率一致）。"""
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
        "48000",
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
        cmd = [
            sys.executable,
            str(ROOT_DIR / "workers/banquet_worker.py"),
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(extracted_wav),
            "--output-dir",
            str(output_dir),
            "--device",
            target_device,
        ]
        output_lines: List[str] = []
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        def _drain_banquet(stream: Any = proc.stdout, lines: List[str] = output_lines) -> None:
            if stream is None:
                return
            for raw_line in stream:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("[banquet] "):
                    _log("Stage1", line.removeprefix("[banquet] ").strip())
                else:
                    lines.append(line)

        reader = threading.Thread(target=_drain_banquet, daemon=True)
        reader.start()
        try:
            returncode = proc.wait(timeout=900)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.wait()
            reader.join(timeout=3)
            raise RuntimeError(f"Banquet ({target_device}) 分离超时（>900s）。") from exc
        reader.join(timeout=5)
        if returncode == 0:
            return
        detail = "\n".join(output_lines).strip()
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

    detected_lang = str(result.get("language") or "")
    if detected_lang and result.get("segments"):
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

    if skip_diarization or not config.is_hf_token_configured():
        reason = "已通过参数跳过" if skip_diarization else "未配置有效 HF_TOKEN"
        _log("Stage3", f"{reason}，全部片段默认归属 SPEAKER_00")
        return normalize_segments(segments)

    _enable_pyannote_torch_load_compat()
    try:
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers
    except Exception as exc:
        _log("Stage3", f"无法加载说话人聚类，待人工校对：{str(exc)[:220]}")
        return normalize_segments(segments)

    attempts = [device, "cpu"] if device == "cuda" else ["cpu"]
    for diar_dev in attempts:
        diarize_model = None
        try:
            _log("Stage3", f"启动 pyannote/speaker-diarization-3.1 (device={diar_dev})...")
            diarize_model = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                use_auth_token=config.HF_TOKEN,
                device=diar_dev,
            )
            diarize_df = diarize_model(audio)
            assigned = assign_word_speakers(diarize_df, aligned_result, fill_nearest=False)
            _log("Stage3", f"说话人聚类完成 ({diar_dev})")
            return normalize_segments(assigned.get("segments") or segments, diarized=True)
        except Exception as exc:
            _log("Stage3", f"说话人聚类 ({diar_dev}) 失败：{str(exc)[:220]}")
        finally:
            if diarize_model is not None:
                del diarize_model
            _release_cuda_memory()

    _log("Stage3", "pyannote 聚类不可用，回退单说话人 SPEAKER_00")
    return normalize_segments(segments)


def _project_media_url(path: Path, project_dir: Path, project_id: str) -> str:
    return f"/workspace/{project_id}/{path.relative_to(project_dir).as_posix()}"


def _build_and_save_script(
    args: argparse.Namespace,
    duration: float,
    vocals_wav: Path,
    bgm_wav: Optional[Path],
    raw_segments: List[Dict[str, Any]],
    detected_lang: str,
    cinematic_stems: Optional[CinematicStems] = None,
) -> Path:
    """组装并通过 Pydantic v2 强校验后落盘 workspace/{project_id}/script.json。"""
    project_id = args.project_id
    script_out = args.output_dir / "script.json"
    existing = ProjectScript.model_validate_json(script_out.read_text(encoding="utf-8")) if script_out.is_file() else None
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

    discovered_speakers = list(dict.fromkeys(seg["speaker"] for seg in raw_segments))
    characters = dict(existing.characters) if existing else {}
    for idx, spk in enumerate(discovered_speakers):
        if spk in characters:
            continue
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

    segments = [
        Segment(**seg, target_text=seg["orig_text"], target_lang=args.target_lang, action="idle")
        for seg in raw_segments
    ]

    max_seg_end = max((s.end for s in segments), default=0.0)
    final_duration = round(max(duration, max_seg_end), 3)

    script = ProjectScript(
        project_id=project_id,
        mode=args.mode,
        source_lang=(existing.source_lang if existing else None) or detected_lang or None,
        glossary=existing.glossary if existing else {},
        media=MediaInfo(
            source_video=source_video_url,
            vocals_wav=_project_media_url(vocals_wav, args.output_dir, project_id),
            bgm_wav=_project_media_url(bgm_wav, args.output_dir, project_id) if bgm_wav else None,
            cinematic_stems=cinematic_stems,
            output_video=None,
            duration=final_duration,
            bgm_volume=0.75,
        ),
        characters=characters,
        segments=segments,
    )

    if args.skip_llm or not config.is_llm_configured():
        _log("Stage4", "跳过翻译或未配置 LLM，保留识别原文")
    elif segments:
        translated = translate_script(script, create_openai_client(), lambda message: _log("Stage4", message))
        _log("Stage4", f"已翻译 {len(translated)}/{len(segments)} 条台词")

    with tempfile.TemporaryDirectory(dir=args.output_dir) as temp:
        draft = Path(temp) / script_out.name
        draft.write_text(script.model_dump_json(indent=2), encoding="utf-8")
        draft.replace(script_out)
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


def _cleanup_stale_cinematic_dirs(output_dir: Path, keep: Path) -> None:
    keep_resolved = keep.resolve()
    for candidate in output_dir.iterdir():
        if candidate.is_dir() and candidate.name.startswith("cinematic-") and candidate.resolve() != keep_resolved:
            shutil.rmtree(candidate, ignore_errors=True)


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

    # Publish new audio separately so a failed reparse preserves the previous script's tracks.
    cinematic_dir = Path(tempfile.mkdtemp(prefix="cinematic-", dir=args.output_dir))
    try:
        with tempfile.TemporaryDirectory(prefix="stagetts_work_") as work_tmp:
            extracted_wav = Path(work_tmp) / "extracted_audio.wav"
            duration = _extract_wav_and_probe(args.input, extracted_wav)
            _stage_banquet(extracted_wav, cinematic_dir, device)
            vocals_wav, bgm_wav, cinematic_stems = _finish_four_stems(cinematic_dir, args.project_id)

        audio, aligned_result, detected_lang = _stage_whisperx_asr_align(
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
        review_count = sum(seg["alignment_needs_review"] or seg["speaker_needs_review"] for seg in raw_segments)
        if review_count:
            _log("Stage3", f"{review_count} 条片段的原文范围或说话人待人工校对")
        script_out = _build_and_save_script(
            args=args,
            duration=duration,
            vocals_wav=vocals_wav,
            bgm_wav=bgm_wav,
            raw_segments=raw_segments,
            detected_lang=detected_lang,
            cinematic_stems=cinematic_stems,
        )
    except BaseException:
        shutil.rmtree(cinematic_dir, ignore_errors=True)
        raise

    _cleanup_stale_cinematic_dirs(args.output_dir, keep=cinematic_dir)
    return script_out


def main() -> None:
    args = parse_args()
    try:
        run_pipeline(args)
    finally:
        _release_cuda_memory()


if __name__ == "__main__":
    main()
