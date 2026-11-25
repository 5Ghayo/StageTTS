"""Synthesize, loudness-match, and assemble the classic 120 s seven-voice dub."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path

from pydub import AudioSegment

from backend import config
from backend.schemas import EmotionRef, ProjectScript
from backend.services.media_mixer import _build_vocal_track
from backend.services.speech_loudness import measure_training_audio
from backend.services.synthesis import synthesize_with_duration_check
from backend.services.task_runner import ensure_tts_stopped

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = Path(os.environ.get("STAGETTS_PIPELINE_DIR", REPO_ROOT / "workspace/classic120-pipeline"))
TRAINING = Path(os.environ.get("STAGETTS_TRAINING_DIR", REPO_ROOT / "workspace/classic120-7voices-v5"))
PROJECT_ID = "classic-120s-v5-shared-reference"
PROJECT = config.WORKSPACE_DIR / PROJECT_ID
FULL_VOCALS = PIPELINE / "workspace/liz-full-post-whisperx/vocals.wav"
BGM = Path(os.environ.get("STAGETTS_BGM_PATH", REPO_ROOT / "workspace/classic120-bgm/bgm.wav"))
SCENE_START = 3651.0
DURATION = 120.046


def run(*args: object) -> None:
    subprocess.run([str(arg) for arg in args], check=True)


def configure(script: ProjectScript) -> None:
    picks = json.loads((PROJECT / "reference-selection.json").read_text(encoding="utf-8"))
    voices = json.loads((TRAINING / "voices.json").read_text(encoding="utf-8"))["voices"]
    folder = PROJECT / "references"
    folder.mkdir(exist_ok=True)
    if set(picks) != {seg.speaker for seg in script.segments}:
        raise ValueError("参考音清单与该段的说话人不一致")
    for voice in voices:
        speaker, name = voice["speaker"], voice["voice"]
        pick = picks[speaker]
        wav = folder / f"{name}-{pick['segment_id']}.wav"
        run("ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{pick['start']:.6f}", "-i", FULL_VOCALS,
            "-t", f"{pick['end'] - pick['start']:.6f}", "-ac", "1", "-ar", "32000", "-c:a", "pcm_s16le", wav)
        gpt = TRAINING / "gpt-all-eligible" / f"{name}-e8.ckpt"
        sovits = list((TRAINING / "speaker-weights").glob(f"{name}_e6_s*_l32.pth"))
        if not gpt.is_file() or len(sovits) != 1:
            raise FileNotFoundError(f"Final weights are not ready for {name}")
        character = script.characters[speaker]
        character.gpt_weights = str(gpt)
        character.sovits_weights = str(sovits[0])
        character.emotion_refs = {"neutral": EmotionRef(wav=str(wav), text=pick["text"], lang=pick["lang"])}
    script.project_id = PROJECT_ID
    script.media.vocals_wav = f"/workspace/{PROJECT_ID}/source-vocals.wav"


def match_loudness(script: ProjectScript) -> list[dict]:
    normalized_dir = PROJECT / "tts-normalized"
    normalized_dir.mkdir(exist_ok=True)
    rows = []
    for seg in script.segments:
        if not seg.tts_wav or seg.speech_cue is None or seg.speech_cue.rms_dbfs is None:
            raise RuntimeError(f"Segment {seg.id} lacks TTS audio or source RMS")
        source = config.resolve_media_path(seg.tts_wav)
        before = measure_training_audio(source)
        target = float(seg.speech_cue.rms_dbfs)
        if before is None:
            raise RuntimeError(f"Segment {seg.id} TTS RMS is unavailable")
        gain = target - before
        audio = AudioSegment.from_wav(source).apply_gain(gain)
        peak_limited = False
        if math.isfinite(audio.max_dBFS) and audio.max_dBFS > -0.2:
            audio = audio.apply_gain(-0.2 - audio.max_dBFS)
            peak_limited = True
        output = normalized_dir / f"seg_{seg.id:04d}.wav"
        audio.export(output, format="wav", parameters=["-ac", "1", "-ar", "32000", "-c:a", "pcm_s16le"])
        after = measure_training_audio(output)
        seg.tts_wav = f"/workspace/{script.project_id}/tts-normalized/{output.name}"
        rows.append({
            "segment_id": seg.id, "speaker": seg.speaker,
            "start": seg.dub_start, "end": seg.dub_end,
            "source_rms_dbfs": target, "tts_rms_before_dbfs": before,
            "gain_db": round(gain, 3), "tts_rms_after_dbfs": after,
            "rms_error_db": None if after is None else round(after - target, 3),
            "peak_limited": peak_limited,
            "actual_duration": seg.actual_duration, "applied_speed_factor": seg.applied_speed_factor,
        })
    return rows


def main() -> None:
    PROJECT.mkdir(parents=True, exist_ok=True)
    source_vocals = PROJECT / "source-vocals.wav"
    run("ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{SCENE_START:.3f}", "-i", FULL_VOCALS,
        "-t", f"{DURATION:.3f}", "-ac", "1", "-ar", "32000", "-c:a", "pcm_s16le", source_vocals)

    script = ProjectScript.model_validate_json(
        (TRAINING / "pronunciation-check/script-v5dev.json").read_text(encoding="utf-8"),
    )
    configure(script)
    ids = [seg.id for seg in script.segments]
    try:
        succeeded, duration_review, errors = synthesize_with_duration_check(script, ids, auto_reflect=False)
    finally:
        ensure_tts_stopped()
    (PROJECT / "script-synth-raw.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    if set(succeeded) != set(ids):
        raise RuntimeError(f"TTS incomplete: {len(succeeded)}/{len(ids)}; errors={errors}")
    export(script, ids, succeeded, duration_review, errors)


def export(script: ProjectScript, ids: list[int], succeeded: list[int], duration_review: list, errors: list[str]) -> None:
    loudness = match_loudness(script)
    loudness_by_id = {row["segment_id"]: row for row in loudness}
    timeline_csv = PROJECT / "classic-120s-timeline.csv"
    with timeline_csv.open("w", encoding="utf-8-sig", newline="") as output:
        fields = ["segment_id", "speaker", "role", "start", "end", "dub_start", "dub_end",
                  "orig_text", "reference_zh", "target_text", "source_rms_dbfs", "tts_rms_after_dbfs",
                  "gain_db", "actual_duration", "applied_speed_factor"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for seg in script.segments:
            loud = loudness_by_id[seg.id]
            writer.writerow({
                "segment_id": seg.id, "speaker": seg.speaker,
                "role": script.characters[seg.speaker].display_name,
                "start": seg.start, "end": seg.end, "dub_start": seg.dub_start, "dub_end": seg.dub_end,
                "orig_text": seg.orig_text,
                "reference_zh": seg.speech_cue.reference_zh if seg.speech_cue else None,
                "target_text": seg.target_text, "source_rms_dbfs": loud["source_rms_dbfs"],
                "tts_rms_after_dbfs": loud["tts_rms_after_dbfs"], "gain_db": loud["gain_db"],
                "actual_duration": seg.actual_duration, "applied_speed_factor": seg.applied_speed_factor,
            })
    (PROJECT / "script-final.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    (PROJECT / "loudness-report.json").write_text(json.dumps(loudness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dubbed = _build_vocal_track(script, PROJECT)
    voice_mp3 = PROJECT / "classic-120s-7voices-vocals.mp3"
    run("ffmpeg", "-nostdin", "-v", "error", "-y", "-i", dubbed, "-c:a", "libmp3lame", "-b:a", "192k", voice_mp3)
    if not BGM.is_file():
        raise FileNotFoundError(BGM)
    mix_mp3 = PROJECT / "classic-120s-7voices-mix.mp3"
    run("ffmpeg", "-nostdin", "-v", "error", "-y", "-i", dubbed, "-i", BGM,
        "-filter_complex", "[1:a]volume=0.75[bgm];[0:a][bgm]amix=inputs=2:duration=first:normalize=0[aout]",
        "-map", "[aout]", "-c:a", "libmp3lame", "-b:a", "192k", mix_mp3)
    report = {
        "requested_ids": ids, "synthesized_ids": succeeded, "duration_review": [x.model_dump() for x in duration_review],
        "errors": errors, "loudness_segments": len(loudness),
        "voice_wav": str(dubbed), "voice_mp3": str(voice_mp3), "mix_mp3": str(mix_mp3),
        "timeline_csv": str(timeline_csv),
        "timing": "speech_cue.dub_start", "loudness": "speech_cue.rms_dbfs",
        "duration_seconds": DURATION,
    }
    (PROJECT / "synthesis-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
