"""Translate the current 28 classic-scene units with DeepSeek and subtitle references."""

from __future__ import annotations

import json
from pathlib import Path

from backend.schemas import ProjectScript
from backend.services.translation import create_openai_client, translate_script

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workspace/liz-full-post-whisperx/script.json"
OUTPUT = ROOT / "workspace/classic-120s-7voices"
SCENE_START = 3651.0
SCENE_IDS = set(range(473, 501))
SCENE_DURATION = 120.046
GLOSSARY = {
    "リズ": "利兹", "りズ": "利兹", "青い鳥": "青鸟",
    "みぞれ": "霙", "ミゾレ": "霙", "ミソレ": "霙",
    "鎧塚": "铠冢", "音大": "音乐大学", "フルート": "长笛", "オーボエ": "双簧管",
}


def shift(value: float | None) -> float | None:
    return None if value is None else round(value - SCENE_START, 3)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    full = ProjectScript.model_validate_json(SOURCE.read_text(encoding="utf-8"))
    speakers = {f"DZ_6e5330e3aa4b_{number}" for number in (2, 3, 5, 6, 7, 9, 13)}
    segments = [seg.model_copy(deep=True) for seg in full.segments if seg.id in SCENE_IDS]
    if len(segments) != 28 or {seg.speaker for seg in segments} != speakers:
        raise RuntimeError("Classic scene no longer matches the expected 28 units and seven speakers")

    for seg in segments:
        seg.start, seg.end = shift(seg.start), shift(seg.end)
        seg.source_audio_start, seg.source_audio_end = shift(seg.source_audio_start), shift(seg.source_audio_end)
        if seg.speech_cue:
            seg.speech_cue.speech_start = shift(seg.speech_cue.speech_start)
            seg.speech_cue.speech_end = shift(seg.speech_cue.speech_end)
            seg.speech_cue.latest_end = shift(seg.speech_cue.latest_end)
        for word in seg.words:
            word.start, word.end = shift(word.start), shift(word.end)
        seg.start, seg.end = max(0.0, seg.start), max(0.001, seg.end)
        seg.invalidate_audio()

    scene = ProjectScript(
        project_id="classic-120s-7voices",
        mode=full.mode,
        tts_engine=full.tts_engine,
        media=full.media.model_copy(update={
            "source_video": None,
            "vocals_wav": "/workspace/classic-120s-7voices/source-vocals.wav",
            "bgm_wav": None,
            "cinematic_stems": None,
            "output_video": None,
            "duration": SCENE_DURATION,
        }),
        source_lang=full.source_lang,
        glossary={**full.glossary, **GLOSSARY},
        characters={key: full.characters[key].model_copy(deep=True) for key in sorted(speakers)},
        evidence=None,
        segments=segments,
    )
    (OUTPUT / "translation-input.json").write_text(scene.model_dump_json(indent=2), encoding="utf-8")

    logs: list[str] = []
    errors: list[str] = []
    client = create_openai_client()
    translated = translate_script(scene, client, log=lambda text: logs.append(text), errors=errors)
    (OUTPUT / "script-translated.json").write_text(scene.model_dump_json(indent=2), encoding="utf-8")
    (OUTPUT / "translation.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
    report = {
        "provider": "DeepSeek",
        "model": getattr(client, "_stagetts_model", None),
        "source_units": len(segments),
        "translated_ids": translated,
        "missing_ids": sorted(SCENE_IDS - set(translated)),
        "errors": errors,
        "uses_reference_ja": sum(bool(seg.speech_cue and seg.speech_cue.reference_ja) for seg in segments),
        "uses_reference_zh": sum(bool(seg.speech_cue and seg.speech_cue.reference_zh) for seg in segments),
    }
    (OUTPUT / "translation-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["missing_ids"] or errors:
        raise RuntimeError("Translation did not complete cleanly")


if __name__ == "__main__":
    main()
