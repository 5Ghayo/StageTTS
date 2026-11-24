"""Capability declarations for TTS engines used by StageTTS.

Only explicitly supported controls are advertised. A missing or unknown engine
never inherits emotion/prosody controls from another implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.schemas import CharacterConfig, EmotionRef, EmotionType

SUPPORTED_EMOTIONS = tuple(sorted(EMOTIONS := {"neutral", "angry", "happy", "sad", "whisper"}))
SUPPORTED_LANGS = ("zh", "en", "ja", "ko", "yue")


@dataclass(frozen=True)
class EngineCapabilities:
    name: str
    version_hint: str
    supports_reference_audio: bool
    supports_reference_text: bool
    supports_reference_lang: bool
    supported_reference_langs: tuple[str, ...]
    supported_emotions: tuple[str, ...]
    direct_emotion_control: bool
    direct_prosody_control: bool
    status: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version_hint": self.version_hint,
            "supports_reference_audio": self.supports_reference_audio,
            "supports_reference_text": self.supports_reference_text,
            "supports_reference_lang": self.supports_reference_lang,
            "supported_reference_langs": list(self.supported_reference_langs),
            "supported_emotions": list(self.supported_emotions),
            "direct_emotion_control": self.direct_emotion_control,
            "direct_prosody_control": self.direct_prosody_control,
            "status": self.status,
            "notes": list(self.notes),
        }


GPT_SOVITS_V2 = EngineCapabilities(
    name="gpt_sovits_v2",
    version_hint="api_v2/ref_audio_path+prompt_text+prompt_lang+text_lang+speed_factor",
    supports_reference_audio=True,
    supports_reference_text=True,
    supports_reference_lang=True,
    supported_reference_langs=SUPPORTED_LANGS,
    supported_emotions=SUPPORTED_EMOTIONS,
    direct_emotion_control=False,
    direct_prosody_control=False,
    status="wired",
    notes=(
        "emotion is a reference-routing key only; no numeric emotion/prosody is sent to TTS",
        "speed_factor is a global tempo scalar, not an emotion or F0 control",
    ),
)

COSYVOICE3 = EngineCapabilities(
    name="cosyvoice3_0_5b",
    version_hint="Fun-CosyVoice3-0.5B",
    supports_reference_audio=True,
    supports_reference_text=True,
    supports_reference_lang=True,
    supported_reference_langs=("zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru"),
    supported_emotions=(),
    direct_emotion_control=False,
    direct_prosody_control=False,
    status="planned_not_wired",
    notes=("documented instruction/pinyin controls require version-specific validation; not called by StageTTS",
           "Cantonese/yue is described among dialects, separate from the nine language list",
           "control flags describe controls applied by StageTTS, not unconnected model documentation"),
)

INDEXTTS = EngineCapabilities(
    name="indextts2",
    version_hint="IndexTTS2",
    supports_reference_audio=True,
    supports_reference_text=True,
    supports_reference_lang=True,
    supported_reference_langs=("zh", "en"),
    supported_emotions=(),
    direct_emotion_control=False,
    direct_prosody_control=False,
    status="planned_not_wired",
    notes=("documented separate emotion reference/vector controls are not wired in StageTTS",
           "IndexTTS2 precise duration control is not enabled in that version",
           "IndexTTS2.5 is a separate future candidate; its duration_factor is not IndexTTS2 behavior"),
)

ENGINES: dict[str, EngineCapabilities] = {
    item.name: item for item in (GPT_SOVITS_V2, COSYVOICE3, INDEXTTS)
}


def get_capabilities(name: str) -> EngineCapabilities | None:
    return ENGINES.get(name)


def all_capabilities() -> list[dict]:
    return [engine.to_dict() for engine in ENGINES.values()]


def routed_emotions(engine: EngineCapabilities) -> list[EmotionType]:
    """Only GPT-SoVITS v2 uses the legacy emotion5 reference routing today."""
    if engine.name != GPT_SOVITS_V2.name or engine.status != "wired":
        return []
    return list(engine.supported_emotions)  # type: ignore[return-value]


def resolve_reference_route(character: CharacterConfig, emotion: EmotionType) -> tuple[EmotionRef | None, str]:
    """Mirror legacy GPT-SoVITS reference routing, without claiming direct control.

    Existence/audio-transcript quality checks stay in tts_client. This adapter
    never converts rich source emotion scores into supported-looking API fields.
    """
    if emotion in character.emotion_refs:
        return character.emotion_refs[emotion], 'exact_emotion_reference'
    if 'neutral' in character.emotion_refs:
        return character.emotion_refs['neutral'], 'neutral_reference_fallback'
    if character.emotion_refs:
        return next(iter(character.emotion_refs.values())), 'available_reference_fallback'
    return None, 'automatic_source_reference_required'


def build_gpt_sovits_request(*, text: str, text_lang: str, reference: EmotionRef,
                            ref_audio_path: str) -> dict:
    """Only API fields already supported by the current GPT-SoVITS client.

    Measured source emotion/prosody stay annotations. There is intentionally no
    emotion vector, F0, valence/arousal or invented speaker control in this map.
    """
    if not text.strip() or not reference.text.strip() or not reference.lang.strip():
        raise ValueError('text, actual reference transcript and actual language are required')
    return {'text': text, 'text_lang': text_lang or 'zh', 'ref_audio_path': ref_audio_path,
            'prompt_text': reference.text, 'prompt_lang': reference.lang,
            'speed_factor': 1.0, 'media_type': 'wav'}
