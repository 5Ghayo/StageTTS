# Legacy diarization experiment archive (optional model analysis)

> Historical record only. This document contains discontinued gated-pyannote experiments and setup snapshots. They are **not** current primary-parser instructions. Current parsing uses Banquet, WhisperX and an isolated DiariZen worker; see [DiariZen setup](diarizen_setup.md).

# Optional model setup and offline analysis

Status: the M1/M2/M3 adapters and JSON runners are implemented for review. On 2026-11-23 the user selected emotion2vec+ base for course research under the documented FunASR model license and authorized reversible local model acquisition. The required base files have been downloaded and the 1,118,245,678-byte checkpoint SHA-256 matches the provider's LFS hash. The environment author completed one real CUDA compatibility smoke on a two-second 220Hz PCM tone: the checkpoint loaded with all keys matched and returned nine labels/scores with uncalibrated/unvalidated review status preserved. The process exited and GPU usage returned to zero. A tone has no meaningful speech emotion label; this verifies execution compatibility, not recognition accuracy, natural-speech quality, or full-project performance. Existing parse/TTS environments are not upgraded.

## emotion2vec+ selection

The [official model repository](https://github.com/ddlBoJack/emotion2vec) recommends the plus family for nine-class speech emotion recognition. The [seed](https://huggingface.co/emotion2vec/emotion2vec_plus_seed), [base](https://huggingface.co/emotion2vec/emotion2vec_plus_base), and [large](https://huggingface.co/emotion2vec/emotion2vec_plus_large) model cards were actually read on 2026-11-23.

| Model | Actual documented distinction | `model.pt` bytes from public HF tree metadata | Runtime status |
| --- | --- | ---: | --- |
| plus seed | Fine-tuned on 201 hours of academic emotion data | 1,118,245,358 | Not tested locally |
| plus base | About 90M parameters; 4,788 hours of filtered pseudo-labeled data | 1,118,245,678 | Recommended first comparison candidate, not configured |
| plus large | About 300M parameters; 42,526 hours of filtered pseudo-labeled data | 1,945,790,254 | Optional quality comparison, not configured |

Checkpoint disk sizes are not peak VRAM. Content may include training state. No RTX 5060 8GB or 16GB RAM fit, throughput, cross-language reliability, or recognition accuracy has been measured here. The same model is not guaranteed to perform equally on acted, whispered, mixed, separated, Japanese, Chinese, or overlapping speech.

The model cards explicitly list `license: other`, `license_name: model-license` and link FunASR. The [FunASR model license](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE) actually read was version 1.1. It requires source/author attribution and retention of model names, describes use as "reference and learning purposes only", and contains conduct, termination, and revision provisions. This is distinct from MIT source code. Do not label these weights MIT or assume unrestricted commercial deployment. The user must review the selected weight license before obtaining or using it; the runner requires an explicit approval field but does not accept a license on anyone's behalf.

The classifier returns nine categories, including an unknown/other category according to its tokenizer. These are model labels, not an exhaustive ontology of emotion, delivery, vocal events, or continuous valence/arousal/dominance. No label is automatically converted into a GPT-SoVITS emotion API argument.

## Candidate isolated emotion environment

The [official inference instructions](https://github.com/ddlBoJack/emotion2vec#inference-with-checkpoints) recommend FunASR, not the older fairseq feature-training stack. [PyPI metadata](https://pypi.org/project/funasr/1.4.28/) read on 2026-11-23 lists FunASR 1.4.28, Python >=3.7. Its dependencies include NumPy, SciPy, librosa, soundfile, PyYAML, OmegaConf, hydra-core, ModelScope, huggingface_hub, transformers, and other audio/text packages. Python metadata alone does not demonstrate compatibility of the complete Windows/CUDA stack.

The first isolated analysis environment was prepared at `F:/Download/StageTTS/env_analysis/python.exe`: Python 3.12.10 app overlay with local FunASR 1.4.28 and explicitly referenced existing torch/torchaudio 2.7.1+cu128, NumPy 2.0.2, transformers 4.51.3. The original parse FunASR remains 1.3.22. This first working overlay is retained for rollback.


This supersedes the initial candidate recommendation of a separate Python 3.10/3.11 environment. PyTorch and torchaudio must remain matching builds, with the chosen CUDA/PyTorch wheel supporting the RTX 5060 architecture. Do not install unrestricted latest packages into the current parse environment.

After authorization, obtain the exact approved model revision by the provider's documented process. The approved base acquisition is pinned to revision `b318240bfe67db81a8c572ecb37ce9c3759b81c9`; the official LFS checkpoint SHA-256 is `60710b5aae1dbe69bdac8920028fb05882d4314fd09031922b4b61ee9e7aadbd`. Its dedicated destination is `F:/Download/StageTTS/emotion_models/emotion2vec_plus_base`, with a local `download_manifest.json` recording actual file hashes and the retrieved license content. GitHub's API license-commit lookup was rate limited; the retrieved license content is hashed rather than falsely claiming a verified license commit.

Record its revision/commit, package freeze, license text, source URL, and file hashes. The complete local directory needs `config.yaml`, `configuration.json`, `model.pt`, and `tokens.txt`. The runner only accepts an absolute existing local directory, never a hub ID as a path. Model files are trusted artifacts: review provenance before loading a PyTorch checkpoint.

## Emotion worker JSON contract

Run the worker in the chosen environment as a subprocess. Input/output can be files via `--input` and `--output`, or stdin/stdout. Third-party stdout is redirected to stderr to preserve JSON framing. The worker imports ML packages only after validation and uses a single model, batch one, sequential spans. Exiting the subprocess releases its model before another stage starts.

```json
{
  "model": {
    "id": "emotion2vec/emotion2vec_plus_base",
    "revision": "EXACT_APPROVED_REVISION",
    "local_path": "F:/models/approved-emotion2vec-plus-base",
    "funasr_version": "1.4.28",
    "license_approved": false,
    "device": "cpu"
  },
  "source": {
    "id": "SOURCE_CONTENT_HASH",
    "kind": "dialogue",
    "audio_path": "F:/experiments/source-dialogue-16k.wav"
  },
  "spans": [{
    "id": "source-turn-id",
    "start": 0.0,
    "end": 2.0,
    "speaker": "SPEAKER_00",
    "source_token_ids": ["source-token-id"],
    "source_turn_ids": ["source-turn-id"],
    "overlap": false,
    "speaker_needs_review": false,
    "alignment_needs_review": false
  }],
  "max_span_seconds": 30
}
```

`license_approved: false` is intentionally not runnable. Configuration must reflect the user's actual approval. Revision `main`, `master`, or `latest` is rejected. The installed FunASR version must match exactly. Optional `weight_sha256` is checked; the actual checkpoint hash is always returned after loading preflight. No weights are downloaded; remote model code is disabled, version checking disabled, local model paths preflighted, and documented HF/transformers offline settings enabled.

Input is local PCM16 mono WAV at 16kHz with explicit immutable source spans. Convert an experiment copy explicitly before requesting analysis; the worker never silently resamples, truncates long spans, or writes project media. A span outside the available audio is an error. Longer-than-configured spans are `not_run`; unknown speakers, overlap, or speaker/alignment review flags are blocked before model loading. Silence is unclassified. Original mixed audio may contain music/effects, so it retains a review reason. Source speaker IDs and token/turn links are carried through.

Output retains exact model `labels` and `scores`. The verified FunASR implementation applies softmax; `score_kind=model_softmax_uncalibrated` records this without claiming calibrated probability of a true human emotion. `selected_label` is only the maximum score. Classification always retains `needs_review=true` and `reliability=unvalidated`; no empirically justified acceptance threshold exists yet. Missing dependency, missing model, unapproved license, or incompatible version yields `not_run`, with no invented neutral score. Per-span exceptions yield errors without erasing other successful spans.

The backend integration supplies an explicit user action, serial stage execution, and validated source evidence. Legacy five-way reference routing remains independent. Model output must not overwrite source transcript, speaker turns, or translation units.

The environment author also exercised the real CPU pYIN path on two-second synthetic signals: 220Hz sine yielded median 220.08097Hz with range zero; seeded broadband noise and silence yielded null with `no_reliable_pyin_voiced_frames`. Cold subprocess times were approximately 7.218s, 3.203s, and 3.578s respectively. This establishes installed dependency and simple voiced-gating behavior; real-speech F0 accuracy and long-span throughput remain unmeasured.

## Original/dialogue comparison and evaluation

The comparison runner accepts the same model/spans with `sources` instead of `source`: one original mix and one dialogue file, with distinct IDs and shared source timeline. It launches one worker subprocess per source in sequence. Results and raw scores remain separate; agreement across sources is a diagnostic, not ground truth or reliability calibration. It writes only the requested experiment output, never a project script.

Optional gold JSON is `{"labels": {"source-span-id": "exact model label"}}`. Without gold, accuracy and macro F1 are null and status states `not_evaluated_no_gold`. With matching human labels, metrics cover only successfully classified matched spans and explicitly list omitted gold spans. Label mapping, actor/speaker stratification, language, overlap, confusion matrices, and representative listening review must be designed before interpreting scores as production acceptance.
