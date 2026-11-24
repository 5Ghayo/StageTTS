# Legacy diarization experiment archive (not the MOSS primary parser)

> Historical record only. This document contains discontinued gated-pyannote experiments and setup snapshots. They are **not** current primary-parser instructions. MOSS is the active default; see [MOSS setup and verification](moss_setup.md). Do not install WhisperX/pyannote to get the default parser working.

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

The migration overlay is now ready at `F:/Download/StageTTS/env_analysis_moss/python.exe`, sharing only the new MOSS site-packages and no old parse path. Its own 25 FunASR extras plus core occupy 130,717,647 bytes (~124.7MiB). Recorded versions: Python 3.12.10, torch/torchaudio 2.8.0+cu128, transformers 5.6.0, huggingface_hub 1.33, NumPy 2.2.6, librosa 0.11, FunASR 1.4.28. AutoModel/Emotion2vec registration and pip check passed, actual CPU pYIN sine/noise/silence parity passed, and one real CUDA synthetic tone classification returned nine finite raw scores with unvalidated review retained. The process exited and GPU returned to zero; this establishes migrated runtime compatibility only. The integration author has now pointed local `ENV_ANALYSIS_PYTHON` to this migrated path without restarting services. Old overlays and parse environment are not deleted. The first overlay consumes about 31.3MiB; the migrated overlay's size is recorded above. These are not true venvs: the embedded runtime lacks venv, so documented `._pth` application isolation and pip `--target --no-deps` were used. Later shared-base package changes can affect it. Environment manifest, package versions, and constraints are retained beside the interpreter. Launch with `-I -B` to suppress user-site imports and avoid writing bytecode into shared packages. Python 3.12 no longer bundles distutils; the environment author enabled the standard site initialization and copied the shipped setuptools `distutils-precedence.pth` into the new local overlay, without altering the base environment. Emotion2vec class registration then succeeded; the earlier missing-distutils attempt was `not_run` before weights were loaded.

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

## MOSS selected as the new main parser; installation/inference pending

The user first selected deployment A for [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize): an independent F-drive environment, official precision, batch one, initial 10–30 second samples to measure output and peak memory. The user subsequently explicitly selected MOSS as the new main ASR/diarization dependency and authorized the large CUDA-runtime download. This is the target configuration, not a completed production readiness claim; environment, Banquet compatibility, official-model inference, and parse-route integration must pass before switching the local production interpreter. The [official model](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) is pinned to revision `10304d926740cf5a5cee05a268086e3265197910`; its single safetensors file is 1,817,113,576 bytes, about 1.69GiB on disk. License is Apache-2.0. This is a 0.9B joint ASR/speaker/timestamp model, not an 8B model or audio source separator. The card lists 50+ languages and explicitly includes English and Japanese in its challenge language table, despite metadata listing only en/zh.

Authorized acquisition target is now complete: `F:/Download/StageTTS/diarization_models/moss-transcribe-diarize/`, 20 files totaling 1,833,165,385 bytes. The 1,817,113,576-byte checkpoint's actual SHA-256 matches upstream: `9a0ceb4ab7330357db3ff583dba8d83625d5b733b00e1d55d6970e11b07026c4`. The diarization author completed four sequential official BF16/SDPA speech runs on authorized Japanese/English dialogue copies:

| Sample | CUDA allocated peak | CUDA reserved peak | Windows peak RSS | Model load + generation |
| --- | ---: | ---: | ---: | ---: |
| Japanese 10s | 1.742GiB | about 1.793GiB | unavailable in first measurement | 22.99s |
| English 10s | 1.742GiB | about 1.799GiB | 2.371GiB | 24.17s |
| Japanese 30s | 1.807GiB | 1.893GiB | 2.371GiB | 38.54s |
| English 30s | 1.811GiB | 1.912GiB | 2.372GiB | 72.22s |

All four reached EOS without the generation cap or OOM and exited separately; the GPU returned to zero residency. Hash/preflight time is excluded from the durations above. Subsequent complete single-scope samples also ran successfully:

| Sample | CUDA allocated peak | CUDA reserved peak | Windows peak RSS | Model load + generation |
| --- | ---: | ---: | ---: | ---: |
| Japanese 60.000375s | 1.942GiB | 1.963GiB | 2.372GiB | 50.66s |
| English 60.000s | 1.942GiB | 1.965GiB | 2.372GiB | 94.27s |

The 60-second runs reached EOS without the 4096-token cap or OOM, returned 234/507 tokens and five/two-speaker versus seventeen/one-speaker segments respectively, and exited with GPU zero. These are anonymous speaker counts, not accuracy measurements. The physical input clips end mid-sentence: EOS is not proof of semantic completeness. The Japanese WAV exceeds the 60-second nominal budget by 0.000375s, retained under a separately explicit <=0.05s input-tail policy with review. This is distinct from correcting model-generated output timestamps. Production failfast uses extracted WAV frames/rate, not container metadata; >60.05s rejects before expensive separation. This sample coverage supports the current 60-second budget only, not arbitrary full-video durations. Japanese 30s generated an endpoint 30.02s for 30s source audio; its complete raw stream is preserved, and CPU-only reparse applies an explicit <=0.05s derived clamp with original `generated_end=30.02` and review reason retained. Larger errors remain invalid, not silently cropped. Japanese 30s yielded five segments/two anonymous speakers, English 30s twelve segments/one anonymous speaker. No gold labels or listening review were performed, so these counts establish neither ASR nor diarization accuracy. The measured 8GB feasibility applies to these at-most-sixty-second samples only.

Banquet also completed an independent new-runtime CUDA compatibility test after MOSS exited: its real 238MB checkpoint loaded, and one second of synthetic 220Hz mono 48k input produced four finite `[1,48000]` stems. CUDA peaks were 219,019,776 allocated bytes and 325,058,560 reserved bytes, with load+inference 21.06s; the process exited. This validates loading/shapes/finiteness, not separation quality.

The diarization author's manifest records every file/source/hash, and `custom_code_review.json` binds the three completely reviewed official Python files to their exact hashes and approved scope.

A separate real ten-second scratch main-pipeline run completed FFmpeg → actual Banquet GPU separation → actual MOSS GPU transcription → skipped LLM → typed script/source-evidence save. It produced one Japanese source utterance at 8.36–9.76s, retained raw generation/speaker scope/source links, and kept empty word arrays and unknown source-token times. All five output WAVs were 480,000 frames at 48kHz stereo. Independent QA found the initial evidence builder had marked joint-generated activity as reviewed; the builder was corrected and a pure-CPU derived snapshot was created without overwriting the original. The corrected snapshot preserves identical source/raw/stable IDs while all joint activity/turn review flags remain true. This correction and before/after evidence are explicit; no second GPU run was needed. This verifies a short parse/evidence chain, not translated dubbing or a completed video: LLM translation and TTS were not run.

Runtime artifacts record the actual checks; long-video policy, cross-window identity reconciliation, and semantic/voice/dubbing quality evaluation remain separate acceptance gates. The current route rejects inputs beyond its explicit 60-second budget plus narrowly recorded tail tolerance rather than automatically splitting or truncating them.

The [official dependency specification](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/pyproject.toml) requires transformers >=5.6,<6; the runtime route uses torch/torchaudio >=2.8, librosa >=0.11 and av >=14. The repository supports Python >=3.10 and documents 3.12. Existing parse torch 2.7.1 and transformers 4.51 must remain intact. The environment author preflighted `F:/Download/StageTTS/env_moss/python.exe`: independent embedded Python 3.12.10 with its own site-packages, currently only pip 26.1.2 and metadata/logs, totaling 34,118,550 bytes. It is not a venv and does not share parse site-packages. Proposed pins are torch/torchaudio 2.8.0+cu128, transformers 5.6.0, librosa 0.11. The official Windows cp312 CUDA torch wheel alone is 3,461,384,651 bytes; torchaudio is 4,673,203 bytes. Estimated total environment/wheel/model footprint is 12–15GB, not yet measured as an installed environment. The material CUDA download and all required runtime dependencies are now explicitly authorized and downloading asynchronously into F-drive storage with pinned official hashes. The independent runtime is now installed: Python 3.12.10, torch/torchaudio 2.8.0+cu128, transformers 5.6.0. Eighty-three retained wheels totaling 3,733,593,257 bytes were SHA-verified; the large official torch wheel hash matched after standard same-source Range resume. `pip check`, scientific imports, PyTorch Lightning 2.6.5/torchmetrics 1.5.2 imports, CUDA 12.8 on RTX 5060 Laptop native `sm_120`, and a tiny finite BF16 SDPA allocation passed. The validation subprocess exited and released GPU memory. This runtime readiness was subsequently complemented by the separate MOSS speech and Banquet synthetic checks above; full production parsing and long-video quality remain unverified. See `runtime_validation.json`, `environment_manifest.json`, `package_versions.txt`, and `setup/wheel_manifest.json` in the environment. Actual measured environment storage is 12,248,359,805 bytes (11.41GiB), including installed packages 8,458,901,473 bytes (7.88GiB) and the retained wheel set 3,733,593,257 bytes (3.48GiB); the model snapshot is separate at 1,833,165,385 bytes. F-drive free space was 16,847,953,920 bytes (15.69GiB) after installation. These measured figures supersede the earlier 12–15GB estimate. Download took approximately 19 minutes; full download/install was 19:17:44–19:39:04, approximately 21m20s including the initial failed temporary pip transfer and two retained Range continuations. Runtime validation took 26.25s. `storage_manifest.json` records these measurements. Do not reuse the small emotion overlay to overwrite incompatible shared packages. Banquet separation remains part of the target pipeline, so its Lightning/bandit dependencies and actual short compatibility test must also pass with torch 2.8. The analysis migration overlay has now passed separate compatibility checks against the new MOSS runtime; it no longer needs old parse packages. Keep both old environments for rollback until production configuration and broader workloads are verified.

Official Transformers inference uses BF16 plus SDPA. Long-sequence eager attention has quadratic memory cost; there is no verified official RTX 5060 8GB/16GB minimum or tested quantization guarantee. The 128k context/90-minute model claim is not a claim that 90-minute inference fits this laptop. Configured 28 layers × 8 KV heads × head dimension 128 at BF16 implies approximately 114,688 bytes/token for a standard KV cache alone; this is an architectural estimate, excluding weights, encoder, activations, generated text, and framework overhead. Short samples and measured CUDA peak/device usage must precede larger scenes. Third-party GGUF/C++ quantization is a separate option rather than an official model capability accepted by default.

The pinned official custom Python must be reviewed and loaded from the approved local snapshot only. `trust_remote_code=True` is necessary in the documented route; the user authorized only the fixed official local code, not future remote changes. Exact runtime/package/model revisions and source identity must accompany each result.

Output resembles `[start][Sxx]text[end]`: preserve the raw tagged generation, generated segment ranges, and anonymous speaker assignments with provenance/review. These are not word timestamps, calibrated speaker probabilities, an independent overlap detector, or globally verified speaker identities. Store MOSS as its own immutable ASR snapshot; do not paste its speaker IDs onto a different WhisperX transcript. Token times remain unknown until separately forced-aligned; never interpolate characters into fake word alignment. S01 from independent windows is not automatically the same character. Malformed markers/ranges remain reviewable rather than silently dropped. CER/cpCER challenge results are not diarization DER.

Target production configuration is `PARSE_BACKEND=moss` with the validated new `ENV_PARSE_PYTHON`. These keys are now actually configured locally: `ENV_PARSE_PYTHON=F:/Download/StageTTS/env_moss/python.exe`, fixed model/revision, torch 2.8.0+cu128, transformers 5.6.0, approved pinned code, `MOSS_MAX_AUDIO_SECONDS=60`, and `MOSS_MAX_NEW_TOKENS=4096`. The integration author changed only authorized keys and did not restart services; the next backend restart reads them. Legacy is retained as an explicit rollback option, never a silent fallback when MOSS configuration is missing or fails. community-1 acquisition/environment work is paused as a backup option; the requirements below are retained as historical, verified setup information, not the selected main parser.

The [official pyannote 4.0.1 requirements](https://github.com/pyannote/pyannote-audio/blob/4.0.1/pyproject.toml) were read by the diarization author: Python >=3.10, torch/torchaudio >=2.8, torchcodec >=0.6, pyannote-core >=6 and metrics >=4. Current main has different minimums; do not install unspecified latest. A separate Python 3.12 + pyannote.audio 4.0.1 candidate environment must be tested for matching CUDA wheels, Windows torchcodec, and FFmpeg. Keep existing pyannote 3.1 default and dependencies intact.

The [official community-1 instructions](https://github.com/pyannote/pyannote-audio#community-1-open-source-speaker-diarization) require a Hugging Face account, the user accepting the model's access conditions, and an appropriate token for acquisition. Access conditions were not accepted on the user's behalf; the gated model card was not bypassed. A later authenticated HEAD request to the official `resolve/main/config.yaml` returned HTTP 403 with a configured token. This establishes that access is currently denied; it does not prove whether the account has previously accepted terms or whether the token lacks fine-grained gated-model permission. Stop that resource and have the user check the provider page/account/token permissions before retrying. Download only after the user explicitly selects the environment/model and completes required provider access. Do not put access tokens in tracked files, worker output, or experiment reports.

The isolated diarization worker accepts `--request FILE --output FILE`. Request fields: `protocol_version="1.0"`, `backend="legacy-3.1"|"community-1"`, `source_id`, `audio_source="raw"|"dialogue"`, absolute local `audio_path` and `model_path` (existing directory or `config.yaml`, no HF ID), exact `expected_package_version`, and `device="cpu"|"cuda"`. The worker resolves directories to their existing `config.yaml` before passing a local file to pyannote for 3.x compatibility. Output includes `pipeline_config_sha256` with an explicit config-only hash scope; this is not a full-model weight hash. Output uses `status=completed|not_run|error`, overlap-preserving `diarization_spans`, separate derived `exclusive_spans`, null scores where the provider supplied none, and actual elapsed/optional CUDA peak measurements. Missing local child checkpoints report failure rather than downloading.

The diarization comparison manifest has `workers=[{backend,python,model_path,package_version,device}]` and `samples=[{id,raw_audio?,dialogue_audio?,gold_rttm?,uem?}]`. It runs serial local comparisons. Without gold, DER stays null; without optional pyannote.metrics, metric evaluation explicitly remains unavailable. The benchmark uses hardware such as an H100 and does not establish 5060 timing. community-1 is not universally more accurate than 3.1 on every listed dataset. Preserve overlapping diarization activity separately from the derived exclusive convenience track. DER requires gold RTTM and an explicitly documented collar/overlap policy; no gold means no DER claim.
