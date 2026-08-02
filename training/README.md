# Itqan — Tajweed Model Training (Python / PyTorch)

This directory contains everything needed to re-train **TajweedNet**, the on-device
tajweed classifier that ships inside the Android app at
[`../app/src/main/assets/model_weights.json`](../app/src/main/assets/model_weights.json).

It is a self-contained copy of the training pipeline so that contributors can
reproduce or improve the model from this repo alone. **No heavy data or model
artifacts are committed** — those are regenerated locally from the raw audio corpus
(`dataset/`, `dataset_merged/`, `dataset_p13/` and `*.pt` are intentionally ignored).

## Layout

| Path | What it is |
|------|-----------|
| `src/train.py` | Data loading + training loop (Conv1d ×3 → BiGRU(64) → 8 per-rule heads). |
| `src/prepare_dataset.py` | Builds per-word `(audio, expected-rule, evidence)` clips from the Tadabur corpus. |
| `src/export_weights.py` | Flattens the trained `state_dict` to the JSON the app parses. |
| `src/tajweed_rules.py` | Python mirror of the Kotlin `TajweedRules.detectRulesForWord`. Must agree with the app. |
| `src/rule_acoustics.py` | Per-clip acoustic evidence features (nasality, vowel length, …) fed to the heads. |
| `src/tadabur_adapter.py`, `src/download_tadabur.py`, `src/extract_shard.py` | Fetch / decode the Tadabur alignment + audio corpus. |
| `src/prepare_and_train_merged.py`, `src/auto_resume_train.py`, `src/launch_merged_when_ready.py` | Orchestration helpers for the multi-phase dataset build. |
| `src/generate_fixture.py`, `src/gru_probe.py` | Debug / parity fixtures for the Kotlin ↔ PyTorch forward pass. |
| `TRAINING.md` | Full training-side guide: pipeline, phases, thresholds, known issues. |
| `DATASET.md` | Dataset construction, per-rule clip counts and distribution notes. |

The app's view of this pipeline is in [`../calibration.md`](../calibration.md); the
runtime architecture is in [`../tajweed.md`](../tajweed.md).

## Dependencies

Python 3.10+ with `torch`, `numpy`, `soundfile` (or `torchaudio`), `pandas`,
`pyarrow`, `requests`, `tqdm`. `onnx` / `onnx2tf` are only used by an optional
conversion path and are not required to export the JSON artifact.

## Reproduce

```bash
cd training

# 1) Fetch + prepare the labeled corpus (creates dataset*/ under here, ignored by git)
python src/prepare_dataset.py

# 2) Train TajweedNet (writes dataset/model_best.pt when done)
python src/train.py

# 3) Export the compact JSON the Android app loads at runtime
python src/export_weights.py          # -> dataset/model_weights.json

# 4) Ship it into the app
cp dataset/model_weights.json ../app/src/main/assets/model_weights.json

# 5) Rebuild the APK (see ../README.md)
```

After exporting, re-tune the per-head thresholds in
[`../app/src/main/java/com/example/ui/RecitationApp.kt`](../app/src/main/java/com/example/ui/RecitationApp.kt)
(`decisionThresholdForHead`) using the eval fold — see
[`../calibration.md`](../calibration.md) §5 and `TRAINING.md` §7.

## Parity with the Android port

The app re-implements `TajweedNet` in Kotlin and **asserts bit-for-bit (float32)
parity** against the PyTorch reference via `TajweedNetParityTest`. Keep
`src/train.py`'s model definition and the Kotlin `TajweedNet` in lock-step, and keep
`src/tajweed_rules.py` consistent with the Kotlin
`TajweedRules.kt` (both must produce the same label for the same word).
