# TajweedNet Calibration & Training

This document explains how the in-app model **actually** gets its weights and thresholds.
(The older DTW-based `tajweed_thresholds.json` / `calibration.json` / `reference_features.json`
pipeline described in earlier revisions has been retired; only the TajweedNet flow is live.)

The single artifact the app loads at runtime is **`assets/model_weights.json`**,
a JSON blob produced by `training/src/export_weights.py` (full training pipeline in `training/`). It contains:

- the `meta` block — rule order, conv channels, GRU units, tensor shapes;
- the `weights` block — `conv0_W/b`, `conv1_W/b`, `conv2_W/b`, `gru_Wih/Whh/bih/bhh`,
  `gru_Wih_rev/Whh_rev/bih_rev/bhh_rev`, and the eight per-rule linear heads
  (`head_MADD_W/b`, `head_GHUNNAH_W/b`, …).

`TajweedNet.kt` parses it eagerly at analyzer init, runs the forward pass in pure Kotlin,
and a parity test (`TajweedNetParityTest`) keeps the Kotlin port numerically aligned with
the training-time PyTorch model.

---

## 1. Where the weights come from (offline)

The weights are exported from a small supervised training pipeline that lives OUTSIDE
this repo:

```
src/export_weights.py     # PyTorch -> JSON exporter, ships to assets/model_weights.json
src/train.py              # data loading + training loop
src/prepare_dataset.py    # builds (audio, expected-rule-label, evidence) pairs
calibration corpus        # multi-reciter Quran audio (Tadabur et al.)
```

Roughly:

1. Collect multi-reciter Quran recitation audio with alignment metadata.
2. For every word in the corpus, derive the expected tajweed rules from the Arabic text
   using `TajweedRules.detectRulesForWord` (the Kotlin version is at `app/src/main/java/com/example/ml/TajweedRules.kt`; the training repo has its own port that must agree).
3. Extract the **identical** feature set the Android app uses at inference time:
   - 40-band log-mel spectrogram per word,
   - a normalized duration scalar,
   - a 5-dim evidence vector.
4. Train Conv1d x3 → BiGRU(64) → 8 per-rule linear heads in PyTorch.
5. Evaluate against a held-out fold to learn the per-head threshold used by
   `decisionThresholdForHead()` (see `tajweed.md` §3 for the current values).
6. `export_weights.py` flattens the trained state_dict into `model_weights.json`.

---

## 2. Shipping a new artifact into the app

When a new checkpoint is trained:

```bash
# from the training repo
python src/export_weights.py  --out model_weights.json

# copy it into the Android repo (this repo)
cp model_weights.json <this-repo>/app/src/main/assets/
```

Rebuild the APK; the singleton-free `RecitationAnalyzer.get(context)` will pick it up on
the next cold start. No schema/code change is needed as long as `TajweedNet`'s parsing
rules (meta block shape, weight names) match the exporter.

---

## 3. Per-head calibration thresholds

The 8 decision thresholds used at verdict time live in `decisionThresholdForHead()` in
`app/src/main/java/com/example/ui/RecitationApp.kt`. Their current values were tuned
against the same held-out fold used during training, where each rule's correct-clips
empirical probability mass is maximised while keeping out-of-class rates acceptable.
Highlights:

- **MADD** — `0.56`. The MADD head is fairly confident on both correct and mispronounced
  clips; the bar is higher than the naive 0.5 to catch faint lengthenings.
- **GHUNNAH** — `0.5`. User-requested strict revert; matches the original Youden-optimal
  flattening.
- **QALQALAH** — `0.15`. The QALQALAH logits for correct clips span roughly 0.09 to 0.97
  with std 0.19; a flat 0.5 wrongly fails ~12% of correct recitations, so the bar is on
  the low end of the actual safe range.
- **ALLAM** — `0.47`.
- **IZHAR** — `0.17`. Same reason as QALQALAH: this head's output scale is lower than
  the others.

To re-tune, re-run the offline evaluation fold and copy the new thresholds into
`decisionThresholdForHead`. The remaining thresholds (GHUNNAH, IDGHAM, IKHFA,
MEEM_SAKINAH) are pinned to 0.5 as of the most recent user request and stay there unless
explicitly overridden.

---

## 4. Notes & invariants

- `RecitationAnalyzer` is a process-wide singleton; weights parse happens once per
  process lifetime on `Dispatchers.IO`. The lazy trigger inside `analyzeRecitationDetails`
  is fine for background work; for UX purposes `MainViewModel.preloadTajweedModel()` is
  called as soon as the user enters a Tajweed-aware surface, so the first mic tap never
  stalls.
- The `ev` (evidence) vector layout and the `dur` scaling must remain consistent between
  `TajweedFeatures.kt` (Android) and the training repo. Any change produces silently bad
  scores — if you change one, change both and re-run the parity test.
- The 8 rule names that ship in `meta.rules` are:
  `["GHUNNAH","MADD","QALQALAH","IDGHAM","IKHFA","MEEM_SAKINAH","ALLAM","IZHAR"]`.
  The user-visible labels (e.g. `"Ghunnah & Shaddah"`, `"Al-Lam"`) are produced by the
  UI layer via `ruleLabelToHead` / `getAppString`, NOT by the model.
- Parity: `TajweedNetParityTest` runs the same fixture through both Kotiln and the
  PyTorch reference and asserts forward-pass agreement to within float32 tolerance. Run
  `./gradlew :app:testDebugUnitTest --tests com.example.ml.TajweedNetParityTest` after any
  weights refresh.

---

## 5. Threshold policy on the deployed side

The values in `decisionThresholdForHead` (in the Android UI
`RecitationApp.kt`) are the *runtime* operating points. They were tuned
against the calibration fold and are not necessarily the same numbers the
offline training/eval pipeline prints:

| Head | Runtime threshold | Reason |
|---|---|---|
| QALQALAH | 0.15 | Correct-clip sigmoid spans 0.09–0.97 std 0.19; lower cut preserves recall. See TRAINING.md §7.2 (training repo) for training-side improvements intended to lift the distribution back up. |
| MADD | 0.56 | Higher than naive 0.5 because the MADD head is fairly confident on both classes. |
| GHUNNAH | 0.50 | Pinned to Youden-optimal. Earlier Youden-emphasising change was reverted. |
| ALLAM | 0.47 | Slightly lower than 0.5. |
| IZHAR | 0.17 | Low-scale head, similar rationale to QALQALAH. |
| IDGHAM / IKHFA / MEEM_SAKINAH | 0.50 | Default Youden-optimal. |

These thresholds are the user-visible source of truth; the training loop's
internal threshold evaluations are separate.
