# Tajweed Scoring (TajweedNet)

This document explains how the app evaluates Tajweed on-device today. **The previous
MFCC/DTW "SimilarityEngine" pipeline has been retired** — it could not generalise across
reciters and had no concept of cross-word rules. The current implementation uses a single
compact neural model (`TajweedNet`), one forward pass per word, with per-head thresholds
learned from the calibration corpus.

## 1. High-level pipeline

For every word the user utters:

1. **Word-level detection (rule expectation):** `getTajweedRuleForWord(word, nextWord, riwayah)`
   scans the Arabic word (plus the next word for cross-word rules) and returns a label like
   `"Madd"`, `"Al-Lam"`, `"Ikhfa"`, etc. This is a deterministic rules table implemented in
   `TajweedRules.kt`, and it is **identical to what the calibration + training pipeline
   used**, so the rule label seen by the model matches the label the rule-detector expects.
2. **Audio slicing:** the captured audio is sliced around the matched word.
   In Memorize+Tajweed mode the audio comes from the Sherpa-ONNX pipeline's
   **UN-GATED** accumulator (`SherpaPipeline.snapshotRawAudio()` live, or
   `pause()`'s return value post-stop): every 250 ms mic chunk including
   inter-word silence, in real time. (The RMS-gated STT buffer is NOT used
   for scoring — it glues words together and compresses the timeline, which
   the E2E harness proved is out-of-distribution for TajweedNet and
   misaligns the wall-clock word timings.) The Tajweed Rules Coach card gets
   the same un-gated buffer via `pause()`. The slice geometry matches the
   TRAINING clips (`prepare_dataset.py`):
   - **single-word rules** (MADD, GHUNNAH, QALQALAH, ALLAM): the word's own
     span ±100 ms, widened to a 600 ms minimum, capped at 2 s (the model's
     log-mel view is MAX_FRAMES=200 ≈ 2 s);
   - **cross-word rules** (IDGHAM, IKHFA, IZHAR, MEEM_SAKINAH): from the
     trigger word's start to the NEXT word's end (+100 ms margins) — the cue
     lives at the boundary, and the training clips span exactly this cut
     (~1.8–2.1 s median).
   The slice is then silence-trimmed (`trimSilence`, RMS-gated,
   loudness-independent) so the model sees speech, not silence padding.
3. **Feature extraction:** `TajweedFeatures.extract(audio)` builds:
   - a 40 × T log-mel spectrogram,
   - a normalized duration scalar (`dur`),
   - a 5-dim evidence vector (`ev`) — summary stats the training corpus showed to be
     head-discriminative (mean/var of MFCC band, dur, …).
4. **Forward pass:** `TajweedNet.forward(logmel, dur, ev)` runs the 3-layer Conv1d + BiGRU
   + per-rule linear heads, and returns a single logit per rule. A sigmoid per rule turns
   it into `P(correct)`.
5. **Verdict:** each word's score is compared against `decisionThresholdForHead(rule)`
   (see §3). `P(correct) < threshold` ⇒ `WordAnalysis.status = TAJWEED_ERROR`, and the
   corresponding `tajweedRule` + `tajweedAcousticFeedback` are attached.

   **Bounded confirmation search (false-popup guard).** The `SpokenWordTiming`
   wall clock is the EMISSION time of the word, which can be offset from the
   word's true audio position by up to the 4 s decode window (and merge/repair
   re-emissions can shift it in either direction). A single window at the
   nominal position therefore sometimes lands on the wrong audio and would
   flag a correct recitation. So before a `TAJWEED_ERROR` is issued, the
   scorer slides a training-sized window (word-span 0.4–1.6 s for single-word
   rules, 2.0 s for cross-word rules) over ±6 s around the nominal position
   and takes the BEST score; the error is only flagged if every position
   stays below threshold. A genuine mispronunciation fails everywhere; a
   timing misalignment recovers. The extra cost is paid only for words about
   to be flagged, and the search breaks as soon as any position passes.
   Validated 15/15 by `PcTajweedFalsePositiveTest` (zero false TAJWEED_ERROR
   on correct Alafasy recitations across 15 surahs).

## 2. TajweedNet architecture (`TajweedNet.kt`)

```
logmel (40 × T) ──> Conv1d(+ReLU,+MaxPool2) ──> Conv1d x2 ──> BiGRU(64)
                                                              │
                                                     temporal mean-pool
                                                              │
                                                     [pooled(128) + dur? + ev]  ──> per-rule Linear head
                                                                                    │
                                                                            sigmoid(logit) = P(correct)
```

- `dur` is appended before the MADD head only (per the trained graph).
- `ev` is appended before every head.
- The weights live in `assets/model_weights.json` (~1 MB), exported by `training/src/train.py` /
  `training/src/export_weights.py` (see the `training/` directory in this repo).
- `TajweedNetParityTest` asserts that the Kotlin forward pass matches the PyTorch reference
  on a fixed input bit-for-bit inside float32 tolerance.

The model weights are loaded **once per app process** through the `RecitationAnalyzer`
singleton (`RecitationAnalyzer.get(context)`), so the expensive JSON parse + tensor
allocation happens a single time. To avoid paying that cost the first time the user taps
the mic, `MainViewModel.preloadTajweedModel()` is triggered on:

- entering the **Tajweed** tab (rules coach), and
- flipping **Memorize** to **Tajweed** mode,

both on `Dispatchers.IO` with a "Loading Tajweed Model…" popup that disappears on its own
once the singleton is warm.

## 3. Per-head thresholds (`decisionThresholdForHead`)

A flat 0.5 cut misclassifies Qalqalah and Izhar heavily, because those heads output low
|logit| even on correct clips. Each rule has its own calibrated bar:

| Rule  | Head key       | Threshold | Comment                                  |
|-------|----------------|-----------|------------------------------------------|
| Madd  | MADD           | 0.56      | tuned for Hafs+Tadabur calibration fold  |
| Ghunnah | GHUNNAH      | 0.50      | user-requested strict revert             |
| Qalqalah | QALQALAH   | 0.15      | low-scale head — flat 0.5 fails correct cases |
| Idgham | IDGHAM         | 0.50      |                                          |
| Ikhfa | IKHFA          | 0.50      |                                          |
| Al-Lam | ALLAM         | 0.47      |                                          |
| Meem Sakinah | MEEM_SAKINAH | 0.50 |                                          |
| Izhar | IZHAR          | 0.17      |                                          |

The melodic note: the *displayed* "%" is `P(correct) × 100`; the green/red verdict is
purely the calibrated threshold.

## 4. Mistake surfacing (the new UX)

**Memorize + Tarteel (the default):** when the user speaks, word-tracking via the
SpeechRecognizer advances through the target text. On a wrong or skipped word the existing
banner appears, an `errorIndex` is computed, and the alignment stops advancing until the
user recites correctly — no tajweed analysis is engaged, so the user pays no battery tax.

**Memorize + Tajweed:** word-tracking runs on the **Sherpa-ONNX** offline
recognizer (`SherpaPipeline`): a single `AudioRecord` feeds both the STT
stream (RMS-gated buffer) and the TajweedNet scorer sink (un-gated buffer).
`analyzeRecitationDetails` ALSO passes the per-word audio to
`RecitationAnalyzer.analyze(..., [rule], threshold)` for any word that the
rule-detector tagged. When the model judges `P(correct) < threshold` (after
the bounded confirmation search, §1 step 5):

- the word's `status` becomes `TAJWEED_ERROR`,
- a separate piece of UI state (`latestTajweedPopup`) is set to `(ruleLabel, wordIndex)`,
- a small non-modal non-pausing dialog appears ("Tajweed Mistake — <rule>") for 4 seconds
  so the user can keep reading without losing their place; a `lastTajweedPopupIndex` guard
  prevents the same error from re-firing every recomposition,
- the recitation flow does **not** pause (this is a deliberate change from earlier v1
  behaviour, in response to the user request: "I want him to continue reading and if there
  is an error a pop up is shown indicating his tajweed error exactly like the skipped/wrong
  word mechanism" — the popup mirrors that flow but is tajweed-specific),
- the per-word chip colouring still displays "Tajweed: <rule>" in the analysis list, so the
  user can scroll back and see a full audit trail.

**Surah / ayah-range change mid-session (state-reset contract).** When the user changes the
target (different surah, or the start/end ayah pickers), the `LaunchedEffect` on
viewModel.ayahs / selectedSurah / rangeStart / rangeEnd stops the live recognizer session,
wipes every speech/analysis variable (`finalSpokenText`, `partialSpokenText`, `spokenTimings`,
`recognize{dAudio,AudioUri,StartMs}`, `alignment`, `detailedAnalysis`, `latestTajweedPopup`,
`errorMessage`) and — crucially — flips `recognizerResultsSuppressedUntil` to `now + 500 ms`.
The partial/final recognizer consumers refuse to emit results while that window is live; this
defeats the previous bug where the recognizer's pending whole-transcript ("Bismillah…" from the
old range) got re-analyzed against the new/empty target and the UI popped "wrong word + tajweed
mistakes" before the user had read a single word of the new range. If the user was listening when
they moved the range, the session auto-resumes ~520 ms later against the new target; otherwise it
stays stopped and they tap the mic to start cleanly.


**Tajweed (+Tarteel cross-tracking):** the same `TAJWEED_ERROR` row feeds the Stats tab:
`triggerGoalRealProgress` collects per-rule mistake counts (`violatedRules`) and occurrence
counts (`encounteredRules`) so the "Correct applications of <rule>" breakdowns stay accurate.

### 4.1 Tajweed Rules coach — random example generator

Each Tajweed card previously bundled a fixed list of 9–12 curated Quran phrases. The user asked
for fresh material on every expand/refresh, like the practice_recitation-main build. Because the
card colouring and the acoustic coach are both driven by `getTajweedRuleForWord` /
`buildTajweedAnnotatedString`, the generator reuses those exact functions rather than
duplicating colouring logic. On expand and on the refresh icon, the card calls
[`generateRandomQuranExample(head)`](app/src/main/java/com/example/ui/RecitationApp.kt), which:

1. picks a random surah (1..114),
2. fetches it from `https://api.alquran.cloud/v1/surah/{n}/ar.alafasy` (OkHttp),
3. walks the ayahs' tokens and returns the first one whose
   `ruleLabelToHead(getTajweedRuleForWord(token, next, "Hafs"))` matches the card's head
   (GHUNNAH / MADD / QALQALAH / IDGHAM / IKHFA / ALLAM / MEEM_SAKINAH / IZHAR).
   The Qalun-only **Silah** card is excluded (it has no dedicated model head and maps to
   IZHAR in `ruleLabelToHead`, so the generic generator would fetch plain Izhar examples for
   it); Silah keeps its curated Qalun pool.

The resulting phrase is coloured with the same `buildTajweedAnnotatedString` per token used
elsewhere, so the rule marking inside it shows the same colour as the card chip. If the
network fetch fails (offline, or the surah happens to lack the rule), the card falls back
to the bundled no-repeat random pick from its curated examples list — same behaviour as
before, so offline UX is unchanged.

## 5. Why the previous pipeline is gone

The old acoustic path (`MaddAnalyzer`/`GhunnahAnalyzer`/`QalqalahAnalyzer`/`SimilarityEngine.kt`)
used hand-tuned duration windows + 39-dim MFCC DTW against a per-rule reference template.
It worked for one reciter but had two structural problems:

1. It could not handle cross-word rules (Idgham, Ikhfa, ALLAM merges across tokens),
2. It had no concept of an "incorrect input" — giving a harsh score to a totally different
   surah looked exactly the same as a mispronounced word.

`TajweedNet` was trained on paired (correct / incorrect) clips per rule, has explicit
cross-word evidence (`ev` carries neighbour statistics), and is robust to speaker
variation because it was calibrated on a multi-reciter corpus.

## 6. Colour palette font-lock

The Read / Memorize highlighted word coloring and the Tajweed coach share
`buildTajweedAnnotatedString(...)` — same palette:

| Rule     | Colour                |
|----------|-----------------------|
| Qalqalah | Blue `0xFF1E88E5`     |
| Al-Lam (Solar) | Blue-Grey `0xFF607D8B` |
| Al-Lam (Lunar) | Teal  `0xFF00ACC1`  |
| Madd     | Red `0xFFE53935`      |
| Idgham   | Orange `0xFFFB8C00`   |
| Ikhfa/Iqlab | Cyan `0xFF00ACC1` |
| Izhar    | Purple `0xFF8E24AA`   |
| Ghunnah  | Green `0xFF43A047`    |
| Meem Sakinah | Deep Purple `0xFF7E57C2` |
| Silah (Qalun only) | Purple `0xFF8E24AA` |
