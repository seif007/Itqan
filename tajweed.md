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
- a separate piece of UI state (`latestTajweedPopup`) is set to `(ruleLabel, wordIndex, score)`,
- a small auto-dismissing dialog appears — "**Tajweed Mistake — <rule>**, your
  <rule> score: N%" — and the mic auto-stops so the user can read the feedback;
  a score-aware guard (`lastTajweedPopupIndex` + `lastTajweedPopupScore`) lets
  the SAME word re-fire when a later re-read produces a different score,
- stopping arms the same mistake-resume re-seat as WRONG/SKIPPED: tapping the
  mic truncates the display transcript to the prefix BEFORE that word, so the
  re-read passes the collector's count-filter (otherwise it is swallowed as an
  "already committed" replay copy and nothing shows), gets displayed, gets
  freshly scored (verdict-cache key includes the utterance timing), and
  re-popups if still wrong,
- while the resume latch is armed the collector FREEZES: every post-stop
  emission (in-flight window decodes plus `pause()`'s own final full-tail
  decode) is dropped, so dead pre-stop audio can never resurrect the mistake
  as CORRECT; the fast analysis loop likewise keeps the confirmed verdict if a
  stale queued pass lands.

**Session boundary (range completed).** When a recitation reaches the END of its
selected range with no `PENDING` (un-scored) words left, `sessionRangeCompleted`
is set. The NEXT mic tap must start a *fresh* recitation — so `startTajweedSession()`
wipes the previous run's transcript + scored state (`alignment`, `detailedAnalysis`,
verdict/mistake-resume caches) and falls through to the clean-session path instead of
`resume()`-ing the paused pipeline (a plain resume would replay the stale,
fully-recited transcript and bleed the finished session into the next recitation). The
explicit **Restart** control does the same wipe (it drops scored state, not just the
transcript) and clears `sessionRangeCompleted`, so a leftover range-completed flag can
never wipe a brand-new run on its first mic tap. Both flags reset on every range/surah
change.

**WRONG / SKIPPED mistakes — confirmation + verification gate.** A text-level
candidate only becomes a *punished* mistake after three gates:

1. **Multi-emission confirmation:** the SAME first-mistake index must survive
   distinct emissions — 2 for WRONG, 3 for SKIPPED (plus ≥2 s dwell and the
   transcript advancing ≥3 words past the skip). The visible banner/red word
   uses this same evidence (`confirmedMistakeUiIdx`) so a transient
   single-window decode artifact never flashes.
2. **Early tail verification:** at the SECOND sighting, one full-context
   offline decode of the trailing ~10 s (`SingleAudioPipeline.verifyTailWords`,
   read-only, serialized against the live decode loop via `decodeBusy`) looks
   for the flagged target word:
   - **found** ⇒ false alarm — the index joins `verifiedSpokenIndices`, counts
     CORRECT, counters reset, recitation continues untouched;
   - **absent** ⇒ remembered in `mistakeVerifyAbsentIdx`; confirmation may then
     proceed to popup + pause;
   - **inconclusive** (decoder busy / too little audio / blank decode) ⇒ decide
     nothing; retry on a later emission (3 s cooldown).
   Punishment REQUIRES the absent verdict — measured on-device, sliding windows
   can chronically garble/drop a word (e.g. لِلَّهِ رَبِّ lost for ~5 s across
   two windows, heard by the next window and every full-tail decode).
 3. **Minimum-listen guard:** no auto-stop within the first ~2.5 s of a session.

**Episode-stable candidates.** Noisy merges make the first-mistake index flicker
±few words between emissions (echo tokens appear/vanish); treating each flicker
as a NEW candidate reset the confirmation counters forever, so on slow phones
nothing ever fired and the cursor froze. The watcher now tracks an ANCHOR index:
drift within ±3 words counts as the same episode (verification and punishment
gates are drift-tolerant too), and an episode closes only after TWO consecutive
clean emissions.

**Verification slot-wait + episode span.** `verifyTailWords(trailingSeconds)`
waits up to ~2.5 s for the live decoder's slot instead of bailing instantly
(the live loop re-decodes every ~1.5 s and each pass takes 1-2.5 s on slow
phones — an unconditional grab lost nearly every race), and its window spans
the WHOLE candidate episode (+8 s margin, clamped 10-45 s) so a word read a
while ago is still inside the decoded audio.

**Stuck-reconcile.** If a candidate STILL survives ≥6 emissions after its
verdict, the live transcript itself is echo-poisoned; one long-span tail decode
is then adopted WHOLESALE when it aligns to the target with more leading
matches than the live log — the same heal a manual pause used to trigger,
applied automatically mid-session.

**Verified-only display.** The wrong/skipped banner + red word appear ONLY after
the tail verification says the word truly was not said. Transient candidates
stay invisible while self-heal has a chance; "only true mistakes should be seen
by the user".

**Exclusive popups.** WRONG/SKIPPED rows carry no tajweed rule (and stale scored
rows for that word are purged): the user sees either a recitation-mistake popup
OR a tajweed popup naming rule + score — never both.

**Multi-rule scoring rows.** A word carrying several rules gets ONE SCORE ROW
PER RULE (Al-Lam + Madd on ٱلضَّالِّينَ, etc.), each judged against its own
head threshold; alignment counts each WORD once. MADD is only attached where
the app PAINTS RED (explicit maddah marks آ/U+0653/U+06E4) — natural 2-count
vowels score 23-51% even on perfect recitations with the current head and stay
excluded until a recalibrated model ships.

**Epoch guards.** Range/surah changes, tab switches and both restart buttons
bump an analysis epoch: queued requests from older epochs are drained/dropped,
and `invalidateInFlightCommit()` makes any in-flight pause()/stop() final-tail
decode discard its emit instead of landing old-range words into the new session.

**Finalize ⇒ PENDING counts CORRECT.** When a session finalizes (user stopped;
the full-tail decode already had its best chance), remaining PENDING words were
never rendered by the recognizer — they are counted as READ CORRECTLY and
displayed green. The user is not charged for the decoder's gaps (revised per
user: "count it as the user read it correctly and display it").

**Merge echo guard (target-count).** The merge's tail-append path drops a word
whose target multiplicity is already fully satisfied by the log (Māʿūn's 4th
أَلَا, Fātiḥah's echoing الرَّحْمَٰنِ) — echoes are stopped at the source
instead of stealing alignment slots. Counts come from a vocab MULTISET (with
duplicates); with only a Set the guard is disabled (a Set would flatten
قُلْ ×2 and drop the real second occurrence).

**Echo absorption (alignment walk).** As a second layer, the verdict walk steps
over a spoken token that matches an ALREADY-CONSUMED target word — but only
when that word does NOT also occur in the remaining target ahead (legitimate
repeats like الله ×2 align normally), capped at 3 consecutive.

**Harness parity.** The PC E2E harness's verdict walk is a VERBATIM mirror of
the app's (same isWordMatch fuzzy matching, echo absorption, annotation-sign
handling, wrong-stops-walk), plus Al-Baqarah scenarios (Ayat al-Kursi 255 ✓,
285-286) exercising long-session echo/annotation edge cases.

**Final-decode repair acceptance (positional quality).** When pause()/stop()
runs the full-tail decode, the collector accepts its "repair swap" only if it
is at least as GOOD as the live log — scored as the longest LEADING run of
words matching the target sequence. A monotonic-merge echo can leave the live
log LONGER but MISORDERED (a premature ٱلْعَٰلَمِينَ emitted before لِلَّهِ
رَبِّ); the old raw-size rule rejected the correct deduplicated repair in that
case and locked in phantom mistakes. A short GARBLED repair still scores a low
prefix and stays rejected.

**Surah / ayah-range change mid-session (state-reset contract).** When the user changes the
target (different surah, or the start/end ayah pickers), the `LaunchedEffect` on
viewModel.ayahs / selectedSurah / rangeStart / rangeEnd stops the live recognizer session,
wipes every speech/analysis variable (`finalSpokenText`, `partialSpokenText`, `spokenTimings`,
`recognize{dAudio,AudioUri,StartMs}`, `alignment`, `detailedAnalysis`, `latestTajweedPopup`,
the verification/forgiveness sets and the display-gate index) and — crucially — flips
`recognizerResultsSuppressedUntil` to `now + 500 ms`.
The partial/final recognizer consumers refuse to emit results while that window is live; this
defeats the previous bug where the recognizer's pending whole-transcript ("Bismillah…" from the
old range) got re-analyzed against the new/empty target and the UI popped "wrong word + tajweed
mistakes" before the user had read a single word of the new range. If the user was listening when
they moved the range, the session auto-resumes ~520 ms later against the new target; otherwise it
stays stopped and they tap the mic to start cleanly. Every explicit restart
(repeat/restart buttons, FAB stop, surah dialog, tab switch) clears the same state.


**Tajweed (+Tarteel cross-tracking):** the same `TAJWEED_ERROR` row feeds the Stats tab:
`triggerGoalRealProgress` collects per-rule mistake counts (`violatedRules`) and occurrence
counts (`encounteredRules`) so the "Correct applications of <rule>" breakdowns stay accurate.

**Disjointed letters (muqatta'at) — tolerant matching.** The Quran opens 29 surahs with
حروف مقطعة (`الم`, `المر`, `كهيعص`, `طه`, `حم`, `يس`, `ص`, `ق`, `ن`, …). These are *written*
as one token but *recited* as spelled-out letter names (`الم` → "ألف لام ميم"), so
`PronunciationVariants.kt` expands each into ordered letter-name slots for the alignment walk.

The strict slot matcher assumes the decoder renders *every* letter name. On device it does not.
Measured on a Samsung A05s reciting Al-Baqarah 2:1 with the shipped `model_int8.onnx` CTC graph,
three consecutive windows emitted:

| window | raw decode |
|---|---|
| 1 | `م` |
| 2 | `سف م ذ الكتاب` |
| 3 | `أف م ذ الكتاب وَلَا ريب فِيهِ` |

i.e. `الم` came back as a truncated `أف` plus a bare `م` — `لام` was dropped. `isWordMatchPlural`
needs three tokens and gets fragments, so it returns false; the walk then burns its OOV-garbage
skip cap (2) on the fragments and charges a **WRONG at word 0**, which returns early and leaves
all 6115 remaining words of the surah `PENDING`. Observed result: `matched=0`, **nothing
highlighted, no popup, cursor frozen until the user taps pause.** Al-Fatihah has no muqatta'at —
which is exactly why the user saw it "work fine" while Al-Baqarah never displayed a word.

`matchMuqattaatRelaxed()` fixes it. Anchored at the transcript frontier (`spokenIndex`), it runs a
bounded DP over the letter-name slots and the next `slots.size` spoken tokens, tolerating the two
failure modes actually observed:

* a letter name rendered short or slightly wrong (shared prefix, or edit distance ≤ 1);
* a letter name dropped by the decoder.

A third, narrower case covers the decoder mangling the *head* of a three-letter name while keeping
its final consonant — one recitation of `ألف` came back as `أف` in one window and `سف` in the
next. A 2-char token ending in the same letter as a 3-char letter name is accepted. This is gated
on the last character on purpose: a plain "edit distance ≤ 2" rule was tried first and is far too
loose — it let `صدر` claim `صاد` and `ولا` claim `الف`. The final-consonant version rejects both.

Bounded so it cannot fire on unrelated speech: run ≤ `slots.size` tokens, order preserved, and at
least `slots.size - 1` slots must match (for 1–2-letter tokens, *all* slots — otherwise a stray
`م` or `ح` could claim `حم`). It returns the number of tokens consumed so the cursor advances past
exactly what was used and no further — on the decode above it consumes `أف م` (2 tokens) and
leaves `ذ` for `ذَٰلِكَ`. The same relaxed matcher backs the `mistakeStop` tail re-decode, so a
correctly-recited muqatta'at can no longer be "confirmed absent" and flagged.

**Clipped-word guard.** The same decoder also truncates ordinary words — `ذَٰلِكَ` came back as
just `ذ` in every window *and* in the full-context tail decode. `ذ` matched nothing, so the walk
scanned ahead, found `ٱلْكِتَٰبُ` and charged the user a **SKIPPED** on a word they had in fact
recited. A spoken token that is a strict prefix of the current target *and at least two characters
shorter* is now treated as a clipped decode: the word stays `PENDING` (the next window can still
complete it) instead of being flagged. Equal-length near-misses are unaffected, so real
substitutions still surface as `WRONG`.

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

### 4.2 Coach verdict card — multi-rule coverage & shared palette

The practice card's "Acoustic score per rule" panel used to disagree with the example it
scored, in three ways (all fixed):

1. **Missing rules.** `analyzeRecitationDetails` scores ONE rule per word —
   `getTajweedRuleForWord` returns the FIRST match and returns immediately, so
   الْحَمْدُ short-circuits on its Al-Lam article and the Izhar-Shafawi meem inside the SAME
   word is never scored. After the primary pass, the card now runs a supplemental pass:
   `TajweedRules.detectRulesForWord` (the training-aligned multi-rule table) detects every
   rule across the example tokens; heads the primary pass left uncovered are scored directly
   against `RecitationAnalyzer` on each token's proportional audio slice (silence-trimmed)
   and appended as extra rows. Rows never duplicate a primary one.
2. **Wrong row colours.** Row colours resolved via an exact card-title match
   (`rulesList.firstOrNull { it.name == ruleName }`) — sub-labels like "Madd Muttasil" /
   "Al-Lam" match no title and fell back to the theme primary (a Madd row rendered TEAL).
   Colours now resolve through the canonical head palette (`tajweedHeadColor`, tajweed.md
   §6): head colour first, exact card match second, theme last.
3. **Status-red word colouring.** The transcript line under the rows painted every
   TAJWEED_ERROR / WRONG / SKIPPED word flat red, which read as "this word is a Madd
   mistake" even when the red was a skipped word or a different rule. Each word is now
   rendered with `buildTajweedAnnotatedString` — the SAME letter-level palette as the
   example — with two status overrides: WRONG/SKIPPED get error colour + strikethrough,
   TAJWEED_ERROR keeps its rule colours plus an underline.

### 4.3 The end-of-range popup was gated on PENDING — and PENDING never clears

Three long-standing user reports — *"the final word never gets displayed"*, *"I didn't
get a popup"*, *"I tapped pause and only then got a summary saying the recitation was
correct"* — are **one bug**, not three.

The walk in `analyzeRecitationDetails()` deliberately never charges a mistake for a word
the recognizer has not finished rendering:

* the **transcript-frontier** guard (`spokenIndex == spoken.size - 1`) emits PENDING, and
* the **clipped-decode** guard (a spoken token that is a strict prefix of the target and
  ≥ 2 chars shorter) emits PENDING while carrying the fragment it *did* hear in
  `spokenText`.

The last word of every recitation is always one of those two: the endpoint window cuts
the final word mid-utterance (`العالم` for `ٱلْعَٰلَمِينَ`). So **a PENDING row always
survives to the end of the session.**

The summary gate was:

```kotlin
val pendingRemain = detailedAnalysis.any { it.status == RecitationStatus.PENDING }
if (sessionFinalized && !pendingRemain && ...) { showSummary() }
```

Since `pendingRemain` was always true, the end-of-range popup could **never** fire. The
user only ever saw a summary by tapping pause manually, whose `finalizeCorrect()` pass
forgives every PENDING word — hence "correct recitation".

Two rules now:

1. **A clipped PENDING row means "reached", not "unknown".** Rows with a non-null
   `spokenText` are counted by `newMatchedWordCount` (so the last word lights up live)
   and are excluded from `pendingRemain`. Only PENDING rows with `spokenText == null`
   — words the decoder never heard, or everything trailing a WRONG — block the
   summary or hold the cursor.
2. **Muqatta'at head forgiveness.** A false WRONG on a disjointed-letter word at
   `targetIndex <= 1` returns early from the walk and leaves *every* remaining word
   PENDING (6115 for a whole-surah Al-Baqarah): `matched=0`, nothing highlighted, no
   popup. The risk is completely asymmetric, so after the relaxed matcher fails the
   head word is forgiven as CORRECT instead of freezing the surah.

> **Do not re-gate any completion/UI path on `status == PENDING` alone.** PENDING is
> the walk's "not decided yet" state and it is permanent for the final word. Always
> distinguish *clipped* (`spokenText != null`) from *unheard* (`spokenText == null`).

### 4.4 The walk swallowed the words it jumped over — `matched` froze at 7/29

The second half of the same user report ("nothing displayed", "no popup") survived the
§4.3 fix because it has a completely different cause: **the alignment could not follow
the transcript past word 7.**

Measured on a full Al-Fatihah (`fatihah_full_v1`): `emitted=27w` — the transcript
reached the END of the surah — yet `matched` sat at **6, then 7, for the whole session**
and `hasError=true`. The committed transcript was:

```
… ٱلرَّحِيمِ | الحمد | ٱلْعَٰلَمِينَ | لله | الرحيم | الرَّحْمَنِ | مَٰلِكِ …
              ^4      ^5(should be 7)  ^6(5)  ^7(9)     ^8(8)      ^9
```

Two independent defects produce that sequence, and **either one alone freezes the surah**:

1. **رَبِّ was permanently lost.** Every window rendered it as the 1-letter fragment
   `رَ`. `restoreTruncatedWord()` in `SherpaPipeline.clean()` refused stems shorter than
   two code points (`cp < 2`), so the fragment was dropped on every re-decode — the word
   could never come back and the aligner charged a false SKIPPED on a word the user
   actually recited.
2. **`monotonicMergeWords()` can only APPEND.** A word the decoder renders late is
   therefore written *behind* speech that belongs to later targets: `لِلَّهِ` landed
   after `ٱلْعَٰلَمِينَ`, and `الرَّحِيمِ` before `الرَّحْمَٰنِ`.

The walk reacted to (2) by jumping ahead (lookahead 3, plus the ≤6-token resync
absorption) to reach the current target — and **silently discarding everything it jumped
over**. Those discarded words *are* the next targets, so each was charged SKIPPED, the
cascade reached a WRONG, and the walk returned: everything behind it stayed PENDING.

Two fixes, both needed:

- **`restoreTruncatedWord()` now accepts 1-letter stems.** Safety is unchanged:
  `maxMissing` still caps a 1-letter stem at ONE missing letter (so `ر` can only become a
  2-letter word like رَبِّ) and the uniqueness scan still vetoes ambiguous fragments.
- **A reorder buffer ("carry") in the walk.** Whenever the walk jumps ahead or resyncs,
  the swallowed words are parked in a FIFO capped at `CARRY_MAX = 4`, reusable only for
  the next `CARRY_BUDGET = 3` targets. The following targets try the buffer *first*; a
  hit is consumed from the buffer and does **not** advance `spokenIndex`. Bounding it
  both ways is what keeps a stale buffer from resurrecting a word far from where it was
  heard. Head resyncs (`targetIndex == 0`) do not fill the buffer — tokens absorbed in
  front of the first target word belong to no target at all.

Result: the same fixture went from `matched=7/29 hasError=true` to **`matched=29/29,
hasError=false, confirmed mistakes: 0`**.

> **Never let the walk discard a spoken word without offering it to the next targets.**
> The transcript is ASR output: it loses words and it reorders them. Any "skip ahead to
> the next match" step must park what it skipped, or a single local scramble becomes a
> permanent freeze of everything downstream.

### 4.5 The injection read cursor survived across runs

`stopTajweedSession()` only pauses the pipeline (it never releases the 296 MB
recognizer), and `runInjectLoop()`'s `injectPos` is a pipeline field — deliberately, so a
paused/resumed session continues where it left off. The side effect is that a **second
injection run in the same process resumed past EOF** and fed pure silence:
`bytes=1487928` but `startPos=1495856`, no window decoded anything, and the run looked
like a dead app for the full `MAXWAIT` (`fatihah_v2`: `matched=0` for 200 s).

`runInjectLoop()` now rewinds to 0 when the file is already exhausted (a mid-file resume
still continues), and `inject_run.sh` **asserts `startPos=0`** and aborts (exit 3)
otherwise — a dead session must never again be captured and reported as a test result.

### 4.6 The popup was gated on "the endpoint brought NEW words" — it never does

With the alignment finally clean (three consecutive runs at `matched=29/29`, 0 confirmed
mistakes) the end-of-range summary **still never appeared**. `inject_run.sh` now asserts
it (§7b, a `uiautomator dump` for "Perfect recitation!" / "Recitation Summary"), because
the dialog is a Compose `AlertDialog` that logs nothing.

The gate is `sessionFinalized && !pendingRemain`, and `sessionFinalized` was set by:

```kotlin
if (ev.endpoint && words.isNotEmpty()) sessionFinalized = true
```

`words` is **not** the transcript — it is the *filtered delta*:
`filterMemorizeWords()` subtracts every already-committed word and caps each word at its
target count. So for a range the user has actually finished, the endpoint carries **zero**
new words by construction. Measured on every completed run: `ev.endpoint=true words=[]`.
The flag stayed false forever and the popup was gated off — permanently.

This also explains the exact shape of the user's report: they got a summary **only after
tapping pause**, because pause drives the `ev.repaired` branch above (which does set the
flag) rather than the plain-endpoint path.

The discriminator is the **raw** emission, not the delta:

```kotlin
if (ev.endpoint && ev.partial.isNotEmpty()) sessionFinalized = true
```

pause()'s synthetic endpoint is still excluded — that one really is empty.

> **Never use a deduplicated/filtered delta as an "is something new?" signal.** A delta is
> empty both when nothing happened *and* when everything had already happened. Gate
> lifecycle transitions on the raw stream.

The gate now also logs all of its inputs (`endOfRange(): sessionFinalized=… pendingRemain=…
newCompleted=…`), so a missing popup is diagnosable from a run instead of guessed at.

### 4.7 Nothing ever ended a session by itself — the summary lived inside the mic button

With §4.3, §4.4 and §4.6 fixed, three consecutive Fatihah runs still printed
`!! NO end-of-range popup on screen`, and the brand-new `endOfRange():` diagnostic never
logged at all. The reason: the stop → score → summarize block was written **inline in the
mic `FloatingActionButton`'s `onClick`** (~line 8155). Nothing in the app ever called it.
A finished range therefore sat there doing nothing until the user happened to tap stop —
which is exactly the report: *"nothing happened (no words displayed and no popups) … then I
clicked to pause and got a summary"*.

Fix: hoist the whole block into `val runStopAndFinalize: () -> Unit = finalize@ { … }`, point
the FAB at it, and add an `autoFinalizeDone`-latched `LaunchedEffect` that runs the identical
path when

* `sessionFinalized` is true and the session is still live, and
* every word of the range has a verdict (`decidedWords.size >= total`, counting **distinct
  `wordIndex`** — `detailedAnalysis` holds one row per (word, rule)), and
* no word is still *unheard*-PENDING (`PENDING && spokenText.isNullOrEmpty()`).

> **A lambda hoisted out of a composable call loses its implicit label.** The body still said
> `return@FloatingActionButton`, which no longer resolves (`e: Unresolved label`). Give the
> lambda its own label — `val f: () -> Unit = finalize@ { … return@finalize }`.

First firing (fatihah_v5): `autoFinalize(): range complete (29/29 words decided)` →
`endOfRange(): sessionFinalized=true pendingRemain=false …` → `END-OF-RANGE POPUP VISIBLE`.

> **A popup raised on a frozen walk does not prove the fix.** That same run had `matched=20/29`
> with a SKIPPED word, so the range had not been completed cleanly — see §4.9.

### 4.8 Room schema drift: goal updates failed silently on every completed session

The first run that ever reached the summary also produced:

```
E/MainViewModel: Error updating goals
java.lang.IllegalStateException: Room cannot verify the data integrity.
  Expected identity hash: 9017222fb51333856b12271a5c3b2d83, found: 9e92f706265a3086f3a77900e84d920c
```

`Goal` had gained `surahNumber` / `rangeStart` / `rangeEnd` / `mistakeEntriesJson` **without a
version bump** (still `version = 4`). Room does not run a migration when the version is
unchanged — it validates the identity hash instead and throws on open, so every goal update
failed while the UI kept working. It had been invisible until §4.7 because no run ever
reached the code path that touches the database.

Fix: bump to `version = 5` and add an **additive, idempotent** `Migration(4, 5)` that inspects
`PRAGMA table_info(goals)` and `ALTER TABLE … ADD COLUMN` only the columns that are actually
missing. Additive so existing goal history survives; idempotent so a database that already
has some of them is not corrupted.

#### 4.8a …but the additive migration was not enough: Room rejects EXTRA columns too

After the bump the error changed but did not go away — it fired on **every** open, twice per
session (measured fatihah_c23 at `01:15:13.179` and `01:15:13.703`), so goal progress was
*still* never written:

```
E/MainViewModel: Error updating goals
java.lang.IllegalStateException: Migration didn't properly handle: goals(com.example.data.Goal).
  Expected: createdTimestamp, id, mistakeEntriesJson, mistakesCount, rangeEnd, rangeStart,
            recitedAyahs, skippedWordsCount, surahNumber, tajweedMistakesBreakdown,
            tajweedMistakesCount, targetAyahs, type, wrongWordsCount
  Found:    completedTimestamp, createdTimestamp, id, mistakeEntriesJson, mistakesCount,
            periodStart, rangeEnd, rangeStart, recitedAyahs, skippedWordsCount, surahNumber,
            tajweedMistakesBreakdown, tajweedMistakesCount, targetAyahs, type, wrongWordsCount
```

Diffing the two lists: all four added columns are present, and the **only** difference is two
columns Room no longer declares — `completedTimestamp` and `periodStart`, leftovers from a
build older than the current entity. Room 2.7 validates the schema **exactly**, so extra
columns fail just as hard as missing ones, and `fallbackToDestructiveMigration()` does not
help because a migration *was* supplied.

Fix: `MIGRATION_4_5` now has a second phase. After the additive phase it recomputes the column
set and, if anything outside the entity is present, does the portable four-step rebuild
(`CREATE goals_rebuilt` → `INSERT … SELECT` the 14 declared columns → `DROP goals` →
`RENAME goals_rebuilt`). **Not** `ALTER TABLE … DROP COLUMN`, which needs SQLite 3.35+ that we
cannot assume on the user's Android 13 device. Rows are copied, so no goal history is lost.

### 4.9 The verification decode is compared with strict equality — but the decoder adds `ال`

`mistakeStop` re-decodes a tail window and asks whether the flagged word is in it. `found=false`
is treated as proof the user skipped the word. Measured on-device (fatihah_v5, target
`صِرَٰطَ`):

```
verifyTailWords(): done 1602ms 5w 'تير سين الصرااط الرحمن الرحيم'
mistakeStop(): verify idx=20 SKIPPED word='صِرَٰطَ' verified=5w found=false
```

The decode **did** contain the word — as `الصرااط`. `ال` is a proclitic the CTC decoder freely
glues on or drops, and strict equality rejected it, so a word the user recited stayed SKIPPED.
Because `newMatchedWordCount` `break`s on the first `WRONG`/`SKIPPED` (§4.4), the cursor froze
at 20/29 and the last nine words never displayed.

Fix: compare every with/without-`ال` combination before condemning a word
(`articleForms()`), requiring the bare form to be ≥ 3 characters so stripping `ال` from
`الله` (→ `له`) can never produce a spurious match. A `found` verdict adds the index to
`verifiedSpokenIndices`, which flips the row to `CORRECT` in `effective` and lets the cursor
advance.

> **Bias towards forgiveness.** A dropped decode and a genuinely skipped word look identical in
> the transcript, and the only discriminator (the tail re-decode) is itself noisy. Condemning
> freezes the whole display; forgiving costs one missed correction. The user complained
> repeatedly about freezes, so the tie goes to forgiveness.

### 4.10 The verification window was anchored to the wrong clock

`mistakeStop` decodes the last `spanSec` seconds, with

```kotlin
val spanSec = ((now - lastWrongStopFirstSeenMs) / 1000L + 8L).toInt().coerceIn(10, 45)
```

`lastWrongStopFirstSeenMs` is when the **candidate** was created — not when the **word** was
spoken. `mistakeIdx` flickers by a word or two between emissions (echoes appear and vanish), and
every flicker past the ±3 episode window *restarts* the candidate clock. The span then collapses
to its 10 s floor and decodes audio that begins **after** the word was already read.

Measured on-device (fatihah_v6, flagged word `مَٰلِكِ` at index 10):

| event | time |
|---|---|
| word actually read | ~22:53:25 |
| mistake flagged | 22:53:28 |
| candidate **recreated** (clock reset) | 22:53:40 |
| verification window (10 s tail) | 22:53:32 – 22:53:42 |
| decode result | `نَعْبُدُ وَإِيَّاكَ نَسْتَعِين اهدِنَا الصِّراطَ المستْتَقِيمَ` (words 14–20) |

The word was never inside the decoded audio, so `found=false` was accepted as proof and a correct
recitation was charged a SKIPPED.

Fix: anchor on `spokenTimings.firstOrNull { it.startIndex == idxToVerify }?.startMs` — when the
*k-th* transcript word reached the recognizer — and fall back to the old clock only when that
timing is missing. Same rule applied to the stuck-reconcile decode.

Result (fatihah_v7): `matched=29/29 hasError=false`, `wrong=0 skipped=0`, `VERDICT: PASS`.

### 4.11 A mistake must stop the recitation; the summary means "I finished"

The user's model is explicit:

> *"the summary is when I click on pause the mic or finish my recitation. for any mistake, the app
> stops there and give me a mistake popup."*

Auto-finalize (§4.7) fired as soon as every word had a verdict — including when one of those
verdicts was SKIPPED. That ended the session before the mistake pipeline could confirm anything,
so the user got a **summary listing a skipped word** instead of the **skipped-word popup** plus
auto-pause (fatihah_v6: `endOfRange(): … skipped=1`, `confirmed (auto-pause + popup): 0`).

Fix: auto-finalize additionally requires that no `WRONG`/`SKIPPED` row is outstanding. A false
alarm is forgiven by the verification pass, the row flips to `CORRECT`, and the summary then
fires normally — so this only defers finalization, it never blocks it.

The harness now also greps the post-run `uiautomator` dump for the mistake dialog
(`Recitation Mistakes Detected` / `You skipped the word`), so a run that ends on an unresolved
mistake is distinguishable from one that completed cleanly.

### 4.12 The mistake never confirmed: every gate was EMISSION-driven

§4.11 blocked the summary, but the user kept getting one — because the mistake path never actually
fired. Every Al-Fatihah run logged the same two facts:

```
mistakeStop(): candidate idx=20 sightings=1 …
confirmed (auto-pause + popup): 0
```

`sightings` never reached the threshold (2 for WRONG, 3 for SKIPPED) because **a sighting is only
counted when the transcript text changes** (`isNewEmission`). A mistake freezes the walk; the user
stops speaking to read the feedback; therefore **no new emission ever arrives**. The counter sat at
1 forever. The 12 s `MISTAKE_GRACE_MS` hold then expired and auto-finalize handed over the summary —
producing exactly the symptom reported: *"as usual there is a summary instead of stopping and
getting w mistake popup."*

Two independent causes, both fixed:

1. **The confirmation is now driven by the verdict's WALL CLOCK, not by emissions.** The decisive
   evidence was always the full-tail `verifyTailWords()` decode; the counters were only a proxy for
   it. A new `LaunchedEffect(currentTab, recitationMode)` ticks every 400 ms and stops the
   recitation as soon as an ABSENT verdict has held 1.5 s — no new emission required. The
   emission-counted path is kept as a secondary trigger (and no longer demands `isNewEmission`).
   Verification also now launches at the **first** sighting (+800 ms dwell) instead of the second,
   so the verdict lands at all.
2. **Row index ≠ word index.** `detailedAnalysis` holds ONE ROW PER (word, rule)
   (`analyzeRecitationDetails` adds one `WordAnalysis` per rule), so
   `indexOfFirst { status == WRONG || SKIPPED }` returns a **row** position that drifts ~6 by word
   20. That row index was being fed to every WORD-indexed structure:
   * `spokenTimings.firstOrNull { it.startIndex == idxToVerify }` — always missed, so the
     §4.10 anchoring fix silently fell back to the candidate clock and decoded the wrong audio;
   * `verifiedSpokenIndices.add(rowIdx)` — never matched any `wordIndex`, so the **false-alarm
     forgiveness that heals a false skip was completely dead**;
   * `confirmedMistakeUiIdx` — highlighted the wrong word.
   The effect now derives `mistakeWord = detailedAnalysis[mistakeIdx].wordIndex` and uses it
   everywhere; the row index is used only to read that row's own text.

The stop itself is hoisted into one `stopForMistake(wordIdx, skipped)` lambda so the emission path
and the ticker cannot diverge.

### 4.13 The stop works — now it must not stop on a CORRECT word

fatihah_c6 proved the mechanism: `matched=29/29`, `wrong=0 skipped=0`, popup visible. fatihah_c7
proved the other half of the requirement (*"we shouldn't get a skipped word at all"*) was still
missing — it auto-paused on `مَٰلِكِ`, a word the user read correctly:

```
verify idx=10 SKIPPED word='مَٰلِكِ' verified=3w found=false   -> INCONCLUSIVE
verify idx=10 SKIPPED word='مَٰلِكِ' verified=5w found=false   -> accepted
SKIPPED CONFIRMED wordIndex=10 word='مَٰلِكِ' -> auto-pause + popup
```

Three separate holes, all in the "absent ⇒ real mistake" inference:

1. **The span was anchored to a clock that does not exist.** The anchor was
   `spokenTimings.firstOrNull { startIndex == idxToVerify }` — but a SKIPPED word was, by
   definition, never heard, so it has **no timing** and the lookup always fell through to
   `lastWrongStopFirstSeenMs` (when the candidate appeared, i.e. *now*). The resulting ~12 s slice
   started **after** the word had been read. Now: anchor on the last word we *did* hear
   (`lastOrNull { startIndex <= idx }`) and, when the word has no timing of its own, use a
   **growing span** (20 s → 28 s → 36 s → 44 s) so every retry looks at MORE audio instead of
   re-decoding the same short slice.
2. **A 3–5 word decode was accepted as proof.** `saneHits == 0` was the only rejection test, so one
   shared token licensed a condemnation. Now the decode must be both large enough
   (`>= 8` words for a 29-word target, `>= 3` for short ranges) **and** share `>= 2` words with the
   ±8 neighbourhood; otherwise INCONCLUSIVE → retry.
3. **One sample is not evidence.** A **two-strike rule**: two independent decodes of *different*
   audio spans must both miss the word before it is condemned. Because the span grows between
   strikes, a word the decoder merely clipped is heard on the retry and forgiven.

Plus a decoder artifact that made the whole thing academic: the tail decode actually *did* contain
the word, as **`مك`** — the streaming model dropped the interior `ل` of `ملك`. `isWordMatch` rejected
it: for `length <= 3` it demands `dist == 1` **and** a prefix/suffix relation, and a middle deletion
has neither. It now also accepts a one-letter interior deletion when a `>= 3`-letter word leaves a
`>= 2`-letter remnant — deliberately NOT for two-letter particles (`من`/`عن`/`ما`), whose
substitutions must stay real errors.

### 4.14 The grace must not fall through to a summary

fatihah_c10 showed the stop/popup path is now safe (no false accusation) but the *fallback* was
still wrong:

```
verify idx=10 … (1w)  INCONCLUSIVE — tail decode (1w, need >=8)
verify idx=10 … (7w)  INCONCLUSIVE — tail decode (7w, need >=8)
verify idx=10 … (6w)  INCONCLUSIVE
verify idx=10 … (6w)  INCONCLUSIVE
verify idx=10 … (15w) found=true          <- forgiven
autoFinalize(): range complete (29/29 words decided)
endOfRange(): … wrong=0 skipped=2 tajweed=1
```

The 12 s grace expired with two SKIPPED words still outstanding and **finalized into the summary**,
which merely listed them. That is the user's complaint verbatim, one layer deeper: the mistake
pipeline had not accused anyone, so auto-finalize concluded the recitation had finished.

The grace expiry now branches on what the verification actually saw:

* **at least one substantial decode missing the word** (`mistakeVerifyAbsentStrikes >= 1`) ⇒ there is
  evidence ⇒ stop and raise the **MISTAKE popup** for the first flagged word. Never a summary.
* **otherwise** every decode was too short/garbled to judge ⇒ we cannot prove anything, and a false
  accusation is worse than a missed skip ⇒ **forgive** the word and let the session finish normally.

Two supporting changes:

* `MISTAKE_GRACE_MS` 12 s → **20 s**: two independent strikes are now required (§4.13), each costing
  1.5–3.6 s plus a cooldown, and at 12 s the second strike was still in flight when the grace fired.
* INCONCLUSIVE no longer clears the verify cooldown. Retrying immediately ran five decodes back to
  back; they share the recognizer, and the injected audio only played **32 s of its 46.5 s**, so the
  final words were never read at all — which is what produced the two "skipped" words in the first
  place. Retries now use the normal 3 s cooldown.

### 4.15 The verification loop starves the recognizer it is judging

The mistake-verification decode runs on the **same** `QuranSttEngine` as the live sliding
windows, and an **INCONCLUSIVE** result (decode too short, or fewer than 2 words shared with the
target neighbourhood) does **not** count as a strike (§4.13). So a candidate that never resolves
was retried forever — one decode every 3 s, for the rest of the session.

Measured baqarah_c14: ذَٰلِكَ was flagged at t+14 s, and four decodes were launched on it between
t+14 s and t+36 s of a 54.5 s recitation. The live windows starved and silently dropped six words.
That is the actual origin of the "skipped" words the mistake pipeline was chasing: **the detector
caused the defect it then tried to verify.**

Fix — a hard per-candidate budget, `MAX_MISTAKE_VERIFY_ATTEMPTS = 3`, added to the verify-launch
gate in the `mistakeStop` effect. 3 is the smallest number that can still reach the two-strike
verdict. Past it we stop spending the recognizer; the word is left **unproven** and the §4.14
grace-expiry branch **forgives** it. Worst case is a missed skip — never a false accusation.

### 4.16 The dropped-word cascade — the merge can only append

`monotonicMergeWords()` is documented as *"never reordered, shrunk or rewritten — only the
not-yet-shown tail of `n` is appended"*. Every repair branch that breaks that rule assumes the
session-covering decode is **not longer** than the live log:

| branch | size test |
|---|---|
| full re-render (anchor 0) | `n.size == L` |
| FINAL-decode repair | `n.size <= L + 2` |
| head-anchored log poison | `n.size <= L` |
| FINAL-decode prefix trim | `n.size` unconstrained but `bestAnchor in 1..2` |
| `sameWordBagFuzzy` | tolerates only **2 unpaired decode** words |

So the single most common long-audio failure had **no branch at all**: the streaming windows drop
words, the log ends up strictly **shorter** than the whole-session decode, every size test fails,
and the merge bolts the missing words onto the **tail — out of reading order**. The aligner's
reorder buffer has a 3-target budget, so losing 6 words freezes the walk permanently.

Measured baqarah_c17 (Al-Baqarah 1–5, 36 targets, 54.5 s):

```
live log 30w:  الم ٱلْكِتَٰبُ لَا ريب فِيهِ هُدًى للمتقين وَمَآ(!) بِالْغَيْبِ …
               … ٱلصَّلَوٰةَ وَمِمَّا ٱلْمُفْلِحُونَ(!) بِمَا أُنْزِلَ إِلَيْكَ …
54.5 s decode 37w, correct:  الم ذَلِكَ الْكِتَابُ لَا رَيْبَ فِيهِ هُدًى لِلْمُتَّقِينَ الَّذِينَ …
merge appends:  … هُمُ ذَلِكَ الَّذِينَ يُؤْمِنُونَ رَزَقْنَٰهُمْ يُنفِقُونَ وَالَّذِينَ يُؤْمِنُونَ
result:         walk freezes at matched=9, session finalizes 9/36
```

Three separate things had to be fixed, and each was invisible until the previous one was:

1. **A branch for the superset case.** New: when `fullReRender && coversSession`, the decode is
   longer than the log (`n.size > L`), and **every log word is accounted for inside the decode**
   (fuzzy ≤1 edit, at most `maxOf(2, L/10)` unpaired — window-edge phantoms), the decode heard
   everything the log heard and more, in the right order ⇒ adopt it. New helper
   `logCoveredByDecode()` — the mirror of `sameWordBagFuzzy()`: iterate the **log**, match into
   the **decode**. Order-blind on purpose; the decode's order is the trustworthy one.
   *Do not* use an ordered-subsequence test — the live log has displaced words (وَمَآ at slot 8
   in c17) and the chain breaks immediately.
2. **The head check must tolerate a junk leading token.** `spansLogHead` requires the decode to
   contain `lastNorm.first()`. Measured baqarah_c18: the live log **opened with the garbage
   phantom الاسني**, so `head=false` and the repair was skipped on a run that needed it. Now the
   decode only needs to contain **one of the log's first three** words (`headCovered`) — still
   rejects a head-truncated decode, tolerant of one junk token. The tail is not checked at all:
   `logCoveredByDecode` already forbids more than a couple of unpaired log words.
3. **`repaired` must be true when the log GREW.** The collector's FINAL-DECODE REPAIR SWAP (the
   only thing that actually overwrites the scrambled transcript) is gated on `ev.repaired`, which
   was computed as *"the append-only merge only grows the log — a same-size or smaller result
   means it was replaced"*. A superset repair legitimately **grows** it, so `repaired` was false,
   the swap was skipped, the new words were appended to the scrambled log and the walk froze at
   11/36 — with `superset repair … adopting the decode's order` sitting right there in the log.
   Now `repaired = merged != last && (merged.size <= prev || merged == cleaned)`.

### 4.17 The end-of-range popup lost a race with `finalizeCorrect()` — and never retried

Measured fatihah_c23 (`01:15:13`, 42 ms apart):

```
.148 autoFinalize?(): rows=28 decided=25/29 unheardPending=1
.149 autoFinalize(): range complete (29/29 words decided) -- running the finalize path
.162 endOfRange(): sessionFinalized=true pendingRemain=true  ... tajweed=1   <-- popup SKIPPED
.204 finalizeCorrect(): 1 trailing PENDING word(s) counted CORRECT (matched=27, total pending=1)
```

No `Dialog` line ever followed — the user got **nothing on screen** for the whole session.
c22/c24/c25 (which did show the popup) have `pendingRemain=false`; c23 is the same code with
one unheard trailing word.

Cause: `runStopAndFinalize()` sets `sessionFinalized` and reads `detailedAnalysis` **in the same
pass**, while `finalizeCorrect()` is a separate `LaunchedEffect` that only runs on the **next**
recomposition. So the popup gate saw a PENDING row that was about to be turned CORRECT 42 ms
later, vetoed the popup — and because the gate is one-shot inside the click/finalize lambda,
nothing ever re-evaluated it after the conversion.

Note the ordering also means the veto fired *before* the forgiveness it depends on, so the
`pendingRemain` guard was suppressing the popup on evidence that was already stale.

Fix: `endOfRange` now applies `finalizeCorrect()`'s **own** forgiveness window when it counts
pending rows —

```kotlin
val matchedSoFarNow = detailedAnalysis.count { CORRECT || TAJWEED_ERROR || HINTED }
val forgivableTail  = matchedSoFarNow + 8
val pendingRemain   = detailedAnalysis.any {
    it.status == PENDING && it.spokenText.isNullOrEmpty() &&
    (matchedSoFarNow < 3 || it.wordIndex > forgivableTail)
}
```

— so the two readers agree on what "still pending" means even when they look one recomposition
apart. A row inside the window is about to be counted CORRECT and can no longer veto the popup.

### 4.18 The FORGIVE branch re-armed its own grace window — an infinite loop, no popup ever

Fixed §4.17, re-ran, and got a *different* failure on the same screen. fatihah_c27:

```
01:43:36.197  autoFinalize(): holding 0ms/20000     (decided=29/29 unheard=0 mistake=true)
01:43:38.199  autoFinalize(): holding 2002ms/20000
   … 20 s of holding …
01:43:56.239  autoFinalize(): grace expired with an UNPROVEN mistake at wordIndex=20
              (no substantial decode ever judged it) -> forgiving rather than accusing
01:43:58.247  autoFinalize(): holding 0ms/20000     <-- window STARTED OVER
01:44:00.249  autoFinalize(): holding 2002ms/20000
```

`autoFinalizeDone` was never set, `runStopAndFinalize()` was **never called**, the walk stayed
frozen at 20/29 and the screen showed nothing until the harness's 40 s tail ran out. The user
gets exactly the "nothing happened" outcome §4.14 exists to prevent.

Cause: the forgive branch cleared only `mistakeHoldSinceMs`. That resets the **countdown**, but
`mistakeOutstanding` counts **rows**, and `verifiedSpokenIndices.add(...)` suppresses only the
*candidate*, not the row — the flagged word kept `status = SKIPPED`. So the next pass saw the
same outstanding mistake, found a zeroed clock, and started another 20 s window. The branch was
re-arming the very timer it was supposed to retire.

Fix (three parts, all needed):

1. **Erase the row**, don't whitelist the word —
   `w.copy(status = CORRECT, spokenText = null)` for that one `wordIndex`. This is what
   `mistakeOutstanding` actually counts, and it is the same thing `finalizeCorrect()` does for a
   trailing PENDING word. Only the unproven row is touched; a **proven** mistake still stops
   with the popup on the branch above.
2. **`graceExpiredOnce`** — a session-scoped latch, set the moment the window expires and
   re-armed only when the session goes live again. The hold condition becomes
   `held < MISTAKE_GRACE_MS && !graceExpiredOnce`, so the clock can never restart mid-session
   and the fork below is guaranteed to reach a real decision.
3. **Stop resetting `mistakeHoldSinceMs`** in the forgive branch — the expired clock is left in
   place on purpose, so even if an erase somehow fails to take, the next pass falls straight
   through to the finalize instead of looping.

Verified: fatihah_c28 — `range complete (29/29 words decided)`, `pendingRemain=false`,
`END-OF-RANGE POPUP VISIBLE`, `wrong=0 skipped=0`, **0** verification decodes,
**0** migration errors.

#### The three ways this popup could vanish (read this before debugging "no popup")

All three produced the same empty screen and were fixed in different places:

| Run | Log signature | Real cause | Fix |
| --- | --- | --- | --- |
| c23 | `endOfRange(): … pendingRemain=true` then `finalizeCorrect(): 1 trailing PENDING … CORRECT` 42 ms later | popup gate read `detailedAnalysis` one recomposition **before** the forgiveness pass | §4.17 — apply `finalizeCorrect()`'s `matched+8` window in the gate |
| c27 | `grace expired … forgiving` then `holding 0ms/20000` **again** | FORGIVE resets the countdown but never clears the row, so `mistakeOutstanding` stays true forever | §4.18 — erase the row + `graceExpiredOnce` latch |
| — | `range complete` never logged at all | `autoFinalize` held on `mistake=true` / `unheard` | §4.14 / §4.15 — bounded grace + verify-attempt cap |

The tell for c27 vs c23 is the second `holding 0ms` — a **repeat** of the countdown start means
the window was re-armed, which only the forgive branch could do.

### 4.19 The anchor-less repair was guarded by size — one word short and it never ran

fatihah_c29 (45 s, 29 targets) showed the popup — so §4.17 and §4.18 held — but it read
**15/29** on a correct recitation. `mergeDiag`:

```
mergeDiag L=26 n=25 vocab=26 head=false headCovered=true end=true anchor=-1 consumed=0 covered=false bag=false
    log='بسم الله الرحمن الرحيم الحمد رب العلمين لله الرحيم الرحمن يوم الدين نعبد واياك نستعين اهدنا الصرط المستقين …'
    dec='م الله الحمد رب العلمين الرحمن الرح يوم الدين اياك نعبد واياك نستعين اهدنا الصرط المستقيم صرط … الضالين'
finalDecodeAndEmit(): done 8039ms new=2 w emitted=28w
```

`anchor=-1` is the signal: the session decode re-heard the whole range in one pass and shared
**no** usable alignment with the log — the single strongest evidence that the live log is
poisoned. And it is.

The decode is genuinely better here. It **reaches past the log** (… ولا **الضالين**), so it
heard words no window ever emitted, and it drops the log's window-edge duplicates
(`الرحيم الرحمن` re-appended after `لله`). Nothing was repaired, though, because two guards
closed by a single word each:

| guard | required | actual |
| --- | --- | --- |
| anchor/poison branch size | `n.size >= L - 2` → **24** | `n = 25`… but `sizeOk` also demanded `n*2 >= L` and the pair was gated with a flat tolerance |
| `sameWordBagFuzzy` | ≤ **2** unpaired decode words | **4** |

Why 4 unpaired? Two CTC **fusions**, each costing exactly one unpaired word:

* `الصرط` + `المستقيم` (two log words) → `الصرط_المستقيم` (one decode word)
* `الرحمن` + `الرح` (two log words) → `الرَّح*` (one decode word)

plus the two words the windows genuinely dropped (`بسم`, `الرحيم`). The size guard and the bag
guard therefore have to loosen **together** — loosening only the size leaves the branch
unreachable in exactly the case it was written for.

Fixes:

1. **`forwardTolerant = (n + 4 >= L && n >= L - 2) || n <= L + 2`** and
   `sizeOk = n + 4 >= L && n * 2 >= L` when the decode covers the session.
2. **`sameWordBagFuzzy` tolerance is proportional: `maxOf(2, log/6)`** — 4 at 26 words, 5 at 30.
   Still far too tight for a *different* recitation: An-Nās and Al-Ikhlāṣ both score 5 unpaired
   against a Fatihah log and are correctly rejected.
3. **`headCovered` (one of the log's first three words) replaces strict `spansLogHead`** on this
   branch too — `spansLogHead` was **false** because the decode opened with the fragment `م`
   (all that survived of `بِسْمِ ٱللَّهِ`), so the strict test blocked the repair on a run that
   needed it. Same trap that forced `headCovered` on the superset branch (§4.16).
4. **`spansToLogEnd` added to the gate.** Growth and truncation are now distinguished by
   *reaching the log's last word*, not by size — a decode that simply stopped early is not a
   tidier transcript, it is a decode that never heard the tail. Without this check the loosened
   size guard would have let a 20-word truncation of a 26-word log replace it, deleting six
   real words.

`MonotonicMergeTest` gained two tests pinned to this exact data — the positive case
(`dec` must win, `الضالين` must be last) and the negative control (a 20-word truncation must
**not** win, `ولا` must survive). Both pass; the class is now `@RunWith(RobolectricTestRunner)`
because the repair branches log through `android.util.Log`.

**Gotcha for anyone testing this:** `monotonicMergeWords(…, vocab = null)` **skips the entire
anchor scan** (`maxK` is 0 when `vocab == null`), so every `bestAnchor`-keyed repair is dead and
the merge just appends. A regression test for these branches MUST pass a realistic `vocab`, or
it silently tests the append path instead.

### 4.20 The summary could still bury a mistake — the two entry points were indistinguishable

The user's contract, stated repeatedly and now enforced structurally:

> *"the summary is when i click on pause the mic or finish my recitation. for any mistake, the
> app stops there and give me a mistake popup. i made this clear many times."*

`runStopAndFinalize` is called from **two** places — the mic FAB (`onClick`) and the end-of-range
effect — and both ran the same body, which raised `sessionSummaryToShow` whenever
`sessionFinalized && !pendingRemain`. Nothing in that body knew *why* it had been called, so the
auto path could latch the summary on a recitation that still had `wrong=1` / `skipped=1`
flagged, and the mistake popup the user was waiting for never appeared. Every earlier fix
(§4.11–§4.18) attacked the *timing* of the mistake pipeline; this one attacked the **precedence**
between the two dialogs, which is the part the user actually described.

Fix — `runStopAndFinalize(userInitiatedStop: Boolean)`:

* **Mic tap → `true`.** The user asked to stop, so the summary is the correct response and
  flagged words are simply listed inside it. No mistake popup is forced on them.
* **End-of-range → `false`.** A real, unresolved mistake must win:
  `mistakeMustWin = !userInitiatedStop && (wrong > 0 || skipped > 0)` suppresses the summary
  entirely, leaving the mistake popup as the only thing on screen.

This is safe precisely because the auto path only reaches `runStopAndFinalize(false)` **after**
`autoFinalize` has already settled every outstanding mistake — either by stopping for it
(`stopForMistake`, popup) or by forgiving a word no decode could judge (§4.18). So a mistake
still flagged at that point is one the pipeline genuinely failed to resolve, which is exactly
when the contract says the popup must take over.

### 4.21 ROOT CAUSE: the candidate's dwell clock was reset forever, so no word was ever verified

Everything above treats *symptoms*. fatihah_c30 finally exposed the mechanism, and it is a single
inverted condition:

```
20:50:10.657  candidate row=11 word=10 sightings=1 absentIdx=-1 ... sinceVerify=…  jobActive=false
20:50:11.566  candidate row=10 word=10 sightings=1 absentIdx=-1 ... sinceVerify=909ms  jobActive=true
20:50:13.361  candidate row=11 word=10 sightings=1 absentIdx=-1 ... sinceVerify=224ms  jobActive=false
20:50:14.163  candidate row=10 word=10 sightings=1 absentIdx=-1 ... sinceVerify=1026ms jobActive=false
...
20:50:30.290  candidate row=10 word=10 sightings=5 absentIdx=-1 ... sinceVerify=2554ms jobActive=false
20:50:30.422  stallWatchdog(): candidate idx=10 unresolved for 20017ms -> forgiving
```

Note `sightings=1` even after twenty seconds, `mistakeVerifyAttempts` never advancing, and
**zero `verify idx=` lines in the entire run.** The candidate was flagged on a word the user
demonstrably recited, and it was never once tested.

Two defects compound here:

1. **The candidate-identity branch re-ran on every recomposition.** Its guard is
   `!sameEpisode && mistakeWord != lastWrongStopIndex` — but both `mistakeEpisodeAnchor` and
   `lastWrongStopIndex` are assigned `mistakeWord` inside that very branch, *and the branch then
   sets `mistakeEpisodeAnchor = -1` on its last two lines*. So `sameEpisode` was permanently
   false, `mistakeWord != lastWrongStopIndex` stayed true, and the branch re-executed endlessly.
   Its first statement is `lastWrongStopFirstSeenMs = System.currentTimeMillis()`.

   The dwell the verification needs is
   `System.currentTimeMillis() - lastWrongStopFirstSeenMs >= 800L` (plus
   `lastWrongStopSightings >= 1`). Both pre-conditions were reset by every pass, so the decode
   was **never launched**. `mistakeStop()` "fired at the first sighting" on paper and never once
   in practice.

2. **The stall watchdog then fabricated the verdict.** After 20 s it added the word to
   `verifiedSpokenIndices` and moved on. With `mistakeVerifyAbsentStrikes == 0` — because no
   decode had run — `autoFinalize` classified it as an "UNPROVEN mistake" and *forgave* it
   (§4.18). `mistakeOutstanding` then read false, `cleanFinish` read true, and the user got a
   **summary**: no skipping, no wrong word, no popup.

   That is precisely the report — *"there is no skipping, there is no wrong words"* on a correct
   recitation. The row is only in the display list at all because `detailedAnalysis` carries one
   row per (word, rule), so a PENDING/forgiven trailing word still shows; the walk itself was
   frozen because the aligner had no verdict to advance on.

Fixes, all three needed:

* **`else if (mistakeCleanStreak > 0 || mistakeEpisodeAnchor < 0)` — a STABLE RE-ENTRY branch.**
  When `mistakeWord` matches the candidate already being tracked but the per-recomposition state
  describes it as new, re-seat the identity **without touching
  `lastWrongStopFirstSeenMs`**. The dwell clock now survives a flicker, so the verification
  decode actually launches. `mistakeEpisodeAnchor = -1` was also removed from the new-candidate
  branch — that line, not the guard, was what made `sameEpisode` unreachable.
* **Last-chance verification in `stallWatchdog`.** Before forgiving, if
  `mistakeVerifyAttempts == 0 && mistakeVerifyAbsentIdx < 0 && mistakeVerifyAbsentStrikes == 0`,
  the watchdog hands the confirm loop the dwell it never reached
  (`lastWrongStopFirstSeenMs = now - 800`) and loops. At most once per stall, and only when
  nothing was spent — so a word that really *is* a skip now gets an absent strike, `autoFinalize`
  finds evidence, and the **MISTAKE popup** is raised instead of a summary.
* **`unverdictedStall` blocks a clean finish.** `cleanFinish` now also requires
  `!(lastWrongStopIndex >= 0 && not in verifiedSpokenIndices && strikes == 0)`, so a word that
  was only ever *guessed* to be fine cannot be what lets the range be declared complete.

The lesson generalises: **every "forgive" path must be gated on evidence, and every evidence path
must be reachable.** Two independent resets of the same clock turned "verify then forgive" into
"never verify, always forgive", and the forgiving looked like a clean recitation.

### 4.22 A repair that REPLACES the log must not be allowed to lose a word

fatihah_c32 froze at 23/29 with the popup — and this time the merge's own diagnostics were
enough to see why, because the decode was a **trade**, not an improvement:

```
mergeDiag L=25 n=24 vocab=26 head=false headCovered=true end=true anchor=-1 covered=false bag=false
    log='… اهدنا الصرط المستقيم عليهم غير المغضوب عليهم ولا الضالين'      <- صرط الذين انعمت MISSING
    dec='… اهدنا الصرط المستقيم صرط الذين انعمت عليهم المغضوب عليهم ولا الضالين'  <- غير MISSING
```

Pairing the two word by word makes the trade explicit:

| side | orphans |
| --- | --- |
| log words with no decode partner (6) | `بسم` `الرحيم` `لله` `الرحيم` `الرحمن` `غير` |
| decode words with no log partner (5) | `م` `الرح` `صرط` `الذين` `انعمت` |

The first five log orphans are window-edge duplicates/fragments the decode *correctly* refused to
re-hear. **`غير` is not** — it is a real word of the āyah that the decode dropped, and the user
pronounced it. Meanwhile `صرط الذين انعمت` are three real words **no window ever emitted**.

Two consequences, and both needed fixing:

1. **`logCoveredByDecode`'s budget was too tight.** `maxOf(2, log/10)` allowed 2 orphans for a
   25-word log; this decode orphans 6. Raised to `maxOf(2, log/4)`, calibrated against real
   positives *and* negatives rather than guessed:

   | case | L | n | unpaired | verdict |
   | --- | --- | --- | --- | --- |
   | c29 real | 26 | 25 | 5 | accept |
   | c32 real | 25 | 24 | 6 | accept |
   | An-Nās (wrong surah) | 26 | 20 | 7 | reject |
   | Al-Ikhlāṣ (wrong surah) | 26 | 15 | 7 | reject |
   | c32 truncated | 25 | 12 | 7 | reject |

   The margin is **one word wide**. That is uncomfortably thin, so the strong guards are the
   caller's `headCovered`, the tail span and the size margin — this budget only rejects grossly
   wrong decodes. `sameWordBagFuzzy` keeps its own `log/6`, so the two checks can no longer
   disagree about a decode they both examine without one of them being deliberately looser.

2. **A replace is only safe when the decode contains everything real the log did.** It did not,
   so adopting `n` wholesale would have traded three false skips for one real one. The repair now
   pairs every log word with its decode counterpart, and **re-injects each orphaned log word at
   its log-derived position** — immediately before the decode word that matched its successor in
   the log:

   ```kotlin
   val successor = (j + 1 until L).firstOrNull { logPartner[it] >= 0 }
   if (successor != null && logPartner[successor] == d) out.add(last[j])
   ```

   Both sides keep everything they heard: the decode's reading order wins, the three restored
   words appear, and `غير` stays between `عليهم` and `المغضوب` where it belongs.

The general rule this run establishes: **when the app chooses between two transcripts, it must
never silently drop a word that only one of them has.** Replacing wholesale is a heuristic about
which transcript is better; re-injecting the survivors is a statement about what was actually
recited.

### 4.23 The rejected repair threw away the only copy of the word — and the analyzer left the last word row-less

fatihah_c33 produced the user's report verbatim: *"the last word is not displayed and yet i dont
have a mistake popup but have a summary popup indicating no mistakes. i shoud either have a
mistake popup and mistakes in the summary or just display the word."* Two independent bugs had to
line up for that, and both were real.

**Bug 1 — the collector discarded the endpoint's delta.** The tail windows only ever rendered the
last word as window-edge fragments (`وَلَاضًّا`, `الضَّا`, `ض م`), all rejected by the merge, so the
live log stopped one word short at `وَلَا`:

```
21:31:57.458 collector(): ev.endpoint=true words=[] partialStr=''
             ^ stop()'s synthetic endpoint commits the 28-word live tail
21:32:07.095 finalDecodeAndEmit(): done 9271ms new=-4 w emitted=24w raw='م اللَّهِ الحمد لِ رَ …'
21:32:07.097 collector(): ev.endpoint=true words=[ٱلضَّآلِّينَ] partialStr='ٱلضَّآلِّينَ'
21:32:07.098 collector(): final-decode repair REJECTED (repair 24w/prefix0 vs live 28w/prefix8) — keeping live transcript
```

Rejecting the repair is **correct** — the 24-word render is a garbled ordering and must not replace
a live transcript with an 8-word correct prefix. But the *delta* (`words=[ٱلضَّآلِّينَ]`) is computed
against the **committed** log, so anything that survives it is by construction a word **no sliding
window ever rendered**. It was thrown away together with the repair: one word of real speech, gone
with no trace, which is exactly what §4.22's rule forbids ("never silently drop a word that only one
of them has").

Fix: after committing the uncommitted live tail, **recompute** the delta against the now-committed
log and append whatever survives:

```kotlin
val recovered = filterMemorizeWords(ev.partial, endpoint = true)
if (recovered.isNotEmpty()) { finalSpokenText = "$finalSpokenText ${recovered.joinToString(" ")}".trim() }
```

Recomputing *after* the tail commit is what keeps the recovered word from being counted twice (the
same count-aware filter that governs every other emission now sees the tail as committed).

**Bug 2 — the analyzer left the last target word with no row at all.** `analyzeRecitationDetails`'s
walk can only emit a row while it still has a spoken word to judge; when the transcript ran out it
`break`ed outright:

```kotlin
if (spokenIndex >= spoken.size) break      // every remaining target gets NO row
```

With 28 spoken words against a 29-word target, index 28 got nothing — and every downstream consumer
counts **rows**:

| consumer | saw | should have seen |
| --- | --- | --- |
| `autoFinalize?(): rows=32 decided=28/29` | 28 distinct indices | 29 |
| `unheardPending=0` | nothing to count | 1 |
| `finalizeCorrect()` | **no log line at all** — no row to forgive | 1 converted |
| UI colouring (`globalWordIndex < matchedWordCount`) | 25 %-opacity fallback = "not recited" | coloured |
| `endOfRange(): wrong=0 skipped=0` | clean summary | — |

So the word was at once invisible (no row), uncoloured (past `matched`), unforgivable (no row) and
unaccused (no row to flag), and the summary claimed a clean recitation. Fixed by filling PENDING
rows (`spokenText = null`) for every unreached target word, exactly like the frontier case:
**every target word always has a row**, unreached rows stay PENDING so they cannot advance
`matched`, and `finalizeCorrect()` / the mistake pipeline can finally see them.

This is the structural invariant that makes the user's contract hold on *every* path: a target word
is either displayed, or flagged, or forgiven **with a trace** — never silently absent.

Both fixes are pinned by `MemorizeTranscriptTrackerTest.rejectedRepair_stillContributesWordsTheLiveLogNeverHeard`
(+ the `doesNotDoubleCount` variant, which exercises the no-synthetic-endpoint ordering) and by the
new `AnalyzerRowCompletenessTest` (`everyTargetWordHasARow_evenWhenTheTranscriptIsShort`,
`anEmptyTranscriptStillYieldsEveryRow`, `aFullyRecitedTargetHasNoPendingRows`).

### 4.24 A repair the collector could not recognise, and an orphan budget that could not see past `L/4`

fatihah_c34 exposed three more places where a *correct* decision was being discarded.

**1. `repaired` was inferred from the result's SHAPE — and the splice has neither shape.**
`finalDecodeAndEmit()` decided whether the merge had produced an authoritative transcript with

```kotlin
val replacedByDecode = mergedClean == cleaned
val repaired = mergedClean != lastEmitClean &&
    (mergedClean.size <= prevCleanSize || replacedByDecode)
```

which only recognises "the log shrank" or "the merge handed back the decode itself". The
superset repair with orphan re-injection (§4.22) returns the decode's order **plus spliced log
survivors**, so it is neither — and when it also GROWS the log, `repaired=false`, the collector's
REPAIR SWAP is skipped, and the delta is appended on top of a transcript that had already been
reordered. `monotonicMergeWords` now **reports the branch it took** through an `onRepair`
callback, and `mergeClean` records it in `mergeWasRepair` for `finalDecodeAndEmit` to read
(immediately — `mergeRaw` overwrites it). The two shape tests stay as a net.

**2. The orphan budget was flat `L/4`, but a decode LONGER than the log cannot be a truncation.**
fatihah_c34: `mergeDiag L=21 n=24 anchor=-1 covered=false bag=false` — the decode restored
`نعبد اهدنا الصرط المستقيم` (which no window emitted) and the range's last word, but dropped
seven window-edge duplicates against a budget of `21/4 = 5`. `covered=false` blocked the repair,
the merge appended its eight new words at the **tail**, and the walk froze at 10/29.

The only failure mode a count budget has to catch is a decode of a **different recitation**, and
that is already `headCovered`'s job (one of the log's first three words must appear in the decode);
every leftover log word is **re-injected**, so a false accept cannot lose a word. So the budget is
now conditional:

| decode vs log | budget | rationale |
| --- | --- | --- |
| `n <= L` | `maxOf(2, L/4)` | a short decode can be a TRUNCATION — keep the calibrated budget |
| `n > L` | `maxOf(3, L/2)` | longer than the log ⇒ cannot be a truncation; `headCovered` rejects a different recitation |

Negative control added: `a longer decode of a different recitation is still rejected`.

**3. The collector scored candidates by their longest leading RUN — too harsh for a missing early
word.** c34's live log scored a prefix of 3 while the whole-session repair scored 0 (its opening
token was the fragment `م`), so the repair was rejected even though it held 27 of the 29 target
words in reading order against the log's 20. The score is now the **longest common subsequence with
the target** — how many target words the transcript holds *in order* — with the leading run as the
tie-break (so the original "correct-but-shorter repair replaces a misordered log" case still
wins), then size:

| case | live | repair | winner |
| --- | --- | --- | --- |
| c34 (log missing 8 mid-range words) | 20 | 27 | repair ✓ |
| c33 (repair is a worse ordering) | 28 | 22 | live ✓ |
| An-Nās / truncated controls | high | low | live ✓ |
| misordered 8w log vs shorter in-order repair | 7 (run 4) | 7 (run 7) | repair ✓ |

**4. One premature grace expiry disabled the end-of-range popup for the whole session.** When the
auto path reached `runStopAndFinalize(false)` and `pendingRemain` withheld the summary (the range
genuinely is not complete — the user is most likely still reading), `autoFinalizeDone` was already
set to `true`, so the end-of-range path never fired again even after the user finished the range.
`runStopAndFinalize` now reports the withholding (`finalizeWithheld`), and the gate re-arms
(`autoFinalizeDone = false`, `graceExpiredOnce = false`) so a later completion still fires.

### Verified on device

Three consecutive Al-Fatihah injection runs on the built + installed APK:

| run | result | note |
| --- | --- | --- |
| c35 | PASS 29/29, popup visible | repair REJECTED; the live log already held all 29 |
| c36 | **PASS 29/29, popup visible** | **§4.23's fix in action** — `rejected repair still contributed 1 word(s) the live log never heard -> transcript now 29w`, `REJECTED (repair 25w/prefix24 vs live 28w/prefix26)`, `endOfRange(): pendingRemain=false newCompleted=7 wrong=0 skipped=0` |
| c37 | PASS 29/29, popup visible | one candidate (word 15) verified → `found=true` → cleared; no false accusation |
| c38 | PASS 29/29, popup visible | nothing flagged at all |

c36 is the decisive run: it is exactly the c33 shape (live log one word short, whole-session repair
correctly rejected) and the recovered word pushed the transcript to 29 words, so `matched=29`, the
display transcript ends `… وَلَا ٱلضَّآلِّينَ` — **the last word is displayed** — and the summary's
"no mistakes" is now TRUE. On-device UI dump of the last run confirms the end-of-range popup:
"Masha'Allah — no mistakes in this recitation." / "Perfect recitation!" / "Surah 1, Ayahs 1 - 7".

A note on the harness output: `[3b] coversSession: no (tail slice only)` is a **misleading harness
inference**. `window=45,0s` with `FINAL_DECODE_MAX_SEC = 90` means `fromIdx == 0`, i.e. the decode
DID cover the session (and `mergeDiag` only prints when `coversSession`). And `[3] FINAL TRANSCRIPT`
prints the last `collector()` DELTA, not the transcript — read `partialStr` for the display text.

### Al-Baqarah 1–5 (36 words) — also clean

Same build, `baqarah_start.wav` (55.25 s), `SURAH_SEARCH=Baqara SURAH_LABEL=Al-Baqarah
RANGE_START=1 RANGE_END=5`:

| run | result | finalize path |
| --- | --- | --- |
| c1 | PASS 36/36, popup visible | `final-decode repair swap (36w)`, `newCompleted=5 wrong=0 skipped=0` |
| c2 | PASS 36/36, popup visible | `final-decode repair swap (36w)`, same |
| c3 | PASS 36/36, popup visible | `final-decode repair swap (37w)`, same |

`confirmed (auto-pause + popup): 0` in all three — zero false accusations. On-device UI dump:
`Surah 2, Ayahs 1 - 5` / "Masha'Allah — no mistakes in this recitation."

**c3 is the strongest evidence yet that §4.24's `mergeWasRepair` was load-bearing:**

```
mergeDiag L=30 n=36 vocab=31 head=false headCovered=true end=true anchor=-1 covered=true bag=false
superset repair — session decode 36w covers the 30w live log; adopting its order and
    re-injecting 1 log-only word(s): الْأَسْنَى
final-decode repair swap (37w)
```

`merged(37) != cleaned(36)` and `37 > prevCleanSize(30)`, so the OLD shape-based rule computed
`repaired = false` — the collector would have APPENDED the 7-word delta onto the 30-word log
instead of swapping, producing exactly the out-of-order transcript §4.16 describes. With the
merge's own branch report it swaps and matches 36/36.

Two things this run leaves on the table (neither causes a failure, so they were NOT changed on a
verified build):

1. The orphan re-injection can re-inject an **OOV garbage** log word (`الْأَسْنَى`, a CTC wreck of
   `الم`) because it only tests "has no decode partner", not "is in the target vocabulary". The
   walk's OOV-garbage skip absorbs it, so all 36 still matched — but the transcript is target+1
   words. Skipping non-target orphans would be tidier.
2. The repair-swap path deliberately does NOT run `filterMemorizeWords` (its committed-strip would
   truncate the transcript it is replacing), so an over-length swap is possible. Benign while the
   extra word is at the edge or OOV.

And the standing weakness, now measured on Baqarah too: **`verifyTailWords` starvation.** c2's
`verify idx=1` returned 6 / 7 / 4 words for 10 / 15 / 21 s spans against `need >= 8` → three
INCONCLUSIVEs, so the candidate lingered until the final-decode swap cleared it; c3 returned
5 / 11 / 11 and logged `ABSENT strike 1/2` (never confirmed). No false accusation either way, but
each doubt burns ~20 s and delays resolution.

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
