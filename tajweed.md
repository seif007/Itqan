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

**Memorize + Tarteel (the default):** word-tracking runs on the platform Android
`SpeechRecognizer` (`recognitionMode == "Tarteel"`), so its recognition accuracy is exactly what
that recognizer gives — no tajweed analysis is ever engaged, so no tajweed rule, score or popup can
appear. It does get the *same session contract* as Memorize + Tajweed: on a wrong or skipped word
the walk freezes at that word and raises the same must-dismiss popup, the display rewinds to it, a
pause shows the honest "Recitation Paused" summary, finishing the range shows the completion
summary, and dismissing the popup lets the next mic tap continue from the flagged word. Because
there is no pipeline and no tail decode to arbitrate, **the freeze itself is the verdict** — in
particular a skipped word is confirmed immediately (a jump is visible to the reader), not after a
multi-emission wait. See §4.58–§4.65 for the parity work and §4.69 for the cursor-pinning defect
that made Tarteel display words past a mistake.

**Memorize + Tajweed:** word-tracking runs on the **Sherpa-ONNX** offline
recognizer (`SherpaPipeline`): a single `AudioRecord` feeds both the STT
stream (RMS-gated buffer) and the TajweedNet scorer sink (un-gated buffer).
`analyzeRecitationDetails` ALSO passes the per-word audio to
`RecitationAnalyzer.analyze(..., [rule], threshold)` for any word that the
rule-detector tagged. When the model judges `P(correct) < threshold` (after
the bounded confirmation search, §1 step 5):

- the word's `status` becomes `TAJWEED_ERROR`,
- one dialog at a time is raised for that `(word, rule)` pair — "**Tajweed Mistake — <rule>**, your
  <rule> score: N%" — and the mic auto-stops so the user can read the feedback; the popup is NEVER
  auto-dismissed, and `shownTajweedPopups` (a SET of pairs) means each verdict speaks at most once
  per session and two standing mistakes cannot take turns (§4.53); the re-read re-arms it,
- a verdict is only spoken if it is ISOLATED — a failing neighbour within `TAJWEED_ISOLATION_SLACK`
  (= 2) words means the scorer's cross-word boundary is offset, not a mistake (§4.55); and one
  acoustic pass scores at most `TAJWEED_WORDS_PER_PASS` (= 6) words (§4.56),
- stopping rewinds the display to the flagged word IN THE SAME INSTANT (§4.52) and arms the same
  mistake-resume re-seat as WRONG/SKIPPED: tapping the mic truncates the display transcript to the
  prefix BEFORE that word, so the re-read passes the collector's count-filter (otherwise it is
  swallowed as an "already committed" replay copy and nothing shows), gets displayed, gets freshly
  scored (verdict-cache key includes the utterance timing), and re-popups if still wrong,
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
candidate only becomes a *punished* mistake after three gates. These gates describe the surface
that HAS a confirm loop — **Memorize + Tajweed**. There, a tentative candidate is RIDDEN OVER (the
walk keeps tracking the voice) for at most `RIDE_OVER_BUDGET_MS` (= 8 s, §4.67); if the gates below
resolve it as absent the mistake pops, and if they never resolve, the budget expiry makes the app
the decider and it pops anyway. On **Tarteel** there is no confirm loop, so the walk's freeze IS the
verdict: skip AND wrong raise the popup immediately from that freeze (§4.58/§4.61).

1. **Multi-emission confirmation:** the SAME first-mistake index must survive
   distinct emissions — 2 for WRONG, 3 for SKIPPED (plus ≥2 s dwell and the
   transcript advancing ≥3 words past the skip). The visible red word highlight
   uses this same evidence (`confirmedMistakeUiIdx`) so a transient
   single-window decode artifact never flashes — and **§4.57**: a candidate is
   released only when its WORD has really left the analysis (a re-read, a range
   change, a re-seat); a brief absence no longer wipes the sightings, dwell and
   absent strikes, which is what used to make a real mistake unprovable.
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

**Verified-only display.** The popup + red word appear ONLY after
the tail verification says the word truly was not said. Transient candidates
stay invisible while self-heal has a chance; "only true mistakes should be seen
by the user". On a surface with no confirm loop (Tarteel) the walk's freeze is
the decision itself — see §4.58/§4.61.

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
[`generateRandomRecitationExample(head)`](app/src/main/java/com/example/ui/RecitationApp.kt), which:

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

1. **Wrong rule set (missing *and* phantom rows).** `analyzeRecitationDetails` scores ONE
   rule per word — `getTajweedRuleForWord` returns the FIRST match and returns immediately, so
   الْحَمْدُ short-circuits on its Al-Lam article and the Izhar-Shafawi meem inside the SAME
   word is never scored. The card fills that gap with a supplemental pass, but the ROW SET is
   bounded by what the example actually DISPLAYS: `tajweedHeadsInPhrase` runs the example
   colouriser (`buildTajweedAnnotatedString`) with a heads sink and returns the rule heads it
   paints, and **that painted set decides which rows exist**. Heads the primary pass left
   uncovered are scored directly against `RecitationAnalyzer` on each token's proportional
   audio slice (silence-trimmed) and appended; a primary row for a head the example does not
   paint is dropped; rows never duplicate a head, and they are ordered by first appearance in
   the example. (Before this, the supplemental pass unioned `TajweedRules.detectRulesForWord`
   with `getTajweedRuleForWord`. Those two detectors disagree with the colouriser — the
   table's noon-saakinah branch pairs the noon with the NEXT word's first letter even when a
   letter follows it inside the same word, inventing IKHFA/IZHAR/IDGHAM; its tanween branch
   emits GHUNNAH for any tanween; its MADD fires on natural 2-count vowels the colouriser
   never paints red — so the user reported *"the example has only 2 tajweed rules but I got
   the percentages of 4 rules"*. Pinning the rows to the painted set also fixes the opposite
   bug: `مِّن مَّآءِ` now correctly gains the IDGHAM row the old union missed.)
2. **Wrong row colours.** Row colours resolved via an exact card-title match
   (`rulesList.firstOrNull { it.name == ruleName }`) — sub-labels like "Madd Muttasil" /
   "Al-Lam" match no title and fell back to the theme primary (a Madd row rendered TEAL).
   Colours now resolve through the canonical head palette (`tajweedHeadColor`, tajweed.md
   §6): head colour first, exact card match second, theme last.
3. **Status-red word colouring.** The transcript line under the rows painted every
   TAJWEED_ERROR / WRONG / SKIPPED word flat red, which read as "this word is a Madd
   mistake" even when the red was a skipped word or a different rule. Each word is now
   rendered with `buildTajweedAnnotatedString` — the SAME letter-level palette as the
   example — with two status overrides: WRONG/SKIPPED get error colour, TAJWEED_ERROR
   keeps its rule colours plus an underline. The annotated line now lives in the card's
   "Your recitation" block (replacing the recognizer's raw transcript, which is never
   shown) instead of under the per-rule percentages.

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
java.lang.IllegalStateException: Migration didn't properly handle: goals(itqan.app.data.Goal).
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

The mistake-verification decode runs on the **same** `SttEngine` as the live sliding
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

### §4.25 — `verifyTailWords` starvation: the verifier was decoding the span as ONE long buffer

**Symptom.** When the live windows missed a word, the app fired a verification decode. It came back
too thin to satisfy the caller's sanity guards, so the verdict was INCONCLUSIVE, the candidate
lingered, and the doubt was only ever resolved by the end-of-session repair swap — ~20 s later.
Measured baqarah_c2 (`verify idx=1`, word `ذَٰلِكَ`):

```
verify idx=1 SKIPPED word='ذَٰلِكَ' verified=6w found=false    span=10s
verify idx=1 INCONCLUSIVE — tail decode (6w, need >=8) saneHits=0
verify idx=1 SKIPPED word='ذَٰلِكَ' verified=7w found=false    span=15s
verify idx=1 INCONCLUSIVE — tail decode (7w, need >=8) saneHits=0
verify idx=1 SKIPPED word='ذَٰلِكَ' verified=4w found=false    span=21s
verify idx=1 INCONCLUSIVE — tail decode (4w, need >=8) saneHits=0
```

Note the inversion: the WIDER the wall-clock span, the FEWER words came back.

**Root cause 1 — a long buffer is a different, worse decoder.** `verifyTailWords` ran the recognizer
ONCE over the whole span, on the theory that a long full-context pass is "the live decode, looking
harder". It is not. In the SAME trace, at the SAME moment, the live 4 s window decoded real words out
of audio that the 10 s verify window turned into filler:

```
decodeWindow(): begin window=4,0s
transcribe(): 400f  raw='فِيهِ هُدًى لِلْمُتَّق'                 <- live 4 s window, REAL
verifyTailWords(): begin window=10,0s
transcribe(): 1000f raw='تفسير سرضةككبابلور ر سب الرحمن الرحيم'   <- verify 10 s, HALLUCINATED
```

Every clean decode in that trace came from a ~4 s slice; every long slice hallucinated. `تفسير…` is
this model's degradation signature.

**Root cause 2 — the window was anchored to the write cursor, so it filled with trailing silence.**
`fromIdx = audioSize - sampleRate * spanSec`, and `audioSize` keeps advancing while the user waits
silently for a verdict. Widening the span — which the two-strike rule explicitly relies on ("the
second strike looks at MORE audio") — therefore looked at MORE SILENCE, and silence is exactly what a
CTC head hallucinates on.

**Fix — `SherpaPipeline.verifyTailWords` now stitches the span out of live-window-sized slices.**

1. The span is decoded as `verifySliceSec`-second slices (4 s, 3 s stride ⇒ 1 s overlap, capped at
   `verifyMaxSlices` = 6) whose words are POOLED, instead of one long buffer. This is "the live
   decoder, looking harder" done properly: it uses the length the live path proves this model decodes
   well. Pooling does NOT weaken soundness — a wider pool of RELIABLE words makes "the word is
   absent" stronger evidence, not weaker.
2. The window's END is anchored to `lastSpeechFrame` (+0.5 s pad) rather than `audioSize`, and its
   HEAD is snapped to the first speech frame (keeping a 0.25 s run-up). `spanSec` now buys ACTUAL
   speech. While the user is still reading — the normal case — `lastSpeechFrame` tracks `audioSize`,
   so both are no-ops.
3. Each slice must clear a 25 % speech-ratio gate, the same rule `snapshotWindow()` uses.
4. `planVerifySlices(...)` is a pure, unit-tested helper (`VerifySlicePlanTest`). It walks BACKWARDS
   from the newest audio — the frozen candidate always sits in the most recent speech, so when the
   slice cap binds it must be the OLDEST audio that is dropped — and guarantees:
   - **contiguous coverage**: consecutive slices abut or overlap. A gap would leave words undecoded,
     and an undecoded word looks ABSENT — manufacturing exactly the false evidence this verifier must
     never produce;
   - **the last slice ends exactly at `toIdx`**. A forward integer-division stride lands a few frames
     short and would leave the newest audio undecoded; the unit test caught this (span 15 s produced
     239998 instead of 240000) before it ever reached the device.
5. `appendWithoutOverlap(...)` drops the ≤2 words repeated at a seam, compared on normalized forms, so
   pooling cannot inflate the word count the caller's `saneHits` / `need >= 8` guards read.

**Caller-side soundness (`RecitationApp`).** Because the slice cap can drop the oldest audio,
"absent from the pool" is only evidence for a candidate the pool actually SPANNED. Condemnation now
additionally requires the decode to BRACKET the candidate — a decoded word matching a target BEFORE
`idxToVerify` and one matching a target AFTER it. `saneHits` alone was satisfiable by words far from
the candidate. This never weakens the forgiveness path and never permits a false accusation.

### Verified on device — §4.25

Built + installed on the A05s, `fatihah.wav` / `baqarah_start.wav`. Three consecutive runs each:

| run | target | result | verification behaviour |
| --- | --- | --- | --- |
| c39 | Al-Fatihah 29w | **PASS 29/29** | 3 slices → `pooled=10w` → `verify idx=9 found=true`; **0 INCONCLUSIVE, 0 strikes** |
| c40 | Al-Fatihah 29w | **PASS 29/29** | no verification needed at all (`[6] (none)`) |
| c41 | Al-Fatihah 29w | **PASS 29/29** | 3 slices → `pooled=10w` → `verify idx=10 found=true`; 0 INCONCLUSIVE, 0 strikes |
| c4 | Al-Baqarah 1–5 36w | **PASS 36/36** | `verify idx=1` word `ذَٰلِكَ` → 3 slices → `pooled=9w` → **`found=true`** |
| c5 | Al-Baqarah 1–5 36w | **PASS 36/36** | identical to c4: `pooled=9w` → `found=true` |
| c6 | Al-Baqarah 1–5 36w | **PASS 36/36** | TWO candidates forgiven: `idx=1 ذَٰلِكَ` (`pooled=8w`) and `idx=16 وَٱلَّذِينَ` (`pooled=7w`) |

**All six runs: 0 INCONCLUSIVE, 0 `ABSENT strike`, 0 false accusations** (`confirmed (auto-pause +
popup): 0`). Before this fix the same two targets logged three INCONCLUSIVEs (c2) and a `strike 1/2`
(c3) per run.

**c4 is the decisive run: it is the exact c2 failure case.** Same word, same index, same 10 s span —
and where the old build returned `verified=6w found=false` → three INCONCLUSIVEs → ~20 s of doubt,
the new build returns `verified=9w found=true` and forgives on the FIRST attempt:

```
span=10s tailSilence=0,0s -> 3 slice(s) of 4,0s
slice 1/3 4,0s -> 2w 'تفسير سر'                          <- hallucinated, tolerated
slice 2/3 4,0s -> 5w 'ذَلِكَ الْكِتَابُ لَا رَيْبَ فِيهِ'  <- the flagged word IS decoded here
slice 3/3 4,0s -> 2w 'هُدًى لِلْمُتَّقِينَ'
done 1864ms pooled=9w from 3 slice(s)
verify idx=1 SKIPPED word='ذَٰلِكَ' verified=9w found=true viaArticle=false
```

Every run finished `autoFinalize(): grace expired with N/N words decided (unheard=0 mistake=false)`
→ `endOfRange(): pendingRemain=false wrong=0 skipped=0`, and `confirmed (auto-pause + popup): 0` —
zero false accusations. Cost: `done ~1.9 s` for 3 slices (~0.6 s per slice), comparable to the single
long decode it replaced.

One detail worth keeping in mind: c6 forgave `idx=16 وَٱلَّذِينَ` on a `pooled=7w` decode, i.e.
BELOW `minDecodeWords = 8`. That is correct, not a hole — the count/sanity guards gate CONDEMNATION
only. A thin decode may always FORGIVE; it may never CONDEMN. The asymmetry is the whole point:
a false "no mistake" costs nothing, a false mistake popup accuses the user of a recitation error.

### 4.26 The Madd card never refreshed, and two bundled Meem Sakinah examples taught the wrong rule

The user asked for an audit of the coach's example data, because both failure modes are silent —
nothing crashes, the card just teaches the wrong thing:

> "if the random examples we get are really containing the rule (for example if I want to train for
> madd it always gives me an example containing madd) and if there is other rules in the example I
> get the score for each rule in the result."

The card list was extracted out of the composable into a top-level
`val tajweedRuleCards: List<TajweedRuleInfo>` so the **shipped** data is unit-testable, and
`TajweedCardExampleAuditTest` now runs the two real detector chains over every bundled example.

**Finding 1 — `ruleLabelToHead` had no `"Madd"` case, so the Madd card was frozen.**
The rule cards carry *display* titles; the detectors emit *sub-labels*. The mapping covered
`"Madd Muttasil"`, `"Madd (Vowel Extension)"` and `"Madd Munfasil (Qalun: Qasr/Tawassut)"` but not
the bare card title `"Madd"`. Three consequences, all real:

1. The card-expand effect starts with `val head = ruleLabelToHead(rule.name) ?: return@LaunchedEffect`,
   so for Madd it returned early: `generateRandomRecitationExample` was never called and the card stayed
   on `rule.examples.first()` forever. The user's *"it always gives me an example containing madd"*
   was true but for the wrong reason — it was always the SAME example.
2. The supplemental score rows resolve a head to a display name with
   `rulesList.firstOrNull { ruleLabelToHead(it.name) == head }?.name ?: head`. For MADD that lookup
   missed and fell through to the raw head string, so the row rendered as `MADD:` — untranslated
   (`getAppString(true, "MADD")` has no entry) and in the theme colour instead of madd-red, because
   `tajweedHeadColor(ruleLabelToHead("MADD"))` was also null.
3. Fixed by adding `"Madd"` to the MADD branch of `ruleLabelToHead`. Purely additive:
   `getTajweedRuleForWord` never emits the bare label, so the generator's matching is unchanged —
   only the card-title lookup starts working.

**Finding 2 — two bundled Meem Sakinah examples did not contain Meem Sakinah.**

| example | what the detector actually found | why |
|---|---|---|
| `مَا لَّبَلِهِمْ` | *(nothing)* | the only `مْ` is the last token's, and a trailing meem-sakinah with **no next word** yields no rule at all |
| `مِن نَّبَإِهِمْ` | `IDGHAM, GHUNNAH` | `نْ + ن` fires Idgham on the FIRST token, and the walk **returns on the first match**, so the meem-sakinah at the end of the second token is never reached |

Both replaced with real Quranic pairs whose first match is the izhar-shafawi branch of MEEM_SAKINAH
(`مْ` + a non-`م`/`ب` letter), a sub-case the card previously had only one example of:

```
TajweedExample("لَهُمْ فِيهَا", "For them therein")        // مْ + ف
TajweedExample("عَلَيْهِمْ غَيْرِ", "Not of those")         // مْ + غ  (Al-Fatihah 1:7)
```

**Finding 3 — the multi-rule reachability check was a tautology.** The first version of the audit
asserted `all ⊆ primary ∪ (all - primary)`, which is true by construction and can never fail. The
row *set* is now pinned by the PAINTED set instead: `tajweedHeadsInPhrase` is the single authority
for which rows exist (see §4.2), and two tests guard the contract — every bundled example must
PAINT its own card's rule, and every head the primary pass scores must also be painted (otherwise
the painted-only row filter would drop a rule the user can see). What can still break is how a row
**resolves for display** (head → card label → Arabic string → rule colour → calibrated threshold),
which a missing `"Madd" -> "MADD"` case broke; that stays pinned too, along with the
`headToFriendlyLabel` → `ruleLabelToHead` round-trip.

**Verified** — `TajweedCardExampleAuditTest`, 7/7 passing:

```
tests=7 failures=0 errors=0
  PASS every scoreable head is fully resolvable for display
  PASS head to friendly label round-trips through ruleLabelToHead
  PASS every bundled example exhibits its own card's rule
  PASS every bundled example paints its own card's rule
  PASS primary heads are a subset of the painted heads
  PASS report the rules each bundled example actually contains
  PASS every card maps to a trained model head
```

The audit dump is also the reference for **which rules each example really teaches** — it prints
`painted`, `primary` and `detected` per example together with the row count under each policy, so
a detector-alignment regression is visible at a glance (e.g. `مَقْطُوعَةٌ` showed 3 rows under the
old `primary ∪ detected` union and 1 under the painted set; `مِّن مَّآءِ` goes the other way, 2 → 3,
because its painted IDGHAM was previously missed).

### 4.27 The popup named only the rule, the summary was withheld by a rebuild, and the acoustic pass starved the decoder

The user's first **real-microphone** run (not injection) produced three reports in one message. All
three were reproduced from the on-device log (`build/tools/logs/app_only.txt`, 484 app-only lines
extracted from a 323 k-line raw dump):

> "for fatihah when i paused there is some words arent displayed then when i opend the mic again they
> got displayed and finished fatihah without errors. so i think there is a problem of time of
> processing words, it needs to be faster if possible. also when i paused i didnt see a summury popup
> even when i finished. so you have to fix it. then i tried baqarah and got a tajweed error and i
> noticed it didnt write the word in the popup. just the rule and the score. i want the words to be
> included in the mistakes popups and in the summury."

**Finding 1 — the tajweed popup never named the word.** `latestTajweedPopup` is a
`Pair(String, Int)` whose first element is the **rule** for `kind="tajweed"` and the **word** for
`kind="wrong"/"skipped"`. The renderer read `ruleLabel` in all three branches, so the tajweed body
could only ever say which *rule* failed. The log shows it plainly:

```
tajweedPop(): TAJWEED_ERROR wordIndex=18 rule=Madd (Vowel Extension) score=46 -> pushing popup
```

`wordIndex=18` was in hand the whole time — the popup simply never resolved it. Fixed by deriving
`popupWord` from `detailedAnalysis.forWordIndex(wordIdx)?.text` (**resolved by word index, never by
list position**, so a multi-rule word still names the right text) and prefixing the body with
`The word "%s": ` (with an Arabic entry added to `getAppString`). The wrong/skipped branches already
carried the word via `ruleLabel`, so only the tajweed branch changed.

> The **summary** needed no change at all: `MistakeEntry.word` was already populated and the dialog
> already renders it through `SessionSummaryRow` / `SessionSummaryTajweedRow`. The user simply never
> reached the summary — Finding 2.

**Finding 2 — the end-of-range summary was withheld because a late analysis rebuild resurrected the
forgiven tail.** The chain, in order, from the log:

```
finalDecodeAndEmit(): done 5531ms
finalizeCorrect(): 4 trailing PENDING word(s) counted CORRECT (matched=25, total pending=4)
analyze(tajweed): done 4335ms … changed=12      <- REBUILT detailedAnalysis wholesale
analyze(tajweed): done 4550ms … changed=10      <- REBUILT it again
autoFinalize?(): rows=33 decided=29/29 unheardPending=4
endOfRange(): … pendingRemain=true              <- summary withheld
```

`finalizeCorrect()` had already latched `pendingConversionDone = true`, so when the deferred tajweed
loop replaced `detailedAnalysis` wholesale from a fresh `analyzeRecitationDetails` call, the four
PENDING rows came back and **nothing forgave them again**. `unheardPending` stuck at 4, the
`pendingRemain` gate vetoed the summary, `autoFinalize` burned its full 20 s grace, and the user saw
no popup. Fixed by freezing the forgiven index set and re-applying it **idempotently** whenever
`detailedAnalysis` changes:

```kotlin
val forgivenTailIndices = remember { mutableStateOf<Set<Int>>(emptySet()) }
LaunchedEffect(… , detailedAnalysis) {
    if (!pendingConversionDone) { …freeze the set, convert once… }
    else { …re-apply PENDING->CORRECT for the FROZEN set only… }
}
```

Only `PENDING -> CORRECT` is ever written, and only for the frozen set, so a late pass that turns one
of these words into a **real** mistake still wins (the mistake-popup path is untouched) and a rebuild
that merely re-created the rows cannot resurrect them. The effect settles after one pass: once no
PENDING row in the set remains, nothing is written and it stops re-firing. The `matched + 8` bound
from §4.19/§4.17 is retained, so a whole-Baqarah target with 2.5 ayahs recited still cannot convert
six thousand rows green.

**Finding 3 — the acoustic pass, not the text pass, was the latency — and it starved the decoder.**
Measured on this phone:

| loop | per-pass time |
|---|---|
| `analyze(fast)` (text verdicts) | **~210 ms** |
| `analyze(tajweed)` (acoustic verdicts) | **2 009 – 6 047 ms** (2.0/2.0/2.3/2.4/2.8/3.2/3.4/4.3/4.6/6.0 s) |

The cost is dominated by the bounded confirmation search (§1 step 5): a word about to be flagged
sweeps ±6 s in 150 ms steps — up to ~80 model evaluations — *per flagged word*. Because passes were
emitted far faster than they could complete, the loop saturated a core and starved the decode loop
directly:

```
decodeLoop(): alive busy=true lastDecodeAge=5052ms
```

which is exactly the reported symptom — *"when i paused there is some words arent displayed then when
i opend the mic again they got displayed"*. Fixed by rate-limiting **live** passes to at least one per
3 s (`TAJWEED_MIN_INTERVAL_MS`) while letting the **final** pass (`latest.finalized`, after the user
stops) skip both the throttle and the 2 s settle delay — that delay is dead time once no more audio is
coming, and removing it is a direct 2 s off the stop-to-report latency. Dropping a live pass is safe:
the fast text loop keeps the transcript current, the next emission re-triggers a pass, and the final
pass always scores everything.

**Broken-code flag:** the summary was withheld *silently* — no log line said "summary suppressed
because a rebuild resurrected 4 PENDING rows". The `pendingRemain` veto is correct (it exists so the
summary cannot bury an unrecited tail) but it is undebuggable from the log alone. Any future
regression here should first grep for the new line
`finalizeCorrect(): re-applied N forgiven tail word(s) after a late analysis rebuild`.

#### 4.27.1 Two more popup bugs the new test found — the body was leaking internal keys

Writing `TajweedPopupWordTest` immediately paid for itself: pinning the *exact* body composition
exposed two defects that no log line could ever show, because they are pure string-table bugs.

**Bug A — the English popup printed the raw key `review_your_rule_template`.**
`getAppString` opens with `if (!isArabic) return key`, and the two verdict templates
(`good_rule_application_template` / `review_your_rule_template`) exist **only** in the Arabic branch
of the `when`. So the English body literally read:

```
The word "ٱلضَّآلِّينَ": review_your_rule_template Your Madd (Vowel Extension) score: 35%.
```

Fixed by moving the English prose into `formatPerRuleVerdict` itself (`"Review your %s application."`
/ `"Good %s application."`) instead of asking the table for a key it does not contain. The Arabic
path still resolves through the table, so it is byte-identical to before.

**Bug B — the English popup printed a doubled `35%%`.**
The score key is `"Your %s score: %d%%."`; the `%%` is an escape for the caller's own `%d`
substitution, but because English returns the key verbatim that escape was never collapsed, while
the Arabic translation uses a single `٪`. Fixed with a trailing `.replace("%%", "%")` (a no-op for
Arabic, which has no `%%`).

Both are pinned by regression tests (`the English verdict template is real prose, never a raw key`,
`the score renders one percent sign in both languages`), and the helper composes the string exactly
as the composable does, so a future edit cannot silently drop the word, re-leak a key, or double a
percent sign.

**The popup does NOT auto-dismiss — and an earlier comment claimed it did.**
`RecitationApp.kt` carried a comment asserting the dialog "auto-dismisses after 4 s" and referring
to a "4 s auto-timer". There has never been one. Auditing every write to `latestTajweedPopup` gives
exactly three clears: the user's Close/back (6768/6792), a new range or surah selection (3942), and
leaving the Memorize tab (4501). Nothing is time-based, which is the contract the user restated:
*"the popup stays until the user dismiss it."* The stale comment was corrected in place, because the
next reader would otherwise "restore" a timer that must not exist.

#### 4.27.2 Verified on device (Al-Fatihah, real-mic path + injection)

**Fix B — the summary is back.** A clean injection run on Al-Fatihah produced:

```
autoFinalize?(): rows=29 decided=29/29 unheardPending=1 alreadyShowing=false done=false
autoFinalize?(): rows=29 decided=29/29 unheardPending=0 alreadyShowing=false done=false
autoFinalize(): grace expired with 29/29 words decided (unheard=0 mistake=false) -- running the finalize path
endOfRange(): sessionFinalized=true pendingRemain=false newCompleted=7 wrong=0 skipped=0 tajweed=0
```

and the dialog was confirmed on screen by `uiautomator dump`:

```
Perfect recitation!
Surah 1, Ayahs 1 - 7
Masha'Allah — no mistakes in this recitation.
Close
```

The user's broken run had `unheardPending=4` -> `pendingRemain=true` -> **no dialog at all**. The
pause/resume run also exercised the re-apply directly:

```
finalizeCorrect(): 9 trailing PENDING word(s) counted CORRECT (matched=21, total pending=9)
finalizeCorrect(): re-applied 9 forgiven tail word(s) after a late analysis rebuild
```

**Fix C — the throttle holds and the decoder recovers.** Across the same window:

| | user's run (before) | after |
|---|---|---|
| live `analyze(tajweed)` passes in ~45 s | 30+ (one per emission) | **8-16** (one per >=3 s) |
| worst `lastDecodeAge` | **5 052 ms** | 774-1 388 ms once the first pass settles |
| per-pass time | 2.0-6.0 s | 2.0-3.5 s (typical) |

**Fix A — the push path is airtight, the detector is simply conservative.** Across 7 injection runs
only 2 produced a `TAJWEED_ERROR` at all, and **in both the popup was pushed** — never a scored error
without a popup:

| run | tajweed passes | passes with errors | popups pushed |
|---|---|---|---|
| fix27_fatihah_1 | 8 | 0 | 0 |
| fix27_pauseresume_1 | 10 | **1** | **1** |
| fix27_popcap_1 | 10 | 0 | 0 |
| fix27_popcap_2 | 11 | 0 | 0 |
| fix27_popcap_3 | 10 | **1** | **1** |
| fix27_popcap_4 | 16 | 0 | 0 |
| fix27_popup_proof | 9 | 0 | 0 |

Both firings were `wordIndex=28 rule=Madd (Vowel Extension)` with `score=35` and `score=53`, i.e. the
last word (ٱلضَّآلِّينَ) when the recitation was imperfect enough to flag — the same rule/word class
as the user's Baqarah report. `wordIndex 28` resolves through the same
`detailedAnalysis.forWordIndex(28)` row that supplied the score, so the body now reads
`The word "ٱلضَّآلِّينَ": Review your Madd (Vowel Extension) application. Your Madd (Vowel Extension)
score: 53%.`

**Found while testing — a false-positive `SKIPPED`.** One run aborted at
`SKIPPED CONFIRMED wordIndex=17 word='ٱهْدِنَا'` while the transcript had in fact reached the end of
Al-Fatihah, and the full-context pass decoded `وَإِيَّاكَ نَعْبُدُ` and `ٱلْمُسْتَقِيمَ` around it. This is
the documented verification-flakiness (the same bytes score very differently run to run), **not** a
regression from §4.27 — but it is recorded here because a false SKIPPED costs the user a false popup,
which the MEMORY.md invariant forbids. **Fixed in §4.28** — the coverage guard now requires proof the
verification pool actually spanned the candidate (`poolSpansCandidate`), and a device re-run of the
same input yields `VERDICT: PASS — clean recitation, no mistakes` with zero confirmed verdicts.

### 4.28 The false SKIPPED: the verification decode never covered the word

The run recorded in §4.27.2 ended with a false `SKIPPED` on ٱهْدِنَا and auto-paused the user. This
section is its root cause and fix. The failure was NOT the "same bytes decode differently" flakiness
— it was a **span geometry** bug that made several guards read "absent" from a decode that had never
covered the word at all.

**The evidence.** The final full-session decode, 13 s later, plainly heard the word:

```
verify idx=17 SKIPPED word='ٱهْدِنَا' verified=10w found=false   <- condemned here
finalDecodeAndEmit(): done ... w emitted=29w
  dec='... واياك نستعين اهدنا الصرط المستقيم صرط الذين انعمت عليهم ...'
```

**Root cause 1 — the timing anchor resolved to a LATER word.** `spokenTimings.startIndex` is the
**spoken** position, which only coincides with the target index while the transcript tracks the
target. The walk had frozen at `matched=17` (that is what a live mistake candidate *is*), so
`firstOrNull { it.startIndex == idxToVerify }` missed and the fallback
`lastOrNull { it.startIndex <= idxToVerify }` returned the timestamp of a **later** word. The span
was then measured from a point *after* the candidate, so it slid entirely off the word:

```
verify idx=17 launched attempt=1 span=11s ownTiming=true
verifyTailWords(): span=11s -> 4 slice(s)
  slice 1/4 -> 'صِرَاطَ الَّذِينَ أَنْعَم'          <- word 20
  slice 2/4 -> 'الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوب'
  slice 3/4 -> 'الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالّ'   <- word 28
```

*Every* decoded word was ≥ 20; ٱهْدِنَا (17) was never in the audio. Fixed by (a) requiring the
timed word to actually **match** the candidate's normalized form, (b) taking the fallback only from
**strictly earlier** words, and (c) growing the span with each attempt so a retry can never converge
on a window that excludes the candidate.

**Root cause 2 — the coverage guard (`bracketsCandidate = sawBefore && sawAfter`) cannot prove
coverage, and the condemning decode was admitted through it.** The old guard asked only whether *any*
matched word sat before the candidate and *any* after. It is not a proof that the candidate's own
audio was inside the span, because the verifier stitches its window from short slices and **drops the
oldest audio when its slice cap binds** — so a pool can hold early words and late words and still
never contain the candidate.

Reconstructed after the fact by replaying the **recorded pool** through both guards
(`build/tools/replay_guard28.py`: real Al-Fatihah tokens + the real `normalize`/`isWordMatch` rules +
the real `appendWithoutOverlap` pooling):

| pool | words | matched target positions | `sawBefore` | `sawAfter` | OLD guard | new `poolSpansCandidate` |
|---|---|---|---|---|---|---|
| attempt 0, span=20s | 17 | 10,11,12,13,16,24,6,3 | **true** | **true** | **passes → strike 1** | true (16→24) |
| attempt 1, span=11s | 10–12 | 20,21,22,21,22,23,24,25,23,27 | ? | true | ? | false |

**What is proven.** Attempt 0's pool matched word 16 (`نَسْتَعِينُ`) and word 24 (`غَيْرِ`) — a genuine
straddle (16 < 17 ≤ 24, gap 8) — plus two stray early words (t=6, t=3) hallucinated out of the
20-second window's misaligned slices. It therefore passed every guard and paid down **strike 1**
(`ABSENT strike 1/2` at 21:04:49.126). The candidate's own audio was nowhere in it: the pool runs
10→12, then 13→16, then jumps to 24, then back to 6 and 3.

**What is NOT proven, and is flagged rather than asserted.** Attempt 1's pool, as printed, matched
only positions ≥ 20, so under BOTH guards it should have been inconclusive (`sawBefore=false`). Yet
the log shows **no `INCONCLUSIVE` line anywhere in the run** and strike 2 fired silently at
21:04:57.403. Two facts explain the discrepancy and neither has been closed out:
1. **The `strikes >= 2` path emits no log at all** — only `strikes < 2` logs
   (`if (strikes < 2) { Log…; return@launch }`, then it falls through to set the absent verdict
   silently). So a veto-vs-condemn decision on the second attempt is **invisible in the log**.
2. **`matcher.clean()` rewrites pool tokens to global-vocabulary forms before matching.** The pool the
   app matched was not the raw slice text this replay used; a garbage token can be restored to *any*
   Quran word, including an early one, which can flip `sawBefore` to true. That is the most likely
   reason attempt 1 was admitted — **not verified**, because the log does not record the cleaned pool.
   This is a real logging gap: `verifyTailWords()` prints the raw slice text, never the cleaned
   tokens or the matched positions, so an admitted condemnation cannot be audited after the fact.

**The fix — `poolSpansCandidate` + logging.** A pure, unit-testable function that demands *local*
evidence: an adjacent pair of matched positions straddling the candidate within a short gap
(default 10), or a matched word immediately beside it. Applied in line with the existing size/saneHits
guards; the INCONCLUSIVE log now reports `straddles=`, and the strike path logs
`ABSENT strike n/2` **unconditionally** so a silent second strike can no longer hide a veto decision.

> **Correction to an earlier draft of this section.** It asserted that the *"pool of words 20–28"*
> (the 11 s attempt) satisfied the old guard via `الرَّحِيمِ` (t=9) / `الضَّالّ` (t=28). The replay
> contradicts that: the t=9/t=28 pairing appears in **neither** recorded pool, and attempt 1's matched
> positions are all ≥ 20 so it is `sawBefore=false` under the old guard too. The pool that actually
> passed a guard was **attempt 0**, via the genuine 16→24 straddle plus stray t=6/t=3 words. The
> attempt-1 admission remains **unexplained and unlogged** — see item 2 above.

> **Why this direction is the safe one.** Per the standing invariant, the guards gate
> **condemnation only** — a false "no mistake" costs nothing, a false mistake popup accuses the user
> of a recitation error. Tightening the coverage proof can only ever move a verdict from "condemned"
> to "inconclusive, retry", which is exactly the trade this project has chosen every time.

### 4.29 The summary showed the rule but not the score

The user's second request in the same message:

> "when we have the summury : if there is mistakes it shows as the mistake (wrong or skipped word or
> tajweed mistake) and shows the word + tajweed rules + the tajweed score for the tajweed mistakes"

The routing was already correct — `entries.isEmpty()` chooses "Perfect recitation! 🌟" and any
mistake switches the title to "Recitation Summary", with wrong / skipped / tajweed rendered as three
separate sections. Two things were nonetheless wrong:

1. **`SessionSummaryTajweedRow` never rendered the score.** The data was already there:
   `MistakeEntry.score = wa.tajweedScore?.toFloat()` is populated at construction and survives the
   `encodeToJson` / `parseMistakeEntries` round-trip — the row simply ignored it. The composable now
   takes an optional `score` and prints `46%` beside the rule, matching the mistake popup (§4.27.1).
   It stays optional so a row whose score never landed still renders.
2. **The word was never at risk, but nothing pinned it.** Both this dialog and the popup read
   `entry.word` / the resolved row text; both are now covered by tests so a future edit cannot blank
   the column silently.

Verified by `SessionSummaryContentTest` (5 tests): all three kinds carry a non-empty word, only
tajweed carries rule+score, `(wordIndex, rule)` grouping yields one row per rule, the title routes on
`isEmpty`, and a score survives the persistence round-trip while a genuinely absent score stays
`null` rather than becoming `0`.

### 4.30 The verification decodes starve the walk — a GLOBAL budget

Found while re-verifying §4.28 (`fix30_verify_1`, Al-Fatihah 1–7, 46.5 s injection). The §4.28 fix
itself behaved correctly — **0 confirmations, 0 false popups, 1 unmaturable `strike 1/2`** — but the
run exposed a separate, real problem:

```
verify idx=9  launched attempt=0 span=21s ownTiming=false anchorAge=8s   t=+24s
verify idx=13 launched attempt=0 span=21s ownTiming=false anchorAge=2s   t=+31s
verify idx=14 launched attempt=0 span=21s ownTiming=false anchorAge=8s   t=+37s
verify idx=15 launched attempt=0 span=21s ownTiming=false anchorAge=2s   t=+44s
```

Four verification decodes, each `~3.7–3.9 s`, launched on a **fresh candidate every ~6.5 s**. That
is ~60 % of wall time spent inside `verifyTailWords()`. The recognizer is single-slot (`decodeBusy`),
and the live window only ever covers the **trailing** samples — so recognizer time spent verifying is
audio the walk can never go back and decode. On a 46.5 s clip the walk fell behind the injection and
words 16–29 were never decoded cleanly.

**Why the existing caps did not stop it.** `MAX_MISTAKE_VERIFY_ATTEMPTS = 3` bounds ONE candidate.
A long range presents a *fresh* candidate every few seconds, and each one spent its own three
decodes — the budget was per-word, never global. `TAJWEED_MIN_INTERVAL_MS` gates the *tajweed*
pass, not this one.

**The fix — a rolling-window global budget.** `verifyBudgetAfterRequest(windowStart, spent, now,
request, window, max)` is a pure function; the live values are `VERIFY_BUDGET_WINDOW_MS = 20_000`
and `VERIFY_BUDGET_MAX_MS = 10_000`, i.e. **at most half of any 20-second window may be spent inside
verification, across all candidates**, leaving the live decode loop the other half.

- The launch gate calls the read-only probe `verifyBudgetAllowsMs(...)`, which reserves
  `ASSUMED_VERIFY_COST_MS = 4000`; the reservation keeps back-to-back launches from over-committing
  before any decode has reported.
- On launch the reservation is **debited**; when the decode returns, the job **settles** it to the
  real measured cost, clamped to the reservation so a long decode or a clock jump cannot retroactively
  blow the budget. The settle logs
  `verify budget spent=<n>ms/10000ms per 20000ms (last decode <n>ms, reserved 4000ms)`.
- The window **self-heals**: once it is ≥ 20 s old the spend resets, so no explicit timer is needed.
  It is also reset explicitly on a new range/surah and on a fresh surface (both sites that already
  clear `lastMistakeVerifyMs`).

**Why this direction is safe.** A declined request only ever *withholds* a verification. The
candidate stays UNPROVEN, and the existing stall-watchdog / auto-finalize forgiveness paths handle it
(`stallWatchdog(): ... -> forgiving so the cursor can advance`). The budget can never move a verdict
toward condemnation — consistent with the standing invariant that a false "no mistake" costs nothing
while a false mistake popup accuses the user.

**Note on the harness FAIL.** The run's "only 4 of 29 matched" verdict is the harness metric, not a
verdict error: `collector(): ev.endpoint=true words=[]` wipes the transcript at auto-stop, and the
tail had never been decoded. No mistake was ever confirmed in that run. When judging such a run,
check `confirmed (auto-pause + popup)` and the `SKIPPED/WRONG CONFIRMED` count *before* calling it a
regression.

Verified by `VerifyBudgetTest` (9 tests): a fresh budget allows the first request; spends accumulate;
a third decode in-window is declined without spending or moving the window; an expired window rolls
and resets; the window does NOT roll one millisecond early; a request exactly at the cap is allowed
(`<=`); a negative request is clamped and never refunds; the probe reserves the nominal cost; and the
probe rolls a stale window rather than blocking forever.

**Device evidence.** Measured on Al-Fatihah 1–7 (same injection as the §4.30 diagnosis):

| run | clip | verification decodes | max `matched` | false confirmations |
|---|---|---|---|---|
| `fix30_verify_1` (before) | 46.5 s | **4** | 15 | 0 |
| `fix31_budget_1` (after) | 46.5 s | **2** | 17 | 0 |
| `fix31_budget_2` (after) | 61.5 s padded | **1** | **27** (final `hasError=false`) | 0 |

The settle log confirms the budget arithmetic in the field:
`verify budget spent=3930ms/10000ms per 20000ms (last decode 3930ms, reserved 4000ms)`.

> **Harness note (not an app defect).** `fix31_budget_1` still reported FAIL at 21/29 words, but the
> live window had decoded **24 words** (`emitted=24w`, reaching `وَلَا الضَّالّ`). The session was cut
> short by the harness's own `runInjectLoop(): auto-stop on 4000ms silence`, which fires the moment the
> injected file EOFs — so the walk never got the ~20 s it needs for the stall-forgiveness path to
> advance the cursor through the tail. Appending **15 s of silence** to the WAV
> (`fatihah_pad.wav`, 61.5 s) removed the artifact: `matched` rose to **27/29**. When reading an
> injection FAIL, always check the live `emitted=` count and the `auto-stop on …ms silence` line
> before concluding the app fell behind. On a real microphone the user simply keeps reading.

### 4.31 "Perfect recitation" for a recitation the app never heard

Reported by the user after watching a live run:

> "there was many words not displayed and yet there is a summury popup indicating that you have
> finished without any errors"

Both halves are real, and the second is the more serious: the app told the user their recitation was
perfect when it had never verified the tail. **Two independent defects, one chain.**

**Defect A — the budget fix lengthened the freeze (a regression from §4.30).** The §4.30 budget
correctly declines a second verification once 10 s of a 20 s window are spent. But the walk freezes
on a flagged candidate until it is RESOLVED, and the only resolver left was the stall watchdog at
**20 s**. Before the budget, the second strike resolved the candidate in a few seconds; after it, the
cursor sat on the word for the full 20 s and **every later word stayed unrendered** — the user's
"many words not displayed". Measured `fix31_budget_1`: the walk held on ٱهْدِنَا (17) while words
18–29 never displayed.

*Fix.* The stall watchdog now forgives **promptly (3 s)** when the budget cannot fund another decode
and at least one decode was already spent — the two-strike verdict is unreachable, so the hold
achieves nothing but a frozen display. Requiring `attempts > 0` keeps a candidate whose verification
simply has not started from being cut off. Direction is the safe one: forgiving can only ever produce
a false "no mistake", never a false accusation.

**Defect B — the summary was raised by the AUTO path at all.** My first attempt at this defect was
WRONG and the user corrected it in one line:

> "if it stoppd then it should be a skipped/wrong word popup or tajweed popup else it should be
> displayed. not getting a summury popup (unless i click pause)"

That is the whole contract. The auto path must **never** raise the summary. It was raising one
because the walk stalled mid-range, `finalizeCorrect()` forged the tail CORRECT, `pendingRemain`
read false, and a "finished" dialog appeared for a recitation the app had not finished hearing.
Measured `fix32_coverage_1`: `endOfRange(): sessionFinalized=true pendingRemain=false newCompleted=6
wrong=0 skipped=0 tajweed=0` → dialog, no mistake, mid-range.

*Fix.* The summary is now gated on `userInitiatedStop`:

```kotlin
if (sessionFinalized && !pendingRemain &&
    viewModel.sessionSummaryToShow == null && !mistakeMustWin &&
    userInitiatedStop                       // <- only the pause button
) { … raise summary … }
```

`runStopAndFinalize(true)` is the pause button; the auto path calls `runStopAndFinalize(false)`. So
the summary can only ever come from the user tapping pause, and the auto path either shows a mistake
popup (when one is genuinely flagged) or shows nothing and leaves the words on screen.

**My first fix for this was rejected, and correctly so.** I had added `unheardWordCount` /
`totalWordCount` to `SessionSummary` and printed *"N of M words were not heard clearly"* inside the
dialog. The user's instruction was explicit: the summary shows **the mistake words + rules + the
tajweed score**, not a coverage note. That line is removed; `perfect` is back to
`entries.isEmpty()` (safe now, because the dialog only appears when the user asks for it). The two
coverage fields are kept **for logging only** — the raise site logs
`endOfRange(): user pause -> summary raised (entries=N unheard=N/N)`, which is how a stalled walk
becomes visible without polluting the UI.

`List<WordAnalysis>.unheardWordCount()` (pure) survives as the diagnostic: PENDING words, plus
CORRECT-with-null-`spokenText` words (the `finalizeCorrect()` rubber-stamp), distinct by word index.
Verified by `SummaryCoverageTest` (7 tests): a fully heard recitation reports 0; PENDING counts; a
CORRECT word with no `spokenText` counts; the measured `fix31_budget_2` shape reports 2 of 29; a word
with two rule rows counts **once**; a real WRONG/SKIPPED is NOT counted as unheard; empty reports 0.

### 4.32 The pause-only summary gate was too strict — a genuine finish must still show it

§4.31's fix gated the summary on `userInitiatedStop` alone. That satisfied *"not getting a summary
popup (unless i click pause)"* but broke the other half of the contract: a recitation that runs
cleanly from the first word to the last and then **ends on its own** raised nothing, and the user
reported it — *"no summary shown when finished in this test ! fix it"*.

The two requirements are not in conflict once "finished" is defined precisely. A walk that stopped
is one of three things:

| What stopped the walk | `unheardWordCount()` | Correct behaviour |
|---|---|---|
| The user tapped pause | any value | **summary** (the user asked for it) |
| The walk decided every target word | **0** | **summary** (the recitation is over) |
| A mid-range stall / budget-blocked verify | **> 0** | **no summary** — leave the words displayed |

So the gate is `(userInitiatedStop || genuinelyFinished)` where
`genuinelyFinished = detailedAnalysis.unheardWordCount() == 0`. This reuses the §4.31 helper rather
than inventing a second notion of "done", and it cannot fire on a stall: a stall is by construction a
walk that has NOT reached the tail, so it always has PENDING or rubber-stamped rows and
`unheardWordCount() > 0`.

`RecitationApp.kt` (~line 9336):

```kotlin
val genuinelyFinished = detailedAnalysis.unheardWordCount() == 0
if (sessionFinalized && !pendingRemain && viewModel.sessionSummaryToShow == null &&
    !mistakeMustWin && (userInitiatedStop || genuinelyFinished)) {
    sessionRangeCompleted = true
    viewModel.sessionSummaryToShow = MainViewModel.SessionSummary(
        surahNumber = chargeSurah, rangeStart = chargeRangeStart, rangeEnd = chargeRangeEnd,
        entries = mistakeEntries, recitedAyahCount = newCompletedCount,
        totalAyahCount = (chargeRangeEnd - chargeRangeStart + 1).coerceAtLeast(1),
        unheardWordCount = detailedAnalysis.unheardWordCount(),   // logging only
        totalWordCount = detailedAnalysis.distinctWordCount())    // logging only
    Log.i("ItqanSherpa", "endOfRange(): summary raised (userPause=$userInitiatedStop " +
        "finished=$genuinelyFinished entries=${mistakeEntries.size} " +
        "unheard=${detailedAnalysis.unheardWordCount()}/${detailedAnalysis.distinctWordCount()})")
} else if (sessionFinalized && !pendingRemain && !mistakeMustWin &&
           !userInitiatedStop && !genuinelyFinished) {
    Log.i("ItqanSherpa", "endOfRange(): auto stop but NOT genuinely finished -> " +
        "no summary; words stay displayed")
}
```

Both branches log, so the UI behaviour is auditable from the log alone — which is how the two
divergent failure modes (§4.31's stall-summary and this section's missing-finish-summary) can be told
apart without a screen capture.

**Device-verified** on `fix34_fatihah_1` (Al-Fatihah 1–7, padded WAV): the walk reached
`matched=29 hasError=false`, then

```
endOfRange(): sessionFinalized=true pendingRemain=false newCompleted=7 wrong=0 skipped=0 tajweed=0 alreadyShowing=false
endOfRange(): summary raised (userPause=false finished=true entries=0 unheard=0/29)
```

i.e. the summary appeared on a **genuine finish** with every word heard, with no user pause — the
exact case that was broken. `build/tools/check_run.sh fix34_fatihah_1` → `CLEAN (all 29 words
matched, no confirmed mistake)`.

### 4.33 The stale injection cursor: a session that ends before EOF poisons the NEXT run

`clean_fatihah_2` aborted before it produced a single decode:

```
runInjectLoop(): INJECT MODE file=inject.pcm bytes=2127928 startPos=1592000
ABORT: injection did not start at position 0 -- stale inject cursor
```

The inject read cursor (`injectPos`) is a `@Volatile var` on the pipeline — deliberately **not**
persisted, so a pause→resume continues mid-file. The old rewind guard was

```kotlin
if (injectPos >= f.length() && f.length() > 0L) { injectPos = 0L }   // only if EXHAUSTED
```

which assumes every run ends by playing the file to the end. It does not. `fix34_fatihah_1` finished
because the walk **decided all 29 target words**, and the session was finalized there — at 49.75 s of
the 66.5 s file. `injectPos` was therefore 1 592 000, not 2 127 928, the `>=` guard did not fire, and
the next run started 50 s in: it fed ~17 s of audio, hit EOF, and then fed pure silence for the whole
session. `inject_run.sh` catches exactly this shape (`startPos=0` assertion, exit 3) so it failed
loudly instead of silently reporting a dead run — but the underlying cursor was still wrong.

**Fix (app).** Reset the cursor in `startReadLoop()` instead of guessing from the file length:

```kotlin
private fun startReadLoop() {
    injectPos = 0L   // fresh session => replay the injection file from the top
    readJob = scope.launch { runReadLoop() }
    decodeJob = scope.launch { decodeLoop() }
}
```

`startReadLoop()` is the **only** fresh-utterance entry point (`start()` and the prewarm
continuation). `resume()` deliberately re-spawns the two workers itself (line ~1701) and never calls
`startReadLoop()`, so pause→resume continuity is untouched — which is the property the old heuristic
was trying to protect. The exhausted-file guard in `runInjectLoop` is kept as a second line of
defence for the residual case (a session that reaches EOF and is then *resumed* rather than freshly
started); its comment now says so.

**Fix (harness).** `inject_run.sh` step 0 now `am force-stop`s the package before `am start`. A warm
process carries in-memory session state across runs — the inject cursor, a leftover live session, a
Compose popup that makes the navigation taps land on the wrong surface. A cold start removes the
whole class of stale-state artifacts, not just the cursor, and the script already re-navigates to
Memorize + Tajweed and re-selects the surah/range, so nothing is lost. The home-screen wait was
widened from 6 to 8 retries (30 s → 40 s) to cover the cold-start recognizer load.

**Lesson.** A test harness that assumes "the run ends at EOF" silently couples two runs together the
moment the app can finish early. Every "start from a known state" step has to be explicit, not
inferred from the artifact.

### 4.34 The clipped tail word was counted as reached but never displayed

Reported on device: *"last word not displayed and there is no popup"*. Reproduced on the very next
run, `clean_fatihah_2` (Al-Fatihah 1–7, padded WAV):

```
02:37:06.558 runInjectLoop(): auto-stop on 4000ms silence
02:37:15.036 collector(): ev.endpoint=true words=[الضَّالِّينَ] partialStr='الضَّالِّينَ'
02:37:15.072 autoFinalize?(): rows=31 decided=29/29 unheardPending=1 alreadyShowing=false done=false
02:37:15.073 autoFinalize(): holding 0ms/20000 (decided=29/29 unheard=1 mistake=false)
02:37:15.177 analyze(tajweed): done 103ms scored=17 tajweedErrors=0 changed=2
02:37:15.201 autoFinalize?(): rows=32 decided=29/29 unheardPending=0 alreadyShowing=false done=false
02:37:15.202 autoFinalize(): grace expired with 29/29 words decided (unheard=0 mistake=false)
02:37:15.212 endOfRange(): sessionFinalized=true pendingRemain=false newCompleted=6 wrong=0 skipped=0 tajweed=0
02:37:15.213 endOfRange(): auto stop but NOT genuinely finished (unheard=1/29) -> no summary; words stay displayed
```

and the harness's own verdict: `FAIL — only 28 of 29 target words matched`, while `analyze(fast)`
reported `matched=29 hasError=false targetWords=29`.

**Two counts disagreed, and the wrong one won.** `unheardPending` counted only PENDING rows with a
**blank** `spokenText`. The last word `ٱلضَّآلِّينَ` was PENDING **carrying a fragment** — the shape
`clippedWord()` emits:

```kotlin
// clippedWord(): target 28 'ٱلضَّآلِّينَ' spoken prefix 'ٱلضَّآ' -> PENDING (clipped decode, not a skip)
```

The CTC decoder returned only the opening letters of the word; the token is a strict prefix of the
target, so the analyzer **consumes** the word and moves on (`matchedWordCount` counts it as reached,
and `endOfRange()`'s `pendingRemain` already treats it as forgiven — see the §4.17 note at
`RecitationApp.kt:9244`). But the row's **status** stayed PENDING, and the transcript colours by
status, so the word rendered at the 25 %-opacity fallback. The cursor said "reached", the screen said
"nothing" — and `cleanFinish` (which tested `decidedWords.size >= total && unheard == 0` with the
blank-token definition) went true, finalizing the session with no popup and no summary.

Three fixes:

**1. A clipped decode is a heard word — show it.** New pure helper
`WordAnalysis.promoteClippedPending()` (plus the row-wise `promoteClippedPendingRows()`), applied in
the analysis merge right where the code already counts the row as reached:

```kotlin
if (w2.status == RecitationStatus.PENDING && !w2.spokenText.isNullOrEmpty()) {
    w2 = w2.promoteClippedPending()          // PENDING -> CORRECT, fragment KEPT
}
```

The fragment is deliberately **kept**, not nulled: `unheardWordCount()` skips rows that carry a
token, so the word still counts as heard and a genuine finish can still raise its summary. Only the
COLOUR changes. A PENDING row with **no** token is a word never heard and is left alone — promoting
it would colour a word the user never said.

**2. `cleanFinish` must not call an unresolved candidate "decided".**

```kotlin
val decidedWords = detailedAnalysis.filter { it.status != RecitationStatus.PENDING }
    .map { it.wordIndex }.toHashSet()                 // a PENDING row is not a decision
val unheard = detailedAnalysis.filter { it.status == RecitationStatus.PENDING }
    .map { it.wordIndex }.toHashSet().size            // candidate or not, it is unheard
```

The old `unheard` was `count { PENDING && spokenText.isNullOrEmpty() }`, so a row holding a candidate
token read as **heard** and the range was declared complete while a word was still undecided.

**3. On grace expiry, unresolved rows are promoted so they DISPLAY.** A PENDING row that survived the
whole grace is a judgement the verifier never returned a verdict for. There is no evidence the word
was wrong (a confirmed mistake takes the branch above), and a false accusation is worse than a missed
skip — so we do not accuse. But we must not leave the word invisible either. The new branch promotes
every surviving PENDING row via `promoteClippedPendingRows()`; rows that carry a fragment stay
"heard", rows with no fragment keep counting in `unheardWordCount()` and therefore still block a
summary they did not earn (§4.31).

**Diagnostics.** `autoFinalize?()` now prints `unheardPending` AND `unheardWords`, and dumps every
undecided row as `w<idx>/<text>:<status>:spoken='<token>'`. The two shapes — PENDING+token and
CORRECT+no-token — are indistinguishable from the counts alone and need opposite handling, so the run
now says which one it is instead of leaving it to be inferred.

**Lesson.** "Reached" and "displayed" were two different predicates maintained in two different
places, and nothing tied them together. When the cursor advances past a word, that word must be
VISIBLE — the invariant is one-directional and has to be asserted at the point the cursor advances,
not re-derived later from a status field that a different code path owns.

### 4.35 The confirm loop is emission-driven, so a frozen walk can never finish a verdict

`clean_fatihah_3` (the first run on the §4.34 build) came back **`NOT CLEAN`, `maxMatched=17/29`** —
much worse than the 28/29 it replaced. The tail word was now displayed correctly (§4.34 works), but
the walk had fallen 12 words behind. The trace explains why:

```
03:06:47.993 mistakeStop(): verify idx=15 launched attempt=0 span=23s ownTiming=false anchorAge=3s
03:06:50.500 stallWatchdog(): candidate idx=15 unresolved for 4003ms (attempts=1 strikes=0) -> forgiving
03:06:51.721 mistakeStop(): verify budget spent=7555ms/10000ms per 20000ms
03:06:51.726 mistakeStop(): verify idx=15 SKIPPED word='وَإِيَّاكَ' verified=13w found=true    <- acquitted, too late
…
03:07:01.026 mistakeStop(): verify idx=14 launched attempt=0 span=25s
03:07:04.744 mistakeStop(): verify idx=14 WRONG word='نَعْبُدُ' verified=16w found=false
03:07:04.761 mistakeStop(): verify idx=14 ABSENT strike 1/2 — needs one more decode before condemning
03:07:05.770 mistakeStop(): candidate row=14 word=14 sightings=2 … sinceVerify=1009ms
03:07:07.586 mistakeStop(): candidate row=17 word=14 sightings=2 … sinceVerify=2825ms   <- 175 ms short
   … nothing for nine seconds …
03:07:16.533 stallWatchdog(): candidate idx=14 unresolved for 20025ms (attempts=1 strikes=1) -> forgiving
```

**Two defects, both in the stall watchdog.**

**(a) The second decode is never launched.** The confirm loop is a `LaunchedEffect` keyed on
emissions. A candidate freezes the walk, the user stops speaking, no emission arrives — so the loop
never re-runs, and after `ABSENT strike 1/2` the decode that would decide condemnation (or acquit the
word) is never spent. The loop's last re-run missed the 3 s cooldown by **175 ms** and nothing ever
tried again. The candidate sat WRONG for the full 20 s and was forgiven UNPROVEN, with the cursor
frozen behind it the entire time.

The watchdog already polls every 2 s, so it is the natural driver. New pure helper:

```kotlin
fun stallShouldNudge(isListening, pipelineRunning, jobIdle, cooldownOver, budgetOk,
                     attempts, nudges, maxAttempts): Boolean =
    isListening && pipelineRunning && jobIdle && cooldownOver && budgetOk &&
        attempts < maxAttempts && nudges <= maxAttempts
```

On a nudge the watchdog only re-opens the confirm loop's 800 ms dwell gate (it rewrites
`lastWrongStopFirstSeenMs`, which the loop observes, so the loop re-runs). **Every** other guard still
applies — attempt budget, 3 s cooldown, global budget, job-active, and the §4.28 bracket+straddle
coverage test on the RESULT — so a nudge can never move a verdict toward condemnation on its own; it
can only make the pipeline finish a judgement it already started. `stallNudges` is bounded by
`maxAttempts` so a nudge the loop never honours cannot hold a candidate forever (the §4.18
infinite-freeze class). It **replaces** the old "last-chance verification" block, which did the same
thing for the `attempts == 0` case but reset the forgiveness clock with no bound — and it subsumes
that case, because `cooldownOver` is defined as `lastMistakeVerifyMs == 0L || now - it >= 3000`.

**(b) The budget short-circuit forgave a candidate while its own decode was in flight.** idx=15 was
forgiven at 4003 ms; the in-flight decode returned `found=true` 1,2 s later. The cause: the nominal
cost of an ACTIVE verification is already debited, so `budgetOk` is false for the whole duration of
that decode, and §4.31's short-circuit read that as "the budget is exhausted, the verdict is
unreachable". New pure helper:

```kotlin
fun stallForgiveMsFor(budgetOk, attempts, jobIdle,
                      normalMs = 20_000L, blockedMs = 3_000L): Long =
    if (!budgetOk && attempts > 0 && jobIdle) blockedMs else normalMs
```

`jobIdle` is the new condition. `attempts > 0` was already there (a candidate whose verification has
not started must not be cut off early).

**Device status.** `clean_fatihah_3` = pass attempt 1 on the §4.34 build → **NOT CLEAN**
(`maxMatched=17/29`, 3 stall-forgives, 1 tajweed error at word 22). It did stop on a valid
**tajweed popup** (`Tajweed Mistake — Izhar (Clarity)`, score 9) — i.e. the §4.32 stop contract held,
and the §4.34 tail fix worked (word 28 was decided, not left greyed). The failure was the freeze, not
the display. `StallNudgeTest` (14 tests) pins both helpers.

**Lesson.** A state machine driven by *emissions* silently dies when the thing it is waiting for
freezes the emission source. Any watchdog whose whole purpose is to break a freeze must also be able
to *restart* the machinery it depends on — not just observe it.

### 4.36 My own §4.34 grace-expiry promotion was an infinite loop

`p1_fatihah` (attempt 1 on the §4.35 build) reached `maxMatched=27/29` — the §4.35 nudge works — and
then never settled. The screen looped:

```
autoFinalize(): grace expired with 7 unresolved word(s) [22, 23, 24, 25, 26, 27, 28] and NO
  evidenced mistake -> promoted to CORRECT so they DISPLAY (0 of them carry a spoken fragment, so
  they still count as heard; the rest stay 'unheard' and block a summary)     <- every 2 s, forever
```

**Cause.** §4.34's grace-expiry branch called `promoteClippedPendingRows()`. That helper
*deliberately* leaves a token-less PENDING row alone — a word never heard must not be coloured as if
the user said it. But the rows the branch exists for are exactly those: the log's own parenthetical,
*"0 of them carry a spoken fragment"*, was the branch telling us it had promoted nothing. So
`unresolvedIdx` was non-empty on every pass, the branch logged and `return@LaunchedEffect`'d without
changing a single row, and the effect re-fired 2 s later. The session never finalized.

**Fix.** A separate, correctly-scoped helper:

```kotlin
fun List<WordAnalysis>.promoteUnresolvedPendingRows(): List<WordAnalysis> =
    map { if (it.status == RecitationStatus.PENDING) it.copy(status = RecitationStatus.CORRECT) else it }
```

`spokenText` is left exactly as it was, which keeps the two cases apart downstream: a row carrying a
fragment stays inside `unheardWordCount()`'s "heard" set (a genuine finish can still raise its
summary), while a token-less row keeps counting as unheard and therefore can never earn a summary the
app did not earn (§4.31). Either way the word is now **VISIBLE**, which is what the contract asks for
once the session has stopped.

Plus `gracePromoteCount`, because "promote then return" is only safe if the promotion actually lands:
the FIRST promotion returns (the UI repaints), and if a later analysis rebuild resurrects the rows it
promotes again but this time **falls through to the finalize** — so a rebuild loop can never hold the
session open (§4.18: "nothing ever happens" is the worst outcome). Reset to 0 wherever
`graceExpiredOnce` is reset.

`ClippedPendingDisplayTest` is now 15 tests; **57 tests, 0 failures** across the six suites.

**Lesson.** A helper that deliberately skips a case is the wrong tool for a branch whose entire input
IS that case — and the loop was the tell. A branch that logs the same line every 2 s is not making
progress, and `return@LaunchedEffect` with no state change is an infinite loop, not a retry. When a
"promote/forgive" branch is added, assert that it CHANGED the list, or the next reader has no way to
tell a no-op from a fix.

### 4.37 The transcript cursor went BACKWARDS after the summary — a seen word was un-rendered

**Symptom (user, attempt 2 on the §4.36 build).** *"the last word isnt displayed and yet we have a
summary popup indicating no mistakes."* The summary said **"Perfect recitation! 🌟 — Masha'Allah, no
mistakes in this recitation"** while the final word of Al-Fatihah (`ٱلضَّآلِّينَ`) sat at the
25 %-opacity grey fallback. Attempt 1 on the same build was clean (`p1_fatihah`, 29/29).

**The renderer's predicate is the CURSOR, not the status.** `RecitationApp.kt` colours a word iff
`globalWordIndex < matchedWordCount`, and falls through to `onSurface.copy(alpha = 0.25f)` otherwise:

```kotlin
if (globalWordIndex < matchedWordCount) { /* primary / hinted / tajweed */ }
else if (globalWordIndex == matchedWordCount && hintActive) { … }
else if (!hintActive && globalWordIndex == errorIndex && …) { /* error colour */ }
else { withStyle(SpanStyle(color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.25f))) { … } }
```

`matchedWordCount = alignment.first`, and `alignment` is recomputed **from scratch, unconditionally**
on every analysis pass (`alignment = Pair(newMatchedWordCount, newHasError)`). There is **no monotonic
guard**, so any later pass that scores fewer words moves the cursor BACKWARDS and un-renders the tail.

**Device evidence (`p2_fatihah`).** Every line below is the same frozen 29-token transcript:

| time | line |
|---|---|
| 22:01:35.180 | `finalDecodeAndEmit(): done 8320ms new=-3 w emitted=25w` |
| 22:01:35.185 | `collector(): ev.endpoint=true words=[ٱلضَّآلِّينَ] partialStr='ٱلضَّآلِّينَ'` |
| 22:01:35.200 | `collector(): rejected repair still contributed 1 word(s) … -> transcript now 29w` |
| 22:01:35.201 | `collector(): final-decode repair REJECTED (repair 25w/prefix24 vs live 28w/prefix26)` |
| 22:01:35.249 | `autoFinalize(): undecided rows = w28/ٱلضَّآلِّينَ:PENDING:spoken='-'` |
| 22:01:35.384 | `autoFinalize?(): rows=33 decided=29/29 unheardPending=0 unheardWords=0` |
| 22:01:35.396 | `endOfRange(): summary raised (userPause=false finished=true entries=0 unheard=0/29)` |
| 22:01:35.680 | `analyze(fast): done 429ms matched=29 hasError=false targetWords=29` |
| 22:01:35.828 | `analyze(tajweed): done 184ms scored=19 tajweedErrors=0 changed=5` |
| 22:01:35.899 | `analyze(fast): done 218ms matched=28 hasError=false targetWords=29` ← **backwards** |

The two fast passes see the SAME transcript and differ only in the request they carry. The aligner is
not a pure function of the text: when `fullAudio` is attached (`hasRealAudio = fullAudio != null &&
fullAudio.size >= 256`) it re-decodes each candidate word's audio slice, and that slice decode is
flaky — the same flakiness that makes a long full-context pass hallucinate (§4.25/§4.30). So a word
the aligner decided a moment ago can come back PENDING, `newMatchedWordCount` drops, and the tail goes
grey. `finalReport` is **not** the cause: it is an unused parameter in `analyzeRecitationDetails`.

**Why this is a contract violation and not just a cosmetic glitch.** The project invariant is
**REACHED ⇒ DISPLAYED (one-directional)**: once the cursor has advanced past a word the user must be
able to see it. On a FINALIZED session the transcript is frozen — no further audio can arrive — so a
lower cursor cannot be a correction. It is only decoder noise, and rendering it takes away a word the
user already earned.

**Fix — two pure helpers (both unit-tested).**

```kotlin
fun resolveTranscriptCursor(newMatchedWordCount: Int, previousCursor: Int,
                            hasError: Boolean, finalized: Boolean): Int =
    if (finalized && !hasError) maxOf(newMatchedWordCount, previousCursor)
    else newMatchedWordCount
```

applied as

```kotlin
val resolvedCursor = resolveTranscriptCursor(newMatchedWordCount, alignment.first,
                                             newHasError, sessionFinalized)
alignment = Pair(resolvedCursor, newHasError)
```

and it logs `analyze(fast): cursor held at 29 (walk wanted 28)` so the hold is visible in a run.

Two deliberate escapes, both necessary:

- **`!finalized`.** A live cursor must stay free. It has to follow the user backwards when they
  re-read from a flagged word, and a resume must not inherit a stale high-water mark.
- **`hasError`.** A WRONG/SKIPPED is a deliberate verdict that must render in its error colour, and
  the renderer's error branch only runs for `globalWordIndex >= matchedWordCount`. Clamping there
  would hide the very mistake the app is stopping for.

```kotlin
fun keepFinalizedDecision(fresh: WordAnalysis, previous: WordAnalysis?, finalized: Boolean): WordAnalysis {
    if (!finalized || previous == null) return fresh
    if (fresh.status == RecitationStatus.PENDING && previous.status != RecitationStatus.PENDING)
        return previous.copy(tajweedScore = fresh.tajweedScore ?: previous.tajweedScore)
    return fresh
}
```

plus `List<WordAnalysis>.keepFinalizedDecisions(previous, finalized)`, keyed by `(word, rule)`.

This is the same defect one layer down: the analysis list is rebuilt wholesale on every pass, so a
word the finalized list had already decided can come back PENDING. That matters because
`unheardWordCount()` is what `endOfRange()` uses for `genuinelyFinished` — a row that flips back to
PENDING makes a recitation the app just called "perfect" read as one word unheard, which is exactly
the two-sources-of-truth disagreement that produced this bug family. Only PENDING is blocked (the
caller already refuses to downgrade WRONG/SKIPPED), and the fresh `tajweedScore` is carried over so
the acoustic result is not lost. Applied in **both** merge sites — the fast merge and the deferred
tajweed merge (whose `changed=5` was the pass immediately before the 29 → 28 drop) — and the §4.34
`promoteClippedPendingRows()` now runs **last**, after the guard, so a freshly PENDING row is first
allowed to be overridden by the row it is replacing.

`FinalizedCursorMonotonicTest` adds 25 tests pinning the measured `p2` sequence, both escapes, the
multi-rule keying, idempotence, and that the guard **cannot forge a decision** (a word the previous
pass never decided stays unheard, so a recitation the app could not judge still cannot earn a
"perfect" summary — §4.31). **82 tests, 0 failures** across the seven suites.

**Lesson.** A cursor derived from a fresh recomputation is not monotonic, and a renderer that keys on
a cursor inherits every wobble of the engine behind it. When a value both *drives a verdict* and
*drives a display*, the display must be pinned to the strongest thing already proven — here: "the
session ended and no mistake was found, therefore every word stays on screen." The same trap produced
§4.34 (status flapped, cursor did not) and this section (cursor flapped, status did not); the honest
form of the rule is the one-directional invariant, asserted at the point where the value is written.

### 4.38 A false "Skipped Word" popup — the endpoint decode was never allowed to acquit

**Symptom (first run on the §4.37 build, `p3_fatihah`).** The app auto-paused on a **"Skipped Word"**
popup for word 17 of Al-Fatihah (`ٱهْدِنَا`) — a word the recitation contained. The user's standing
complaint class: a false mistake popup accuses them of a recitation error, which project policy ranks
as the worst outcome.

**Device evidence.** The run stopped at word 17 and never rendered words 18–28 live
(`VERDICT: FAIL — only 17 of 29 target words matched`). But the app's own evidence contradicts the
verdict:

| time | line |
|---|---|
| 22:51:39.683 | `mistakeStop(): idx=17 pool=16w positions=[-1, 12, 13, 14, 13, 16, -1, 16, -1, 18, 19, -1, 20, 12, 22, 23] straddles=true candTarget='اهدنا'` |
| 22:51:39.683 | `verify idx=17 ABSENT strike 1/2 — needs one more decode before condemning` |
| 22:51:47.895 | `verify idx=17 ABSENT strike 2/2 — CONDEMNING` |
| 22:51:48.487 | `runInjectLoop(): auto-stop on 4001ms silence` |
| 22:51:48.863 | `finalDecodeAndEmit(): begin endpoint=true window=45,0s` |
| 22:51:49.764 | `verdictTicker(): absent verdict held 1868ms for wordIndex=17 -> auto-pause + popup` |
| 22:51:49.764 | `mistakeStop(): SKIPPED CONFIRMED wordIndex=17 word='ٱهْدِنَا' (sightings=2)` |
| 22:51:57.174 | `log='… اياك نعبد واياك نستعين الصرط المستقيم …'` ← **live log: `اهدنا` missing** |
| 22:51:57.174 | `dec='… اياك نعبد واياك نستعين اهدنا الصرط المستقيم …'` ← **endpoint decode: `اهدنا` PRESENT** |
| 22:51:57.178 | `finalDecodeAndEmit(): done 8316ms new=6 w emitted=28w` |
| 22:51:57.179 | `collector(): mistake-freeze dropping endpoint=true 28w (resume latch armed @ word 17)` |

**Cause — two failures stacked.**

1. **A 4 s-slice decode that DROPS a word is indistinguishable, to the aligner, from a skip.** The
   live window rendered `…نَسْتَعِينُ` and then `ٱلصِّرَٰطَ` with no `ٱهْدِنَا` between them; the walk can
   only represent that as a SKIPPED. The verification decodes are the SAME model over the SAME kind
   of short slice, so they dropped it again — and because the neighbours appeared (`positions`
   contains 16, 18, 19, 20, 22, 23), §4.28's `poolSpansCandidate` straddle test read the coverage as
   proven. Two strikes, condemn.
2. **The one decode that heard the word was never allowed to speak.** The whole-session endpoint
   decode is strictly better evidence and it was **already running** — it began 0.9 s before the
   confirm and landed 8 s after it, with `اهدنا` in its transcript. The confirmed-mistake freeze
   (§4.28) then discarded that emission on purpose, because it exists so a re-decode cannot
   resurrect a mistake and auto-dismiss the popup.

So the popup could not be taken back — popups are never auto-dismissed. **The confirm itself had to
be withheld.**

**Fix — `mistakeMayBeConfirmed(pipelineRunning, endpointDecodeLanded, sessionStoppedForMs, holdMs)`.**

```kotlin
fun mistakeMayBeConfirmed(pipelineRunning: Boolean, endpointDecodeLanded: Boolean,
                          sessionStoppedForMs: Long, holdMs: Long = 15_000L): Boolean =
    pipelineRunning || endpointDecodeLanded || sessionStoppedForMs >= holdMs
```

Two wall clocks feed it, both recorded in the collector **above** the mistake-freeze (that freeze
`return@collect`s and would otherwise swallow the very evidence this section exists for):

- `pipelineStoppedAtMs` — first emission seen while `tajweedPipelineHolder.running == false`.
- `endpointDecodeLandedAtMs` — the first post-stop emission carrying words. That is the endpoint
  decode's result; it is the acquittal.

Both are reset at every session boundary (range/surah change, fresh `startTajweedSession`, the
collector's `ev.freshSession`, and an explicit Restart) so a previous run's evidence cannot gate the
next one.

Holding the confirm is safe in both directions, and that is the point:

- **False positive.** With no latch armed the endpoint decode is committed like any other emission,
  the transcript gains the missing word, the walk re-aligns it CORRECT, and there is no candidate
  left to condemn. The popup never fires.
- **Real skip.** The word is absent from that decode too, so the confirm still happens — once the
  decode lands. The hold only delays it.

**The same hold was added to the watchdog's FORGIVE path.** Without it the budget-blocked
short-circuit (§4.31) forgives at 3 s, the endpoint decode needs ~8 s, and forgiving first turns a
real skip into a silent "no mistake" — the row becomes CORRECT, so the ticker has no candidate left
to condemn, and injection tests exist precisely to catch wrong/skipped words. Holding *both*
verdicts until the decode lands keeps the two directions symmetric: whatever the endpoint decode
heard decides it. The hold is bounded by the same clock, so **forgiveness is delayed, never blocked**
— the §4.31 asymmetry (a false "no mistake" is cheap, a false accusation is not) is preserved.

`EndpointAcquittalTest` adds 12 tests: the measured p3 cell must not confirm, a running pipeline is
untouched, a landed decode releases the hold, a real skip still confirms, the bound holds and
expires, and the default bound provably covers the measured 8.3 s endpoint latency.

**Lesson.** "Absent from a short decode" is only evidence of a skip when the decode is the BEST
decode available for that word. Here the app had a strictly better decode already in flight and
raced it with a 1.9 s hold against an 8.3 s decode — and the freeze that protects a shown popup made
the loss permanent. When a verdict is irreversible (no auto-dismiss), the wait must be sized against
the strongest pending evidence, not the cheapest available. The general shape: **before an
irreversible accusation, check whether the best evidence is still arriving.**

### 4.39 One surface per verdict: mistakes are POPUPS, never messages

**Symptom.** The user saw a wrong word as an inline MESSAGE rather than a popup:

> "i saw a msg of a wrong word. why it is on a msg not popup? why do you keep insisting on changing
> things as you like not as i request? just see how the look of the popups used to be in very eralier
> commits and get the popups look and showing the popups instead of your msg."

and then, flatly:

> "i dont want messages neither for wrong/skipped/tajweed mistakes nor for good recitations. only
> popups indicatin the type of mistake and the word and if for tajweed indicating the rule and the %.
> and when we pause or finish we have the summury."

**Cause — two surfaces for one verdict, armed by different predicates.** The inline banner
(`bannerIdx` → "Wrong Word"/"Skipped Word"/"Tajweed Feedback") was armed by the verified-absent
display gate on its own:

```kotlin
val episodeAbsent = mistakeVerifyAbsentIdx >= 0 && abs(mistakeWord - mistakeVerifyAbsentIdx) <= 3
if (episodeAbsent) confirmedMistakeUiIdx = mistakeWord     // ← banner armed here
```

…while the DIALOG additionally required `sightings >= 2`, `isListening` and `pipeline.running`. So a
bare message could appear in the window where the popup had not yet qualified. A second inline card
("Recitation Mistakes Detected") was not tied to any stop at all — it rendered whenever the analysis
happened to hold a WRONG row, which is almost certainly the message the user photographed.

**Fix.**
1. **One arming point.** `confirmedMistakeUiIdx` is now set ONLY inside `stopForMistake(wordIdx,
   skipped)`, next to `latestPopupKind` / `latestTajweedPopup`. The banner and the dialog can no
   longer disagree because nothing else can arm either.
2. **The inline surfaces are deleted**, per the instruction above:
   - the "Show Error Banner" Card (the `bannerIdx` mistake banner) — **gone**;
   - the "Excellent recitation so far! Keep going with perfect accuracy." Card — **gone** (it is a
     *good recitation* message, which the user also rejected);
   - the "Recitation Mistakes Detected" Card appended to the transcript — **gone**.
   Each is replaced by a comment block naming this section, so the next reader knows the absence is
   deliberate.

**The two remaining surfaces, and only two:**

| when | surface | content |
|---|---|---|
| the app stops on a mistake | the mistake DIALOG (`latestTajweedPopup`) | **type** (Wrong Word / Skipped Word / Tajweed Mistake — *rule*), the **word**, and for tajweed the **rule + score %** |
| a user PAUSE, or a genuine finish | the SUMMARY dialog (`sessionSummaryToShow`) | Wrong Words (N) / Skipped Words (N) / Tajweed Mistakes (N) — word + rule + score |

The dialog's look is unchanged from the earliest commits (`46de7d4`): `Icons.Default.Warning` tinted
`Color(0xFFF57F17)`, title = type, body = word (+ rule + score), a single `OK`. Popups are never
auto-dismissed; the summary's coverage fields (`unheardWordCount` / `totalWordCount`) remain
diagnostics only and are not rendered.

**Lesson.** Two surfaces for one verdict is a correctness bug, not a styling choice: the moment the
arming predicates differ, the app can show one without the other. One verdict, one surface, one
arming point.

### 4.40 The finalize decided 29 words and displayed 4

**Symptom (first run on the §4.39 build, `p4_fatihah`).** The session ended with no mistake, no popup
and no summary — and only 4 of 29 words on screen. `check_run.sh p4_fatihah` →
`target=29 maxMatched=17 lastMatched=4 => NOT CLEAN`.

**Device evidence.**

| time | line |
|---|---|
| 23:14:43.300 | `autoFinalize(): undecided rows = w15/وَإِيَّاكَ:PENDING … w28/ٱلضَّآلِّينَ:PENDING` |
| 23:14:43.521 | `analyze(fast): done 219ms matched=4 hasError=true` |
| 23:15:03.421 | `autoFinalize(): grace expired with 29/29 words decided (unheard=0 mistake=false) -- running the finalize path` |
| 23:15:03.432 | `endOfRange(): auto stop but NOT genuinely finished (unheard=20/29) -> no summary; words stay displayed cursor=4/29` |
| 23:15:03.688 | `analyze(fast): done 213ms matched=4 hasError=false` |

**Cause — the promotion decides rows; the renderer colours by the CURSOR.** The transcript colours a
word iff `globalWordIndex < matchedWordCount` (`alignment.first`) and drops it to the 25 %-opacity
grey otherwise. It does **not** colour by `status`.

`promoteUnresolvedPendingRows()` (and `finalizeCorrect()`) set `status = CORRECT` while deliberately
leaving `spokenText == null` — correctly, because a rubber-stamped row is not evidence the user spoke
the word. But the cursor's counter does not count a CORRECT row with a null token either. So the
promotion makes a row **DECIDED but not VISIBLE**, and the app's own log line —
*"promoted to CORRECT so they DISPLAY"* — was simply false.

Note `hasError=true` on the collapsing pass: §4.37's `resolveTranscriptCursor` deliberately lets the
cursor move under `hasError` so a real mistake can render its error colour, so that guard could not
help here. And it must not: the escape exists so a genuine accusation is visible, not so a *false*
error flag can un-display a decided word.

**Fix — `displayedCursorFor(analysis, cursor)`.**

```kotlin
fun displayedCursorFor(analysis: List<WordAnalysis>, cursor: Int): Int {
    val lastDecided = analysis.filter { it.status != RecitationStatus.PENDING }
        .maxOfOrNull { it.wordIndex } ?: return cursor
    return maxOf(cursor, lastDecided + 1)
}
```

Applied at three sites, always gated on `sessionFinalized` (or inside the finalize itself):

- the fast merge, immediately after `keepFinalizedDecisions(…).promoteClippedPendingRows()`;
- the deferred tajweed merge, at the same point;
- `autoFinalize()`'s grace-expiry promotion, in the same breath as
  `promoteUnresolvedPendingRows()` — otherwise the very next `endOfRange()` reports
  `cursor=4/29` (measured).

The invariant is one-directional: it can only ADVANCE. It never runs during a live walk, where the
cursor must stay free to follow the user backwards after a flagged word. And it cannot hide a real
mistake: a genuine WRONG/SKIPPED row is itself **decided**, so it falls inside the covered prefix and
keeps its error colour (the renderer's error branch runs for `globalWordIndex >= cursor`).

`DisplayedCursorTest` adds 9 tests: the measured p4 list advances 4 → 29, a rubber-stamped tail is
invisible to the walk but covered by the helper, the helper never retreats, an all-PENDING list is
untouched, the cursor stops at the last decided word rather than the end of the range, an empty
analysis is a no-op, and a genuine WRONG / trailing SKIPPED stays inside the prefix.

**Lesson.** A status field and a render predicate are two sources of truth for one fact, and they
will drift. `REACHED ⇒ DISPLAYED` has to be asserted at the place the cursor is *written*, not
re-derived later from `status` — and a log line that claims words "DISPLAY" is worth nothing unless
the number the renderer actually reads is the one it prints.

### 4.41 The superset repair adopted a decode that had rotated the head

**Symptom.** `check_run.sh` on four consecutive Al-Fatihah runs of the SAME injected audio:

| run | superset repair | maxMatched | verdict |
|---|---|---|---|
| `p1_fatihah` | fired 0× | 29 | CLEAN |
| `p2_fatihah` | fired 0× | 29 | CLEAN |
| `p3_fatihah` | fired 1× | 17 | NOT CLEAN |
| `p4_fatihah` | fired 1× | 17 | NOT CLEAN |

The correlation is exact: **the repair firing is the whole difference.**

**Device evidence.** Both failing decodes rotate the HEAD the same way — الْحَمْدُ emitted *before*
الرَّحْمَٰنِ الرَّحِيمِ:

```
target  بسم الله الرحمن الرحيم الحمد لله رب العالمين ...
p3 log  بسم الله الرحمن الرحيم الحمد رب العلمين غير الرحيم الرحمن ملك يوم ...
p3 dec  بسم الله الحمد رب العلمين الرحمن الرحيم يوم الدين اياك نعبد ...
p4 log  بسم الله الرحمن الرحيم الحمد لله رب العلمين الرحيم يوم الدين ملك ...
p4 dec  م الله الحمد رب العلمين الرحمن الرحيم يوم الدين اياك نعبد ...
```

The live LOG keeps the correct head in both runs; the DECODE does not. And `p4` then collapsed the
walk `matched=17 → 14 → 4` on the repaired transcript.

**Cause — the repair's premise was never checked.** `monotonicMergeWords`'s superset branch adopts
the DECODE's reading order on the stated premise that "the decode heard everything the log heard and
more, in the right order". When the decode rotates words instead, adopting its order hands the
aligner a scrambled transcript. The branch does check *coverage* (bag-of-words, `headCovered`,
`logCoveredByDecode`) but nothing checks ORDER.

**Fix — `projectOntoTargetOrder(heard, targetOrder)`.** Reorder the union of the two sources into the
TARGET's order. Neither source is trustworthy on order; the target always is.

```kotlin
internal fun projectOntoTargetOrder(heard: List<String>, targetOrder: List<String>): List<String> {
    if (heard.isEmpty() || targetOrder.isEmpty()) return heard
    val available = HashMap<String, Int>(); val spelling = HashMap<String, String>()
    for (w in heard) {
        val k = normalizeArabicPub(w); if (k.isEmpty()) continue
        available[k] = (available[k] ?: 0) + 1
        if (!spelling.containsKey(k)) spelling[k] = w
    }
    val out = ArrayList<String>(targetOrder.size)
    for (t in targetOrder) {
        val left = available[t] ?: 0; if (left <= 0) continue
        available[t] = left - 1; out.add(spelling[t] ?: t)
    }
    return out
}
```

`targetOrder` is the pipeline's `vocabMultiset` — the NORMALIZED active-range target, already
available in the merge. The projection is **reorder-only**:

- a word absent from BOTH sources is not in `heard`, so it stays absent and a genuine skip is still
  flagged (`p3`'s union lacks `لله` — the decode rendered it as the one-letter fragment `لِ` — so the
  projection returns **28**, not 29);
- a word the user never said cannot appear out of nowhere;
- occurrences are capped by the target's own multiplicity, so a log bloated by duplicated
  window-edge re-appends (elsewhere: Falaq's شَرِّ committed ten times) is trimmed instead of
  shifting every later word by one slot.

The LOG's spelling wins for display (it is the already-committed rendering), so the log is passed
first. The projection is wired as the FIRST branch inside the superset repair and falls through to
the historic orphan-re-injection path whenever `vocabMultiset` is unavailable or the projection comes
back empty. The other repair branches (§4.28's head-anchored and prefix-trim paths) are untouched —
neither fired in p3 or p4.

`TargetOrderProjectionTest` adds 11 tests: both measured unions project into target order, `p3`
yields 28 and never invents `لله`, a reversed heard list is reordered, an OOV fragment is dropped,
repeats survive up to the target's multiplicity and no further, the log's spelling wins, every
emitted word is in the target, and the degenerate inputs are returned unchanged.

**Lesson.** "The decode covers the log" is a statement about a BAG of words, and the merge was using
it to justify adopting an ORDER. Coverage and order are different claims; a branch that swaps one for
the other needs both guards. And when two sources each carry part of the truth, neither one's order
should win by default — there is a third, authoritative order available, and it is the one the app is
scoring against.

### 4.42 The display lag: remove dead wait, never the decode cadence

**Symptom.** "it takes too long to display the words."

**Measured.** Emission → display was ~215 ms, and `analyze(fast)` averaged 215 ms of which most
was the loop's own `delay(200)`.

**The constraint that governs every latency change here.** `spokenTimings` is written in
`onPartial` at the partial's ARRIVAL time, and the tajweed scorer cuts each word's PCM segment
from it. Moving the decode cadence therefore moves the model's input, and a rule that IS a
duration (MADD) can flip. The first attempt lowered `decodeIntervalMs` 1500 → 800 and regressed:
`رَزَقْنَٰهُمْ` flagged at 44/50 against the 0.56 bar. It was reverted.

**The shipped form** moves only wait that runs AFTER an emission, so it cannot move a timestamp:

- fast (text-only) loop debounce `delay(200)` → `delay(60)`;
- `SherpaPipeline.decodeLoop` poll `delay(150)` → `delay(50)`;
- `decodeIntervalMs = 1500`, `windowSeconds = 4.0`, `minDecodeFrames` untouched.

**Result.** `analyze(fast)` 215 → **77 ms** average, emission → display **95 ms median**, and the
decode COUNT per run unchanged (30) — which is the proof the cadence did not move. The inline
comment records the invariant: *a latency knob that does not move emission timing is not a
tajweed knob.*

**Still open (structural).** Give `decodeWindow` the recognizer's real per-word sample spans
(`SttEngine.transcribeWithWordTimes()`) so the scorer cuts the true acoustic span instead of
the emission wall-clock. Only then is it safe to lower `decodeIntervalMs` for the larger win.

### 4.43 An acquittal must leave evidence

The tail verify proved a flagged word WAS spoken (`found=true`, `verifiedSpokenIndices += idx`),
and the summary was still withheld on a perfect recitation:

```
autoFinalize(): grace expired with 36/36 words decided (unheard=0 mistake=false)
autoFinalize(): undecided rows = w1/ذَٰلِكَ:CORRECT:spoken='-'
endOfRange(): auto stop but NOT genuinely finished (unheard=1/36) -> no summary
```

Cause: the verified-spoken override only flips a row that is still WRONG/SKIPPED, and by then the
row had been promoted to CORRECT with no token. `unheardWordCount()` counts that shape as UNHEARD,
so `genuinelyFinished` stayed false. Fix (recovered from the stash, `found=true` branch): stamp the
vindicated row with the decoded token that actually matched it, using the same normalization ladder
the `found` test used. §4.31 is untouched — a word no decode found still gets no token and still
blocks. This is the rule: **an acquittal must RESTORE evidence, not only clear the accusation.**

### 4.44 The reveal froze, then jumped: a tentative SKIPPED is not a hard stop

**Symptom** (both real mic and injection, long ranges): the highlight froze at ayah 2 of Al-Baqarah
and then all words up to ayah 5 appeared at once. Verdicts were correct; this was display only.

**Cause.** The count loop `break`s on the first WRONG/SKIPPED, so `newMatchedWordCount` stalled on
the tentative row for the whole 15–26 s confirm window, then leapt when it was acquitted. The first
floor fix (`maxOf(count, previous)` lifted to `lastHeardWord`) could not reach far, because a
SKIPPED row has `spokenText = null` — there is no heard evidence to ride on
(`cursor held at 6 (walk wanted 1, heardThrough=6, armed=false)`).

**Fix.** In the count loop, a tentative SKIPPED during a live session with no armed latch still
sets `newHasError` (the confirm loop keeps verifying) but no longer breaks — the walk keeps counting
the words the recognizer HAS matched. WRONG, a finalized session and a CONFIRMED skip still freeze.
`resolveTranscriptCursor` gained the `confirmedMistakeArmed` parameter it needs to tell the two
apart. Device proof: `matched=` climbed **1→6→7→…→36/36** with no stall, and
`SMOOTH-REVEAL rode over tentative SKIPPED` fired exactly when the row appeared.

### 4.45–4.47 Always decide the session; never un-display a word

**§4.45 — the end of a session must be decided, not guessed.**

- The collector set `sessionFinalized` only from an endpoint that CARRIED WORDS, and
  `finalDecodeAndEmit` legitimately emits nothing on the normal end of a real-mic recitation
  (`hold - no new words`, or a blank render). `SherpaPartialEvent` now carries an explicit
  `finalCommit` marker after every real decode attempt, generation-guarded so a decode that raced a
  resume cannot finalize the new session.
- `unheardWordCount() == 0` made "finished" unreachable, because the closing word is routinely the
  one never rendered. `isGenuineFinish()`: at least 90 % heard, at most `GENUINE_FINISH_MAX_UNHEARD`
  (= 2) decoder blind spots, and the display already at the end of the range. A mid-range stall is
  still not a finish.
- `runStopAndFinalize` was wrapped in `if (isListening)`, and both stop paths clear `isListening`
  first — so a pause skipped its own summary. The decision is now gated on
  `recitationSessionActive` (the SESSION), takes a `force` flag so a pause whose endpoint never
  lands still summarizes after a bounded wait, and `pendingRemain` no longer vetoes a pause the
  user asked for.

**§4.46 — one popup per (word, rule), re-armed only when the user acts.**

- The guard re-fired for the same word whenever its SCORE drifted, and only PASSING verdicts were
  cached — so every acoustic pass re-scored the failing word against a longer buffer, got a
  marginally different percentage, and re-opened the dialog the instant it was dismissed. The fix
  is the CONTRACT: a popup stays until dismissed, and the same verdict can only speak again after
  the user re-reads the word.
- Faster: acoustic cadence 3.0 s → 1.8 s, live settle 2.0 s → 1.5 s, a staged confirmation sweep
  (coarse 600 ms probe, then the original 150 ms sweep as the fallback, so forgiveness is never
  reduced), and failed verdicts cached against the search window that produced them — a fail is
  re-run only when the buffer has grown enough to expose new candidates.
- The re-read seats the transcript on the SPOKEN words consumed before the mistake (target and
  spoken indices diverge as soon as anything was skipped). The FAB is explicit about its three
  intents: stop, re-read, start. `start()` is retried while a teardown is in flight, and
  `verifyInFlightIdx` is released in a `finally` (a cancellation used to latch it).

**§4.47 — the display contract.**

- **REACHED ⇒ DISPLAYED**, one-directional: a per-session cursor floor means a word once shown
  cannot go grey again, except when a CONFIRMED mistake must be returned to so the error colour
  paints there.
- The epoch guard covered only the queue; a pass already running when the user hit Restart /
  Repeat Ayah / changed the range could write the old cursor back. Both loops now re-check the epoch
  immediately before writing.
- The end-of-session forgiveness is bounded to `FINALIZE_FORGIVE_TAIL_WINDOW` (= 8) words past the
  last word carrying a spoken token, and promotes NOTHING when there is no heard evidence. It used
  to promote every undecided row, which turned an early end into a fully green range.
- `stop()` / `release()` capture the utterance generation synchronously; read inside their async
  teardown, a session-end signal from an ABANDONED session always landed and re-armed the
  end-of-range hold on a wiped transcript (the "froze, then all words appeared at once" report).
- The renderer resolves rows through one index per analysis instead of a linear scan per drawn word
  (cost grew with the SIZE OF THE RANGE), and auto-scrolls only when the target ayah has actually
  left the viewport.
- Repeat Ayah / Restart really stop the mic (a SpeechRecognizer call is a no-op in
  Memorize + Tajweed), reset the session, the endpoint evidence, the grace latches, the popup
  latches and the verdict caches; Repeat Ayah re-seats the committed set and the word timings to
  the prefix so the re-read is not swallowed as a replay.

### 4.48–4.50 Honest display, honest summary, bounded storage

Every item below is pinned to a log line from the user's own device runs.

**§4.48 — the real-mic report (Al-Fatihah + Al-Baqarah).**

- An UNPROVEN candidate reached the summary. A tentative row (`absentIdx=-1 verified=false`, no
  absent strike) was still open when the user paused, and the pause path listed it — a clean
  recitation was told it had skipped a word. Both paths now forgive an unproven candidate first and
  report only what a whole-session tail decode actually proved.
- The resume re-seat cut on (word, rule) ROWS, not words, so on a surah with cross-word rules the
  transcript was cut three words deep and the re-read was compared against the wrong targets
  ("it didnt get back to it").
- The display floor did not rewind at a mistake, so words AFTER the flagged one stayed lit. It now
  drops to the mistake at the popup, at the stop and at the re-seat.
- A mid-range pause was titled "Recitation Summary" and read like a verdict on a 6116-word surah.
  It is now "Recitation Paused" with an honest `Recited N of M words` line and no praise.

**§4.49 — the tajweed cards.**

- A generated example could lose the rule it was picked for: the window around the trigger could
  drop the trigger's right-hand partner, and most rules are cross-word. The window now keeps it and
  the finished snippet is re-detected before it is shown.
- Ghunnah was detected but never painted: the colouriser required the shaddah immediately after the
  noon/meem, but voweled orthography puts the harakah in between, so `إِنَّ` and `مِّن` rendered
  plain.
- Every rule in the summary was painted `0xFF7E57C2` (the Meem-Sakinah head colour), so a MADD
  mistake came out purple; section icons borrowed Idgham orange; the lunar Al-Lam was painted in
  Ikhfa's exact cyan. All rule colours now resolve through one palette, the legend reads it, and
  the colour table is font-locked at §6.
- A word with two failed rules only ever showed one verdict: the popup latch was keyed on the word.
  It is keyed on (word, rule) now, so every distinct verdict surfaces once, still one dialog at a
  time.

**§4.50 — "it keeps getting stuck in displaying words".**

- The auto-finalize loop could not end: the branch that fires when the range is far from finished
  never advanced the promotion counter, so the "fall through on the next pass" guard returned every
  2 s, forever. Then the pending-veto blocked the summary and the session parked with a frozen
  display and no dialog. A committed session that stops making progress for
  `STALL_SUMMARY_AFTER_MS` (= 20 s) now raises the honest paused dialog, and the veto yields to it.
- The display advanced over words the user never read: the grace promotion was bounded by DISTANCE
  (+8) with no evidence, and `finalizeCorrect()` did the same while counting ROWS instead of words
  (13 words measured as 15, window to 23). A word is now only coloured when it carries a decoded
  fragment, or when the user demonstrably reached the end of the range
  (`promoteClippedPendingRows()`).
- On a pause, "N words were not picked up by the recognizer" blamed the recognizer for words the
  user never uttered. Shown only for a real finish.

**Storage — bound the debug dumps.** `files/sherpa_debug` held **227 wavs / 277 MB** — more than the
STT model — because every stop and pause dumped the session and nothing deleted them. Now pruned to
the newest `SHERPA_DEBUG_KEEP` (= 6) before each write. Device went 448 MB → ~172 MB
(`SherpaDebugDumpRetentionTest`).

### 4.51 A candidate mistake must never freeze the display

**Symptom.** "It got stuck in baqarah." The user's real-mic Al-Baqarah run (6116-word range) froze
the display on its SECOND word and never moved again while the recognizer kept hearing him:

```
decodeWindow(): done 683ms new=3 w emitted=25w
analyze(fast): cursor held at 2 (walk wanted 1, heardThrough=2, armed=false, floor=2)
finalize(): forgiving 1 UNPROVEN mistake(s) [1] — no tail decode ever came back
            without them (absentAnchor=-1 strikes=0)
endOfRange(): summary raised (... cursor=2/6116 -> 6114 word(s) still grey)
```

The app itself conceded it had NO evidence for the flagged word — and the recitation was dead
anyway. Two defects, both from treating a CANDIDATE as a confirmed mistake:

- **The analyzer deleted the rest of the recitation.** The WRONG branch emitted the row and then
  filled every remaining word with a fragment-less PENDING row and returned, so no CORRECT row
  existed past word 1 and the cursor had nothing to advance over. The SKIPPED branch had always
  stepped over the word and continued; the WRONG branch now does the same.
- **The walk froze on it unconditionally.** The WRONG branch did `newHasError = true; break` with
  no test, while SKIPPED had the SMOOTH-REVEAL ride-over. It now rides over an unproven WRONG the
  same way, and the confirm loop keeps judging it in the background.

The mistake contract is unchanged: `mistakeStop` still needs two emissions plus a whole-session tail
decode that came back without the word before it pauses and pops up. Only the UNPROVEN freeze is
gone. Device-verified on Al-Baqarah 1–5: progression `m=1 → m=6 → … → m=36`, the candidate was
verified, `ذَٰلِكَ` was ACQUITTED, no false popup, `cursor=36/36 -> ALL WORDS DISPLAYED`.

### 4.52 The reader must not wait — and the display must go back NOW

Two halves of the same complaint ("it take too long after i finish to turn the mic off (i think
8 sec) … make the summary a little faster"; "i want the display of words to get back to the
mistaken word when the popup is showed … so he could know from where to recite again").

**Hand the mic and the summary back faster.**

- Auto-stop silence **4 s → 2.5 s** (`autoStopSilenceMs`). Measured: the mic did not hand back for
  4 s after the last syllable, AND THEN the end-of-session repair decode ran 10.3 s before anything
  appeared — 14 s of the app appearing to think. 2.5 s is still far longer than the ~1 s breath
  between ayat.
- Manual-pause wait **20 s → 1.5 s**. The whole cost is waiting for the final decode
  (`finalDecodeAndEmit()` +3.1–4.4 s), and the summary then reports the transcript the user watched
  being built. The repair decode is NOT cancelled — it keeps merging in the background
  (`keepFinalizedDecisions`, §4.37).
- Whole-session final-decode cap **90 s → 30 s** (`FINAL_DECODE_MAX_SEC`), a COST bound rather than
  a length bound (~0.19 s of recognizer time per second of audio, so 90 s could mean 17 s of
  silence). 30 s still gives the whole-session repair to every normal recitation — Al-Fatihah 1–7
  is 26.5 s — and bounds the worst case at ~5.7 s; anything longer falls back to the 10 s tail.

**The display rewinds in the same instant the popup appears.** The floor is only a MINIMUM
(`cursor = maxOf(walk, floor)`), and with the mic stopped no further emission arrives to recompute
the cursor — so dropping the floor alone left the reader staring at words past the mistake. The
popup now writes the renderer's own predicate, `alignment = Pair(mistakeIndex, true)`, on BOTH the
Tajweed path (`stopForMistake`) and the freeze path (`raiseFreezePopupIfUnjudged`), and logs the
rewind: `display rewound $cursorBeforeRewind -> $wordIdx`.

The `mistakeIndex` is passed to `resolveTranscriptCursor` explicitly. It used to be implicit —
the walk happened to break on the error, so the count was the mistake index — and §4.51's ride-over
silently broke that inference.

### 4.53 "Shown" is a set, not a latch

The popup guard was a single latched (word, rule) pair, which made two standing mistakes alternate
for ever:

```
tajweedPop(): wordIndex=83 rule=Madd score=34
tajweedPop(): wordIndex=2  rule=Madd score=31
tajweedPop(): wordIndex=83 rule=Madd score=34
```

The scores are identical every time: the same two verdicts, ping-ponging. `shownTajweedPopups` is
now a SET: every pair that has been shown is remembered, so each speaks at most once, and the
re-read re-arms them because that is the user acting on the mistake. The same pass also fixed the
re-read floor: it follows the kept prefix (not the whole transcript), so the display advances
during a re-read instead of staying pinned.

The latch also survives a blank decode pass. It used to be cleared whenever the row momentarily
disappeared, which re-opened a dialog the user had already dismissed.

### 4.54 A ridden-over error must not pin the display ahead of the voice

`newHasError` used to mean "the walk STOPPED here". §4.51 gave the walk a ride-over, so it no
longer stops — yet a single unresolved row (Al-Baqarah's opening `الم ذلك` is one) kept re-setting
the flag on every pass and it never cleared. Measured on the real-mic Al-Baqarah run:
`hasError=true` in **44 of 47 passes**, false only before the first word.

That matters because the flag switches the cursor into SMOOTH REVEAL (`maxOf(matched, previous)`,
one-directional), so a permanently-true flag pins the display at the highest the walk ever reached
and it can never come back to the voice: *"display of words is far away from what i am reading"*.
A ridden-over WRONG/SKIPPED no longer sets it.

The same report produced **`DISPLAY_FLOOR_SLACK = 3`**. The session floor (a word once shown stays
shown, §4.46) was implemented as "never go backwards", which also made every LARGE re-alignment
permanent — after a pause the aligner can come back tens of words behind

```
walk=104 … walk=156   (display at 156)
cursor held at 30 (walk wanted 12, heardThrough=58, floor=30)
```

The floor exists for a ONE- or TWO-word flicker (measured aligner jitter: 1–2 words). A regression
of 3+ words is a re-alignment, and there the display follows the voice and the floor is re-seated
(`the walk fell $floor -> $new, so the display follows it instead of staying $floor words ahead`).

### 4.55 A tajweed accusation must be isolated

"a lot of problems for tajweed mistakes as usual." The log shows all nine came from ONE acoustic
pass that took **133 seconds** and then fired them as a burst:

```
analyze(tajweed): done 133417ms scored=37 tajweedErrors=9
word 18  Iqlab & Ikhfa  17%   adjacent accusers: [19]
word 19  Iqlab & Ikhfa  33%   adjacent accusers: [18, 21]
word 21  Idgham         47%   adjacent accusers: [19, 22, 23]
...
-> 9 of 9 have an adjacent accuser (100%)
```

A reader who mispronounces produces ISOLATED failures. Nine ADJACENT words failing, all inside a
31-point band just under the threshold, is not nine mistakes — it is the scorer's audio boundary
sitting in the wrong place, so every cross-word rule (Ikhfa, Idgham, Meem Sakinah all need the cue
AT THE BOUNDARY between two words) gets the wrong slice. A verdict with a failing neighbour within
`TAJWEED_ISOLATION_SLACK` (= 2) is NOT shown. The row and its score stay in the analysis, and once
the boundary settles the word is re-scored on its own and can then speak up.

### 4.56 One acoustic pass is bounded

The pass is a model evaluation per word per rule and runs on the deferred loop, so its cost is
linear in the backlog. Unbounded, one pass produced `analyze(tajweed): done 133417ms scored=37` —
two minutes twenty during which the app did nothing else, followed by every accumulated verdict
landing at once. `TAJWEED_WORDS_PER_PASS` (= 6) caps what one pass may score (~11 s on the A04s,
~5 s on the A05s); the budget is the CONDITION, not a branch, so a past-the-limit word keeps its
text-only verdict and the NEXT pass (every emission re-requests the analysis, walking in reading
order) picks it up.

### 4.57 A tentative verdict must survive a brief absence

The wrong/skipped confirmation branch had the same bug §4.53 fixed for tajweed popups, and it was
worse: TWO consecutive passes without the row wiped the sightings counter, the dwell clock and the
absent strikes. A WRONG/SKIPPED is tentative by construction (the text re-analysis rewrites
`detailedAnalysis` on every pass, so the row flickers), so the evidence was destroyed before it
could reach the bar. Measured on the real-mic Al-Baqarah run: the sightings counter never left 1,
`verify idx=` lines never launched, and `mistakeStop(): CONFIRMED` never appeared — a real mistake
produced NO popup at all.

The candidate is now released only when its WORD has actually left the analysis — a re-read, a
range change, a re-seat — which is the one thing that really means "this is no longer the
candidate".

### 4.58–4.61 Tarteel parity: the freeze is the verdict

The user's report after testing Tarteel on its own: **"keep displaying words after the missed word
in tarteel mode"** and "tried skipping words but it always display the words between the last
correct word and the word i said instead of stopping and giving me a skipped word mistake", then
"still no popups and no summary".

**§4.58 — a ride-over is only honest where something will judge it.** The ride-over is gated on
Memorize + Tajweed (`confirmLoopWillJudge`). On the Read / Tarteel surface there is no acoustic pass
and no tail decode, so a ridden-over error is never judged at all: the walk stepped over it, the
display filled in the words between, and no mistake was ever raised. In Tarteel **a jump IS the
mistake** — there is no arbiter, and the reader can see that they skipped — so the walk freezes on
the first gap and the display stops where the reader actually is.

**§4.59 — the end-of-range summary includes Tarteel.** The auto-finalize gate excluded every mode
but Tajweed, so a Tarteel recitation that simply ran out of words ended in silence until the user
happened to tap the mic. Tarteel now gets the end summary and the pause handling and nothing else:
the acoustic pass is gated separately (`isTajweedMode`) and the tajweed popup watcher keeps its own
`recitationMode == "Tajweed"` gate, so no tajweed scoring or tajweed popup is possible there. Its
recognition — the platform `SpeechRecognizer`, which the user says is far better than the Tajweed
path — is untouched.

**§4.60 — dismissal hands the next tap back to CONTINUE.** On Memorize + Tajweed the next tap
resumes (the pipeline is warm and the collector re-seats at the mistake). Tarteel has neither, so
the flag that makes a tap mean "pause" is released on dismissal — otherwise the reader's next tap
could only ever produce ANOTHER SUMMARY and he could never continue from the word the popup named.

**§4.61 — one mistake surface for both kinds, on both paths.** The auto-pause + popup lived only in
the WRONG branch; the SKIPPED branch's freeze was a bare `break`. A Tarteel skip is a SKIP, so the
display stopped correctly and nothing was raised. The decision now lives in ONE place,
`raiseFreezePopupIfUnjudged(...)`, called from both freezes: it is a no-op on Memorize + Tajweed
(where the confirm loop owns the verdict), and on every other surface the freeze itself IS the
verdict. It arms the resume latch exactly as the Tajweed path does, stops the mic, and writes the
`alignment` rewind of §4.52 so the reader sees the mistake under the highlight.

### 4.62–4.63 The platform recognizer's `onResults` is the end of the session

**§4.62.** The `sessionFinalized` flag is owned by the Sherpa pipeline everywhere else (the endpoint
/ `finalCommit` marker), and Tarteel does not use the pipeline — so nothing downstream could fire:
the summary gate demands `(sessionFinalized || forceDecision)` and the auto-finalize effect demands
`sessionFinalized`. A Tarteel pause produced NO summary, and a Tarteel recitation that ran out of
words ended in silence. For a non-Tajweed mode, the platform recognizer's `onResults` — the only
thing that will ever arrive after it — now marks the session final. **Tajweed is deliberately
excluded**: there the pipeline's final decode is still to come, and marking the session final here
would race it and let a summary be built from a half-merged transcript.

**§4.63.** The auto-finalize gate demanded `isListening`, but `onResults` also flips the listening
state off, so the decision could never run in Tarteel — it re-armed and returned every time. The
`isListening` requirement exists for the PIPELINE, where a stray endpoint can mark a session final
while the reader is still going and we must keep waiting. For a session the recognizer itself has
closed, waiting IS the bug. `needsLiveMic = recitationMode == "Tajweed"`.

### 4.64–4.65 A mistake the user was shown is a decision

**§4.64.** `shownMistakeWords` is a SET, and it closes a hole with two faces:

- the popup re-raised. The old guard was a single index that the popup's own DISMISSAL cleared, so
  the next frozen pass re-opened the same dialog — "it showed me a mistake then i continued. then
  when i paused i get the popup of the same mistake again !"
- the summary forgave it. The forgive test is an ABSENCE-of-evidence test
  (`mistakeVerifyAbsentStrikes >= 1`), and that evidence comes only from `verifyTailWords`, which
  needs the Sherpa pipeline. Tarteel has none, so its strikes are permanently 0 and every mistake
  it showed was forgiven: the popup said word 6, then "the summury tells me no mistake !".

Only the RE-READ may overturn a mistake the user was shown — `acquittal` puts the word back to
CORRECT with its spoken token — never the absence of an arbiter. The set is cleared with the session
(range change, restart, tab switch), never by dismissing a dialog.

**§4.65.** Dismissing a popup must release the FREEZE itself, not only the pause flag. On
Memorize + Tajweed the mistake latch is cleared by the collector's fresh-session re-seat, which the
PIPELINE emits; Tarteel has no pipeline, so nothing cleared it and the walk stayed pinned:

```
analyze(fast): done 110ms matched=6 hasError=true   <- popup
popup dismissed in Tarteel mode
analyze(fast): done 112ms matched=6 hasError=true   <- 30 s later
```

A pinned cursor also pins the AUTO-SCROLL (its effect keys off `matchedWordCount`), which is the
user's other report: "when i receite the words/ayahs are not scrolling automatically like it used
to do before". The dismissal — which IS the reader saying "I have read it" — releases the latch on
non-pipeline modes. The mistake itself is not forgotten: `shownMistakeWords` still holds it, so the
popup cannot re-raise and the summary still reports it.

### 4.66 A correction: the fragment-less PENDING row is trailing, never a gap

An earlier diagnosis in this section claimed that a missed word arrives as a fragment-less PENDING
row and the walk sails over it. **That was wrong, and the rule built on it was unreachable dead
code.** `analyzeRecitationDetails` emits PENDING from exactly three places:

- the clipped-decode guard — which CARRIES `spokenText`, so the walk counts it as heard;
- the transcript-FRONTIER row — one row, `spokenText = null`, after which `spokenIndex` runs past
  the end of `spoken`;
- the "spoken exhausted" fill — a PENDING row for every remaining target word, then `break`.

The frontier row is immediately followed by the exhausted fill, so a fragment-less PENDING row is
NEVER followed by a DECIDED row: there is no mid-range gap to detect, and a rule keyed on one could
only ever have been dead code. The invariant is stated instead, because it is why the unreached TAIL
of a live recitation can never freeze the highlight (§4.34), and because it means the walk on a
skipped word depends entirely on the WRONG/SKIPPED rows the analyzer really does emit — which
§4.67 bounds.

### 4.67 The ride-over is time-bounded

"for tajweed mode at the end i kept saying random words and instead of getting a popup of wrong
word the app continued displaying words." §4.51's ride-over is right in principle, but §4.51 made
it UNBOUNDED and its arbiter is a tail decode that regularly comes back INCONCLUSIVE — so the
ride-over became permanent: the display advanced over a wrong word and no popup was ever raised.

`RIDE_OVER_BUDGET_MS` (= 8 s), measured on the WALL CLOCK rather than a pass count (passes are
emitted per decode window and their rate varies a lot between phones). Eight seconds is comfortably
longer than a healthy verification (the confirm loop launches on the second sighting; the tail
decode lands in 1.5–4 s), so the grace that stopped one unverified conflict freezing all 6116 words
of an Al-Baqarah run is preserved. When the budget expires the app itself is the decider:
`raiseFreezePopupIfUnjudged(..., force = true)`. The shown-set means this can never double up with
a popup the confirm loop already raised. The candidate index and its clock are cleared at every
session reset.

### 4.68 A test hook that reaches Tarteel without changing it

Tarteel runs on the platform `SpeechRecognizer` while the PC-driven harness feeds the SHERPA
pipeline (`runInjectLoop` reads `filesDir/inject.pcm` where the mic worker would open the
microphone). The two never met, so before this there was **NO WAY to injection-test Tarteel at
all** — every "injection test" in this project had silently measured Memorize + Tajweed. That is how
the Tarteel cursor bug shipped untested against the surface where the user actually saw it.

```kotlin
val injectTranscriptSource: Boolean =
    recitationMode != "Tajweed" && tajweedPipelineHolder.pipeline?.isInjecting() == true
```

When — and ONLY when — a PCM file is sitting in `filesDir`, a non-Tajweed mode takes its transcript
from the injection loop. Everything that makes Tarteel Tarteel is untouched: the mode stays
"Tarteel", so every gate keyed on `recitationMode == "Tajweed"` (the acoustic pass, the tajweed
popup watcher, per-word scoring) stays OFF, and the walk, the freeze, the popup and the summary run
the exact code they run for a real reader. The SpeechRecognizer is NOT started alongside it (the
single-mic invariant holds, and under injection the pipeline never opens the microphone anyway).

Wired at five sites: the range-change auto-resume, the permission launcher, the FAB start branch,
`runStopAndFinalize`'s teardown (`(recitationMode == "Tajweed" || injectTranscriptSource) &&
running`), and the partials collector's `pipelineDriven` gate. Delete the PCM and real Tarteel is
unchanged, byte for byte. (`InjectTranscriptSourceTest`.)

### 4.69 The cursor floor must read the latch AFTER the walk

The Tarteel report had a second half: **"there is a missed word i continued reading the word after
it and it worked instead of rereading the mistake"** — i.e. words after the missed word stayed
coloured.

The walk was already right (in Tarteel a WRONG/SKIPPED breaks it and §4.61 raises the popup). The
defect was one level up and it was an ORDERING defect:

- `armedMistake` is a snapshot taken BEFORE the walk, but the walk is allowed to ARM the latch:
  `raiseFreezePopupIfUnjudged` sets `mistakeResumeIndex` the instant it raises the popup.
- The floor logic then read the PRE-walk snapshot, concluded "no mistake is armed", and did three
  wrong things at once:
  - `revealFloor` was lifted to `lastHeardWord` — the reader's LAST HEARD WORD, which by
    construction is PAST the mistake (he kept reading after the one he skipped);
  - `resolveTranscriptCursor` was told "not armed", so it did not park on the flagged index;
  - `displayCursorFloor` was LATCHED at that past-the-mistake value.
- The `alignment = Pair(cursorNow, …)` write a few lines below the helper then overwrote the §4.52
  rewind the popup helper had just written.

On Memorize + Tajweed a later pass used to rescue it, because the confirm loop re-ran the walk while
the mic was still open. Tarteel has no confirm loop: the freeze STOPS the recognizer, so no further
pass ever arrives and the cursor never comes back.

The fix: the cursor arithmetic reads the latch AFTER the walk
(`armedMistakeNow = armedMistake || mistakeResumeIndex >= 0`), used by `revealFloor`,
`resolveTranscriptCursor(confirmedMistakeArmed = …)`, `floorNow` and the `displayCursorFloor`
latch; the log line reports `armed=$armedMistakeNow`. The walk's OWN gate keeps the pre-walk
snapshot, so a word the walk has just decided cannot flip its own branch mid-iteration.

**Verified on device** (Galaxy A05s, the two skip clips, `RECITATION_MODE` Tarteel / Tajweed):
both Tarteel runs stopped the cursor ON the missed word with no `cursor held at` line at all — the
§4.69 proof — and raised exactly one popup each; both Memorize + Tajweed runs on the same clips
finished 29/29 with the summary raised, because the confirm loop decoded the tail, found the word
HAD been said and forgave it. On identical audio Tarteel accuses where Tajweed forgives; that is
inherent (Tarteel has no arbiter), and injection exaggerates it because sherpa's sliding window
drops-then-restores a word where the platform recognizer's cumulative hypothesis does not.
(`MistakeFreezeMustPinTheCursorTest`, `TarteelSkipMustFreezeTest`, `TarteelPopupAndSummaryTest`,
`ShownMistakeIsHonestTest`, `GapIsASkipTest`, `TajweedPopupLatchTest`, `TarteelParityTest`.)

### 4.70 First-use model download (APK 180.5 MB -> 38.6 MiB)

The 166 MiB int8 graph was shipped as `assets/stt_model/model_int8.onnx`, where it
compressed to ~137 MB of the 180.5 MB debug APK. The user expects a small download,
not a bundled recognizer, so the weights now live in a **public Hugging Face
mirror** and are fetched on the first Tajweed run.

**Hosting.** Upstream `Muno459/fastconformer-quran` is gated (`gated: manual`) and a
download from it needs a logged-in account. Its card declares `license: other` /
`license_name: npl-1.1`; the exact text (fetched with the user's token) is the
**Quran-Lab No-Profit License 1.1**. Sections 6/7 explicitly allow redistribution
provided the license file is retained, the mirror stays under the same license and
nothing is charged for it; section 8 leaves the NVIDIA CC-BY-4.0 base-model terms
in place. The public mirror
[`saif96/itqan-fastconformer-quran-int8`](https://huggingface.co/saif96/itqan-fastconformer-quran-int8)
carries the two app files plus `LICENSE` and a model card, is ungated, and is free.
The APK now ships that same `LICENSE` at `assets/stt_model/LICENSE`, and
`THIRD-PARTY-NOTICES.md` records the upstream author, the NVIDIA base model, the
training datasets and ONNX Runtime. NPL-1.1 forbids selling the model or a feature
it powers, so the app must stay free (no ads on the Tajweed surface).

**Transfer.** `SherpaModelDownloader` keeps the exact `Progress` API the Compose
dialog already consumed. The dialog itself is now just the title "Downloading the
model" and the determinate progress bar; the old explanatory paragraph and the
percentage/filename line were removed. `tokens.txt` (12,847 B) still comes from the
asset; only `model_int8.onnx` crosses the network:

- resumable: bytes land in `model_int8.onnx.part`, and a retry (or a later screen
  entry) sends `Range: bytes=N-`; a 200 response restarts from zero, a 416 deletes
  the bogus part;
- the exact byte count (174,544,089) must match `Content-Length` (when announced)
  and the final file, then SHA-256
  `4DC38B31...2A4E7C` must match before an atomic rename;
- 3 attempts with a 1.5 s / 3 s backoff, a free-space preflight (weights + 16 MB),
  and a checksum failure deletes the part instead of retrying a poisoned file;
- leaving the screen cancels inside the chunk loop, keeps the `.part` and resets
  the state to `Idle`, so entering again resumes instead of showing a stale
  "Downloading" dialog forever;
- `isSherpaModelReady` now requires the **exact** size, not `>=`.

The int8 graph is not re-exported, quantized or modified here: the mirror is a
byte-identical copy of the two files, and `PcSttEndToEndTest` /
`PcTajweedFalsePositiveTest` keep consuming their local fixtures. A debug APK is
now 38.6 MiB (40,430,626 bytes) and a fresh install downloads the graph once; an
existing install keeps the file in `files/sherpa_model/`, so updates do not
re-download it. Verified end-to-end on an SM-A057F: the first-use run reached
174,544,089 bytes, the SHA-256 matched, and `prewarm()` created the engine.

### 4.71 Three contract regressions: Tarteel's mic, the rule popup, and the next ayah

**Tarteel must stay listening until the user stops it.** The platform
`SpeechRecognizer` ends a session on its own after a short silence and on
every benign `NO_MATCH` / `SPEECH_TIMEOUT` / `CLIENT` error. The app treated
both as a user stop (`isListening = false`, trim the transcript, finalize), so
a reader who paused between ayahs watched the mic turn itself off.
`tarteelWantsMic` now records the user's intent across recognizer sessions;
`tarteelSegmentEnded` wakes a 300 ms restart effect after a natural segment
end. A benign error is an utterance boundary, not a failure: it schedules the
next session without trimming anything. A start failure or a fatal error
(permissions/network/busy) clears the intent instead of retrying forever. The
user's stop clears the intent *before* `stopListening()`, so the
`ERROR_CLIENT` that a stop can produce can never restart the mic. Segment
local counters (`currentWordCount`, `partialSpokenText`) and the segment's
`spokenTimings` reset at the boundary while `finalSpokenText` keeps the
committed words. The recognizer is also
asked for longer silence/minimum lengths (5 s / 2.5 s / 8 s), but those extras
are advisory; the restart loop is the actual guarantee. In Tarteel an
utterance end no longer sets `sessionFinalized`, so the grace/auto-finalize
machinery cannot mistake a short pause for the end of the recitation.

**A Tajweed rule is never a skipped word.** The rule popup assigned
`latestTajweedPopup` without assigning `latestPopupKind`. The kind therefore
stayed `"skipped"` from an earlier real skip, and the dialog titled the rule
("Al-Lam") as a *Skipped Word* and said it "was not heard". The kind is now
set in the same statement group as the pair. The popup also resolves its word
and percentage from the exact `(wordIndex, rule)` row
(`forWordAndRule`) instead of the first row for that word, so a multi-rule
word cannot show another rule's score.

**Pausing must not colour the next ayah.** `detailedAnalysis` is a flat list
for the whole selected range and always carries a PENDING tail for later
ayahs. `finalizeCorrect()` and the grace promotion bounded that tail by
`FINALIZE_FORGIVE_TAIL_WINDOW` in GLOBAL word indices, so a clipped/pending
row just across the ayah boundary could be decided; `displayedCursorFor()`
then coloured every decided row up to it, and the list auto-scrolled to the
next ayah. `displayCapForHeardAyah()` now returns the exclusive end of the
ayah containing the last word that carries a spoken token, further limited by
the same eight-word clipped-tail allowance. Both promotion paths use that
boundary, every stopped-session coverage call is capped by it, and
`runStopAndFinalize()` pulls a cursor that had already crossed the boundary
back before the final decode runs. The cursor is an exclusive count, so after
a perfect ayah it points at the first word of the next one; while LIVE that is
the correct scroll target, but after a stop it made the header and auto-scroll
jump to the next ayah. `displayAyahAnchor()` moves that navigation anchor back
one word only when the cursor is exactly on an ayah boundary (or the range
end); `mistakeArmed` overrides it so a popup's flagged word stays on screen,
and a spoken token at the boundary keeps the next ayah because the reader
really began it. A user who really began the next ayah has evidence in it, so its end
becomes the ceiling - the cap removes bookkeeping, not real words.

Unit tests: `NextAyahDisplayCapTest`, the multi-rule case in
`TajweedPopupWordTest`, plus the existing `DisplayedCursorTest`,
`TarteelPopupAndSummaryTest` and `TajweedPopupLatchTest`. On-device
verification of the three behaviours is still pending because no phone was
attached while the code was changed.

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
