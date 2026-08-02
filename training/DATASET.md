# Dataset — corpus, labels, shards

This document is a **human-maintained** companion to the auto-generated
per-dataset `README.md`. The auto-generated one (written by
`prepare_dataset.py`'s `write_readme()` on each rebuild) tells you the *current*
configuration and counts of whatever folder you just rebuilt; this document
explains the corpus, the labelling philosophy, and the shard-scope knob. They
complement each other.

For training, see [`TRAINING.md`](TRAINING.md).

---

## 1. Source corpus

- **Audio:** the `tadabur_part/` tree at
  `C:/Users/seifg/.../tools/calib/tadabur_part` — word-aligned verse sidecars
  (`*.json`) sitting next to their mono `*.wav`s, organised as
  `tadabur_part/shardK/<reciter_id>/<verse>.json|.wav`. Read-only; nothing is
  copied (manifest rows keep *absolute* paths and decode clips on the fly).
- **Reciters:** ~310–360 reciters depending on how many shards are included
  (each shard adds ~30 reciters + ~500 verses). Each reciter is *style-labelled*
  as Tajweed / non-Tajweed based on their median MADD-word elongation ratio vs
  the population tertile (`MADD_RATIO_FLOOR = 1.3`). This classification alone
  would be too coarse for per-rule labels — see §3.

## 2. Shard scope (the `--shards` flag)

`tadabur_part/` currently holds **13 shards (0–12)**. The training model that
became the "C" reference was trained on **shards 0–6 only** (1,809 of 5,969
sidecars); shards 7–12 were downloaded later. `prepare_dataset.py` accepts a
`--shards` argument that restricts the walk to a subset:

```bash
# 7-shard dataset (matches the C model's training corpus — used for the
# apples-to-apples Phase-2 label quality test vs C)
python src/prepare_dataset.py --tadabur_part <...>/tadabur_part \
    --target_dir dataset --shards 0-6

# 13-shard / merged dataset (~1.9× more data per rule; twice the runtime)
python src/prepare_dataset.py --tadabur_part <...>/tadabur_part \
    --target_dir dataset_p13
#   (or --shards 0-12, equivalently)
```

The flag was added **after** the project discovered (the hard way) that early
runs rebuilt silently against `tadabur_part/`, which had only shards 0–6 *at
the time* but later grew to 13 — so the rebuild's input scope silently changed
under us. With `--shards` you state your intent and the rebuild is reproducible.

## 3. The labelling philosophy — per-rule acoustic evidence

Labels are *bucketed per clip* by the rule's own acoustic signal, not by a
single global (reciter-level) flag:

| Rule | Acoustic cue (`rule_acoustics.rule_evidence`) | what it measures |
|---|---|---|
| MADD       | clip elongation ratio (`wdur / reciter_median_nonmadd`) | vowel elongation |
| GHUNNAH    | `nasal_ratio` (200–500 Hz energy / total) | nasal murmur |
| ALLAM      | `lam` (lateral /l/ presence) | lam shamsiyyah/qamariyyah |
| QALQALAH   | `0.5·closure + 0.3·onset + 0.2·tilt` | complete stop + sharp release |
| IDGHAM     | `nasal_ratio` | nasalised merge into و ي ن م |
| IKHFA      | `nasal_ratio` | nasalisation toward non-emphatic consonant |
| MEEM_SAKINAH | `nasal_ratio` | nasal/labial merge |
| IZHAR      | `1 - nasal_ratio` | *clear* noon before throat letters (NO nasal) |

Each rule labels each clip into three buckets:

- **A — CORRECT (positive, `y=1`)**: rule applies in text AND the clip's own
  evidence is in the STRONG side of the population tertile AND the reciter has
  `>= MIN_EVIDENCE[rule]` words of that rule (avoids assuming a madd-good
  reciter is also ghunnah-good).
- **B — MISTAKE (hard negative, `y=0`)**: rule applies AND the clip's evidence
  is WEAK AND evidence OK.
- **C — DISTRACTOR (easy negative, `y=0`)**: rule does NOT apply in text
  (abundant); subsampled to keep pos:neg ≈ 1:2 (`NEG_MULT=2`).

## 4. The per-clip denoising gate (`PER_CLIP_GATE_RULES`)

The label above is computed at the **clip** level using *this clip's own*
acoustic evidence against a **global per-clip tertile** (33rd / 67th percentile
pooled across every clip of that rule), set inside
`prepare_dataset.py:PER_CLIP_GATE_RULES`. The ambiguous middle third (between
`lo` and `hi`) is **excluded from training** — that sharpens the strong-vs-weak
signal by dropping the clips whose labels were noisiest.

### The fix that mattered
For the first generation of the project the gate was enabled only for
`GHUNNAH`, `ALLAM`, `QALQALAH` — exactly the three rules that went from ~0.60 to
~0.90 AUC when it was first added. The four remaining rules (`IDGHAM`, `IKHFA`,
`IZHAR`, `MEEM_SAKINAH`) still carried the coarser *reciter-level* tertile
label, which is why they stayed stuck at 0.66–0.75 AUC regardless of capacity,
LR, scheduler, or augmentation. `rule_acoustics.rule_evidence` already handled
them, so extending the gate set to all 7 non-MADD rules was a one-line change
with an outsized effect (TRAINING.md §7). **All four weak rules now sit in the
0.93–0.98 TEST-AUC band; MADD was unaffected (it has its own duration gate).**

### The cross-word window fix (a latent bug surfaced by the change)
For *cross-word* rules (`CROSSWORD_RULES = {IDGHAM, IKHFA, IZHAR,
MEEM_SAKINAH}`) the rule's acoustic cue lives at the word boundary. The clip
must be **extended into the next word** so the boundary is in the audio. In the
old code, the per-clip gate measured the clip's evidence *before* the cross-word
extension was computed — measuring the cue on the wrong single-word window and
comparing it to tertile thresholds that had been computed in pass1 on the
extended window. With IDGHAM/IKHFA/IZHAR/MEEM_SAKINAH now gated, that
inconsistency would have broken their labels; the extension step was hoisted
above the gate so evidence, tertile thresholds, and the model's input are all
measured on the same window.

## 5. Current datasets on disk

| folder | shards | purpose | current best model |
|---|---|---|---|
| `dataset/` | 0–6 | Phase 2 (denoised labels, 7 shards). The "apples-to-apples vs C" run. | `model_best.pt` (Phase 2, epoch 53, agg val 0.959) |
| `dataset_p13/` | 0–12 | Phase 3 (denoised labels, 13 shards). Gives the thin rules ~2× positives so their TEST AUCs certify. | training in progress |

### Backups on disk
- `dataset/c_model_backup/model_C_best.pt` — the original **C model** (pre-denoise
  4 weak rules), preserved so all later results can be compared to it.
- `phase2_model_backup/model_p2_best.pt` — Phase 2 best, so the in-progress
  Phase 3 run **cannot** destroy Phase 2's deliverable by overwriting
  `dataset/model_best.pt`.
- `dataset/phase1_backup/manifest_*.phase1old` — the pre-Phase-2 manifests
  (reciter-level labels on the 4 weak rules), in case the project ever needs to
  reproduce the old (noise-prone) label distribution for comparison.

## 6. Reproducibility knobs (defaults shown)

| env / flag | default | meaning |
|---|---|---|
| `MADD_RATIO_FLOOR` | 1.3 | reciter-classification gate for "Tajweed" mid-word elongation ratio (median over their madd-word vs non-madd-word durations). |
| `MADD_CLIP_HI` / `MADD_CLIP_LO` | 1.6 / 1.25 | per-clip madd ratio cutoffs above = strong (correct), below = weak (mistake), between = excluded as ambiguous. |
| `MIN_MADD_WORDS` | 3 | a reciter's madd ratio is only trusted once they have this many madd words. |
| `MIN_EVIDENCE[rule]` | {MADD:3, GHUNNAH:3, QALQALAH:3, IDGHAM:1, IKHFA:1, ALLAM:3, MEEM_SAKINAH:2, IZHAR:1} | a reciter's per-rule windows are only trusted after this many of their clips of that rule. Rarer rules use 1 so weak reciters still contribute *mistake* clips. |
| `NEG_MULT` | 2.0 | negatives (B + C) capped at this multiple of positives per rule. |
| `B_FRAC` | 0.6 | share of the negative budget taken from Bucket B (mistakes) vs C (distractors). |
| `SEED` | 1337 | fixed for reproducible balancing + splits. |
| `RIWAYAH` | Hafs | recitation school used by the rule detector. |
| `--shards` | (all) | shard whitelist, e.g. `0-6`. **Always set** if you want a particular corpus scope; otherwise it walks every shard in `tadabur_part/`. |

## 7. What an auto-generated `dataset/README.md` will and won't tell you

**Will:** the *current* corpus scope (reciter count, verses), per-rule A/B/C
counts, the strong-vs-weak group medians and `rel_gap` (`evidence_separability`
in `stats.json`) — useful for sanity-checking that a rule's labels are
actually separable by its evidence. **Won't:** the per-clip gate's history, the
shard-scope choice rationale, or the C→Phase 2→Phase 3 story — those live here
and in `TRAINING.md`.

## 8. QALQALAH distribution note (added post-Phase-3)

The QALQALAH cue (`0.5·closure + 0.3·onset + 0.2·tilt`) separates
strong-vs-weak reciters cleanly — but on inspection, the "strong" bucket A is
itself spread over a wide range. Close to a quarter of Bucket-A clips sit
within a hair of the per-clip gate's lower tertile threshold and the model
inherits that instability: requesting QALQALAH recall at a flat 0.5 sigmoid
cut wrongly rejects ~12% of correct recitations. The AUC ~0.92 is real (the
head *ranks* correct above incorrect) but the absolute sigmoid output for the
correct class is left-skewed.

Practical side-effects:

- The Android app's `decisionThresholdForHead(QALQALAH)` is **0.15** at deploy
  time, not the naive 0.5 — pinned to the bottom of the band where recall is
  preserved.
- For training-side improvements (per-head focal-loss weighting, evidence
  injection at the head input, the per-anchor anchor-cluster hint, and
  Brier-score reporting on positives) see TRAINING.md §7.2.
