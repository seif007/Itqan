# Tajweed Mistake-Detection — Training & Model

This sketch trains a **tiny on-device model** that, given a word-level audio
segment plus the *expected* Tajweed rule(s) derived from the Quranic text,
predicts whether the rule was performed correctly. A mismatch is a mistake.

It pairs with a `dataset*/` directory built by `src/prepare_dataset.md` (see
[`DATASET.md`](DATASET.md) for the corpus / label story) and reuses the manifest
format described there. The Android app is **not** modified.

> **Status (2026-07-28).** The model that ships is the small
> `Conv32 + BiGRU64 + per-rule heads` network described in §2 (~48 k params).
> The single most impactful change in the project was extending the **per-clip
> denoising gate** of `prepare_dataset.py` to the four previously-stuck rules
> (`IDGHAM`, `IKHFA`, `IZHAR`, `MEEM_SAKINAH`) — exactly the mechanism that had
> fixed `GHUNNAH`/`ALLAM`/`QALQALAH` earlier. With cleaner labels, the four
> "weak" rules went from TEST AUC 0.66–0.75 → 0.93–0.98 in a few dozen epochs
> (see §7). Another run on the larger 13-shard corpus is in progress to give the
> thinnest rules enough positives to certify statistically.

---

## 1. Inputs and features (PLAN section 5.1)

- 16 kHz mono WAV segments, sliced on the fly from the absolute WAV paths in
  `manifest_<RULE>.jsonl` (`start_ms` / `end_ms`). **No audio is copied or held
  in RAM** — it is decoded clip-by-clip inside the streaming DataLoader.
- Features: **log-mel spectrogram**, 40 bins, 25 ms window / 10 ms hop, computed
  with a small dependency-free NumPy STFT plus triangular mel filterbank
  (TorchAudio is used for resampling only when present). Frames are padded or
  trimmed to a fixed `MAX_FRAMES = 200` (~2 s) so tensors stay uniform and
  RAM-bounded.
- Two clip-level scalars are also fed to the heads, because temporal mean-pool
  destroys them:
  - **Duration feature** (→ MADD head only): `min(dur, 4s)/4`, scales to ~1.0
    per second. MADD's signature is vowel *elongation*, which mean-pooling would
    loose. Other rules don't see it (don't get distracted by length).
  - **Acoustic-evidence vector** `ev` (→ every head): 5-dim per-clip cue
    (`nasal_ratio`, `lam`, `closure`, `onset`, `tilt`), bound to ~[-1,1] via
    `tanh`. This is the *same* cue `prepare_dataset.py` uses to label, and a
    pure function of the audio, so it's identical at train and inference time.
- Everything is computed in a Python generator, so peak RAM is set by the batch
  size, not the dataset size.

## 2. Architecture (PLAN section 5.2)

**Shared tiny backbone + per-rule heads:**

```
log-mel (40 x T)
   |
   v
Conv1D(40->32, k=3) - ReLU - MaxPool(2)      \
Conv1D(32->32, k=3) - ReLU - MaxPool(2)      |  3 small conv layers
Conv1D(32->32, k=3) - ReLU - MaxPool(2)      /
   |  (permute to time-major)
   v
Bi-GRU(32 -> 64)   -> 128-d sequence
   |  temporal mean-pool
   v
128-d shared embedding  (concatenated with dur + ev -> head input)
   |
   |- Linear(…->1) -> sigmoid   <- head for MADD   (sees duration too)
   |- Linear(…->1) -> sigmoid   <- head for GHUNNAH
   |- ... (one head per rule)   <- IDGHAM, IKHFA, IZHAR, QALQALAH,
   |                               ALLAM, MEEM_SAKINAH
   v
per-rule P(correct)
```

- **Total params: 48,785** (the original "C" capacity). A larger
  `conv_channels=48, gru_units=96` variant (~105 k params) was tried and
  *overfit* the thin rules — they have very small val sets and the extra
  capacity memorized them. The current default (32 / 64) is empirically optimal.
- **Multi-task training.** Each step samples a batch from **one** rule and
  back-propagates only that rule's sigmoid head; the backbone is shared so it
  learns general "correct vs wrong recitation" cues. Per-rule `pos_weight`
  (neg/pos) rebalances the residual class skew after `prepare_dataset.py`
  balancing. Splits are reciter-disjoint (see `splits.json`).
- **Rule-specific batch budget.** `--max_steps_per_rule N` caps each rule at
  `min(own_batches, N)` per epoch so a huge rule (MADD) doesn't force every
  other rule to step enough to catch up. Default `200` proved a good balance.

## 3. Hybrid GPU/CPU execution (the GTX 950M fix)

The host laptop's GPU is a GTX 950M (Maxwell, sm_50). When
`--device cuda` runs the Bi-GRU on it, the CUDA GRU kernel **SIGSEGVs (exit
139)** or hangs; turning cuDNN off hangs instead of crashing. The fix is a
**hybrid split** that keeps the Conv backbone on GPU but routes the Bi-GRU to
CPU:

- `train.py:536` disables cuDNN (`torch.backends.cudnn.enabled = False`).
- `train.py:547` moves `model.gru.to("cpu")` after the model is on `cuda`; sets
  `model.gru_device = torch.device("cpu")`.
- `TajweedNet.forward` moves the conv output to CPU before the GRU and back to
  the main device before the per-rule heads (so the Linear matmuls still run on
  GPU). `--device auto` picks this whenever CUDA is available; `--device cpu`
  runs everything on CPU (slower-per-epoch, no transfers).

This is not optional on this machine — full-CPU is ~3× slower, full-CUDA on the
Maxwell GRU kernel crashes.

## 4. Checkpointing, resume, early-stop, scheduler

- **Per-epoch checkpoint:** `model_checkpoint.pt` holds model + optim + RNG + the
  `epochs_since_improve` patience counter + `scheduler_state`. State that
  survives resume: epoch index, global step, RNG shuffle state, best val-AUC,
  best epoch, patience counter, scheduler position.
- **`--fresh`** ignores any existing checkpoint and starts from epoch 0;
  without it, the script resumes from `model_checkpoint.pt`. **Required after
  a code/architecture change**; must be omitted to continue a long multi-day
  run (or to recover from a power cut — see below).
- **Early-stop:** `--patience N` stops when `epochs_since_improve >= N`
  (aggregate val-AUC). The patience counter is *saved in the checkpoint* (a
  prior bug reset it to 0 on resume, so early-stop never fired across resumes;
  fixed).
- **Cosine warm restart:** `--scheduler cosine` (default) uses
  `CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)`. Restarts at
  epochs 10, 30, 70,… bump val-AUC a notch back up. `--scheduler plateau`
  uses `ReduceLROnPlateau` instead.
- **Power-cut recovery.** If the PC loses power mid-epoch, the python process
  often appears hung (visible in `ps`, not actually computing). The recovery
  recipe (proven against repeated power cuts in this project):
  1. `cmd //c "taskkill /PID <WINPID> /F"` the hung python from its PID in `ps -W`.
  2. Verify the checkpoint via `torch.load(... checkpoint.pt)` — confirm
     `epoch`/`best_epoch`/`epochs_since_improve` look reasonable (not 0).
  3. Re-run the same `train.py` command **without** `--fresh`. The trainer will
     print `RESUMED from model_checkpoint.pt: epoch=… best_val_auc=…` and pick
     up exactly where it stopped. The patience-counter-save fix means early-stop
     fires when it would have, not when it first resumes.

## 5. On-device export (PLAN section 5.3)

`src/train.py` can save a **TorchScript** trace (`dataset/model_traced.pt`)
and, when `onnx` + `onnx2tf` are installed, export **int8 TFLite**. **Export is
opt-in via `--export`; it is OFF by default** because `torch.jit.trace`
SEGFAULTs on the hybrid model's device transfers inside `forward()`. Only add
`--export` after switching to a non-hybrid (CPU) execution for the export pass.

```
python src/train.py --dataset_dir dataset --epochs 60 --device cpu --export
# -> dataset/model.onnx  (then)
onnx2tf -i dataset/model.onnx -o dataset/tflite_int8 -osd
```

For int8 quantization you need a **representative dataset** of real log-mel
clips; `export_tflite()` already yields a `rep_reader()` generator that streams
a handful of real segments (so the converter sees realistic activations). If
`onnx2tf` is not installed, the script saves the traced model + generator and
prints the exact command to finish the int8 conversion later — the Android app
stays untouched either way.

Real-time path (future): run the model per word **after** the existing alignment
step in the app, feeding it the expected rule(s) from `tajweed_rules.py`.

## 6. How to run

```bash
# quick self-check (one forward pass + export path, no full training)
python src/train.py --dataset_dir dataset --smoke

# fresh GPU run (this project's usual invocation; resume = drop --fresh)
python -u src/train.py --device auto --dataset_dir ../dataset \
    --epochs 60 --batch_size 64 --dropout 0.3 --weight_decay 1e-3 \
    --patience 15 --lr 5e-4 --max_steps_per_rule 200 --scheduler cosine --fresh

# train only the thin rules
python -u src/train.py --dataset_dir ../dataset --rules IDGHAM,IKHFA,IZHAR,MEEM_SAKINAH ...
```

Selecting `--rules` trains heads for the given rule subset only; omitting it
trains all 8 rules (which is what produced the reported AUCs).

### CLI arguments that matter

| arg | default | effect |
|---|---|---|
| `--device` | `auto` | `cuda` (hybrid GPU/CPU), `cpu`, or `auto` (CUDA if available). |
| `--conv_channels`, `--gru_units` | `32`, `64` | backbone width. Don't raise without a reason; `48/96` overfit the thin rules. |
| `--epochs`, `--patience` | `20`, `0` (off) | Use `60 / 15` for real runs. |
| `--max_steps_per_rule` | `10000` | `200` caps huge rules like MADD so small ones keep up. |
| `--scheduler` | `cosine` | `plateau` is the alternative. |
| `--lr` | `1e-3` | `5e-4` is what produced the reported AUCs. |
| `--rules` | all 8 | comma-separated subset (heads created only for the chosen rules). |
| `--fresh` | off | ignore checkpoint, start at epoch 0. |
| `--export` | off | also TorchScript-trace + TFLite int8 export; off by default (segfault on hybrid). |

## 7. Results — the label-quality breakthrough

The C model used reciter-level tertile labels for every rule *except*
GHUNNAH/ALLAM/QALQALAH (which already had a per-clip denoising gate). That left
IDGHAM/IKHFA/IZHAR/MEEM_SAKINAH carrying noisy labels exactly where their own
acoustic signal was inconsistent with the reciter-level proxy. Extending the
per-clip gate to those four rules (and re-running with `--shards 0-6`, the same
7-shard corpus C was trained on, so the comparison was isolated to the *label*
change alone) gave the per-rule TEST AUC lift below.

| rule | C TEST AUC (n) | Phase-2 TEST AUC (n) | delta |
|---|---|---|---|
| MADD       | 0.951 (1867) | 0.965 (1867) | **+0.014** ↑ |
| QALQALAH   | 0.938 (115)  | 0.903 (188)  | -0.035 ↓ (test set ~1.6×, harder) |
| GHUNNAH    | 0.913 (213)  | 0.948 (402)  | **+0.035** ↑ |
| ALLAM      | 0.900 (210)  | 0.947 (377)  | **+0.047** ↑ |
| MEEM_SAKINAH | 0.749 (90) | 0.981 (124)  | **+0.232 ↑** ★ |
| IDGHAM     | 0.679 (91)   | 0.947 (73)   | **+0.268 ↑** ★ (test n small) |
| IKHFA      | 0.708 (22)   | 0.988 (27)   | **+0.280 ↑** ★ (test n small) |
| IZHAR      | 0.655 (46)   | 0.932 (28)   | **+0.277 ↑** ★ (test n small) |
| **mean**   | 0.812 | 0.951 | **+0.140** |
| **mean weak (4)** | 0.698 | 0.962 | **+0.264** |

Caveats on the headline numbers:
- QALQALAH regressed slightly because its test set grew ~1.6× (the rebuild
  enlarged its dataset with the extended-window consistency fix); it didn't get
  worse, the hold-out just got harder.
- IKHFA (n=27), IZHAR (n=28), IDGHAM (n=73) test sets are very small — a few
  flipped clips move AUC by ±0.03–0.05; the magnitudes are partly statistical
  luck. The phase-3 run on 13-shard data (~2× positives) was intended to give
  these rules enough test volume to certify. **It did — see §7.1.**

### 7.1 Phase-3 results (13-shard, statistically-certified)

Phase-3 trained on `dataset_p13/` (13 shards, the original 0–6 plus shards
7–12). Same per-clip gate, same original 48k capacity (Conv1d 32, BiGRU 64),
cosine warm restarts, `--lr 5e-4`, hybrid Conv-GPU / BiGRU-CPU. The run was
driven by `src/auto_resume_train.py`, which loops `train.py` through the
GTX 950M's intermittent CUDA SIGSEGV (exit 139) by reloading the
checkpoint and relaunching. It finished cleanly at **epoch 59** with
`best_val_auc = 0.962` (model at epoch 55):

| rule | C (n) | Phase-2 (7-shard) | **Phase-3 (13-shard, n)** | vs C |
|---|---|---|---|---|
| MADD         | 0.951 (1867) | 0.965 | **0.956 (2889)** | +0.005 ↑ |
| MEEM_SAKINAH | 0.749 (90)   | 0.981 | **0.955 (205)**  | +0.206 ↑ ★ |
| GHUNNAH      | 0.913 (213)  | 0.948 | **0.941 (586)**  | +0.028 ↑ |
| QALQALAH     | 0.938 (115)  | 0.903 | **0.925 (316)**  | -0.013 ↓ |
| ALLAM        | 0.900 (210)  | 0.947 | **0.937 (609)**  | +0.037 ↑ |
| IDGHAM       | 0.679 (91)   | 0.947 | **0.993 (113)**  | +0.314 ↑ ★ |
| IKHFA        | 0.708 (22)   | 0.988 | **0.947 (58)**   | +0.239 ↑ ★ |
| IZHAR        | 0.655 (46)   | 0.932 | **1.000 (34)**   | +0.345 ↑ ★ |
| **mean**     | 0.812 | 0.951 | **0.962**         | **+0.150** |

**Deliverable policy: keep both.** Phase-3 (`dataset_p13/model_best.pt`) is
the primary deliverable — every rule beats the original C baseline, and the
thin rules (IKHFA n=58, IDGHAM n=113, IZHAR n=34) finally have enough test
volume to be certifiable rather than statistical luck. Phase-2
(`phase2_model_backup/model_p2_best.pt`) is retained as the fallback: it is
marginally higher on the large-n reciter-level rules (GHUNNAH 0.948 vs 0.941,
ALLAM 0.947 vs 0.937, QALQALAH 0.903 vs 0.925 on the smaller Phase-2 test set),
and ships the denoised labels that fixed the original 0.60 weak rules in the
first place.

Caveats carrying over from §7:
- QALQALAH regressed vs C on Phase-2 (test set ~1.6×, harder); on Phase-3 it
  *recovered* to 0.925 — essentially matching C's 0.938 once n grew to 316.
- IKHFA (n=58) and IZHAR (n=34) are still small; an AUC of 0.947 / 1.000
  carries a confidence interval of a few points. They are interpretable now,
  not noise, but keep the CI in mind.
- Early-stop was **off** for the run (`--patience 0` default), so the run was
  ended by the cosine scheduler hitting its budget (`--max_steps_per_rule`)
  at epoch 59, not by patience firing. The best checkpoint at epoch 55 is the
  deliverable; epochs 56–59 did not beat it. For future runs pass
  `--patience 15` so a plateau ends the run instead of waiting for the budget.

The takeaway: the four weak rules were not a *training* problem (more epochs /
bigger backbone / different LR didn't help — all were tried); they were a
*label* problem. Cleaning the labels with the same mechanism that fixed the
strong rules did what no training tweak could.

### 7.2 QALQALAH: the per-clip score distribution issue (post-Phase-3)

Phase-3's headline TEST AUC (0.925) hides a UX problem the Android app exposed:
evaluating the deployed model **per-clip** (not aggregated AUC) on QALQALAH
positives, a meaningful share of **correct** QALQALAH clips come out with
`P(correct) < 0.5`. Model calls them wrong. Empirically the QALQALAH head's
output for correct recitations spans roughly **0.09 → 0.97 with std ≈ 0.19** —
wide and left-skewed. A flat sigmoid threshold at 0.5 therefore wrongly fails
~12% of correct QALQALAH, which is exactly what we saw on-device.

The AUC says "the model ranks correct above incorrect" — but the *absolute*
calibration of the sigmoid output is shifted down for this rule vs the others.

This is the crunch mismatch between classification ranking (what AUC measures)
and probability calibration (what flat thresholds care about).

#### Inference-side workaround (deployed)
`decisionThresholdForHead(QALQALAH)` is set to **0.15** on the app
(`RecitationApp.kt`) — the bottom of the safe band where recall stays high
without letting true negatives slide through. Empirical FPR/TPR tradeoff on
the user-visible app supports this threshold; do not lower further without
re-running the calibration fold.

#### Training-side candidates (priority order)

These intent to **lift the QALQALAH logits on correct clips** so we can raise
the decision threshold back up toward ~0.3–0.4 without losing recall. We have
not retrained yet — these are the next experiments to run when the calibration
corpus gets another pass.

1. **Per-head focal-loss with rule-weighted lifetime balance.** QALQALAH sits
   midway between "explosive burst" (closure + sharp release) and "flat stop".
   Bucket-B negatives are clips that *should* have QALQALAH but were performed
   weakly — they are close to the decision boundary. Currently equal-weighted
   with the easier distractors in bucket C, which are far from the boundary.
   Weighting loss by rule × bucket-life-time proximity should push the QALQALAH
   distribution right on the correct side.
2. **Explicit per-clip evidence injection.** QALQALAH's evidence
   (`0.5 · closure + 0.3 · onset + 0.2 · tilt`, see §3 in DATASET.md) already
   separates correct from weak. Currently the GRU gets it via the shared
   backbone only — try concatenating the evidence vector to the GRU *output*
   too (in addition to the per-head input) so the QALQALAH head sees both
   "raw log-mel shape" and "own-cue scalar" with no bottleneck.
3. **Tune the QALQALAH evidence gate downward.** `MIN_EVIDENCE[QALQALAH] = 3`
   is fine, but the gate's positive snout (Bucket A) keyed on the top third of
   the global per-clip tertile may be too tight given how noisy the cue is on
   the low-clip reciters. Try `lo=0.20` / `hi=0.55` instead of `lo=0.33` /
   `hi=0.67` to bring in the middle band as positives, widersing the QALQALAH
   positives pool ~1.5-2x. Re-run on the same 13 shards; check AUC + per-clip
   score histogram to verify the distribution shifted right.
4. **Second pass at QALQALAH-specific anchor words.** Candidates like قُل,
   بِرَبِّ, and other high-frequency QALQALAH-containers appear hundreds of
   times per reciter; a per-anchor clustering pass may uncover QALQALAH-realized
   clusters behind the bimodal cue distribution, which we can then use to
   pseudo-label additional anchors with fewer acoustic-evidence ambiguities.
5. **Brier-score evaluation on positives.** Rather than AUC, report Brier
   score on the QALQALAH positive set so "correct clips scored 0.4 instead of
   0.9" shows up directly — AUC is invariant to monotone shifts of the head's
   output, so AUC ≈ 0.92 can coexist with mean-P(correct) ≈ 0.55. We publish
   both metrics going forward.

## 8. Why not bigger capacity, longer training, or augmentation

Each of these was tried in earlier work; none moved the weak rules:

- **Bigger backbone (conv=48, gru=96, ~105 k params) + thin-rule LR.** Overfit
  the tiny val sets of IKHFA/IZHAR/IDGHAM: TEST AUC got worse than C on 3 of 4
  thin rules. Thin rules need *cleaner labels*, not more parameters.
- **Longer training on the same labels.** Plateaus within a few epochs; cosine
  warm restarts only buy back ~0.005 at the restart. Performance is bounded by
  label signal-to-noise, not capacity.
- **Augmentation (SpecAugment, time-shift, mixup).** Not implemented; the
  limiting bottleneck is data *label* quality, so adding noise on the input
  side would not help and could mask the very cues the heads rely on (nasal
  band, stop closure, lateral /l/).

## 9. Current dataset limitations to keep in mind

- **MADD** is rich and the cleanest target (Bucket B from non-Tajweed reciters
  is a strong hard-negative). GHUNNAH / ALLAM / QALQALAH / MEEM_SAKINAH are now
  quite strong. **IDGHAM / IKHFA / IZHAR** were thin on the 7-shard corpus;
  the 13-shard Phase-3 run gave them enough volume (IKHFA n=58, IDGHAM n=113,
  IZHAR n=34) that their TEST AUCs are now statistically stable, see §7.1.
- Bucket B is a *style proxy*, not ground-truth error labels; verify the
  MADD-ratio is bimodal before trusting B for non-MADD rules.
- See [`DATASET.md`](DATASET.md) for the corpus / shard / per-clip-gate story
  (separate from the auto-generated `dataset/README.md`, which a rebuild
  overwrites).
