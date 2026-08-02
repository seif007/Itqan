#!/usr/bin/env python3
"""
prepare_dataset.py - Build a labeled Tajweed mistake-detection training dataset
from the (read-only) Tadabur tadabur_part/ audio corpus.

Design goals (see PLAN.md section 4):
  * READ ONLY from the existing source tadabur_part/ (TADABUR_PART env, absolute).
  * WRITE ONLY to the new repo's dataset/ dir.
  * KEEP ALL reciters.
  * LABEL QUALITY FIX: each rule's CORRECT/MISTAKE label comes from THAT rule's
    OWN measured acoustics (see rule_acoustics.py), NOT from a single global
    MADD-ratio "tajweed reciter" flag.  Previously every rule reused the MADD
    elongation flag, so only MADD learned; now GHUNNAH/IKHFA/IDGHAM/MEEM_SAKINAH
    are labelled by nasal-energy, IZHAR by clarity (low nasal), QALQALAH by
    stop/burst, MADD by elongation.  Reciters are split strong/weak per rule at
    the population median of their acoustic evidence.
  * Cross-word rules (IDGHAM/IKHFA/IZHAR/MEEM_SAKINAH) get clips EXTENDED to
    include the following word, because the relevant acoustic event happens at
    the transition into the next word.
  * PER-RULE evidence gates: a reciter is only trusted as a "correct performer"
    of rule R if they have >= MIN_EVIDENCE[R] words of R (avoids assuming a
    madd-good reciter is also ghunnah-good).
  * For every rule build three buckets per segment:
        A  CORRECT   (positive)   : rule applies AND reciter is acoustically
                                    STRONG for this rule AND evidence OK
        B  MISTAKE   (hard neg)   : rule applies AND reciter is acoustically
                                    WEAK for this rule AND evidence OK
        C  DISTRACTOR(easy neg)   : rule does NOT apply (abundant)
  * Balance classes deterministically (fixed seed); make train/val/test
    RECOITER-DISJOINT.
  * Memory safety: stream sidecars, never hold the audio or the whole row set in
    RAM.  Pass 1 = reciter-level acoustic stats + per-rule evidence (decodes each
    verse WAV once).  Pass 2 = stream per-word rule rows to one temp file per
    rule (bounded RAM, one verse in memory at a time).  Pass 3 = balance per-rule
    (one rule in memory at a time) and emit final outputs.

Outputs under TARGET_DIR (default: ../dataset):
  manifest_<RULE>.jsonl  per rule: {rule,label,reciter,wav_abs,start_ms,end_ms,
                                    reciter_is_tajweed,y,split}
  splits.json            {train:[reciters], val:[...], test:[...]}
  stats.json             per-rule pos/neg counts, reciter counts, madd ratios
  README.md              how labels/splits/gates were derived (reproducible)

Run:
  TADABUR_PART="C:/.../practice_recitation-main/tools/calib/tadabur_part" \
  python prepare_dataset.py
"""
import os
import sys
import json
import random
import statistics
import argparse

# ---- locate sibling modules (tajweed_rules, tadabur_adapter) ----
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tajweed_rules import detect_rules_for_word
from tadabur_adapter import (
    recover_diacritic_words,
    walk_sidecars,
    to_int_reciter,
    median,
)
import soundfile as sf
from rule_acoustics import (
    CROSSWORD_RULES,
    SR,
    load_clip,
    clip_features,
    rule_evidence as rule_evidence_fn,
)

# ---------------------------------------------------------------------------
# Config (env-overridable)
# ---------------------------------------------------------------------------
RULES = ["MADD", "GHUNNAH", "QALQALAH", "IDGHAM", "IKHFA",
         "ALLAM", "MEEM_SAKINAH", "IZHAR"]   # SILAH is intentionally not derived
RULE_SET = set(RULES)

# Rules whose label quality is bottlenecked by the reciter-level tertile gate
# (a "strong reciter" can still mumble one word, so per-clip labels are noisy).
# For these we relabel each CLIP by its OWN acoustic evidence against a global
# per-clip tertile, matching the exact cue the head sees.  GHUNNAH (nasal) and
# ALLAM (lateral /l/) have plenty of volume but only ~0.60 AUC under the
# reciter-level gate; QALQALAH's stop/burst cue is weak (rel_gap ~0.17) so its
# per-clip gate denoises the noisiest clips too.
#
# Extending the gate to the cross-word nasal/clear rules (IDGHAM, IKHFA,
# IZHAR, MEEM_SAKINAH) is the Phase-1 fix for the four weak TEST-AUC rules.
# rule_acoustics.rule_evidence already returns a per-clip scalar for every one
# of them (nasal_ratio for IDGHAM/IKHFA/MEEM_SAKINAH; 1-nasal_ratio for IZHAR),
# so they are denoised by the *same* mechanism that lifted GHUNNAH/ALLAM/QALQALAH
# from ~0.60 to ~0.90 AUC.  MADD keeps its own per-clip duration gate below.
PER_CLIP_GATE_RULES = {"GHUNNAH", "ALLAM", "QALQALAH",
                       "IDGHAM", "IKHFA", "IZHAR", "MEEM_SAKINAH"}

RIWAYAH = os.environ.get("RIWAYAH", "Hafs")
MADD_RATIO_FLOOR = float(os.environ.get("MADD_RATIO_FLOOR", "1.3"))
# Per-CLIP MADD elongation labelling (the real MADD fix).  A MADD word's own
# elongation ratio = word_duration / reciter_median_nonMADD_duration.  This is a
# per-clip acoustic fact, unlike MADD_RATIO_FLOOR which is the reciter MEDIAN of
# that ratio (reciter-level label that fought the per-clip duration feature in the
# model).  We label a clip STRONG (correct madd) only if its ratio is clearly
# elongated, WEAK (mistake) only if clearly under-elongated, and DROP the ambiguous
# middle -- same sharpening philosophy as the tertile split for the other rules.
MADD_CLIP_HI = float(os.environ.get("MADD_CLIP_HI", "1.6"))
MADD_CLIP_LO = float(os.environ.get("MADD_CLIP_LO", "1.25"))
MIN_MADD_WORDS = int(os.environ.get("MIN_MADD_WORDS", "3"))
# Per-rule evidence gate (PLAN: MIN_EVIDENCE[rule]).  It is the minimum number of
# times a reciter must apply a rule before ANY of their clips (strong A or weak B)
# are trusted for that rule.  Common rules keep 3 (stable median evidence); RARE
# rules use 1-2 so that weak reciters -- who by nature apply the rule only a few
# times -- still contribute their MISTAKE (B) clips.  Without this, the rare rules'
# B bucket empties out (B=0) and the task collapses to "rule present vs absent"
# instead of "correct vs mistake", which inflates and invalidates the AUC.
_RARE = {"IDGHAM", "IKHFA", "IZHAR"}
_MID = {"MEEM_SAKINAH"}
_DEFAULT_GATE = int(os.environ.get("MIN_EVIDENCE", "3"))
MIN_EVIDENCE = {r: (1 if r in _RARE else 2 if r in _MID else _DEFAULT_GATE)
                for r in RULES}
# cap negatives (B+C) at NEG_MULT * positives (A)
NEG_MULT = float(os.environ.get("NEG_MULT", "2"))
# fraction of the negative budget filled by Bucket B (mistake); rest from C
B_FRAC = float(os.environ.get("B_FRAC", "0.6"))
SEED = int(os.environ.get("SEED", "1337"))
# when a rule has fewer than this many positives at the strict gate, relax the
# per-rule evidence gate down to gate=1 so it keeps a small balanced set
# (PLAN section 4.4 option a) instead of collapsing to a handful of clips.
MIN_POS_FOR_RULE = int(os.environ.get("MIN_POS_FOR_RULE", "10"))

# split proportions (reciter-disjoint)
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "0.8"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.1"))
# rules with fewer than this many positives are still emitted but flagged thin
THIN_POS = int(os.environ.get("THIN_POS", "30"))


def log(*a):
    print("[prepare]", *a, flush=True)


# ---------------------------------------------------------------------------
# Per-rule evidence gate selection
# ---------------------------------------------------------------------------
def positives_at_gate(rule_quality, rule_evidence, rule, gate):
    """Exact count of positive (A) clips for `rule` if the per-rule evidence
    gate is `gate`: a reciter whose OWN acoustic evidence for `rule` is 'strong'
    AND who has >= `gate` clips of that rule contributes all of those clips as
    positives."""
    total = 0
    for rid, ev in rule_evidence.items():
        if ev.get(rule, 0) >= gate and rule_quality.get(rid, {}).get(rule) == "strong":
            total += ev[rule]
    return total


def choose_rule_gates(rule_evidence, rule_quality, rules, min_ev, min_pos=MIN_POS_FOR_RULE):
    """Pick, per rule, the evidence gate (the minimum number of times a reciter
    must apply the rule before their clips are trusted).  `min_ev` is a per-rule
    dict of *starting* (strict) gates.  For each rule we try gate = strict, then
    strict-1, ... down to 1, and keep the LOWEST gate that still yields >=
    `min_pos` positives -- so common rules stay at 3 (reliable median) while rare
    rules drop to 1-2 (so weak reciters still contribute B clips).  A gate of
    None means even gate=1 yields no positives -> rule is unusable."""
    gates = {}
    relaxed = {}
    for rule in rules:
        base = min_ev[rule] if isinstance(min_ev, dict) else min_ev
        chosen = None
        for g in range(base, 0, -1):
            if positives_at_gate(rule_quality, rule_evidence, rule, g) >= min_pos:
                chosen = g
                break
        if chosen is None:
            p1 = positives_at_gate(rule_quality, rule_evidence, rule, 1)
            chosen = 1 if p1 > 0 else None
            relaxed[rule] = (p1 > 0)
        else:
            relaxed[rule] = (chosen < base)
        gates[rule] = chosen
    return gates, relaxed


# ---------------------------------------------------------------------------
# Pass 1 - reciter-level acoustic stats (tiny, all in RAM)
# ---------------------------------------------------------------------------
def pass1_reciter_stats(tadabur_part):
    madd_dur = {}
    nonmadd_dur = {}
    rule_evidence = {}          # rid -> {rule: count}
    rule_feat = {}              # rid -> {rule: [acoustic evidence scalars]}
    reciter_verse_count = {}

    n_sidecars = 0
    for sidecar in walk_sidecars(tadabur_part):
        n_sidecars += 1
        try:
            with open(sidecar, encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception as e:
            log("sidecar load failed", sidecar, e)
            continue
        rid = to_int_reciter(meta.get("reciter_id"))
        aligns = meta.get("word_alignments", [])
        reciter_verse_count[rid] = reciter_verse_count.get(rid, 0) + 1
        if not aligns:
            continue

        wav_rel = meta.get("wav", "")
        wav_abs = os.path.join(tadabur_part, wav_rel) if wav_rel else ""
        if wav_rel and not os.path.exists(wav_abs):
            wav_abs = os.path.splitext(sidecar)[0] + ".wav"
        have_audio = os.path.exists(wav_abs)

        diac = recover_diacritic_words(meta.get("text_ar_uthmani", ""), aligns)
        ev = rule_evidence.setdefault(rid, {})

        # Decode the verse WAV ONCE (bounded: one verse in memory at a time).
        y_verse = None
        if have_audio:
            try:
                last_end = float(aligns[-1].get("end", 0.0)) * 1000.0 + 50.0
                y_verse, _ = load_clip(wav_abs, 0.0, max(last_end, 1.0))
            except Exception:
                y_verse = None

        for wi, w in enumerate(aligns):
            word = w.get("word", "")
            if not word:
                continue
            start = float(w.get("start", 0.0))
            end = float(w.get("end", 0.0))
            wdur = end - start
            if wdur <= 0:
                continue
            dw = diac[wi] if wi < len(diac) and diac[wi] else word
            nxt_dw = None
            nxt_end = None
            if wi + 1 < len(aligns):
                nxt_dw = diac[wi + 1] if (wi + 1 < len(diac) and diac[wi + 1]) \
                    else aligns[wi + 1].get("word")
                nxt_end = float(aligns[wi + 1].get("end", end))
            rules = detect_rules_for_word(dw, nxt_dw, RIWAYAH)
            is_madd = "MADD" in rules
            if is_madd:
                madd_dur.setdefault(rid, []).append(wdur)
            else:
                nonmadd_dur.setdefault(rid, []).append(wdur)

            # Per-rule ACOUSTIC evidence (the label fix): measure each rule's own
            # acoustic phenomenon instead of reusing the MADD flag.
            applied = [r for r in rules if r in RULE_SET]
            feat_cache = {}
            for r in applied:
                # count evidence for EVERY applied rule (incl. MADD)
                ev[r] = ev.get(r, 0) + 1
                if r == "MADD":
                    continue  # MADD keeps the duration-ratio evidence below
                if r in CROSSWORD_RULES and nxt_end and nxt_end > start:
                    cstart, cend = start, nxt_end
                else:
                    cstart, cend = start, end
                fkey = (round(cstart, 3), round(cend, 3))
                if fkey not in feat_cache:
                    seg = None
                    if y_verse is not None:
                        s_f = max(0, int(round(cstart * SR)))
                        e_f = max(s_f + 1, int(round(cend * SR)))
                        seg = y_verse[s_f:e_f]
                    feat_cache[fkey] = clip_features(seg) if seg is not None else None
                ev_val = rule_evidence_fn(r, feat_cache[fkey])
                if ev_val is not None:
                    rule_feat.setdefault(rid, {}).setdefault(r, []).append(ev_val)
        if n_sidecars % 1000 == 0:
            log("pass1 scanned", n_sidecars, "sidecars;",
                len(reciter_verse_count), "reciters")
    log("pass1 audio decoded for", n_sidecars, "sidecars")

    # acoustic MADD ratio + Tajweed classification (kept for MADD + audit)
    ratios = {}
    is_tajweed = {}
    nonmadd_med = {}
    for rid in set(list(madd_dur.keys()) + list(nonmadd_dur.keys())):
        md = madd_dur.get(rid, [])
        nd = nonmadd_dur.get(rid, [])
        if len(md) < MIN_MADD_WORDS or not nd:
            ratios[rid] = 0.0
            is_tajweed[rid] = False
            nonmadd_med[rid] = 0.0
            continue
        ratios[rid] = median(md) / max(median(nd), 1e-6)
        is_tajweed[rid] = ratios[rid] >= MADD_RATIO_FLOOR
        nonmadd_med[rid] = median(nd)

    log("pass1 done:", n_sidecars, "sidecars;",
        len(reciter_verse_count), "reciters;",
        sum(1 for v in is_tajweed.values() if v), "tajweed,",
        sum(1 for v in is_tajweed.values() if not v), "non-tajweed")

    # GLOBAL PER-CLIP tertile thresholds for the denoising rules (GHUNNAH/ALLAM).
    # Pool every per-clip evidence value across ALL reciters, then take the 33rd
    # and 67th percentiles.  In pass2 each such clip is labelled strong/weak by
    # its OWN evidence vs these thresholds -- so the label matches exactly the
    # acoustic cue the head consumes (unlike the reciter-level median gate).
    clip_thr = {}
    for r in PER_CLIP_GATE_RULES:
        vals = []
        for d in rule_feat.values():
            lst = d.get(r)
            if lst:
                vals.extend(lst)
        if len(vals) >= 12:
            vals_sorted = sorted(vals)
            lo = vals_sorted[len(vals_sorted) // 3]
            hi = vals_sorted[(2 * len(vals_sorted)) // 3]
            clip_thr[r] = (float(lo), float(hi))
            log("per-clip tertile for %s: n=%d lo=%.4f hi=%.4f"
                % (r, len(vals), lo, hi))
        else:
            log("per-clip tertile for %s: too few clips (%d) -> skip"
                % (r, len(vals)))
    return (madd_dur, nonmadd_dur, rule_evidence, rule_feat,
            reciter_verse_count, ratios, is_tajweed, nonmadd_med, clip_thr)


def derive_rule_quality(rule_feat, rule_evidence, ratios):
    """Per-rule CORRECT/MISTAKE label source (the fix for the global-MADD-flag
    problem).  For every rule except MADD, split the reciters that have >=1 clip
    of that rule into a clear top third ('strong' = labelled CORRECT), a clear
    bottom third ('weak' = labelled MISTAKE), and an EXCLUDED ambiguous middle
    third.  Dropping the ambiguous middle makes 'strong vs weak' a much sharper,
    more learnable signal (it removes the reciters whose evidence is closest to
    the population centre, i.e. the noisiest labels).  MADD keeps the proven
    duration-ratio evidence.  Reciters with no usable evidence for a rule are
    mapped to None (excluded from that rule's A/B)."""
    quality = {}
    for r in RULES:
        if r == "MADD":
            continue
        meds = {}
        for rid, d in rule_feat.items():
            lst = d.get(r)
            if lst:
                meds[rid] = statistics.median(lst)
        if not meds:
            continue
        if len(meds) < 6:
            # Too few reciters for a meaningful tertile split: fall back to a
            # median split so we don't throw away the whole rule.
            mid = statistics.median(list(meds.values()))
            for rid, m in meds.items():
                quality.setdefault(rid, {})[r] = "strong" if m >= mid else "weak"
            continue
        # Tertile split: bottom third -> weak, top third -> strong, middle -> None
        items = sorted(meds.items(), key=lambda kv: kv[1])
        n = len(items)
        lo = n // 3
        hi = n - lo  # index where the top third begins
        for i, (rid, m) in enumerate(items):
            if i < lo:
                q = "weak"
            elif i >= hi:
                q = "strong"
            else:
                q = None
            quality.setdefault(rid, {})[r] = q
    for rid in ratios:
        q = "strong" if ratios[rid] >= MADD_RATIO_FLOOR else "weak"
        if ratios[rid] <= 0:
            q = None
        quality.setdefault(rid, {})["MADD"] = q
    return quality


# ---------------------------------------------------------------------------
# Pass 2 - stream per-word rule rows to one temp file per rule (bounded RAM)
# ---------------------------------------------------------------------------
def pass2_stream_rows(tadabur_part, target_dir, rule_evidence, rule_quality,
                      is_tajweed, gates, nonmadd_med, clip_thr):
    tmp_paths = {r: os.path.join(target_dir, "_tmp_%s.jsonl" % r) for r in RULES}
    fhs = {r: open(tmp_paths[r], "w", encoding="utf-8") for r in RULES}

    n_sidecars = 0
    for sidecar in walk_sidecars(tadabur_part):
        n_sidecars += 1
        try:
            with open(sidecar, encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            continue
        rid = to_int_reciter(meta.get("reciter_id"))
        taj = bool(is_tajweed.get(rid, False))
        qmap = rule_quality.get(rid, {})
        ev = rule_evidence.get(rid, {})
        # Per-CLIP MADD label: a MADD word's own elongation ratio vs this
        # reciter's median non-MADD word duration.  Overrides the reciter-level
        # MADD quality from derive_rule_quality so the label matches the model's
        # per-clip duration feature.  Strong=clearly elongated, weak=clearly
        # under-elongated, middle=excluded (ambiguous).  Computed lazily per word
        # below where wdur is known; stored back into a per-sidecar override map.
        wav_rel = meta.get("wav", "")
        wav_abs = os.path.join(tadabur_part, wav_rel) if wav_rel else ""
        if wav_rel and not os.path.exists(wav_abs):
            # fall back to sibling wav next to the sidecar
            wav_abs = os.path.splitext(sidecar)[0] + ".wav"
        if not os.path.exists(wav_abs):
            continue
        aligns = meta.get("word_alignments", [])
        diac = recover_diacritic_words(meta.get("text_ar_uthmani", ""), aligns)

        for wi, w in enumerate(aligns):
            word = w.get("word", "")
            if not word:
                continue
            start = float(w.get("start", 0.0))
            end = float(w.get("end", 0.0))
            wdur = end - start
            if wdur <= 0:
                continue
            dw = diac[wi] if wi < len(diac) and diac[wi] else word
            nxt_dw = None
            if wi + 1 < len(diac) and diac[wi + 1]:
                nxt_dw = diac[wi + 1]
            elif wi + 1 < len(aligns):
                nxt_dw = aligns[wi + 1].get("word")
            rules = detect_rules_for_word(dw, nxt_dw, RIWAYAH)
            applied = [r for r in rules if r in RULE_SET]

            # C (distractor) for rules that do NOT apply (word-only clip)
            for r in RULES:
                if r not in applied:
                    fhs[r].write(json.dumps({"bucket": "C",
                        "reciter": rid, "wav_abs": wav_abs,
                        "start_ms": round(start * 1000.0, 1),
                        "end_ms": round(end * 1000.0, 1),
                        "reciter_is_tajweed": taj}, ensure_ascii=False) + "\n")

            # A / B for rules that apply AND have sufficient evidence AND the
            # reciter has usable acoustic evidence for that specific rule.
            for r in applied:
                # Per-CLIP MADD labelling: override the reciter-level quality with
                # this word's own elongation ratio (matches the model's duration
                # feature).  Other rules keep their reciter-level tertile quality.
                if r == "MADD":
                    nm = nonmadd_med.get(rid, 0.0)
                    if nm and nm > 0:
                        ratio = wdur / nm
                        if ratio >= MADD_CLIP_HI:
                            q = "strong"
                        elif ratio <= MADD_CLIP_LO:
                            q = "weak"
                        else:
                            q = None  # ambiguous -> excluded
                    else:
                        q = None
                else:
                    q = qmap.get(r)
                # extend the clip for cross-word rules so the transition is in
                # audio.  Done BEFORE the per-clip gate so the evidence scalar
                # and the model's input are measured on the SAME window -- the
                # nasal/clear boundary cue of IDGHAM/IKHFA/IZHAR/MEEM_SAKINAH
                # (all CROSSWORD_RULES) lives at the word transition, not inside
                # the single host word.  The reciter-level tertile thresholds in
                # `clip_thr` are likewise computed on extended windows (pass1), so
                # measuring the per-clip label on the non-extended window would
                # create a train-consistency bug.
                cend = end
                if r in CROSSWORD_RULES and wi + 1 < len(aligns):
                    nxt_end = float(aligns[wi + 1].get("end", end))
                    if nxt_end > start:
                        cend = nxt_end
                # Per-CLIP denoising gate: relabel by this clip's OWN acoustic
                # evidence vs the global per-clip tertile, overriding the noisier
                # reciter-level tertile.  This aligns the label with the exact cue
                # the head consumes.  Active for ALL non-MADD rules now (see
                # PER_CLIP_GATE_RULES); reciter-level `q` is the fallback when the
                # rule has too few clips to compute a global tertile (clip_thr).
                if r in PER_CLIP_GATE_RULES and r in clip_thr:
                    cstart_ms = round(start * 1000.0, 1)
                    cend_ms = round(cend * 1000.0, 1)
                    seg = None
                    try:
                        seg, _ = load_clip(wav_abs, cstart_ms, cend_ms)
                    except Exception:
                        seg = None
                    f = clip_features(seg) if seg is not None else None
                    ev_scalar = rule_evidence_fn(r, f)
                    if ev_scalar is not None:
                        lo, hi = clip_thr[r]
                        if ev_scalar >= hi:
                            q = "strong"
                        elif ev_scalar <= lo:
                            q = "weak"
                        else:
                            q = None  # ambiguous -> excluded
                    else:
                        q = None
                g = gates.get(r, MIN_EVIDENCE.get(r, 1))
                if g is None or ev.get(r, 0) < g:
                    continue
                if q == "strong":
                    bucket = "A"
                elif q == "weak":
                    bucket = "B"
                else:
                    continue  # reciter not trusted for this rule -> excluded
                fhs[r].write(json.dumps({"bucket": bucket,
                    "reciter": rid, "wav_abs": wav_abs,
                    "start_ms": round(start * 1000.0, 1),
                    "end_ms": round(cend * 1000.0, 1),
                    "reciter_is_tajweed": taj}, ensure_ascii=False) + "\n")
        if n_sidecars % 2000 == 0:
            log("pass2 streamed", n_sidecars, "sidecars")
    for fh in fhs.values():
        fh.close()
    log("pass2 done:", n_sidecars, "sidecars written to temp files")
    return tmp_paths


# ---------------------------------------------------------------------------
# Pass 3 - balance per rule (one rule in RAM at a time) and emit outputs
# ---------------------------------------------------------------------------
def pass3_balance_and_emit(tmp_paths, target_dir, reciter_split, reciter_verse_count,
                           ratios, is_tajweed, stats, gates, relaxed):
    rng = random.Random(SEED)
    per_rule_summary = {}

    for r in RULES:
        A, B, C = [], [], []
        with open(tmp_paths[r], encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                if o["bucket"] == "A":
                    A.append(o)
                elif o["bucket"] == "B":
                    B.append(o)
                else:
                    C.append(o)
        pos = len(A)
        if pos == 0:
            # nothing positive for this rule -> not usable for a binary head
            per_rule_summary[r] = {
                "positives": 0, "mistakes": len(B), "distractors_raw": len(C),
                "kept": 0, "thin": True,
                "gate": gates.get(r), "gate_relaxed": relaxed.get(r, False),
                "note": "no positive clips (even gate=1); skipped",
            }
            log("rule %-13s: 0 positives -> skipped (no usable binary target)" % r)
            continue

        # negatives = a BUDGETED mix of B (mistake) and C (distractor), capped at
        # NEG_MULT * positives.  We keep up to B_FRAC of the budget from B (the
        # strong hard-negative) and fill the remainder with C (easy negative),
        # subsampling each deterministically.  This guarantees pos:neg ~ 1:NEG_MULT
        # and never discards the abundant C (the earlier bug).
        neg_target = int(round(NEG_MULT * pos))
        budget_B = min(len(B), int(round(B_FRAC * neg_target)))
        budget_C = neg_target - budget_B
        if len(B) > budget_B:
            rng.shuffle(B)
            keep_B = B[:budget_B]
        else:
            keep_B = B
        if len(C) > budget_C:
            rng.shuffle(C)
            keep_C = C[:budget_C]
        else:
            keep_C = C
        neg = len(keep_B) + len(keep_C)

        # write manifest_<RULE>.jsonl with split assigned by reciter
        out_path = os.path.join(target_dir, "manifest_%s.jsonl" % r)
        n_written = 0
        with open(out_path, "w", encoding="utf-8") as out:
            for o in keep_B + keep_C:
                label = "mistake" if o["bucket"] == "B" else "distractor"
                rec = o["reciter"]
                row = {
                    "rule": r,
                    "label": label,
                    "reciter": rec,
                    "wav_abs": o["wav_abs"],
                    "start_ms": o["start_ms"],
                    "end_ms": o["end_ms"],
                    "reciter_is_tajweed": o["reciter_is_tajweed"],
                    "y": 0,
                    "split": reciter_split.get(rec, "train"),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
            for o in A:
                rec = o["reciter"]
                row = {
                    "rule": r,
                    "label": "correct",
                    "reciter": rec,
                    "wav_abs": o["wav_abs"],
                    "start_ms": o["start_ms"],
                    "end_ms": o["end_ms"],
                    "reciter_is_tajweed": o["reciter_is_tajweed"],
                    "y": 1,
                    "split": reciter_split.get(rec, "train"),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1

        pos_neg = (pos / neg) if neg else float("inf")
        thin = pos < THIN_POS
        per_rule_summary[r] = {
            "positives": pos,
            "mistakes": len(keep_B),
            "distractors_used": len(keep_C),
            "distractors_raw": len(C),
            "negatives": neg,
            "pos_neg_ratio": round(pos_neg, 3),
            "kept": n_written,
            "thin": thin,
            "gate": gates.get(r),
            "gate_relaxed": relaxed.get(r, False),
        }
        log("rule %-13s: pos=%d neg=%d (B=%d C=%d) ratio=%.2f gate=%s%s"
            % (r, pos, neg, len(keep_B), len(keep_C), pos_neg,
               gates.get(r), " [THIN]" if thin else ""))

        # cleanup temp
        try:
            os.remove(tmp_paths[r])
        except OSError:
            pass

    stats["per_rule"] = per_rule_summary
    return per_rule_summary


# ---------------------------------------------------------------------------
# Reciter-disjoint splits
# ---------------------------------------------------------------------------
def build_splits(reciter_ids, ratios, is_tajweed):
    rng = random.Random(SEED + 7)
    recs = sorted(reciter_ids)
    rng.shuffle(recs)
    n = len(recs)
    n_train = int(round(TRAIN_FRAC * n))
    n_val = int(round(VAL_FRAC * n))
    train = recs[:n_train]
    val = recs[n_train:n_train + n_val]
    test = recs[n_train + n_val:]
    splits = {"train": train, "val": val, "test": test}
    reciter_split = {}
    for s, lst in splits.items():
        for r in lst:
            reciter_split[r] = s
    return splits, reciter_split


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tadabur_part",
                    default=os.environ.get("TADABUR_PART", ""),
                    help="absolute path to existing source tadabur_part (read-only)")
    ap.add_argument("--target_dir",
                    default=os.environ.get("TARGET_DIR",
                                            os.path.join(HERE, "..", "dataset")),
                    help="output dataset dir (write-only)")
    # Shard whitelist.  The audio corpus (tadabur_part/) now contains shards 0-12
    # but the C model was trained ONLY on shards 0-6 (3,193 sidecars).  Rebuilding
    # without this flag would silently produce the 13-shard merged data (the
    # experiment that already underperformed C on 6/8 rules).  Format: "0-6",
    # "0,1,2", or "0-3,7".  Omit/range-all = walk every shard (merged behaviour).
    ap.add_argument("--shards", default=None,
                    help="restrict walk to the given shards (e.g. '0-6'); default = all")
    args = ap.parse_args()

    # Apply shard whitelist to tadabur_adapter.walk_sidecars before pass1/pass2.
    if args.shards:
        shards = set()
        for tok in args.shards.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                a, b = tok.split("-", 1)
                shards.update(range(int(a), int(b) + 1))
            else:
                shards.add(int(tok))
        import tadabur_adapter as _ta
        _ta.SHARD_WHITELIST = shards
        log("shard whitelist:", sorted(shards))
    else:
        log("shard whitelist: none (walk all shards)")

    tadabur_part = os.path.abspath(args.tadabur_part)
    target_dir = os.path.abspath(args.target_dir)
    if not tadabur_part or not os.path.isdir(tadabur_part):
        log("ERROR: TADABUR_PART not set or not a dir:", tadabur_part)
        sys.exit(1)
    os.makedirs(target_dir, exist_ok=True)
    log("TADABUR_PART =", tadabur_part)
    log("TARGET_DIR   =", target_dir)
    log("params: floor=%.2f min_madd=%d evidence_gates=%s neg_mult=%.1f seed=%d"
        % (MADD_RATIO_FLOOR, MIN_MADD_WORDS, dict(MIN_EVIDENCE), NEG_MULT, SEED))

    # Pass 1
    (madd_dur, nonmadd_dur, rule_evidence, rule_feat, rvc,
     ratios, is_tajweed, nonmadd_med, clip_thr) = pass1_reciter_stats(tadabur_part)

    # Per-rule acoustic quality (the label fix): each rule's CORRECT/MISTAKE now
    # comes from THAT rule's own measured acoustics, not from MADD elongation.
    rule_quality = derive_rule_quality(rule_feat, rule_evidence, ratios)

    # per-rule evidence gate (relax only when a rule would otherwise have 0 positives)
    gates, relaxed = choose_rule_gates(rule_evidence, rule_quality, RULES, MIN_EVIDENCE,
                                       MIN_POS_FOR_RULE)
    log("per-rule gates:", {r: (gates[r], "relaxed" if relaxed[r] else "")
                            for r in RULES if gates[r] is not None})

    # Diagnostic: per-rule evidence separability (strong vs weak group medians)
    log("--- per-rule acoustic evidence separability ---")
    sep = {}
    for r in RULES:
        strong, weak = [], []
        for rid, d in rule_feat.items():
            lst = d.get(r)
            if not lst:
                continue
            m = statistics.median(lst)
            if rule_quality.get(rid, {}).get(r) == "strong":
                strong.append(m)
            elif rule_quality.get(rid, {}).get(r) == "weak":
                weak.append(m)
        if strong and weak:
            sm, wm = statistics.median(strong), statistics.median(weak)
            gap = (sm - wm) / (max(abs(sm), abs(wm), 1e-6))
            sep[r] = {"strong_n": len(strong), "weak_n": len(weak),
                      "strong_med": round(sm, 4), "weak_med": round(wm, 4),
                      "rel_gap": round(gap, 3)}
            log("  %-13s strong(n=%d med=%.4f)  weak(n=%d med=%.4f)  rel_gap=%.2f"
                % (r, len(strong), sm, len(weak), wm, gap))
        else:
            sep[r] = {"strong_n": len(strong), "weak_n": len(weak),
                      "note": "no split (single group)"}
            log("  %-13s no split (strong=%d weak=%d)" % (r, len(strong), len(weak)))

    # splits (need full reciter set)
    splits, reciter_split = build_splits(list(rvc.keys()), ratios, is_tajweed)

    # Pass 2 (stream)
    tmp_paths = pass2_stream_rows(tadabur_part, target_dir, rule_evidence,
                                  rule_quality, is_tajweed, gates, nonmadd_med,
                                  clip_thr)

    # Pass 3 (balance + emit)
    stats = {
        "config": {
            "MADD_RATIO_FLOOR": MADD_RATIO_FLOOR,
            "MIN_MADD_WORDS": MIN_MADD_WORDS,
            "MIN_EVIDENCE": MIN_EVIDENCE,
            "NEG_MULT": NEG_MULT,
            "B_FRAC": B_FRAC,
            "SEED": SEED,
            "MIN_POS_FOR_RULE": MIN_POS_FOR_RULE,
            "RIWAYAH": RIWAYAH,
            "TRAIN_FRAC": TRAIN_FRAC,
            "VAL_FRAC": VAL_FRAC,
        },
        "reciters": {
            "total": len(rvc),
            "tajweed": sum(1 for v in is_tajweed.values() if v),
            "non_tajweed": sum(1 for v in is_tajweed.values() if not v),
            "verses": sum(rvc.values()),
        },
        "splits": {k: len(v) for k, v in splits.items()},
        "gates": {r: gates.get(r) for r in RULES},
        "evidence_separability": sep,
    }
    per_rule = pass3_balance_and_emit(tmp_paths, target_dir, reciter_split, rvc,
                                      ratios, is_tajweed, stats, gates, relaxed)
    stats["per_rule"] = per_rule

    # madd-ratio distribution summary (for the README / audit)
    all_r = sorted(ratios.values())
    if all_r:
        stats["madd_ratio"] = {
            "min": round(min(all_r), 3),
            "median": round(statistics.median(all_r), 3),
            "max": round(max(all_r), 3),
            "n_ge_floor": sum(1 for x in all_r if x >= MADD_RATIO_FLOOR),
        }

    with open(os.path.join(target_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    with open(os.path.join(target_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    write_readme(target_dir, stats, per_rule)
    log("DONE. Outputs in", target_dir)


def write_readme(target_dir, stats, per_rule):
    lines = []
    lines.append("# Tajweed Mistake-Detection Training Dataset\n")
    lines.append("Generated by `src/prepare_dataset.py` from the **read-only** "
                 "Tadabur `tadabur_part/` corpus. Audio is never copied or moved; "
                 "manifest rows reference absolute WAV paths in the source dir.\n")
    c = stats["config"]
    lines.append("## Configuration\n")
    lines.append("- `MADD_RATIO_FLOOR = %s` - acoustic Tajweed gate (median MADD-word "
                 "duration / median non-MADD-word duration)." % c["MADD_RATIO_FLOOR"])
    lines.append("- `MIN_MADD_WORDS = %s` - minimum MADD words before a reciter's "
                 "ratio is trusted." % c["MIN_MADD_WORDS"])
    lines.append("- `MIN_EVIDENCE = %s` - per-rule evidence gate: a reciter is only "
                 "trusted as a *correct performer* of rule R if they have >= this many "
                 "words of R (avoids assuming a madd-good reciter is also ghunnah-good)."
                 % c["MIN_EVIDENCE"])
    lines.append("- `B_FRAC = %s` - share of the negative budget taken from Bucket B "
                 "(mistake); the rest comes from Bucket C (distractor)." % c.get("B_FRAC", B_FRAC))
    g = stats.get("gates", {})
    relaxed_rules = [r for r in RULES if g.get(r) is not None and r in stats.get("per_rule", {})
                     and stats["per_rule"][r].get("gate_relaxed")]
    if relaxed_rules:
        lines.append("- Per-rule gates actually used: " +
                     ", ".join("%s=%s" % (r, g[r]) for r in RULES if g.get(r) is not None) +
                     ".  Rules whose gate was relaxed below its strict per-rule value "
                     "(still too sparse within surahs 1-20): %s.  They keep a *small "
                     "balanced set* per PLAN section 4.4(a) rather than collapsing to "
                     "zero." % (", ".join(relaxed_rules),))
    lines.append("- `NEG_MULT = %s` - negatives (mistake + distractor) are capped at "
                 "this multiple of positives." % c["NEG_MULT"])
    lines.append("- `SEED = %s` - fixed for reproducible balancing/splits." % c["SEED"])
    lines.append("- `RIWAYAH = %s`\n" % c["RIWAYAH"])

    lines.append("## Reciter classification (keep ALL, label by style)\n")
    r = stats["reciters"]
    lines.append("- total reciters: **%d** (%d Tajweed, %d non-Tajweed)"
                 % (r["total"], r["tajweed"], r["non_tajweed"]))
    lines.append("- verses scanned: %d" % r["verses"])
    mr = stats.get("madd_ratio", {})
    if mr:
        lines.append("- MADD-ratio distribution: min=%.2f median=%.2f max=%.2f; "
                     "%d reciters >= floor"
                     % (mr["min"], mr["median"], mr["max"], mr["n_ge_floor"]))
    lines.append("")

    lines.append("## Three buckets per rule (per-rule acoustic evidence)\n")
    lines.append("- **A - CORRECT (positive, y=1)**: rule applies in text AND the "
                 "reciter's OWN acoustic evidence for *this rule* is in the STRONG half "
                 "of the population AND they have >= MIN_EVIDENCE words of the rule.")
    lines.append("- **B - MISTAKE (hard negative, y=0)**: rule applies AND the reciter's "
                 "acoustic evidence for *this rule* is in the WEAK half AND evidence OK. "
                 "The evidence is rule-specific: nasal-energy for GHUNNAH/IKHFA/IDGHAM/"
                 "MEEM_SAKINAH, clarity (low nasal) for IZHAR, stop/burst for QALQALAH, "
                 "duration ratio for MADD. This replaces the old single MADD-ratio flag "
                 "that made every rule's label correlate only with MADD elongation.")
    lines.append("- **C - DISTRACTOR (easy negative, y=0)**: rule does NOT apply in "
                 "text (abundant). Subsampled to keep pos:neg in [1:1, 1:3].\n")
    lines.append("See `evidence_separability` in stats.json for the strong-vs-weak "
                 "group medians per rule (rel_gap is how cleanly the label separates).")

    lines.append("## Per-rule counts (after balancing)\n")
    lines.append("| rule | positives (A) | mistakes (B) | distractors (C) | "
                 "pos:neg | thin |")
    lines.append("|------|-------------:|-------------:|----------------:|--------:|:----:|")
    for rule, s in sorted(per_rule.items()):
        if s.get("positives", 0) == 0:
            lines.append("| %s | 0 | %d | %d | - | yes |"
                         % (rule, s.get("mistakes", 0), s.get("distractors_raw", 0)))
        else:
            lines.append("| %s | %d | %d | %d | %.2f | %s |"
                         % (rule, s["positives"], s["mistakes"], s["distractors_used"],
                            s["pos_neg_ratio"], "yes" if s["thin"] else ""))
    lines.append("")

    lines.append("## Train / Val / Test (reciter-disjoint)\n")
    sp = stats["splits"]
    lines.append("- train reciters: %d, val reciters: %d, test reciters: %d"
                 % (sp["train"], sp["val"], sp["test"]))
    lines.append("- No reciter appears in more than one split (prevents voice "
                 "memorisation). Each manifest row carries its reciter's `split`.")
    lines.append("")

    lines.append("## Files\n")
    lines.append("- `manifest_<RULE>.jsonl` - one row per (segment, rule): "
                 "`{rule,label,reciter,wav_abs,start_ms,end_ms,reciter_is_tajweed,y,split}`. "
                 "`label` is in {correct, mistake, distractor}; `y=1` only for CORRECT.")
    lines.append("- `splits.json` - reciter -> split assignment.")
    lines.append("- `stats.json` - machine-readable copy of the above.")
    lines.append("")
    lines.append("## Notes / limitations\n")
    lines.append("- Labels are *acoustic-style proxies*, not human error annotations. "
                 "Each rule's label is derived from THAT rule's own measurable acoustics "
                 "(see rule_acoustics.py), which is the honest best-available signal and "
                 "is learnable, but it is not ground truth.")
    lines.append("- The MADD-ratio still gates MADD; non-MADD rules use nasal-energy / "
                 "stop / clarity, which may be noisier. `evidence_separability.rel_gap` "
                 "shows how cleanly each rule separates strong vs weak reciters.")
    lines.append("- Thin rules (few positives, e.g. IKHFA/IZHAR/IDGHAM) have tiny "
                 "validation sets; their test AUC is statistically noisy. Adding more "
                 "Tadabur shards is the recommended remedy for volume.")
    lines.append("- Cross-word rules (IDGHAM/IKHFA/IZHAR/MEEM_SAKINAH) use clips extended "
                 "to include the following word so the relevant transition is in audio.")
    lines.append("- SILAH is intentionally not derived from text (see tajweed_rules.py) "
                 "and therefore has no clips.")
    with open(os.path.join(target_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
