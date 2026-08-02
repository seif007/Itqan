#!/usr/bin/env python3
"""
tadabur_adapter.py — Turn the downloaded Tadabur subset into a corpusF-shaped
manifest that the Calibrate.kt runner can consume.

Key points:
  * Tajweed rules are DETERMINISTIC from Arabic text, so we derive them from
    each word via tajweed_rules.detect_rules_for_word (riwayah="Hafs").
  * Tadabur has NO tarteel/tajweed style label. We therefore filter reciters
    ACOUSTICALLY: a reciter who actually performs tajweed prolongs MADD
    noticeably, so their median MADD-word duration is well above their median
    non-MADD-word duration (ratio >> 1). Plain-tarteel reciters have ratio ~1.
    We keep reciters whose ratio is clearly > 1 and drop the rest.
  * Words with rule "NONE" are skipped (no acoustic target to learn).

Output: <in_dir>/manifest.jsonl  with one JSON per (word, rule) row:
  {"rule","reciter","wav","word","start_ms","end_ms"}
(times in ms derived from word_alignments start/end in SECONDS.)
"""
import os, json, sys, statistics
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from tajweed_rules import detect_rules_for_word

# ---------------- diacritic recovery from Uthmani text ----------------
# Tadabur's word_alignment `word` tokens are rasm (no diacritics), so Tajweed
# rules cannot be derived from them directly.  Each sidecar also carries
# `text_ar_uthmani` which HAS full diacritics but uses Uthmani spelling
# variants (alif-maqsurah vs ya, waw-madd vs alif, stray madd-alifs) and a
# slightly different tokenisation.  We recover the diacritic-bearing form of
# each alignment word by greedily matching its normalised-rasm against the
# Uthmani string, then trimming stray leading/trailing marks.
_ALIF = {0x0627, 0x0649, 0x0622, 0x0623, 0x0625, 0x0671, 0x0670}
def _preprocess_uth(s):
    s = s.replace('\u0648\u0670', '\u0627')  # وٰ (waw + superscript alif) -> ا
    s = s.replace('\u064A\u0670', '\u0627')  # يٰ (ya   + superscript alif) -> ا
    s = s.replace('\u0627\u0653', '\u0627')  # آ (alif + maddah)          -> ا
    return s
def _is_letter(c):
    o = ord(c)
    if o in _ALIF: return True
    if 0x0621 <= o <= 0x064A: return True
    return False
def _norm_char(c):
    o = ord(c)
    if o in _ALIF: return 'A'          # alif family collapses for matching
    if o == 0x0640: return ''          # tatweel
    if c == ' ': return ''
    if 0x0621 <= o <= 0x064A: return c # consonant / alif-maqsurah range
    return ''                          # every other mark/annotation -> drop
def _nrasm(s):
    return ''.join(_norm_char(c) for c in _preprocess_uth(s))
def _try_match(uth, s, target):
    L = len(uth); cur = ''; p = s
    while p < L:
        nc = _norm_char(uth[p])
        if nc == '':
            p += 1; continue
        cur += nc
        if cur == target: return min(p + 1, L)
        if not target.startswith(cur): return None
        p += 1
    return None
def recover_diacritic_words(uth, aligns):
    """Diacritic-bearing Uthmani word for each alignment entry (parallel list)."""
    out = []
    uth = _preprocess_uth(uth)
    L = len(uth)
    p = 0
    for a in aligns:
        w = a.get('word', '')
        target = _nrasm(w)
        if not target:
            out.append('')
            continue
        e = None; pp = p
        while pp < L:
            e = _try_match(uth, pp, target)
            if e is not None: break
            while pp < L and _norm_char(uth[pp]) == '':
                pp += 1
            if pp < L: pp += 1
            else: break
        word = uth[p:e] if e is not None else ''
        # strip stray leading/trailing marks/spaces (they belong to neighbours)
        i = 0
        while i < len(word) and not _is_letter(word[i]): i += 1
        j = len(word)
        while j > i and not _is_letter(word[j - 1]): j -= 1
        out.append(word[i:j])
        if e is not None: p = e
    return out

DEFAULT_IN = os.path.join(ROOT, "tadabur_part")
DEFAULT_OUT = os.path.join(DEFAULT_IN, "manifest.jsonl")

RIWAYAH = os.environ.get("RIWAYAH", "Hafs")
MADD_RATIO_FLOOR = float(os.environ.get("MADD_RATIO_FLOOR", "1.3"))
MIN_RECITERS = int(os.environ.get("MIN_RECITERS", "8"))
MAX_RECITERS = int(os.environ.get("MAX_RECITERS", "50"))
MIN_MADD_WORDS = int(os.environ.get("MIN_MADD_WORDS", "3"))

def to_int_reciter(rid):
    try:
        return int(rid)
    except Exception:
        return abs(hash(str(rid))) % 100000

# Optional shard whitelist (set by prepare_dataset.py via --shards).  When set,
# walk_sidecars only descends into <in_dir>/shardK/ for K in this set; other
# shards (e.g. later-downloaded shards 7-12 that did NOT exist when the C model
# was labelled) are skipped.  None = walk the whole tree (the merged behaviour).
SHARD_WHITELIST = None

def walk_sidecars(in_dir):
    for root, _, files in os.walk(in_dir):
        if SHARD_WHITELIST is not None:
            # Only descend into whitelisted shardN subdirs of in_dir itself.
            rel = os.path.relpath(root, in_dir)
            top = rel.split(os.sep)[0]
            if top != "." and top.startswith("shard"):
                try:
                    if int(top[5:]) not in SHARD_WHITELIST:
                        continue
                except ValueError:
                    continue
        for fn in files:
            if not fn.endswith(".json"):
                continue
            if fn == "_shards_done.txt":
                continue
            full = os.path.join(root, fn)
            wav = os.path.join(root, fn[:-5] + ".wav")
            if not os.path.exists(wav):
                continue  # not a verse sidecar
            yield full

def median(xs):
    return statistics.median(xs) if xs else 0.0

def analyze(in_dir):
    rows = []                       # corpusF-shaped manifest rows
    madd_durs = {}                  # rid -> [madd word durations]
    nonmadd_durs = {}               # rid -> [non-madd word durations]
    reciter_verse_count = {}        # rid -> verse count

    for sidecar in walk_sidecars(in_dir):
        with open(sidecar, encoding="utf-8") as fh:
            meta = json.load(fh)
        rid = to_int_reciter(meta["reciter_id"])
        wav_rel = meta["wav"]
        aligns = meta.get("word_alignments", [])
        reciter_verse_count[rid] = reciter_verse_count.get(rid, 0) + 1
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
            # Prefer the diacritic-bearing Uthmani form for rule derivation;
            # fall back to the bare rasm word if recovery failed for this token.
            dw = diac[wi] if wi < len(diac) and diac[wi] else word
            nxt_dw = None
            if wi + 1 < len(diac) and diac[wi + 1]:
                nxt_dw = diac[wi + 1]
            elif wi + 1 < len(aligns):
                nxt_dw = aligns[wi + 1].get("word")
            rules = detect_rules_for_word(dw, nxt_dw, RIWAYAH)
            is_madd = "MADD" in rules
            # MADD words are the acoustic probe; they must NOT pollute the
            # "non-MADD" baseline, otherwise the ratio is artificially < 1.
            if is_madd:
                madd_durs.setdefault(rid, []).append(wdur)
            else:
                nonmadd_durs.setdefault(rid, []).append(wdur)
            for r in rules:
                if r in ("NONE",):
                    continue
                rows.append({
                    "rule": r,
                    "reciter": rid,
                    "wav": wav_rel,
                    "word": word,
                    "start_ms": round(start * 1000.0, 1),
                    "end_ms": round(end * 1000.0, 1),
                })

    # ---- Tajweed-vs-tarteel acoustic filter ----
    # A reciter who performs tajweed prolongs MADD noticeably, so their median
    # MADD-word duration is well above their median non-MADD-word duration
    # (ratio >> 1).  Plain/tarteel reciters sit near or below 1.  We require a
    # minimum number of MADD words so a single-verse reciter can't post a
    # noisy ratio.
    ratios = {}
    for rid in set(list(madd_durs.keys()) + list(nonmadd_durs.keys())):
        md = madd_durs.get(rid, [])
        nd = nonmadd_durs.get(rid, [])
        if len(md) < MIN_MADD_WORDS or not nd:
            ratios[rid] = 0.0
            continue
        ratios[rid] = median(md) / max(median(nd), 1e-6)

    # keep reciters clearly doing tajweed (ratio above floor)
    tajweed = {rid: r for rid, r in ratios.items() if r >= MADD_RATIO_FLOOR}
    kept = set(tajweed.keys())
    if len(kept) < MIN_RECITERS:
        # Soften: keep the top MIN_RECITERS (with enough MADD evidence) by ratio.
        ev = {rid: r for rid, r in ratios.items() if madd_durs.get(rid) and len(madd_durs[rid]) >= MIN_MADD_WORDS}
        top = sorted(ev.items(), key=lambda kv: -kv[1])[:MIN_RECITERS]
        kept = set(r for r, _ in top)
    # cap to MAX_RECITERS (top by ratio) to bound calibration cost
    if len(kept) > MAX_RECITERS:
        kept = set(sorted(kept, key=lambda r: -ratios[r])[:MAX_RECITERS])

    kept_rows = [r for r in rows if r["reciter"] in kept]

    # per-rule counts (after filter)
    rule_counts = {}
    for r in kept_rows:
        rule_counts[r["rule"]] = rule_counts.get(r["rule"],0) + 1

    return kept_rows, ratios, kept, rule_counts, reciter_verse_count

def run(in_dir=DEFAULT_IN, out_path=DEFAULT_OUT):
    kept_rows, ratios, kept, rule_counts, rvc = analyze(in_dir)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in kept_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"[adapter] riwayah={RIWAYAH}")
    print(f"[adapter] total reciters seen: {len(ratios)}  verses: {sum(rvc.values())}")
    print(f"[adapter] tajweed reciters kept: {len(kept)} (floor ratio={MADD_RATIO_FLOOR})")
    print(f"[adapter] manifest rows (after filter): {len(kept_rows)}")
    print("[adapter] per-reciter MADD ratio (kept marked *):")
    for rid, r in sorted(ratios.items(), key=lambda kv: -kv[1]):
        mark = "*" if rid in kept else " "
        vc = rvc.get(rid, 0)
        print(f"   {mark} reciter {rid:>4}  madd_ratio={r:5.2f}  verses={vc}")
    print("[adapter] per-rule clip counts (kept):")
    for rule in sorted(rule_counts):
        print(f"   {rule:>12}: {rule_counts[rule]}")
    print(f"[adapter] wrote {out_path}")
    print("=" * 60)
    return kept_rows

if __name__ == "__main__":
    in_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    run(in_dir, out_path)
