#!/usr/bin/env python3
"""
Deterministic Tajweed-rule derivation from Quranic Arabic text.

Tadabur (and most large recitation corpora) give WORD-LEVEL timestamps + the
Arabic text, but NO Tajweed-rule labels. That is fine: Tajweed rules are a
function of the text itself (noon/tanween -> ikhfa/idgham/izhar, meem saakin ->
meem sakinah, qalqalah letters with sukoon, madd letters, lam shamsiyyah/
qamariyyah, ghunnah contexts). So we derive the rule(s) a word demonstrates and
use the dataset's word timestamp to locate the audio.

Rule names intentionally match the app's TajweedRule enum:
    MADD, GHUNNAH, QALQALAH, IDGHAM, IKHFA, ALLAM, MEEM_SAKINAH, SILAH, IZHAR

SILAH is Qalun-specific and obscure; it is intentionally NOT derived here and is
instead carried over from corpusF via the adapter merge.

For each word we return the LIST of rules it demonstrates (a word can show more
than one). The adapter emits one manifest clip per (word, rule).

Validation: `python tajweed_rules.py --validate` runs the engine over corpusF's
tokens and reports how often the engine agrees with corpusF's given rule.
"""
import os
import sys
import json
import argparse

# ---------------- Arabic Unicode constants ----------------
ALIF        = '\u0627'
ALIF_WASL   = '\u0671'   # ٱ
ALIF_MADDA  = '\u0622'   # آ (alif madda -> inherently madd)
ALIF_MAQ    = '\u0649'   # ى (alif maqsurah)
LAM         = '\u0644'   # ل
YAA         = '\u064A'   # ي
WAW         = '\u0648'   # و
NOON        = '\u0646'   # ن
MEEM        = '\u0645'   # م
HAMZA       = '\u0621'   # ء

FATHA = '\u064E'
DAMMA = '\u064F'
KASRA = '\u0650'
SUKOON = '\u0652'
SHADDA = '\u0651'
TANWEEN_FATH = '\u064B'   # ً
TANWEEN_DAMM = '\u064C'   # ٌ
TANWEEN_KASR = '\u064D'   # ٍ
TANWEEN = {TANWEEN_FATH, TANWEEN_DAMM, TANWEEN_KASR}

# letters that DO carry a diacritic we must strip when finding base consonants
DIACRITICS = set(TANWEEN) | {FATHA, DAMMA, KASRA, SUKOON, SHADDA, '\u0670', '\u0640'}

QALQALAH = {'\u0642', '\u0637', '\u0628', '\u062C', '\u062F'}  # ق ط ب ج د
THROAT   = {'\u0621', '\u0647', '\u0639', '\u062D', '\u063A', '\u062E'}  # ء ه ع ح غ خ (izhar)
IKHFA    = {'\u062A', '\u062B', '\u062C', '\u062F', '\u0630', '\u0632',  # ت ث ج د ذ ز
            '\u0633', '\u0634', '\u0635', '\u0636', '\u0637', '\u0638',  # س ش ص ض ط ظ
            '\u0641', '\u0642', '\u0643', '\u0644'}                     # ف ق ك ل
IDGHAM_GHUNNAH = {'\u064A', '\u0646', '\u0645', '\u0648'}  # ي ن م و (with ghunnah)
IDGHAM_PLAIN   = {'\u0644', '\u0631'}                       # ل ر (without ghunnah)
SUN_LETTERS = {'\u062A', '\u062B', '\u062C', '\u062D', '\u062E', '\u0639', '\u063A',
               '\u0641', '\u0642', '\u0643', '\u0644', '\u0645', '\u0646', '\u0647', '\u064A'}  # شمسية


def strip_diacritics(s: str) -> str:
    return ''.join(c for c in s if c not in DIACRITICS)


def ends_with_tanween(word: str) -> bool:
    return bool(word) and word[-1] in TANWEEN


def first_base_letter(next_word: str or None) -> str or None:
    if not next_word:
        return None
    base = strip_diacritics(next_word)
    # skip a leading alif/wasl/hamza; the connecting letter is the first real consonant
    for c in base:
        if c in (ALIF, ALIF_WASL, '\u0623', '\u0625', '\u0622', '\u0670'):
            continue
        return c
    return base[0] if base else None


def has_sukoon_after(word: str, idx: int) -> bool:
    return idx + 1 < len(word) and word[idx + 1] in (SUKOON, SHADDA)


def detect_rules_for_word(word: str, next_word: str or None = None,
                          riwayah: str = "Hafs"):
    """Return the list of Tajweed rules demonstrated by `word`.

    `next_word` enables cross-word noon-saakin / tanween and meem-saakin rules.
    """
    rules = set()
    if not word:
        return ["NONE"]

    # --- QALQALAH: a qalqalah letter carrying sukoon (or shadda) ---
    for i, ch in enumerate(word):
        if ch in QALQALAH and has_sukoon_after(word, i):
            rules.add("QALQALAH")
            break

    # --- MADD: fatha+alif, kasra+yaa, damma+waw, madda, madd badal, madd maqsurah,
    #          and madd munfasil from tanween + madd-letter (ًا / ٍي / ٌو) ---
    if (FATHA + ALIF in word or FATHA + ALIF_WASL in word or FATHA + ALIF_MAQ in word
            or KASRA + YAA in word
            or DAMMA + WAW in word
            or HAMZA + ALIF in word or HAMZA + WAW in word or HAMZA + YAA in word
            or ALIF_MADDA in word
            or TANWEEN_FATH + ALIF in word or TANWEEN_FATH + ALIF_MAQ in word
            or TANWEEN_KASR + YAA in word
            or TANWEEN_DAMM + WAW in word):
        rules.add("MADD")

    # --- GHUNNAH: nasalization. Present whenever there is tanween (tanween is
    #     produced with ghunnah), or a noon/meem carrying shadda (idgham of a
    #     nasal), or the nasal-transition contexts handled below. ---
    if any(c in TANWEEN for c in word):
        rules.add("GHUNNAH")
    for ch in (NOON, MEEM):
        if ch in word and has_sukoon_after(word, word.rfind(ch)) and SHADDA in word:
            rules.add("GHUNNAH")

    # --- ALLAM: the definite article الـ (lam shamsiyyah/qamariyyah) ---
    if word.startswith(ALIF + LAM) or word.startswith(ALIF_WASL + LAM):
        rules.add("ALLAM")

    nfl = first_base_letter(next_word)

    # --- NOON saakin / TANWEEN connecting to the next word's first letter ---
    noon_saakin = (NOON in word and has_sukoon_after(word, word.rfind(NOON))) \
        or ends_with_tanween(word)
    if noon_saakin and nfl is not None:
        if nfl in IDGHAM_GHUNNAH:
            rules.add("IDGHAM"); rules.add("GHUNNAH")
        elif nfl in IDGHAM_PLAIN:
            rules.add("IDGHAM")
        elif nfl in THROAT:
            rules.add("IZHAR")
        elif nfl in IKHFA:
            rules.add("IKHFA"); rules.add("GHUNNAH")
        else:
            rules.add("IZHAR")  # default: clear

    # --- MEEM saakin connecting to the next word's first letter ---
    meem_saakin = MEEM in word and has_sukoon_after(word, word.rfind(MEEM))
    if meem_saakin and nfl is not None:
        if nfl == MEEM:
            rules.add("MEEM_SAKINAH"); rules.add("GHUNNAH")   # idgham shafawi
        elif nfl == '\u0628':                                   # ب
            rules.add("MEEM_SAKINAH"); rules.add("GHUNNAH")     # ikhfa shafawi
        else:
            rules.add("MEEM_SAKINAH")                           # izhar shafawi

    if not rules:
        return ["NONE"]
    return sorted(rules)


# --------------------------------------------------------------------------
# Validation against corpusF (no network needed)
# --------------------------------------------------------------------------
def validate_corpusf(corpusF_dir: str):
    manifest = os.path.join(corpusF_dir, "manifest.jsonl")
    if not os.path.exists(manifest):
        print(f"validate: manifest not found at {manifest}")
        return
    # group clips by verse so we can supply cross-word context
    verses = {}
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            key = (o.get("reciter"), o.get("surah"), o.get("verse_key"))
            verses.setdefault(key, []).append(o)

    total = 0
    hit = 0
    per_rule = {}
    for key, words in verses.items():
        words.sort(key=lambda o: int(o.get("word_index", 0)))
        for i, o in enumerate(words):
            given = o.get("rule")
            token = o.get("word") or o.get("token") or ""
            nxt = words[i + 1].get("word") or words[i + 1].get("token") or None \
                if i + 1 < len(words) else None
            derived = detect_rules_for_word(token, next_word=nxt)
            total += 1
            per_rule.setdefault(given, [0, 0])
            per_rule[given][1] += 1
            if given in derived:
                hit += 1
                per_rule[given][0] += 1
    print(f"Validation vs corpusF (verse-context): {hit}/{total} clips have their "
          f"given rule in the engine's derived set ({100.0*hit/max(1,total):.1f}%)")
    print("Per given-rule coverage (engine includes the label):")
    for r in sorted(per_rule):
        h, t = per_rule[r]
        print(f"   {r:14s} {h:3d}/{t:3d}  ({100.0*h/max(1,t):.0f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="validate engine against corpusF/manifest.jsonl")
    ap.add_argument("--corpusF", default=os.path.join(os.path.dirname(__file__), "corpusF"))
    args = ap.parse_args()
    if args.validate:
        validate_corpusf(args.corpusF)
    else:
        # quick smoke test
        tests = [
            ("الرَّحْمَـٰنِ", "الرَّحِيمِ"),   # ALLAM + madd; next word starts with ر (idgham plain)
            ("مِنْ", "نُّورٍ"),                 # noon saakin -> depends on next
            ("أَنَّ", None),
            ("يَقُومُ", None),                  # madd (damma+waw)
            ("الْحَقُّ", None),                 # shadda/qalqalah? qaf sukoon -> qalqalah
            ("بِسْمِ", "اللَّهِ"),
        ]
        for w, nxt in tests:
            print(w, "->", detect_rules_for_word(w, nxt))
