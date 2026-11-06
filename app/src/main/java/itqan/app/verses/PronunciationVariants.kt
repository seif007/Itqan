package itqan.app.verses

/**
 * Pronunciation-aware matching layer for the recitation checker.
 *
 * The Quran text we receive from the API is the *display* (Uthmani) script, but
 * several words are not recited letter-for-letter the way they are written:
 *
 *  1. The "disjointed" opening letters of some surahs (الحروف المقطعة): الم at the
 *     start of Al-Baqarah is recited "ألف لام ميم" (three syllable-words), not as a
 *     single "الم" token. There are only 14 such fixed tokens in the whole Quran.
 *
 *  2. The dagger alif (ألف خنجرية, U+0670): الصَّلَوٰة is written with a superscript
 *     alif but pronounced with a waw ("الصَّلَوْة"). Same family covers الرِّبا,
 *     سَلَوٰة, هَٰذَا, أُولَٰئِكَ, ذَٰلِكَ, مِائَة, الْحَيَوٰة and الْزَّكَوٰة.
 *
 *  3. Hamzat al-wasl (ٱ) is silent when preceded by another word, and hamza on
 *     variants (أ/إ/ؤ/ئ) is folded by the base normalizer.
 *
 * We keep small deterministic tables for the truly fixed exceptions (muqatta'at,
 * dagger-alif) and let the base char-level normalizer fold the rest. Recognizers
 * return free-form Arabic, so we produce MULTIPLE accepted spoken-form variants
 * per display word (segmented and concatenated; alternate matres like را/راء), and
 * [isWordMatchAny]/[isWordMatchPlural] try every combination on the target side.
 * The DISPLAY side stays exactly what the API delivered.
 */

/**
 * Muqatta'at expansions. Keys are the BASE-LETTER form of the token after the app's
 * normalizeArabic (diacritics stripped, wasl->ا, ة->ه, ى/ئ->ي, ؤ->و, ء->ا). Each
 * value is a segmentation into syllable-slots; every slot is itself a list of
 * accepted spellings (e.g. را vs راء, طا vs طاء). The matcher also accepts the full
 * concatenation of all slots, for recognizers that write the sequence as one token.
 */
private val muqattaat: Map<String, List<List<List<String>>>> = mapOf(
    // البقرة ، آل عمران ، العنكبوت ، الروم ، لقمان ، السجدة
    "الم" to listOf(
        listOf(listOf("الف"), listOf("لام"), listOf("ميم")),
        listOf(listOf("الف"), listOf("لم"), listOf("ميم"))
    ),
    // الرعد
    "المر" to listOf(
        listOf(listOf("الف"), listOf("لام"), listOf("ميم"), listOf("را", "راء"))
    ),
    // يونس ، هود ، يوسف ، إبراهيم ، الحجر (تُقرأ ألف لام را)
    "الر" to listOf(
        listOf(listOf("الف"), listOf("لام"), listOf("را", "راء"))
    ),
    // مريم
    "كهيعص" to listOf(
        listOf(listOf("كاف"), listOf("ها", "هاء"), listOf("يا", "ياء"), listOf("عين"), listOf("صاد"))
    ),
    // غافر..الأحقاف (حم)
    "حم" to listOf(
        listOf(listOf("حا", "حاء"), listOf("ميم"))
    ),
    // الشورى حم ثم عسق
    "عسق" to listOf(
        listOf(listOf("عين"), listOf("سين"), listOf("قاف"))
    ),
    // طه
    "طه" to listOf(
        listOf(listOf("طا", "طاء"), listOf("هـ", "ها", "هاء"))
    ),
    // الشعراء ، القصص (طسم)
    "طسم" to listOf(
        listOf(listOf("طا", "طاء"), listOf("سين"), listOf("ميم"))
    ),
    // يس
    "يس" to listOf(
        listOf(listOf("ياسين")),                       // commonly recited as one merged name
        listOf(listOf("يا", "ياء"), listOf("سين"))     // segmented letter names
    ),
    // ص
    "ص" to listOf(
        listOf(listOf("صاد"))
    ),
    // ق
    "ق" to listOf(
        listOf(listOf("قاف"))
    ),
    // ن
    "ن" to listOf(
        listOf(listOf("نون"))
    ),
    // الأعراف افتُتحت بـ المص  (تُقرأ ألف لام ميم صاد)
    "المص" to listOf(
        listOf(listOf("الف"), listOf("لام"), listOf("ميم"), listOf("صاد"))
    )
)

/**
 * Dagger-alif (الألف الخنجرية) and other fixed "not-pronounced-as-written" words.
 * Keys are the NORMALIZED base form of the display word (what the app's
 * normalizeArabic produces, which strips the dagger alif U+0670 and فواصل).
 * Values are the accepted spoken forms in the same normalized base.
 */
private val specialWordSpokenForms: Map<String, List<String>> = mapOf(
    // الصَّلَوٰة / الزَّكَوٰة / سَلَوٰة / الْحَيَوٰة -> واو في اللفظ
    "الصلوه" to listOf("الصلوه", "الصلوا", "الصلوة"),
    "الزكوه" to listOf("الزكوه", "الزكوا", "الزكوة"),
    "سلوه"   to listOf("سلوه", "سلوا", "سلوة"),
    "الحيوه" to listOf("الحيوه", "الحيوا", "الحيوة"),
    "علوا"   to listOf("علوا", "علوة"),
    "شكوه"   to listOf("شكوه", "شكوا", "شكوة"),
    "نجوه"   to listOf("نجوه", "نجوا", "نجوة"),
    "منوه"   to listOf("منوه", "منوا", "منوة"),
    // الرِّبَا  (تُقرأ الربا), سَمٰوٰة، لَٰكِنْ, أُولَٰئِكَ, ذَٰلِكَ, هٰذا
    "الربوا" to listOf("الربا", "الربوه", "الربوة"),
    "الربو"  to listOf("الربا"),
    "سموه"   to listOf("سموه", "سموا", "سموة"),
    "السموه" to listOf("السموه", "السموا", "السموة"),
    "لاكن"   to listOf("لاكن"),
    "ولكن"   to listOf("ولكن", "ولكنه"),
    "اوليك"  to listOf("اولييك", "اوليك", "اولئك"),   // أُولَٰئِكَ
    "اولييك" to listOf("اولييك", "اوليك"),
    "ذلك"    to listOf("ذلك"),
    "ذالك"   to listOf("ذلك"),
    "هذا"    to listOf("هذا", "هذي"),
    "ههنا"   to listOf("ههنا", "ههيا"),
    // مِائَة (dagger alif on the mad signal)
    "ميه"    to listOf("ميه", "مايه", "مئة", "مائه"),
    "مايه"   to listOf("مايه", "ميه")
)

/**
 * Base-letter normalizer used by this file. Mirrors RecitationApp's normalizeArabic:
 * strips ALL diacritics (incl. dagger alif U+0670) + punctuation, folds wasl->ا,
 * ة->ه, ى/ئ->ي, ؤ->و, ء->ا, tatweel (ـ) removed.
 */
private fun normalizeForSpokenForms(text: String): String {
    val sb = StringBuilder(text.length)
    for (i in 0 until text.length) {
        val c = text[i]
        // Arabic harakat + dagger alif + maddah-above + honourific/Quranic annotation
        // marks + tatweel + ZWJ/FWJ: all stripped so "الٓمٓ" and "الم" are identical.
        if (c in '\u064B'..'\u065F' ||    // fathatan .. sukun .. hamza-above/below
            c == '\u0670' ||               // dagger alif
            c in '\u06D6'..'\u06ED' ||      // Quranic annotation signs
            c == '\u0653' || c == '\u0654' || c == '\u0655' || // maddah-above + hamza marks
            c == '\uFEFF' || c == '۬' || c == 'ے' || c == 'ـ' ||
            c == '\u200D' || c == '\u200C' ||
            c == '\u0671' && false  // wasl handled in the switch below
        ) continue
        if (".,;:!?()[]{}'\"/\\-_".contains(c)) continue
        if (c == '،' || c == '؛' || c == '؟') continue
        when (c) {
            '\u0622', '\u0623', '\u0625', '\u0671' -> sb.append('ا')
            '\u0629' -> sb.append('ه')
            '\u0649', '\u06CC', '\u0626' -> sb.append('ي')
            '\u0624' -> sb.append('و')
            '\u0621' -> sb.append('ا')
            else -> sb.append(c)
        }
    }
    return sb.toString()
}

private val muqattaatNormalized: Map<String, List<List<List<String>>>> by lazy {
    muqattaat.mapKeys { (k, _) -> normalizeForSpokenForms(k) }
}

private val specialWordSpokenFormsNormalized: Map<String, List<String>> by lazy {
    specialWordSpokenForms.mapKeys { (k, _) -> normalizeForSpokenForms(k) }
        .mapValues { (_, forms) -> forms.map { normalizeForSpokenForms(it) } }
}

/** True when the display word expands into multiple syllable-words (muqatta'at). */
fun hasMuqattaatExpansion(displayWord: String): Boolean {
    val base = normalizeForSpokenForms(displayWord)
    return muqattaatNormalized.containsKey(base)
}

/**
 * Single-token accepted spoken forms for an ordinary (non-muqatta'at) display word.
 * Always includes the plain normalized token itself, plus the dagger-alif alternates.
 */
internal fun singleTokenCandidatesFor(displayWord: String): List<String> {
    val base = normalizeForSpokenForms(displayWord)
    val out = LinkedHashSet<String>()
    out.add(base)
    specialWordSpokenFormsNormalized[base]?.let { out.addAll(it) }
    return out.toList()
}

/**
 * Per-display-word form groups used by the matcher.
 *
 * Each entry is either
 *  - `List<String>` = one-token candidates (normal words + the concatenated muqatta'at), OR
 *  - `List<List<String>>` = an ordered muqatta'at segmentation where each slot accepts its
 *    own alternates (را vs راء, etc.). The alignment loop consumes the slots in order.
 */
fun formGroupsFor(displayWord: String): List<Any> {
    val base = normalizeForSpokenForms(displayWord)
    val muq = muqattaatNormalized[base]
    if (muq != null) {
        val out = mutableListOf<Any>()
        for (seg in muq) out.add(seg)
        // Also accept the full concatenation as ONE token (recognizer did not segment).
        val concatCandidates = mutableListOf<String>()
        for (seg in muq) {
            var combos = listOf("")
            for (slot in seg) {
                val next = mutableListOf<String>()
                for (prefix in combos) for (alt in slot) next.add(prefix + alt)
                combos = next
            }
            concatCandidates.addAll(combos)
        }
        if (concatCandidates.isNotEmpty()) out.add(0, concatCandidates)
        return out
    }
    return listOf(singleTokenCandidatesFor(base))
}

/**
 * Pronunciation variants for a *spoken* token from the recognizer. Today we just
 * run the same base normalizer; the TARGET-side tables carry the alternates. Kept
 * as a function so future spoken-side expansion needn't touch the call sites.
 */
fun spokenVariants(spoken: String): List<String> {
    return listOf(normalizeForSpokenForms(spoken))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tolerant muqatta'at matching
//
// The strict ordered-slot matcher above assumes the decoder renders EVERY letter
// name of a disjointed-letter word correctly. On device it does not.
//
// Measured on a Samsung A05s (SM-A057F) reciting Al-Baqarah 2:1 with the shipped
// `model_int8.onnx` CTC graph, three consecutive decode windows emitted:
//
//     window 1:  م
//     window 2:  سف م ذ الكتاب
//     window 3:  أف م ذ الكتاب وَلَا ريب فِيهِ
//
// i.e. `الم` (ألف لام ميم) came back as a truncated `أف` and a bare `م` — the
// `لام` was dropped outright. The 3-slot matcher needs three tokens and gets
// garbage; [isWordMatchPlural] returns false; the alignment walk falls through to
// the OOV-garbage skip, blows its cap of 2 on the fragments, and charges a WRONG
// at word 0 — which returns early and leaves all 6115 remaining words of the
// surah PENDING. Result: `matched=0`, nothing highlighted, no popup, and the
// cursor frozen until the user taps pause.
//
// Al-Fatihah has no muqatta'at, which is precisely why the user saw it
// "work fine" while Al-Baqarah never displayed a single word.
// ─────────────────────────────────────────────────────────────────────────────

/** Small local edit distance (the app's matcher lives in the UI module). */
private fun editDistance(a: String, b: String): Int {
    if (a == b) return 0
    if (a.isEmpty()) return b.length
    if (b.isEmpty()) return a.length
    var v0 = IntArray(b.length + 1) { it }
    var v1 = IntArray(b.length + 1)
    for (i in a.indices) {
        v1[0] = i + 1
        for (j in b.indices) {
            val cost = if (a[i] == b[j]) 0 else 1
            v1[j + 1] = minOf(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        }
        val tmp = v0; v0 = v1; v1 = tmp
    }
    return v0[b.length]
}

/**
 * Ordered letter-name slots for a muqatta'at display word, or null for an
 * ordinary word. Only the canonical (first) segmentation is returned — the
 * relaxed matcher below tolerates the decoder's deviations from it.
 */
fun muqattaatSlots(displayWord: String): List<List<String>>? {
    val base = normalizeForSpokenForms(displayWord)
    return muqattaatNormalized[base]?.firstOrNull()
}

/** True when [token] plausibly renders the letter name spoken for [slot]. */
private fun relaxedSlotMatch(token: String, slot: List<String>): Boolean {
    val t = normalizeForSpokenForms(token)
    if (t.isEmpty()) return false
    for (alt in slot) {
        if (t == alt) return true
        // The CTC decoder clips letter names to their first letter or two
        // (ميم -> م, الف -> اف) far more often than it invents letters, so a
        // shared prefix is strong evidence — provided the token is at least
        // one letter and the alternation is at least two.
        if (alt.length >= 2 && (alt.startsWith(t) || t.startsWith(alt))) return true
        if (alt.length >= 2 && editDistance(t, alt) <= 1) return true
        // Three-letter names (الف, لام, ميم, كاف, عين, قاف, نون, صاد) come back
        // two edits away surprisingly often: one recitation of `ألف` decoded as
        // `أف` in one window and `سف` in the next — the decoder reliably keeps
        // the FINAL consonant and mangles everything before it. Accept a 2-char
        // token that ends in the same letter as a 3-char alternation.
        //
        // Gated on the last character deliberately. A plain "edit distance <= 2"
        // rule was tried first and is far too loose: it let `صدر` claim `صاد`
        // and `ولا` claim `الف`. Matching the final consonant rejects both.
        if (alt.length == 3 && t.length == 2 && t.last() == alt.last()) return true
    }
    return false
}

/**
 * Tolerant match of a short spoken run against a muqatta'at's letter-name slots.
 *
 * [spokenTokens] must start at the transcript frontier. Returns the number of
 * tokens consumed by the match (so the caller can advance the cursor past
 * exactly what was used), or -1 when the run does not match.
 *
 * Tolerates two decoder failures, both observed on device:
 *  - a letter name rendered short or slightly wrong (prefix / edit distance 1);
 *  - a letter name dropped entirely.
 *
 * Bounded so it cannot fire on unrelated speech: the run is at most
 * `slots.size` tokens, at least `slots.size - 1` of the slots must match, and
 * order is preserved.
 */
fun matchMuqattaatRelaxed(spokenTokens: List<String>, slots: List<List<String>>): Int {
    val n = slots.size
    if (n == 0 || spokenTokens.isEmpty()) return -1
    val run = spokenTokens.take(n)
    // Two-letter muqatta'at (حم, طه, يس, ق, ص, ن) are short enough that
    // accepting `n - 1` would let a single stray letter (م, ح, ي) claim the
    // word; require every slot there. Three-plus-letter tokens may lose ONE
    // letter name to the decoder (measured: `الم` -> `أف م`, `لام` dropped).
    val need = if (n <= 2) n else n - 1

    // best[i][j] = most slots matched by slots[i..] against run[j..]; tie-break
    // on the fewest tokens consumed so the cursor advances as little as possible.
    data class Hit(val matched: Int, val end: Int)
    fun better(a: Hit, b: Hit): Hit =
        if (b.matched > a.matched || (b.matched == a.matched && b.end < a.end)) b else a

    val memo = Array(n + 1) { arrayOfNulls<Hit>(run.size + 1) }
    fun solve(i: Int, j: Int): Hit {
        if (i == n) return Hit(0, j)
        memo[i][j]?.let { return it }
        var best = Hit(0, j)
        if (j < run.size) {
            best = better(best, solve(i, j + 1))          // stray token: skip it
        }
        best = better(best, solve(i + 1, j))              // letter dropped by the decoder
        if (j < run.size && relaxedSlotMatch(run[j], slots[i])) {
            val rest = solve(i + 1, j + 1)
            best = better(best, Hit(rest.matched + 1, rest.end))
        }
        memo[i][j] = best
        return best
    }

    val hit = solve(0, 0)
    if (hit.matched < need) return -1
    // `end` is an index into the run; tokens beyond it were strays the decoder
    // inserted and must stay available for the NEXT target word.
    return hit.end.coerceAtLeast(1)
}