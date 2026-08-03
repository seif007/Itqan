package com.example.quran

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
        if (c in '\u064B'..'\u065F' || c == '\u0670' || c in '\u06D6'..'\u06ED' || c == '\uFEFF' || c == '۬' || c == 'ے' || c == 'ـ') continue
        if (".,;:!?()[]{}'\"/\\-_".contains(c)) continue
        if (c == '،' || c == '؛' || c == '،' || c == '؟') continue
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