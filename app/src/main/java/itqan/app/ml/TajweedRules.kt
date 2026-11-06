package itqan.app.ml

/**
 * Deterministic Tajweed-rule derivation from Quranic Arabic text.
 * Ported from practice_recitation-main/tools/calib/tajweed_rules.py.
 * Rule names match the trained TajweedNet head order (TajweedFeatures.RULES).
 *
 * Arabic letters are referenced via integer code points (Char(cp)) to keep the
 * source ASCII-safe.
 */
object TajweedRules {

    // --- code points ---
    private val CP = object {
        val ALIF = 0x0627
        val ALIF_WASL = 0x0671
        val ALIF_MADDA = 0x0622
        val ALIF_MAQ = 0x0649
        val LAM = 0x0644
        val YAA = 0x064A
        val WAW = 0x0648
        val NOON = 0x0646
        val MEEM = 0x0645
        val HAMZA = 0x0621
        val ALEF_HAMZA_ABOVE = 0x0623
        val ALEF_HAMZA_BELOW = 0x0625
        val MADDA_ABOVE = 0x0670
        val TATWEEL = 0x0640
        val FATHA = 0x064E
        val DAMMA = 0x064F
        val KASRA = 0x0650
        val SUKOON = 0x0652
        val SHADDA = 0x0651
        val TANWEEN_FATH = 0x064B
        val TANWEEN_DAMM = 0x064C
        val TANWEEN_KASR = 0x064D
        val BEH = 0x0628
    }

    private fun ch(cp: Int) = cp.toChar()

    private val ALIF = ch(CP.ALIF)
    private val ALIF_WASL = ch(CP.ALIF_WASL)
    private val ALIF_MADDA = ch(CP.ALIF_MADDA)
    private val ALIF_MAQ = ch(CP.ALIF_MAQ)
    private val LAM = ch(CP.LAM)
    private val YAA = ch(CP.YAA)
    private val WAW = ch(CP.WAW)
    private val NOON = ch(CP.NOON)
    private val MEEM = ch(CP.MEEM)
    private val HAMZA = ch(CP.HAMZA)
    private val MADDA_ABOVE = ch(CP.MADDA_ABOVE)

    private val TANWEEN = setOf(ch(CP.TANWEEN_FATH), ch(CP.TANWEEN_DAMM), ch(CP.TANWEEN_KASR))
    private val DIACRITICS = setOf(
        ch(CP.TANWEEN_FATH), ch(CP.TANWEEN_DAMM), ch(CP.TANWEEN_KASR),
        ch(CP.FATHA), ch(CP.DAMMA), ch(CP.KASRA), ch(CP.SUKOON), ch(CP.SHADDA),
        ch(CP.MADDA_ABOVE), ch(CP.TATWEEL)
    )

    private val QALQALAH = setOf(ch(0x0642), ch(0x0637), ch(0x0628), ch(0x062C), ch(0x062F))
    private val THROAT = setOf(ch(0x0621), ch(0x0647), ch(0x0639), ch(0x062D), ch(0x063A), ch(0x062E))
    private val IKHFA = setOf(
        ch(0x062A), ch(0x062B), ch(0x062C), ch(0x062F), ch(0x0630), ch(0x0632),
        ch(0x0633), ch(0x0634), ch(0x0635), ch(0x0636), ch(0x0637), ch(0x0638),
        ch(0x0641), ch(0x0642), ch(0x0643), ch(0x0644)
    )
    private val IDGHAM_GHUNNAH = setOf(ch(0x064A), ch(0x0646), ch(0x0645), ch(0x0648))
    private val IDGHAM_PLAIN = setOf(ch(0x0644), ch(0x0631))
    private val BEH = ch(CP.BEH)

    private fun stripDiacritics(s: String): String = s.filter { it !in DIACRITICS }

    private fun endsWithTanween(word: String): Boolean = word.isNotEmpty() && word.last() in TANWEEN

    private fun firstBaseLetter(nextWord: String?): Char? {
        if (nextWord.isNullOrEmpty()) return null
        val base = stripDiacritics(nextWord)
        for (c in base) {
            if (c in setOf(ALIF, ALIF_WASL, ch(CP.ALEF_HAMZA_ABOVE), ch(CP.ALEF_HAMZA_BELOW), ALIF_MADDA, MADDA_ABOVE))
                continue
            return c
        }
        return if (base.isEmpty()) null else base[0]
    }

    private fun hasSukoonAfter(word: String, idx: Int): Boolean =
        idx + 1 < word.length && (word[idx + 1] == ch(CP.SUKOON) || word[idx + 1] == ch(CP.SHADDA))

    fun detectRulesForWord(word: String, nextWord: String? = null): List<String> {
        if (word.isEmpty()) return listOf("NONE")
        val rules = LinkedHashSet<String>()

        for (i in word.indices) {
            if (word[i] in QALQALAH && hasSukoonAfter(word, i)) {
                rules.add("QALQALAH")
                break
            }
        }

        if ("${ch(CP.FATHA)}$ALIF" in word || "${ch(CP.FATHA)}$ALIF_WASL" in word || "${ch(CP.FATHA)}$ALIF_MAQ" in word
            || "${ch(CP.KASRA)}$YAA" in word
            || "${ch(CP.DAMMA)}$WAW" in word
            || "$HAMZA$ALIF" in word || "$HAMZA$WAW" in word || "$HAMZA$YAA" in word
            || ALIF_MADDA in word
            || "${ch(CP.TANWEEN_FATH)}$ALIF" in word || "${ch(CP.TANWEEN_FATH)}$ALIF_MAQ" in word
            || "${ch(CP.TANWEEN_KASR)}$YAA" in word
            || "${ch(CP.TANWEEN_DAMM)}$WAW" in word
        ) {
            rules.add("MADD")
        }

        if (word.any { it in TANWEEN }) rules.add("GHUNNAH")
        for (chp in listOf(NOON, MEEM)) {
            val idx = word.lastIndexOf(chp)
            if (idx >= 0 && hasSukoonAfter(word, idx) && ch(CP.SHADDA) in word) rules.add("GHUNNAH")
        }

        if (word.startsWith("$ALIF$LAM") || word.startsWith("$ALIF_WASL$LAM")) rules.add("ALLAM")

        val nfl = firstBaseLetter(nextWord)

        val noonIdx = word.lastIndexOf(NOON)
        val noonSaakin = (noonIdx >= 0 && hasSukoonAfter(word, noonIdx)) || endsWithTanween(word)
        if (noonSaakin && nfl != null) {
            when {
                nfl in IDGHAM_GHUNNAH -> { rules.add("IDGHAM"); rules.add("GHUNNAH") }
                nfl in IDGHAM_PLAIN -> rules.add("IDGHAM")
                nfl in THROAT -> rules.add("IZHAR")
                nfl in IKHFA -> { rules.add("IKHFA"); rules.add("GHUNNAH") }
                else -> rules.add("IZHAR")
            }
        }

        val meemIdx = word.lastIndexOf(MEEM)
        val meemSaakin = meemIdx >= 0 && hasSukoonAfter(word, meemIdx)
        if (meemSaakin && nfl != null) {
            when (nfl) {
                MEEM -> { rules.add("MEEM_SAKINAH"); rules.add("GHUNNAH") }
                BEH -> { rules.add("MEEM_SAKINAH"); rules.add("GHUNNAH") }
                else -> rules.add("MEEM_SAKINAH")
            }
        }

        return if (rules.isEmpty()) listOf("NONE") else rules.toList()
    }

    fun splitWords(text: String): List<String> =
        text.split(Regex("\\s+")).map { it.trim() }.filter { it.isNotEmpty() }

    fun detectForPhrase(phrase: String): List<Pair<String, List<String>>> {
        val words = splitWords(phrase)
        return words.mapIndexed { i, w ->
            val next = if (i + 1 < words.size) words[i + 1] else null
            w to detectRulesForWord(w, next)
        }
    }
}
