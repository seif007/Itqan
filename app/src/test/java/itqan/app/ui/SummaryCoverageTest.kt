package itqan.app.ui

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * §4.31 — the session summary must not claim perfection for words it never heard.
 *
 * The bug: `perfect` was decided from `entries.isEmpty()` alone — "no mistake
 * was CONFIRMED". When the walk stalls, `finalizeCorrect()` forges the trailing
 * PENDING rows to CORRECT with `spokenText = null` (no spoken evidence), so a
 * run that aligned only 27 of 29 words still showed "Perfect recitation! 🌟".
 * Measured `fix31_budget_2`: `pendingRemain=false`, `wrong=0 skipped=0`, yet two
 * words were never heard.
 *
 * `unheardWordCount()` is the coverage signal the dialog now gates on: a word
 * is unheard when it is still PENDING, or when it is CORRECT with no
 * `spokenText` (the recognizer bound nothing to it).
 */
class SummaryCoverageTest {

    private fun row(
        idx: Int,
        status: RecitationStatus,
        spoken: String? = null,
        rule: String? = null,
    ) = WordAnalysis(
        wordIndex = idx,
        text = "w$idx",
        normalizedText = "w$idx",
        status = status,
        spokenText = spoken,
        tajweedRule = rule,
    )

    @Test
    fun `a fully heard recitation has zero unheard words`() {
        val rows = listOf(
            row(0, RecitationStatus.CORRECT, spoken = "بسم"),
            row(1, RecitationStatus.CORRECT, spoken = "الله"),
            row(2, RecitationStatus.CORRECT, spoken = "الرحمن"),
        )
        assertEquals(0, rows.unheardWordCount())
    }

    @Test
    fun `PENDING words count as unheard`() {
        val rows = listOf(
            row(0, RecitationStatus.CORRECT, spoken = "بسم"),
            row(1, RecitationStatus.PENDING),
            row(2, RecitationStatus.PENDING),
        )
        assertEquals(2, rows.unheardWordCount())
    }

    @Test
    fun `a CORRECT word with no spokenText is unheard - it was rubber-stamped`() {
        // This is exactly what finalizeCorrect() produces: status CORRECT,
        // spokenText null. It must NOT be treated as verified.
        val rows = listOf(
            row(0, RecitationStatus.CORRECT, spoken = "بسم"),
            row(1, RecitationStatus.CORRECT, spoken = null),
        )
        assertEquals(1, rows.unheardWordCount())
    }

    @Test
    fun `the measured fix31_budget_2 shape reports the two unheard tail words`() {
        // 29 target words; 27 genuinely heard; the last two forged CORRECT by
        // finalizeCorrect() with no spoken evidence. The old gate said
        // "perfect"; this must report 2.
        val rows = (0 until 27).map { row(it, RecitationStatus.CORRECT, spoken = "w$it") } +
            listOf(
                row(27, RecitationStatus.CORRECT, spoken = null),
                row(28, RecitationStatus.CORRECT, spoken = null),
            )
        assertEquals(2, rows.unheardWordCount())
        assertEquals(29, rows.distinctWordCount())
    }

    @Test
    fun `a word carrying two rule rows is counted once, not twice`() {
        // detailedAnalysis is ONE ROW PER (word, rule): word 5 has both a
        // PENDING row and a CORRECT row. It is still ONE unheard word.
        val rows = listOf(
            row(0, RecitationStatus.CORRECT, spoken = "بسم"),
            row(5, RecitationStatus.PENDING, rule = "Madd"),
            row(5, RecitationStatus.CORRECT, spoken = null, rule = "Ghunnah"),
        )
        assertEquals(1, rows.unheardWordCount())
        assertEquals(2, rows.distinctWordCount())
    }

    @Test
    fun `a real mistake is not counted as unheard`() {
        // WRONG / SKIPPED are their OWN signal (they appear in the summary as
        // mistakes). They must not also inflate the coverage warning, or a
        // genuine error would read as "we did not hear you".
        val rows = listOf(
            row(0, RecitationStatus.CORRECT, spoken = "بسم"),
            row(1, RecitationStatus.WRONG, spoken = "غلط"),
            row(2, RecitationStatus.SKIPPED),
        )
        assertEquals(0, rows.unheardWordCount())
    }

    @Test
    fun `an empty analysis reports nothing unheard`() {
        assertEquals(0, emptyList<WordAnalysis>().unheardWordCount())
        assertEquals(0, emptyList<WordAnalysis>().distinctWordCount())
    }
}
