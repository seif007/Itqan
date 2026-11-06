package itqan.app.ui

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Pins the STOPPED-SESSION ayah ceiling.
 *
 * The report: "for tajweed, after i stop and pause the mic, there are always
 * other words displayed in the next ayah". The flat target is the whole
 * selected range and the analysis carries a PENDING tail for every later
 * ayah; the finalize/promotion windows used global word indices, so a clipped
 * row at the start of the next ayah could be decided and the cursor then
 * coloured up to it.
 *
 * These tests are pure: no Android/Compose state, just the boundary helpers.
 */
class NextAyahDisplayCapTest {

    private fun ayah(
        id: Int,
        start: Int,
        count: Int,
    ) = MainViewModel.AyahDisplay(
        ayahId = id,
        text = (0 until count).joinToString(" ") { "w${start + it}" },
        words = (0 until count).map { "w${start + it}" },
        normalizedWords = (0 until count).map { "w${start + it}" },
        wordStartIndex = start,
        audio = null,
    )

    private fun row(
        index: Int,
        status: RecitationStatus,
        spoken: String? = null,
        rule: String? = null,
        score: Int? = null,
    ) = WordAnalysis(
        wordIndex = index,
        text = "w$index",
        normalizedText = "w$index",
        status = status,
        spokenText = spoken,
        tajweedRule = rule,
        tajweedScore = score,
    )

    @Test
    fun `stopping at the end of an ayah cannot colour the next ayah`() {
        val ayahs = listOf(ayah(id = 1, start = 0, count = 4), ayah(id = 2, start = 4, count = 3))
        val analysis = listOf(
            row(0, RecitationStatus.CORRECT, "w0"),
            row(1, RecitationStatus.CORRECT, "w1"),
            row(2, RecitationStatus.CORRECT, "w2"),
            row(3, RecitationStatus.CORRECT, "w3"),
            // The whole next ayah is present as PENDING bookkeeping.
            row(4, RecitationStatus.PENDING),
            row(5, RecitationStatus.PENDING),
            row(6, RecitationStatus.PENDING),
        )

        assertEquals(4, displayCapForHeardAyah(analysis, ayahs))
        assertEquals(4, ayahEndForWordIndex(3, ayahs))
        assertEquals(7, ayahEndForWordIndex(6, ayahs))
    }

    @Test
    fun `stopping mid-ayah allows the clipped tail inside that same ayah`() {
        val ayahs = listOf(ayah(id = 1, start = 0, count = 4), ayah(id = 2, start = 4, count = 15))
        val analysis = listOf(
            row(0, RecitationStatus.CORRECT, "w0"),
            row(1, RecitationStatus.CORRECT, "w1"),
            row(2, RecitationStatus.CORRECT, "w2"),
            row(3, RecitationStatus.CORRECT, "w3"),
            row(4, RecitationStatus.CORRECT, "w4"),
            // 5..18 were never heard; index 13 is frontier 5 + the 8-word
            // clipped-tail allowance, still inside ayah 2 (which ends at 19).
            row(5, RecitationStatus.PENDING),
        )

        assertEquals(13, displayCapForHeardAyah(analysis, ayahs))
    }

    @Test
    fun `no heard evidence imposes no ceiling`() {
        val ayahs = listOf(ayah(id = 1, start = 0, count = 4))
        val analysis = listOf(row(0, RecitationStatus.PENDING))
        assertEquals(Int.MAX_VALUE, displayCapForHeardAyah(analysis, ayahs))
    }

    @Test
    fun `a real spoken word at the start of the next ayah is evidence, not a bookkeeping row`() {
        val ayahs = listOf(ayah(id = 1, start = 0, count = 4), ayah(id = 2, start = 4, count = 3))
        val analysis = listOf(
            row(0, RecitationStatus.CORRECT, "w0"),
            row(1, RecitationStatus.CORRECT, "w1"),
            row(2, RecitationStatus.CORRECT, "w2"),
            row(3, RecitationStatus.CORRECT, "w3"),
            row(4, RecitationStatus.CORRECT, "w4"),
            row(5, RecitationStatus.PENDING),
            row(6, RecitationStatus.PENDING),
        )
        // The user really began ayah 2; its end is now a legitimate ceiling.
        assertEquals(7, displayCapForHeardAyah(analysis, ayahs))
    }

    @Test
    fun `an out-of-range index falls back to the end of the selected range`() {
        val ayahs = listOf(ayah(id = 1, start = 0, count = 4), ayah(id = 2, start = 4, count = 3))
        assertEquals(7, ayahEndForWordIndex(99, ayahs))
    }

    @Test
    fun `a paused cursor on an ayah boundary anchors the screen to the ayah just read`() {
        val ayahs = listOf(ayah(id = 1, start = 0, count = 4), ayah(id = 2, start = 4, count = 3))
        // Live: the next word is the next ayah (correct).
        assertEquals(4, displayAyahAnchor(4, paused = false, mistakeArmed = false, ayahs = ayahs))
        // Paused at the exact boundary: stay on the completed ayah.
        assertEquals(3, displayAyahAnchor(4, paused = true, mistakeArmed = false, ayahs = ayahs))
        // Paused mid-ayah: do not move anywhere.
        assertEquals(5, displayAyahAnchor(5, paused = true, mistakeArmed = false, ayahs = ayahs))
        // An armed mistake must stay on the flagged word even at a boundary.
        assertEquals(4, displayAyahAnchor(4, paused = true, mistakeArmed = true, ayahs = ayahs))
        // A word of the next ayah was really heard: do not pretend it was not.
        assertEquals(
            4,
            displayAyahAnchor(
                4,
                paused = true,
                mistakeArmed = false,
                ayahs = ayahs,
                hasHeardAtCursor = true,
            ),
        )
        // End of the range (exclusive) anchors to the last word.
        assertEquals(6, displayAyahAnchor(7, paused = true, mistakeArmed = false, ayahs = ayahs))
    }
}
