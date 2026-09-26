package com.example.ui

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * §4.47 — THE END-OF-SESSION FORGIVENESS MUST NOT PAINT THE WHOLE RANGE.
 *
 * The grace-expiry branch promotes PENDING rows so the words the decoder clipped
 * at the end of a recitation stay visible (§4.34 / §4.40). It promoted EVERY
 * PENDING row, so a session that ended early — an auto-stop on the recording's own
 * silence, or a Restart the user had just tapped — decided and DISPLAYED the whole
 * remaining range in one step. On screen a fully recited range is
 * indistinguishable from words nobody read, which is the report: "it froze, and
 * then all the words appeared at once until the end of the ayah range".
 *
 * The bound is the same shape `finalizeCorrect()` uses: the last word that carries
 * spoken evidence, plus [FINALIZE_FORGIVE_TAIL_WINDOW] for the clipped tail.
 */
class ForgiveTailWindowTest {

    private fun row(idx: Int, spoken: String? = null) = WordAnalysis(
        wordIndex = idx,
        text = "w$idx",
        normalizedText = "w$idx",
        status = if (spoken == null) RecitationStatus.PENDING else RecitationStatus.CORRECT,
        spokenText = spoken,
    )

    /** Mirrors the grace-expiry promotion: heard frontier + a fixed tail window. */
    private fun promotable(
        rows: List<WordAnalysis>,
        tailWindow: Int = FINALIZE_FORGIVE_TAIL_WINDOW,
    ): Set<Int> {
        val heardFrontier = rows.filter { !it.spokenText.isNullOrEmpty() }
            .maxOfOrNull { it.wordIndex + 1 } ?: 0
        // No heard evidence at all => no frontier => no forgiveness. Without this a
        // session in which nothing was recognised still displayed the first eight
        // words, which is precisely the "words appear that were never read" bug.
        if (heardFrontier <= 0) return emptySet()
        val limit = heardFrontier + tailWindow
        return rows.filter { it.status == RecitationStatus.PENDING && it.wordIndex < limit }
            .map { it.wordIndex }
            .toSet()
    }

    @Test
    fun `the clipped tail of a finished recitation is promoted`() {
        // 28 of 29 words heard; the last one never rendered. It is within the
        // window, so it is promoted and stays visible (§4.40).
        val rows = (0 until 28).map { row(it, "w$it") } + listOf(row(28))
        assertEquals(setOf(28), promotable(rows))
    }

    @Test
    fun `the rest of the range is NOT promoted when the session stopped early`() {
        // The s3 / b2 shape: 16 words heard, the remaining 13 never reached.
        val rows = (0 until 16).map { row(it, "w$it") } + (16 until 29).map { row(it) }
        val promoted = promotable(rows)
        // Only the clipped tail just past the frontier, and nothing near the end.
        assertEquals((16 until 24).toSet(), promoted)
        assertEquals(false, promoted.any { it >= 24 })
    }

    @Test
    fun `an empty recitation promotes nothing at all`() {
        val rows = (0 until 29).map { row(it) }
        assertEquals(emptySet<Int>(), promotable(rows))
    }

    @Test
    fun `a session with no spoken evidence stops exactly where it is`() {
        val rows = (0 until 5).map { row(it, "w$it") } + (5 until 29).map { row(it) }
        val promoted = promotable(rows)
        assertEquals(false, promoted.any { it > 12 })
    }
}
