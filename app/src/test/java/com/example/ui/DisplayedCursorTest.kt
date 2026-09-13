package com.example.ui

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * §4.40 — REACHED ⇒ DISPLAYED, ENFORCED AT THE FINALIZE.
 *
 * The transcript colours a word by the CURSOR (`globalWordIndex <
 * matchedWordCount`, i.e. `alignment.first`) and drops it to the 25 %-opacity
 * grey otherwise. It does NOT colour by the row's `status`.
 *
 * `finalizeCorrect()` and `promoteUnresolvedPendingRows()` set
 * `status = CORRECT` while deliberately leaving `spokenText == null` — and the
 * cursor's counter does not count a CORRECT row with a null token, because a
 * rubber-stamped row is not evidence the user spoke the word. So the promotion
 * makes a row DECIDED but not VISIBLE.
 *
 * Measured `p4_fatihah` (Al-Fatihah 1–7, on-device):
 *
 *   23:15:03.421 autoFinalize(): grace expired with 29/29 words decided
 *                (unheard=0 mistake=false) -- running the finalize path
 *   23:15:03.432 endOfRange(): auto stop but NOT genuinely finished
 *                (unheard=20/29) -> no summary; words stay displayed
 *                cursor=4/29
 *
 * Every word was decided; four were rendered. The session ended with NO mistake
 * and NO popup, so the UX contract ("if it stoppd then it should be a
 * skipped/wrong word popup or tajweed popup else it should be displayed")
 * requires the whole range on screen.
 *
 * `displayedCursorFor` makes the cursor COVER every decided word. It is
 * one-directional (advance only) and the callers gate it on `sessionFinalized`,
 * so it never runs during a live walk.
 */
class DisplayedCursorTest {

    private fun row(
        idx: Int,
        status: RecitationStatus,
        spoken: String? = null,
    ) = WordAnalysis(
        wordIndex = idx,
        text = "w$idx",
        normalizedText = "w$idx",
        status = status,
        spokenText = spoken,
    )

    // ─────────────────────────────────────────────────────────────────────
    // The measured failure
    // ─────────────────────────────────────────────────────────────────────

    /**
     * The `p4_fatihah` end state: 29 rows all decided, most of them the
     * finalize's rubber stamp (CORRECT with no spoken fragment), and the cursor
     * stuck at 4.
     */
    @Test
    fun `the p4 finalized list advances the cursor from 4 to all 29`() {
        val rows = (0 until 29).map { idx ->
            if (idx < 4) row(idx, RecitationStatus.CORRECT, spoken = "w$idx")
            else row(idx, RecitationStatus.CORRECT)   // rubber-stamped, no token
        }
        assertEquals(29, displayedCursorFor(rows, 4))
    }

    /**
     * The whole point: the rubber-stamped rows are NOT counted by the cursor's
     * own counter, so without this helper the cursor stays where the walk left
     * it and 25 words stay grey.
     */
    @Test
    fun `a rubber-stamped tail is invisible to the walk but covered by the helper`() {
        val rows = listOf(
            row(0, RecitationStatus.CORRECT, spoken = "w0"),
            row(1, RecitationStatus.CORRECT, spoken = "w1"),
            row(2, RecitationStatus.CORRECT),
            row(3, RecitationStatus.CORRECT),
        )
        // What the walk counts (CORRECT *with* a token):
        val walkCount = rows.count { it.status == RecitationStatus.CORRECT && !it.spokenText.isNullOrEmpty() }
        assertEquals(2, walkCount)
        // What the user must SEE:
        assertEquals(4, displayedCursorFor(rows, walkCount))
    }

    // ─────────────────────────────────────────────────────────────────────
    // One-directional
    // ─────────────────────────────────────────────────────────────────────

    /** It can only advance: a cursor already past the decided words is kept. */
    @Test
    fun `a cursor already covering the decided words is unchanged`() {
        val rows = (0 until 5).map { row(it, RecitationStatus.CORRECT) }
        assertEquals(29, displayedCursorFor(rows, 29))
        assertEquals(5, displayedCursorFor(rows, 5))
    }

    /** It never retreats, even when the decided prefix is short. */
    @Test
    fun `the helper never lowers the cursor`() {
        val rows = listOf(row(0, RecitationStatus.CORRECT), row(1, RecitationStatus.PENDING))
        assertEquals(17, displayedCursorFor(rows, 17))
    }

    // ─────────────────────────────────────────────────────────────────────
    // What must NOT be covered
    // ─────────────────────────────────────────────────────────────────────

    /** A list of nothing but PENDING rows leaves the cursor exactly where it is. */
    @Test
    fun `an all-PENDING list does not move the cursor`() {
        val rows = (0 until 29).map { row(it, RecitationStatus.PENDING) }
        assertEquals(0, displayedCursorFor(rows, 0))
        assertEquals(11, displayedCursorFor(rows, 11))
    }

    /**
     * Only the DECIDED prefix is covered — the helper does not assume the whole
     * range. A decided run followed by PENDING rows stops at the last decided
     * word (the finalize's promotion is what extends it further).
     */
    @Test
    fun `the cursor stops at the last decided word, not the end of the range`() {
        val rows = (0 until 29).map { idx ->
            if (idx <= 9) row(idx, RecitationStatus.CORRECT, spoken = "w$idx")
            else row(idx, RecitationStatus.PENDING)
        }
        assertEquals(10, displayedCursorFor(rows, 3))
    }

    /** An empty analysis cannot move the cursor. */
    @Test
    fun `an empty analysis leaves the cursor unchanged`() {
        assertEquals(7, displayedCursorFor(emptyList(), 7))
    }

    // ─────────────────────────────────────────────────────────────────────
    // A real mistake stays visible
    // ─────────────────────────────────────────────────────────────────────

    /**
     * A genuine WRONG row is itself DECIDED, so it falls INSIDE the covered
     * prefix and keeps its error colour. The helper cannot hide the very
     * mistake the app stopped for.
     */
    @Test
    fun `a genuine mistake row stays inside the covered prefix`() {
        val rows = (0 until 29).map { idx ->
            when (idx) {
                20 -> row(idx, RecitationStatus.WRONG, spoken = "w20")
                in 0..19 -> row(idx, RecitationStatus.CORRECT, spoken = "w$idx")
                else -> row(idx, RecitationStatus.PENDING)
            }
        }
        val cursor = displayedCursorFor(rows, 15)
        assertEquals(21, cursor)
        // The renderer's error branch runs for `globalWordIndex >= cursor`, so
        // word 20 must be BELOW the cursor for its error colour to be reachable.
        assertEquals(true, 20 < cursor)
    }

    /**
     * A trailing SKIPPED (the last word of the range was never heard) is also
     * decided, so it is covered and stays renderable.
     */
    @Test
    fun `a trailing skipped row is covered`() {
        val rows = (0 until 29).map { idx ->
            if (idx == 28) row(idx, RecitationStatus.SKIPPED)
            else row(idx, RecitationStatus.CORRECT, spoken = "w$idx")
        }
        assertEquals(29, displayedCursorFor(rows, 28))
    }
}
