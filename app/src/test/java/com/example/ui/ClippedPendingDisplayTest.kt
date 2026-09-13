package com.example.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.34 — a clipped decode is a word the user SAID, so it must be DISPLAYED.
 *
 * `clippedWord()` fires when the CTC decoder returns only the opening letters of
 * the target: the token is a strict PREFIX of the target, at least one char long
 * and at least two chars shorter than it. The analyzer then CONSUMES the word —
 * `matchedWordCount` counts it as reached, and `endOfRange()`'s `pendingRemain`
 * treats it as forgiven. But the row it emits is PENDING, and the transcript
 * colours by status, so the word stayed at the 25 %-opacity fallback.
 *
 * Measured `clean_fatihah_2` (Al-Fatihah 1–7):
 *   analyze(fast): done … matched=29 hasError=false targetWords=29
 *   autoFinalize?(): rows=31 decided=29/29 unheardPending=1   <- nothing to hold on
 *   endOfRange(): auto stop but NOT genuinely finished (unheard=1/29) -> no summary
 * and the on-device report was "last word not displayed and there is no popup".
 * The checker's own verdict was `FAIL — only 28 of 29 target words matched`.
 *
 * `promoteClippedPending()` is the fix: PENDING **with** a spoken fragment
 * becomes CORRECT (visible, still counted as heard); PENDING **without** one is
 * left alone (a word never heard must stay invisible until something decides it).
 */
class ClippedPendingDisplayTest {

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

    // ── the promotion rule ────────────────────────────────────────────────

    @Test
    fun `a clipped PENDING row with a spoken fragment is promoted to CORRECT`() {
        // Exactly what clippedWord() emits: PENDING + the prefix it heard.
        val clipped = row(28, RecitationStatus.PENDING, spoken = "ٱلضَّآ")
        val promoted = clipped.promoteClippedPending()
        assertEquals(RecitationStatus.CORRECT, promoted.status)
    }

    @Test
    fun `the spoken fragment is KEPT, not nulled`() {
        // Load-bearing: unheardWordCount() skips rows carrying a token, so
        // keeping it is what lets a genuine finish still raise its summary.
        val clipped = row(28, RecitationStatus.PENDING, spoken = "ٱلضَّآ")
        val promoted = clipped.promoteClippedPending()
        assertEquals("ٱلضَّآ", promoted.spokenText)
        assertEquals(0, listOf(promoted).unheardWordCount())
    }

    @Test
    fun `a PENDING row with NO token is left untouched`() {
        // A word never heard must stay invisible: promoting it would colour a
        // word the user never said.
        val neverHeard = row(12, RecitationStatus.PENDING)
        val promoted = neverHeard.promoteClippedPending()
        assertEquals(RecitationStatus.PENDING, promoted.status)
        assertNull(promoted.spokenText)
        assertSame(neverHeard, promoted)
    }

    @Test
    fun `a blank spokenText is NOT a fragment`() {
        // "" is what a half-built row can carry; it is not evidence.
        val blank = row(3, RecitationStatus.PENDING, spoken = "")
        assertEquals(RecitationStatus.PENDING, blank.promoteClippedPending().status)
    }

    @Test
    fun `already decided rows are returned unchanged`() {
        val correct = row(0, RecitationStatus.CORRECT, spoken = "بسم")
        val wrong = row(1, RecitationStatus.WRONG, spoken = "غلط")
        val skipped = row(2, RecitationStatus.SKIPPED)
        val tajweed = row(3, RecitationStatus.TAJWEED_ERROR, spoken = "علي")
        val hinted = row(4, RecitationStatus.HINTED)
        for (r in listOf(correct, wrong, skipped, tajweed, hinted)) {
            assertSame(r, r.promoteClippedPending())
        }
    }

    // ── the measured clean_fatihah_2 shape ───────────────────────────────

    @Test
    fun `the measured clean_fatihah_2 shape - last word clipped, no popup, no summary`() {
        // 29 target words; 28 genuinely heard; word 28 is the clipped tail.
        val rows = (0 until 28).map {
            row(it, RecitationStatus.CORRECT, spoken = "w$it")
        } + row(28, RecitationStatus.PENDING, spoken = "ٱلضَّآ")

        // BEFORE: the row is PENDING, so the old gate read `unheardPending=0`
        // (it only counted PENDING with a BLANK token) and declared the range
        // complete while the word sat at the 25 %-opacity fallback.
        assertEquals(1, rows.count { it.status == RecitationStatus.PENDING })

        // AFTER: every word is CORRECT and carries a token.
        val fixed = rows.promoteClippedPendingRows()
        assertTrue(fixed.none { it.status == RecitationStatus.PENDING })
        assertEquals(0, fixed.unheardWordCount())
        assertEquals(29, fixed.distinctWordCount())
    }

    @Test
    fun `a clipped word does not mask a genuinely unheard one`() {
        // Word 28 clipped (visible), word 20 never heard (must stay invisible
        // AND must keep blocking a "perfect" summary).
        val rows = (0 until 29).map {
            row(it, RecitationStatus.CORRECT, spoken = "w$it")
        }.toMutableList()
        rows[28] = row(28, RecitationStatus.PENDING, spoken = "ٱلضَّآ")
        rows[20] = row(20, RecitationStatus.PENDING)

        val fixed = rows.promoteClippedPendingRows()
        assertEquals(RecitationStatus.CORRECT, fixed.first { it.wordIndex == 28 }.status)
        assertEquals(RecitationStatus.PENDING, fixed.first { it.wordIndex == 20 }.status)
        assertEquals(1, fixed.unheardWordCount())
    }

    @Test
    fun `promotion is idempotent`() {
        val rows = listOf(
            row(0, RecitationStatus.CORRECT, spoken = "w0"),
            row(1, RecitationStatus.PENDING, spoken = "w1"),
            row(2, RecitationStatus.PENDING),
        )
        val once = rows.promoteClippedPendingRows()
        val twice = once.promoteClippedPendingRows()
        assertEquals(once, twice)
    }

    @Test
    fun `a word with two rule rows is promoted consistently`() {
        // detailedAnalysis is ONE ROW PER (word, rule); both rows of a clipped
        // word must end up CORRECT or the transcript would colour them apart.
        val rows = listOf(
            row(7, RecitationStatus.PENDING, spoken = "ٱلْعَٰ", rule = "Madd"),
            row(7, RecitationStatus.PENDING, spoken = "ٱلْعَٰ", rule = "Ghunnah"),
        )
        val fixed = rows.promoteClippedPendingRows()
        assertTrue(fixed.all { it.status == RecitationStatus.CORRECT })
        assertEquals(1, fixed.distinctWordCount())
        assertEquals(0, fixed.unheardWordCount())
    }

    // ── promoteUnresolvedPendingRows (the grace-expiry path) ──────────────

    @Test
    fun `the grace-expiry promotion touches TOKEN-LESS rows too`() {
        // THE INFINITE-LOOP REGRESSION. `promoteClippedPendingRows()` is a NO-OP
        // on a token-less PENDING row, and those are exactly the rows the
        // grace-expiry branch exists for. Measured p1_fatihah:
        //   grace expired with 7 unresolved word(s) [22..28] and NO evidenced
        //   mistake -> promoted ... (0 of them carry a spoken fragment)
        // repeated every 2 s for minutes, the session never finalizing.
        val rows = listOf(
            row(21, RecitationStatus.CORRECT, spoken = null),
            row(22, RecitationStatus.PENDING),
            row(23, RecitationStatus.PENDING),
        )
        // The clipped-only helper leaves them PENDING — that is the bug.
        assertTrue(rows.promoteClippedPendingRows().any { it.status == RecitationStatus.PENDING })
        // The unresolved helper makes them VISIBLE.
        val fixed = rows.promoteUnresolvedPendingRows()
        assertTrue(fixed.none { it.status == RecitationStatus.PENDING })
    }

    @Test
    fun `a token-less promoted row is visible but still counts as unheard`() {
        // Visible (CORRECT colours it) yet honest: no spoken evidence means it
        // must keep blocking a "perfect" summary.
        val rows = listOf(row(22, RecitationStatus.PENDING))
        val fixed = rows.promoteUnresolvedPendingRows()
        assertEquals(RecitationStatus.CORRECT, fixed[0].status)
        assertNull(fixed[0].spokenText)
        assertEquals(1, fixed.unheardWordCount())
    }

    @Test
    fun `a fragment-carrying promoted row is visible AND still counted as heard`() {
        val rows = listOf(row(22, RecitationStatus.PENDING, spoken = "أَنْعَ"))
        val fixed = rows.promoteUnresolvedPendingRows()
        assertEquals(RecitationStatus.CORRECT, fixed[0].status)
        assertEquals("أَنْعَ", fixed[0].spokenText)
        assertEquals(0, fixed.unheardWordCount())
    }

    @Test
    fun `the measured p1_fatihah grace-expiry shape becomes fully visible`() {
        // The real log line: `undecided rows = w21/ٱلَّذِينَ:CORRECT:spoken='-' |
        // w22/أَنْعَمْتَ:PENDING:spoken='-' | … | w28/ٱلضَّآلِّينَ:PENDING:spoken='-'`
        // — words 0..21 decided, words 22..28 unresolved. 29 words, 7 unheard.
        val rows = (0..21).map { row(it, RecitationStatus.CORRECT, spoken = "w$it") } +
            (22..28).map { row(it, RecitationStatus.PENDING) }
        assertEquals(29, rows.distinctWordCount())
        assertEquals(7, rows.count { it.status == RecitationStatus.PENDING })
        val fixed = rows.promoteUnresolvedPendingRows()
        assertTrue(fixed.none { it.status == RecitationStatus.PENDING })
        // Still not a "perfect" recitation — the 7 were never judged.
        assertEquals(7, fixed.unheardWordCount())
        assertEquals(29, fixed.distinctWordCount())
    }

    @Test
    fun `the unresolved promotion is idempotent`() {
        val rows = listOf(
            row(0, RecitationStatus.CORRECT, spoken = "w0"),
            row(1, RecitationStatus.PENDING),
            row(2, RecitationStatus.WRONG, spoken = "x"),
            row(3, RecitationStatus.PENDING, spoken = "w3"),
        )
        val once = rows.promoteUnresolvedPendingRows()
        assertEquals(once, once.promoteUnresolvedPendingRows())
    }

    @Test
    fun `the unresolved promotion never touches a decided row`() {
        val decided = listOf(
            row(0, RecitationStatus.CORRECT, spoken = "w0"),
            row(1, RecitationStatus.WRONG, spoken = "x"),
            row(2, RecitationStatus.SKIPPED),
            row(3, RecitationStatus.TAJWEED_ERROR, spoken = "y"),
            row(4, RecitationStatus.HINTED),
        )
        assertEquals(decided, decided.promoteUnresolvedPendingRows())
    }
}
