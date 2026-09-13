package com.example.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.37 — A FINALIZED SESSION MAY NOT UN-DISPLAY A WORD.
 *
 * `alignment.first` is the TRANSCRIPT RENDERER'S CURSOR, not a coverage metric.
 * The renderer colours a word iff `globalWordIndex < matchedWordCount` and drops
 * it to the 25 %-opacity grey otherwise, so a cursor that moves BACKWARDS
 * literally un-renders words the user already saw.
 *
 * The cursor is recomputed from scratch on every analysis pass, and the aligner
 * is not stable: when `fullAudio` is attached it re-decodes each candidate
 * word's slice, and that decode is flaky. Measured `p2_fatihah` (Al-Fatihah 1–7,
 * on-device, all on the SAME frozen 29-token transcript):
 *
 *   22:01:35.200 collector(): rejected repair still contributed 1 word(s)
 *                -> transcript now 29w
 *   22:01:35.384 autoFinalize?(): decided=29/29 unheardWords=0
 *   22:01:35.396 endOfRange(): summary raised (finished=true unheard=0/29)
 *   22:01:35.680 analyze(fast): done 429ms matched=29 hasError=false
 *   22:01:35.828 analyze(tajweed): done 184ms ... changed=5
 *   22:01:35.899 analyze(fast): done 218ms matched=28 hasError=false  <- BACKWARDS
 *
 * The last word of the surah fell to the grey fallback WHILE the "Perfect
 * recitation!" summary was on screen. The user: "the last word isnt displayed
 * and yet we have a summury popup indicating no mistakes".
 *
 * These tests pin the two halves of the fix:
 *  - [resolveTranscriptCursor] — the cursor is monotonic on a finalized session
 *    that found no mistake, and stays fully free otherwise.
 *  - [keepFinalizedDecision] — a fresh PENDING row may not un-decide a word the
 *    finalized analysis already decided, so `unheardWordCount()` cannot flip
 *    back to 1 on a recitation the app just called perfect.
 */
class FinalizedCursorMonotonicTest {

    // ─────────────────────────────────────────────────────────────────────
    // resolveTranscriptCursor — the renderer's predicate
    // ─────────────────────────────────────────────────────────────────────

    /** The measured `p2_fatihah` regression, reduced to its two passes. */
    @Test
    fun `finalized session holds the cursor when a later pass scores fewer words`() {
        // Pass A: matched=29 -> alignment.first = 29
        val afterPassA = resolveTranscriptCursor(
            newMatchedWordCount = 29, previousCursor = 0,
            hasError = false, finalized = true
        )
        assertEquals(29, afterPassA)

        // Pass B: the same frozen transcript now scores 28. The cursor must NOT drop.
        val afterPassB = resolveTranscriptCursor(
            newMatchedWordCount = 28, previousCursor = afterPassA,
            hasError = false, finalized = true
        )
        assertEquals(29, afterPassB)
    }

    /** A live session must stay free: the cursor has to follow the user. */
    @Test
    fun `live session lets the cursor move backwards`() {
        val cursor = resolveTranscriptCursor(
            newMatchedWordCount = 4, previousCursor = 12,
            hasError = false, finalized = false
        )
        assertEquals(4, cursor)
    }

    /**
     * A confirmed mistake must render in its error colour, and the renderer's
     * error branch only runs for `globalWordIndex >= matchedWordCount`. Clamping
     * here would hide the very mistake the app is stopping for.
     */
    @Test
    fun `a finalized session that found a mistake is NOT clamped`() {
        val cursor = resolveTranscriptCursor(
            newMatchedWordCount = 3, previousCursor = 20,
            hasError = true, finalized = true
        )
        assertEquals(3, cursor)
    }

    /** Progress on a finalized session still advances the cursor. */
    @Test
    fun `finalized session still advances when a later pass scores more words`() {
        val cursor = resolveTranscriptCursor(
            newMatchedWordCount = 29, previousCursor = 28,
            hasError = false, finalized = true
        )
        assertEquals(29, cursor)
    }

    /** A finalized session with nothing decided yet is not pinned anywhere. */
    @Test
    fun `finalized session with an empty previous cursor stays at the walk result`() {
        val cursor = resolveTranscriptCursor(
            newMatchedWordCount = 0, previousCursor = 0,
            hasError = false, finalized = true
        )
        assertEquals(0, cursor)
    }

    /**
     * The escape hatches must be independently sufficient. Written out
     * explicitly rather than looped: a heterogeneous `arrayOf(...)` with
     * `as Boolean`/`as Int` casts throws ClassCastException in Kotlin.
     */
    @Test
    fun `live session with a mistake is not clamped`() {
        assertEquals(
            2,
            resolveTranscriptCursor(newMatchedWordCount = 2, previousCursor = 9,
                hasError = true, finalized = false)
        )
    }

    @Test
    fun `live session without a mistake is not clamped`() {
        assertEquals(
            2,
            resolveTranscriptCursor(newMatchedWordCount = 2, previousCursor = 9,
                hasError = false, finalized = false)
        )
    }

    @Test
    fun `finalized session with a mistake is not clamped`() {
        assertEquals(
            2,
            resolveTranscriptCursor(newMatchedWordCount = 2, previousCursor = 9,
                hasError = true, finalized = true)
        )
    }

    @Test
    fun `finalized session without a mistake IS clamped`() {
        assertEquals(
            9,
            resolveTranscriptCursor(newMatchedWordCount = 2, previousCursor = 9,
                hasError = false, finalized = true)
        )
    }

    /**
     * Every word of Al-Fatihah must survive a whole sequence of post-finalize
     * passes, including the `analyze(tajweed) … changed=5` merge in between.
     */
    @Test
    fun `the measured p2 sequence leaves all 29 words displayed`() {
        var cursor = 0
        // the pass that finalized the session
        cursor = resolveTranscriptCursor(29, cursor, hasError = false, finalized = true)
        // analyze(fast) 429ms matched=29
        cursor = resolveTranscriptCursor(29, cursor, hasError = false, finalized = true)
        // analyze(tajweed) changed=5 — does not touch the cursor
        // analyze(fast) 218ms matched=28
        cursor = resolveTranscriptCursor(28, cursor, hasError = false, finalized = true)
        assertEquals(29, cursor)
    }

    // ─────────────────────────────────────────────────────────────────────
    // keepFinalizedDecision — the analysis list
    // ─────────────────────────────────────────────────────────────────────

    private fun row(
        idx: Int,
        status: RecitationStatus,
        spoken: String? = null,
        rule: String? = null,
        score: Int? = null,
    ) = WordAnalysis(
        wordIndex = idx,
        text = "w$idx",
        normalizedText = "w$idx",
        status = status,
        spokenText = spoken,
        tajweedRule = rule,
        tajweedScore = score,
    )

    /**
     * The measured `p2_fatihah` state: word 28 was decided (the grace promotion
     * set it CORRECT), the summary was raised with `unheardWords=0` — and the
     * next aligner pass handed back PENDING.
     */
    @Test
    fun `a fresh PENDING row may not un-decide a decided word on a finalized session`() {
        val decided = row(28, RecitationStatus.CORRECT)
        val fresh = row(28, RecitationStatus.PENDING)
        val result = keepFinalizedDecision(fresh, decided, finalized = true)
        assertEquals(RecitationStatus.CORRECT, result.status)
    }

    /** The acoustic score the fresh pass computed must not be thrown away. */
    @Test
    fun `the fresh tajweed score survives the decision guard`() {
        val decided = row(28, RecitationStatus.CORRECT, rule = "Madd")
        val fresh = row(28, RecitationStatus.PENDING, rule = "Madd", score = 7)
        val result = keepFinalizedDecision(fresh, decided, finalized = true)
        assertEquals(RecitationStatus.CORRECT, result.status)
        assertEquals(7, result.tajweedScore)
        assertEquals("Madd", result.tajweedRule)
    }

    /** A previously-scored row keeps its score when the fresh pass has none. */
    @Test
    fun `an existing score is kept when the fresh row has none`() {
        val decided = row(28, RecitationStatus.CORRECT, rule = "Madd", score = 5)
        val fresh = row(28, RecitationStatus.PENDING, rule = "Madd")
        val result = keepFinalizedDecision(fresh, decided, finalized = true)
        assertEquals(RecitationStatus.CORRECT, result.status)
        assertEquals(5, result.tajweedScore)
    }

    /** During a LIVE session nothing is frozen — the user may re-read. */
    @Test
    fun `a live session accepts the fresh PENDING row`() {
        val decided = row(28, RecitationStatus.CORRECT)
        val fresh = row(28, RecitationStatus.PENDING)
        val result = keepFinalizedDecision(fresh, decided, finalized = false)
        assertEquals(RecitationStatus.PENDING, result.status)
    }

    /** A first sighting (no previous row) is always taken as-is. */
    @Test
    fun `a word with no previous row keeps the fresh verdict`() {
        val fresh = row(3, RecitationStatus.PENDING)
        assertEquals(RecitationStatus.PENDING, keepFinalizedDecision(fresh, null, true).status)
        assertEquals(RecitationStatus.PENDING, keepFinalizedDecision(fresh, null, false).status)
    }

    /** A decided FRESH row always wins — this helper only blocks PENDING. */
    @Test
    fun `a fresh CORRECT row is never replaced by a PENDING previous row`() {
        val previous = row(28, RecitationStatus.PENDING)
        val fresh = row(28, RecitationStatus.CORRECT)
        val result = keepFinalizedDecision(fresh, previous, finalized = true)
        assertEquals(RecitationStatus.CORRECT, result.status)
    }

    /** TAJWEED_ERROR is a decided verdict and must survive too. */
    @Test
    fun `a fresh PENDING row may not un-decide a TAJWEED_ERROR row`() {
        val decided = row(9, RecitationStatus.TAJWEED_ERROR, rule = "Ghunnah", score = 4)
        val fresh = row(9, RecitationStatus.PENDING, rule = "Ghunnah")
        val result = keepFinalizedDecision(fresh, decided, finalized = true)
        assertEquals(RecitationStatus.TAJWEED_ERROR, result.status)
    }

    /**
     * A PENDING previous row is NOT a decision, so the fresh row stands. This
     * is what keeps the helper from freezing an undecided word forever.
     */
    @Test
    fun `PENDING against PENDING stays PENDING`() {
        val result = keepFinalizedDecision(
            row(12, RecitationStatus.PENDING), row(12, RecitationStatus.PENDING), finalized = true
        )
        assertEquals(RecitationStatus.PENDING, result.status)
    }

    // ─────────────────────────────────────────────────────────────────────
    // keepFinalizedDecisions — row-wise, keyed by (word, rule)
    // ─────────────────────────────────────────────────────────────────────

    /**
     * MULTI-RULE: `detailedAnalysis` is one row per (word, rule). A word
     * carrying two rules must have BOTH rows guarded independently, and a
     * decided row for one rule must not resurrect a different rule's row.
     */
    @Test
    fun `the guard is keyed by word AND rule`() {
        val previous = listOf(
            row(5, RecitationStatus.CORRECT, rule = "Madd"),
            row(5, RecitationStatus.PENDING, rule = "Ghunnah"),
        )
        val fresh = listOf(
            row(5, RecitationStatus.PENDING, rule = "Madd"),
            row(5, RecitationStatus.PENDING, rule = "Ghunnah"),
        )
        val out = fresh.keepFinalizedDecisions(previous, finalized = true)
        assertEquals(2, out.size)
        assertEquals(RecitationStatus.CORRECT, out.first { it.tajweedRule == "Madd" }.status)
        assertEquals(RecitationStatus.PENDING, out.first { it.tajweedRule == "Ghunnah" }.status)
    }

    /** Not finalized => the list passes through untouched (identity). */
    @Test
    fun `a live session passes the list through unchanged`() {
        val previous = listOf(row(1, RecitationStatus.CORRECT))
        val fresh = listOf(row(1, RecitationStatus.PENDING))
        val out = fresh.keepFinalizedDecisions(previous, finalized = false)
        assertEquals(RecitationStatus.PENDING, out.single().status)
    }

    /**
     * The end-to-end shape of the measured run: after the guard, a finalized
     * Al-Fatihah has NO unheard words — so `unheardWordCount()` agrees with the
     * "no mistakes" summary that is already on screen.
     */
    @Test
    fun `after the guard a finalized fatihah has no unheard words`() {
        // 28 words decided in the previous pass, the 29th freshly PENDING.
        val previous = (0 until 28).map { row(it, RecitationStatus.CORRECT, spoken = "t$it") } +
            listOf(row(28, RecitationStatus.CORRECT, spoken = "ٱلضَّآلِّينَ"))
        val fresh = (0 until 28).map { row(it, RecitationStatus.CORRECT, spoken = "t$it") } +
            listOf(row(28, RecitationStatus.PENDING))

        // Without the guard the summary and the list disagree.
        assertEquals(1, fresh.unheardWordCount())

        val guarded = fresh.keepFinalizedDecisions(previous, finalized = true)
        assertEquals(29, guarded.distinctWordCount())
        assertEquals(0, guarded.unheardWordCount())
    }

    /**
     * The guard must NOT invent a decision. A word the previous pass never
     * decided stays unheard, so a recitation the app could not judge can still
     * never earn a "perfect" summary (§4.31).
     */
    @Test
    fun `the guard cannot forge a decision for an undecided word`() {
        val previous = listOf(row(0, RecitationStatus.CORRECT, spoken = "a"))
        val fresh = listOf(
            row(0, RecitationStatus.CORRECT, spoken = "a"),
            row(1, RecitationStatus.PENDING),
        )
        val guarded = fresh.keepFinalizedDecisions(previous, finalized = true)
        assertEquals(1, guarded.unheardWordCount())
        assertTrue(guarded.any { it.status == RecitationStatus.PENDING })
    }

    /** Idempotence: applying the guard twice changes nothing further. */
    @Test
    fun `the guard is idempotent`() {
        val previous = listOf(row(2, RecitationStatus.CORRECT, spoken = "x"))
        val fresh = listOf(row(2, RecitationStatus.PENDING))
        val once = fresh.keepFinalizedDecisions(previous, finalized = true)
        val twice = once.keepFinalizedDecisions(previous, finalized = true)
        assertEquals(once, twice)
    }

    /**
     * The two helpers compose: guard first, then the §4.34 clipped promotion —
     * so a word that is decided AND carries a fragment is CORRECT, and a
     * genuinely clipped word (PENDING + token, never decided before) is also
     * coloured while still counting as heard.
     */
    @Test
    fun `guard then clipped promotion leaves both decided and clipped words visible`() {
        val previous = listOf(row(27, RecitationStatus.CORRECT, spoken = "t27"))
        val fresh = listOf(
            row(27, RecitationStatus.PENDING),               // un-decided by a flaky pass
            row(28, RecitationStatus.PENDING, spoken = "ٱلضَّآ"), // clipped tail word
        )
        val out = fresh
            .keepFinalizedDecisions(previous, finalized = true)
            .promoteClippedPendingRows()

        assertEquals(RecitationStatus.CORRECT, out.first { it.wordIndex == 27 }.status)
        assertEquals(RecitationStatus.CORRECT, out.first { it.wordIndex == 28 }.status)
        // Both count as HEARD (a token, or a previously-decided row), so a
        // genuine finish still raises its summary.
        assertEquals(0, out.unheardWordCount())
        assertFalse(out.any { it.status == RecitationStatus.PENDING })
    }

    /** A token-less PENDING row that was never decided stays unheard. */
    @Test
    fun `an undecided token-less row is still not coloured by the promotion`() {
        val previous = emptyList<WordAnalysis>()
        val fresh = listOf(row(4, RecitationStatus.PENDING))
        val out = fresh
            .keepFinalizedDecisions(previous, finalized = true)
            .promoteClippedPendingRows()
        assertEquals(RecitationStatus.PENDING, out.single().status)
        assertEquals(1, out.unheardWordCount())
        assertNull(out.single().spokenText)
    }
}
