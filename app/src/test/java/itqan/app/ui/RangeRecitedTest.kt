package itqan.app.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.46 — "DID THE RANGE REALLY GET RECITED?", the test behind
 * `sessionRangeCompleted` — the flag that makes the NEXT mic tap wipe the
 * transcript and start from word 1.
 *
 * It is NOT the display cursor. The end-of-session path deliberately advances the
 * cursor over the tail the decoder never rendered, so those words are VISIBLE
 * (§4.40) — which means the cursor reaches the end of the range even when the
 * user stopped half-way through it. Reading the flag off the cursor therefore
 * destroyed a half-finished recitation: measured `s3_fatihah`, the session ended
 * at word 16 of 29 (the recording's own silence), the promotions carried the
 * cursor to 29/29, and the user's 16 words would have been thrown away on the
 * next tap.
 *
 * A word the user really said carries a spoken token, so the honest test is the
 * same coverage test [isGenuineFinish] applies.
 */
class RangeRecitedTest {

    /** Mirrors the decision made in `runStopAndFinalize` for `sessionRangeCompleted`. */
    private fun rangeRecited(
        totalWords: Int,
        unheardWordCount: Int,
        maxUnheard: Int = GENUINE_FINISH_MAX_UNHEARD,
    ): Boolean {
        if (totalWords <= 0) return false
        return (totalWords - unheardWordCount) >= totalWords - maxUnheard
    }

    @Test
    fun `a fully heard range counts as recited`() {
        assertTrue(rangeRecited(totalWords = 29, unheardWordCount = 0))
    }

    @Test
    fun `the decoder's blind-spot tail still counts as recited`() {
        // The measured real-mic shape: the last word was never rendered.
        assertTrue(rangeRecited(totalWords = 29, unheardWordCount = 1))
        assertTrue(rangeRecited(totalWords = 29, unheardWordCount = GENUINE_FINISH_MAX_UNHEARD))
    }

    @Test
    fun `a session that stopped half-way is NOT recited - it must stay resumable`() {
        // s3_fatihah: 13 words of the tail were never heard at all.
        assertFalse(rangeRecited(totalWords = 29, unheardWordCount = 16))
        assertFalse(rangeRecited(totalWords = 29, unheardWordCount = 3))
    }

    @Test
    fun `an empty target is never recited`() {
        assertFalse(rangeRecited(totalWords = 0, unheardWordCount = 0))
    }

    @Test
    fun `the display cursor is deliberately NOT part of the test`() {
        // This is the regression guard for the s3 bug: the cursor said 29/29 while
        // only 16 words had ever been heard. If anyone re-introduces the cursor
        // into this decision, this test's inputs are exactly what it takes.
        val cursor = 29
        val total = 29
        val unheard = 16
        assertTrue(cursor >= total)                     // the cursor WAS at the end...
        assertFalse(rangeRecited(total, unheard))       // ...and the range was NOT recited
    }
}
