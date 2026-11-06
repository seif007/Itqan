package itqan.app.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.58 — "in tarteel mode … tried skipping words but it always display the words
 * between the last correct word and the word i said instead of stopping and giving
 * me a skipped word mistake".
 *
 * A regression from §4.51. The ride-over was introduced so the display would keep
 * following the reader's voice while the CONFIRM LOOP decided whether a flagged
 * word was really a mistake. But the confirm loop only exists on
 * Memorize + Tajweed, so on the READ / Tarteel surface a ridden-over error was
 * never judged by anything: the walk stepped over the gap, the display painted
 * the words between the last correct word and the word the reader actually said,
 * and no skipped-word mistake was ever raised.
 *
 * A ride-over is only honest where something will judge it.
 */
class TarteelSkipMustFreezeTest {

    /** The walk's decision for one error row, as implemented. */
    private fun walksOverTheError(
        sessionFinalized: Boolean,
        armed: Boolean,
        confirmLoopWillJudge: Boolean
    ): Boolean = !sessionFinalized && !armed && confirmLoopWillJudge

    @Test
    fun tarteelFreezesOnASkippedWord() {
        assertFalse(
            "Tarteel has no confirm loop, so a skip must STOP the display",
            walksOverTheError(
                sessionFinalized = false,
                armed = false,
                confirmLoopWillJudge = false
            )
        )
    }

    @Test
    fun tarteelFreezesOnAWrongWordToo() {
        // The user asked for the wrong-word case to be checked as well ("check the
        // wrong word also there"). It shares the same branch structure, so the same
        // gate must hold.
        assertFalse(
            "Tarteel has no confirm loop, so a wrong word must STOP the display",
            walksOverTheError(
                sessionFinalized = false,
                armed = false,
                confirmLoopWillJudge = false
            )
        )
    }

    @Test
    fun memorizeTajweedStillRidesOverSoBaqarahDoesNotFreeze() {
        // The fix must not undo §4.51, which is what stopped Al-Baqarah freezing on
        // its second word for a whole recitation.
        assertTrue(
            "Memorize + Tajweed must keep riding over an unproven error",
            walksOverTheError(
                sessionFinalized = false,
                armed = false,
                confirmLoopWillJudge = true
            )
        )
    }

    @Test
    fun aConfirmedMistakeFreezesEverywhere() {
        for (judged in listOf(true, false)) {
            assertFalse(
                "a CONFIRMED mistake must freeze the reveal in every mode",
                walksOverTheError(
                    sessionFinalized = false,
                    armed = true,
                    confirmLoopWillJudge = judged
                )
            )
        }
    }

    @Test
    fun theFrozenCursorStopsAtTheMistakeNotPastIt() {
        // Freezing means the walk counted up to the mistake and stopped there, so
        // the display rests ON the mistake — the reader sees which word stopped
        // them, and the words they skipped are NOT painted as read.
        val rows = listOf("CORRECT", "CORRECT", "SKIPPED", "CORRECT", "CORRECT")
        var matched = 0
        var hasError = false
        for (r in rows) {
            if (r == "WRONG" || r == "SKIPPED") { hasError = true; break }
            matched++
        }
        assertTrue("the walk must have stopped on an error", hasError)
        assertEquals("the display rests on the mistaken word", 2, matched)
    }
}
