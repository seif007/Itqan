package itqan.app.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.45 — "DID THE USER FINISH THE RANGE?", the gate that decides whether the
 * end-of-range summary may be raised on the AUTO (end-of-range) path.
 *
 * The bug it replaces: the gate was `unheardWordCount() == 0`, i.e. EVERY target
 * word must carry spoken evidence. On a real microphone the trailing word of a
 * recitation is routinely the one the recognizer never renders — the last
 * seconds are silence, the RMS gate drops them, and the final whole-session
 * render comes back a subset of the sliding windows' merge — so that row is
 * CORRECT with no token, counts as unheard, and the summary of a plainly
 * FINISHED recitation was withheld:
 *
 *   endOfRange(): auto stop but NOT genuinely finished (unheard=1/29)
 *   -> no summary; words stay displayed
 *
 * while an INJECTED run of the same range decoded that word and did raise the
 * summary. That asymmetry is the whole "works with injection, nothing when I
 * test myself" report.
 *
 * The replacement is a COVERAGE test, and the second half of the contract is
 * that a mid-range stall (the app auto-stopped on 4 s of silence while the user
 * was only half-way) must still NOT be reported as a finish.
 */
class GenuineFinishTest {

    @Test
    fun `every word heard is a finish`() {
        assertTrue(isGenuineFinish(totalWords = 29, unheardWordCount = 0, displayedCursor = 29))
    }

    @Test
    fun `the measured real-mic shape - 28 of 29 heard, display at the end - IS a finish`() {
        // Al-Fatihah 1-7, 29 words. The last word was never rendered by the
        // decoder; finalizeCorrect() promoted it to CORRECT with no token and
        // advanced the cursor over it, so it is DISPLAYED (grey-decided) and the
        // user has plainly read to the end. This must raise the summary.
        assertTrue(isGenuineFinish(totalWords = 29, unheardWordCount = 1, displayedCursor = 29))
    }

    @Test
    fun `a forgiven blind spot in the middle with the display at the end is a finish`() {
        // The grace-expiry forgive stamps a row CORRECT with no token, which
        // counts as unheard. The user is at the end of the range, so the
        // recitation is over and the summary is what they are owed.
        assertTrue(isGenuineFinish(totalWords = 29, unheardWordCount = 1, displayedCursor = 29))
    }

    @Test
    fun `a mid-range stall is NOT a finish`() {
        // Stopped at ayah 2 of 7: 19 words never heard, display far short of the
        // end. The contract forbids a "finished" summary here - the user is
        // mid-range and gets nothing.
        assertFalse(isGenuineFinish(totalWords = 29, unheardWordCount = 19, displayedCursor = 10))
    }

    @Test
    fun `the display short of the end is NOT a finish even with tiny noise`() {
        // Coverage is high and only one word is missing, but the cursor shows the
        // user is still reading - the auto-stop fired on a thinking pause.
        assertFalse(isGenuineFinish(totalWords = 29, unheardWordCount = 1, displayedCursor = 12))
    }

    @Test
    fun `too many unheard words is never a finish`() {
        // 3+ blind spots is a decode problem, not a finished recitation.
        assertFalse(isGenuineFinish(totalWords = 29, unheardWordCount = 3, displayedCursor = 29))
    }

    @Test
    fun `coverage below the floor is never a finish`() {
        // 100 words, 89 heard: below the 90 % floor even at the end of the range.
        assertFalse(isGenuineFinish(totalWords = 100, unheardWordCount = 11, displayedCursor = 100))
    }

    @Test
    fun `an empty range is never a finish`() {
        assertFalse(isGenuineFinish(totalWords = 0, unheardWordCount = 0, displayedCursor = 0))
    }

    @Test
    fun `a long range tolerates the same small blind spots`() {
        // The tolerance is a small ABSOLUTE number of words, not a fraction of
        // the range: a 200-word range with 2 decoder blind spots at the end is as
        // finished as a 29-word one.
        assertTrue(isGenuineFinish(totalWords = 200, unheardWordCount = 2, displayedCursor = 200))
        assertFalse(isGenuineFinish(totalWords = 200, unheardWordCount = 3, displayedCursor = 200))
    }
}
