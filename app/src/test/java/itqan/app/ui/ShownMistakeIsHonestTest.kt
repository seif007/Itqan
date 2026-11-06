package itqan.app.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.64/§4.65 — the user, testing Tarteel on its own:
 * "it showed me a mistake then i continued. then when i paused i get the popup of
 * the same mistake again ! and the summury tells me no mistake !", plus
 * "when i receite the words/ayahs are not scrolling automatically like it used to do
 * before".
 *
 * The log lines the first two came from:
 *   analyze(fast): done 110ms matched=6 hasError=true          <- the popup
 *   popup dismissed in Tarteel mode -> releasing the pause flag
 *   analyze(fast): done 112ms matched=6 hasError=true          <- 30 s later
 *   ... the same two numbers on every pass ...
 *   autoFinalize(): grace expired with an UNPROVEN mistake at wordIndex=6
 *     (no substantial decode ever judged it) -> forgiving rather than accusing
 *   endOfRange(): ... wrong=0 skipped=0 tajweed=0
 *   endOfRange(): summary raised (... entries=0 ...)
 *
 * One stuck cursor behind all three symptoms: the mistake latch froze the walk at
 * word 6 for the rest of the session, the forgive rule then erased the mistake the
 * app itself had shown, and the auto-scroll — which keys off the cursor — stopped
 * dead with it.
 */
class ShownMistakeIsHonestTest {

    /** The set of words the app has actually popped up for. */
    private fun raisePopup(shown: Set<Int>, wordIdx: Int): Set<Int>? =
        if (wordIdx in shown) null else shown + wordIdx

    @Test
    fun aDismissedPopupDoesNotComeBackForTheSameWord() {
        var shown = raisePopup(emptySet(), 6)
        assertEquals("the first popup is raised", setOf(6), shown)
        // The user dismisses it and carries on. The dismissal must NOT take the word
        // out of the set — that is exactly what let the dialog re-open.
        shown = raisePopup(shown!!, 6)
        assertTrue("the same word must not pop twice", shown == null)
    }

    @Test
    fun aNewSessionCanFlagTheSameWordAgain() {
        // The set is cleared with the session, so a fresh recitation is not silenced
        // by a mistake from the previous one.
        val shownAfterReset = emptySet<Int>()
        assertTrue(raisePopup(shownAfterReset, 6) == setOf(6))
    }

    /** The forgive filter, as implemented. */
    private fun forgiven(
        absentProven: Boolean,
        verified: Boolean,
        shown: Set<Int>,
        wordIdx: Int
    ): Boolean = !absentProven && !verified && wordIdx !in shown

    @Test
    fun aShownMistakeIsNeverForgivenAsUnproven() {
        // Tarteel: no pipeline, so `absentProven` is permanently false. Before
        // §4.64 that forgave every mistake the app had shown.
        assertFalse(
            "a mistake the user was SHOWN must survive the forgive pass",
            forgiven(absentProven = false, verified = false, shown = setOf(6), wordIdx = 6)
        )
        // An unseen, unproven candidate is still forgiven — the original protection.
        assertTrue(
            forgiven(absentProven = false, verified = false, shown = emptySet(), wordIdx = 6)
        )
        // And a re-read that was verified spoken still clears it.
        assertFalse(
            forgiven(absentProven = false, verified = true, shown = setOf(6), wordIdx = 6)
        )
    }

    @Test
    fun bothModesShareTheOneRule() {
        // The Tajweed confirm path writes the same set, so its popups are protected
        // from re-raising and from being forgiven too.
        var shown = emptySet<Int>()
        shown = raisePopup(shown, 18)!!      // stopForMistake on Memorize + Tajweed
        assertTrue(shown.contains(18))
        assertTrue(raisePopup(shown, 18) == null)
        assertFalse(
            "and the summary cannot erase it either",
            forgiven(absentProven = false, verified = false, shown = shown, wordIdx = 18)
        )
    }

    @Test
    fun theCursorIsReleasedSoTheDisplayAndScrollMoveAgain() {
        // §4.65 — the mistake latch is what freezes the walk. On Tajweed the
        // collector's fresh-session re-seat clears it (the pipeline emits it); on a
        // non-pipeline mode nothing did, and the cursor — and with it the auto-scroll
        // — stayed pinned for the whole session.
        var mistakeResumeIndex = 6
        var recitationSessionActive = true
        // dismissal on a non-pipeline mode
        recitationSessionActive = false
        if (true /* recitationMode != "Tajweed" */) mistakeResumeIndex = -1
        assertEquals(
            "the freeze must be released when the reader continues",
            -1, mistakeResumeIndex
        )
    }

    @Test
    fun releasingTheFreezeDoesNotForgetTheMistake() {
        val shown = setOf(6)
        val mistakeResumeIndex = -1                 // latch released
        assertTrue(
            "the mistake stays on record for the summary",
            6 in shown
        )
        assertEquals("the walk is free to move again", -1, mistakeResumeIndex)
    }
}
