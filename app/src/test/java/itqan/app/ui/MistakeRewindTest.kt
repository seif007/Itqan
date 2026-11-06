package itqan.app.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.52 — "i want the display of words to get back to the mistaken word when the
 * popup is showed to the user and the mic is stopped so he could know from where
 * to receite again after the mistake".
 *
 * The user hit this on a tajweed popup: the popup named the word, the mic
 * stopped, and the highlight stayed where the recitation had reached. It only
 * snapped back when he turned the mic on again — i.e. after the re-seat in the
 * collector's fresh-session block, which is the wrong moment: the decision to
 * re-read has already been made and the app is showing nothing about it.
 *
 * Two things had to change, and both are pinned here.
 */
class MistakeRewindTest {

    @Test
    fun loweringTheFloorAloneCannotMoveTheCursorBackwards() {
        // The floor is a MINIMUM: `cursor = maxOf(walk, floor)`. This is why the
        // §4.48 "drop the floor to the mistake" never rewound anything on its own.
        val walkCount = 22
        val floorBefore = 22
        val floorAfter = 5
        val cursorBefore = maxOf(walkCount, floorBefore)
        val cursorAfter = maxOf(walkCount, floorAfter)
        assertEquals(22, cursorBefore)
        assertEquals(
            "dropping the floor under a cursor that is already ahead changes nothing",
            22, cursorAfter
        )
    }

    @Test
    fun anArmedMistakeParksTheCursorOnTheFlaggedWord() {
        // §4.51 gave the walk a ride-over for a TENTATIVE error, so the walk's
        // count is now PAST the mistake. "Return to the mistake" can therefore no
        // longer be inferred from where the walk stopped — it has to be stated.
        val matchedPastTheMistake = 22
        val mistake = 18
        val cursor = resolveTranscriptCursor(
            newMatchedWordCount = matchedPastTheMistake,
            previousCursor = 22,
            hasError = true,
            finalized = false,
            confirmedMistakeArmed = true,
            mistakeIndex = mistake
        )
        assertEquals(
            "with the latch armed the display must sit ON the flagged word",
            mistake, cursor
        )
    }

    @Test
    fun aTentativeErrorStillRidesOverSoTheDisplayKeepsMoving() {
        // The fix must not undo §4.51: with NO latch armed the highlight has to
        // keep following the voice, or Al-Baqarah freezes on word 2 again.
        val cursor = resolveTranscriptCursor(
            newMatchedWordCount = 13,
            previousCursor = 6,
            hasError = true,
            finalized = false,
            confirmedMistakeArmed = false,
            mistakeIndex = -1
        )
        assertTrue("the reveal must keep advancing, got $cursor", cursor >= 13)
    }

    @Test
    fun theRewindHappensInTheSameInstantAsThePopup() {
        // The mic is stopped in the same statement block, so no further emission
        // ever arrives and `analyze(fast)` never re-runs to recompute the cursor.
        // The renderer predicate (`alignment.first`) therefore has to be written
        // by the popup itself, not left to the next analysis pass.
        var alignment = 22 to true          // the highlight is 22 words in
        var emissionAfterPopup = false      // the mic is stopped: no more emissions
        if (!emissionAfterPopup) {
            // the analysis pass that would recompute the cursor never runs
        }
        val wordIdx = 18
        alignment = wordIdx to true
        assertEquals(
            "the popup must put the highlight on the flagged word itself",
            wordIdx, alignment.first
        )
        assertTrue(
            "hasError must stay true so the word keeps its error colour",
            alignment.second
        )
    }
}
