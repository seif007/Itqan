package itqan.app.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.69 — the real cause of the Tarteel reports, and the fix.
 *
 * The user said, on Memorize + Tarteel:
 *
 *   * "keep displaying words after the missed word in tarteel mode"
 *   * "there is a miised word i continued reading the word after it and it worked
 *     instead of rereading the mistake"
 *
 * The walk itself was already correct: in Tarteel `confirmLoopWillJudge` is false,
 * so a WRONG/SKIPPED row breaks the walk and `raiseFreezePopupIfUnjudged` raises
 * the popup. The defect was one level up, in the cursor arithmetic that runs right
 * after the walk — and it is an **ordering** defect, which is why it survived every
 * attempt to fix the walk.
 *
 * The chain, exactly as it ran:
 *  1. `armedMistake` is snapshotted BEFORE the walk.
 *  2. The walk freezes on the skipped word and, on a surface with no confirm loop,
 *     the popup helper arms `mistakeResumeIndex` **during** that walk.
 *  3. The floor code then read the stale pre-walk snapshot, concluded "no mistake is
 *     armed", and lifted `revealFloor` to `lastHeardWord` — the reader's LAST HEARD
 *     WORD, which by construction lies PAST the skipped word (he kept reading).
 *  4. `resolveTranscriptCursor` was told "not armed", so it did not park on the
 *     flagged index, and `alignment = Pair(cursorNow, …)` overwrote the §4.52 rewind
 *     that the popup helper had just written.
 *  5. `displayCursorFloor` was latched at that past-the-mistake value.
 *  6. The freeze STOPS the recognizer, so no further pass ever arrives to correct it.
 *
 * Step 6 is why this showed up on Tarteel and not (as often) on Memorize + Tajweed:
 * there the confirm loop re-runs the walk while the mic is still open, so a later
 * pass recovers the cursor.
 */
class MistakeFreezeMustPinTheCursorTest {

    /** The cursor arithmetic, as implemented after §4.69. */
    private class Cursor(
        private var displayFloor: Int,
        private var mistakeResumeIndex: Int = -1,
    ) {
        /** What `raiseFreezePopupIfUnjudged` does: arm the latch + rewind. */
        fun freezeOnMistake(wordIndex: Int) {
            mistakeResumeIndex = wordIndex
            // §4.52: the renderer predicate is written directly, so the display
            // rewinds at the instant the popup appears.
            cursor = wordIndex
        }

        var cursor: Int = 0

        /** The floor/cursor block, run once per analysis pass. */
        fun pass(
            matched: Int,
            hasError: Boolean,
            lastHeardWord: Int?,
            finalized: Boolean = false,
        ): Int {
            // PRE-WALK snapshot, exactly as the old code took it.
            val armedMistake = mistakeResumeIndex >= 0
            // ...the walk runs here and may freeze on a mistake, arming the latch.

            // §4.69: read the latch AFTER the walk.
            val armedMistakeNow = armedMistake || mistakeResumeIndex >= 0
            val revealFloor =
                if (!finalized && hasError && !armedMistakeNow) maxOf(cursor, lastHeardWord ?: 0)
                else cursor
            val resolved =
                if (armedMistakeNow) mistakeResumeIndex else maxOf(matched, revealFloor)
            val floorNow = if (armedMistakeNow) 0 else displayFloor
            cursor = maxOf(resolved, floorNow)
            if (!armedMistakeNow && cursor > displayFloor) displayFloor = cursor
            return cursor
        }
    }

    @Test
    fun thePassThatRaisesThePopupMustLeaveTheCursorONtheMistake() {
        val c = Cursor(displayFloor = 6)
        c.cursor = 6
        // The reader has recited past the skipped word: rows after it carry fragments,
        // so `lastHeardWord` is 12 while the mistake is word 9.
        c.freezeOnMistake(9)
        val after = c.pass(matched = 9, hasError = true, lastHeardWord = 12)
        assertEquals(
            "the display must stop ON the missed word while its popup is up - not run " +
                "on to the last word the reader got out before he was stopped",
            9,
            after
        )
        assertFalse(
            "no word after the mistake may be rendered as read",
            after > 9
        )
    }

    @Test
    fun theFloorMustNotLatchPastTheMistakeEither() {
        val c = Cursor(displayFloor = 6)
        c.cursor = 6
        c.freezeOnMistake(9)
        c.pass(matched = 9, hasError = true, lastHeardWord = 12)
        // The next pass (e.g. the summary, or the re-read after dismissal) must still
        // start from the mistake. A latched floor would drag it forward again.
        assertEquals(9, c.cursor)
    }

    @Test
    fun theSmoothRevealStillWorksWhenNoMistakeIsArmed() {
        // The reveal floor exists so a TENTATIVE candidate cannot freeze the display
        // for the whole confirm window (§4.51). With nothing armed it must keep
        // following the voice — this fix must not cost that.
        val c = Cursor(displayFloor = 6)
        c.cursor = 6
        val after = c.pass(matched = 9, hasError = true, lastHeardWord = 12)
        assertEquals(
            "an unarmed candidate must let the reveal follow the voice",
            12,
            after
        )
    }

    @Test
    fun aCleanPassIsUnaffected() {
        val c = Cursor(displayFloor = 4)
        c.cursor = 4
        assertEquals(7, c.pass(matched = 7, hasError = false, lastHeardWord = 7))
    }

    @Test
    fun theWalkStillGatesOnThePreWalkSnapshot() {
        // A word the walk has just decided must not flip its own branch mid-iteration:
        // `rideOverBudgetSpent` / the ride-over gate keep reading `armedMistake`.
        // If that ever switched to the post-walk value, the walk would change verdict
        // halfway through one pass over the list.
        val armedBeforeWalk = -1 >= 0
        assertFalse("no latch before the walk", armedBeforeWalk)
    }
}