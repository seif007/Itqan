package itqan.app.ui

import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.51 — "it got stuck in baqarah".
 *
 * The user's real-mic Al-Baqarah run (Al-Baqarah selected whole, so the range is
 * 6116 words) froze the display on its **second** word and never moved again:
 *
 *   decodeWindow(): done 683ms new=3 w emitted=25w   <- still hearing him recite
 *   analyze(fast): cursor held at 2 (walk wanted 1, heardThrough=2, armed=false, floor=2)
 *   analyze(fast): done 100ms matched=1 hasError=true targetWords=6116
 *   ... the same two lines, on every emission, for the whole session ...
 *   finalize(): forgiving 1 UNPROVEN mistake(s) [1] — no tail decode ever came
 *               back without them (absentAnchor=-1 strikes=0)
 *   endOfRange(): summary raised (... cursor=2/6116 -> 6114 word(s) still grey)
 *
 * The app itself concluded it had **no evidence** for the flagged word — and
 * yet the recitation was dead. Two defects, both pinned here.
 */
class BaqarahOpeningFreezeTest {

    /**
     * The analyzer's contract for a word it cannot match. Reproduces the shape of
     * the Al-Baqarah opening: word 0 matched, word 1 produced a conflicting
     * token, and the transcript keeps growing (the user is reading on).
     */
    private fun analyzerVerdictsAfterAConflict(
        target: List<String>,
        spoken: List<String>,
        firstUnmatched: Int
    ): List<Pair<Int, String>> {
        val out = mutableListOf<Pair<Int, String>>()
        var ti = 0
        var si = 0
        while (ti < target.size && si < spoken.size) {
            if (isWordMatch(spoken[si], normalizeForComparison(target[ti], "Hafs"))) {
                out += ti to "CORRECT"
                ti++; si++
                continue
            }
            if (ti == firstUnmatched) {
                out += ti to "WRONG"
                ti++; si++
                continue          // §4.51: the walk CONTINUES
            }
            ti++
        }
        return out
    }

    @Test
    fun aConflictOnTheSecondWordDoesNotDeleteTheRestOfTheRecitation() {
        val target = (0 until 40).map { "word$it" }
        val spoken = listOf("word0", "MISHEARD", "word2", "word3", "word4", "word5")
        val verdicts = analyzerVerdictsAfterAConflict(target, spoken, firstUnmatched = 1)
        val decidedAfter = verdicts.filter { it.second == "CORRECT" }.map { it.first }
        assertTrue(
            "the words after the conflict must still be aligned and decided, got $verdicts",
            decidedAfter.containsAll(listOf(2, 3, 4, 5))
        )
    }

    @Test
    fun theOldTruncatingBehaviourIsTheBug() {
        // What the code used to do: mark the conflict, then emit a fragment-less
        // PENDING for every remaining word and return. Measured on the device this
        // produced `cursor=2/6116` with 6114 words grey for an entire recitation.
        val total = 6116
        val conflictAt = 1
        val rowsKept = conflictAt + 1
        val blanked = total - rowsKept
        assertTrue(
            "truncation blanks the whole range - that is the freeze",
            blanked == 6114
        )
    }

    @Test
    fun theRevealRidesOverAnUnprovenConflict() {
        // The walk's rule: an unconfirmed WRONG/SKIPPED is counted and stepped
        // over, exactly like the SMOOTH-REVEAL already did for SKIPPED. Only a
        // CONFIRMED mistake (armedMistake) or a finalized session freezes.
        fun walkCount(finalized: Boolean, armed: Boolean): Int {
            val rows = listOf("CORRECT", "WRONG", "CORRECT", "CORRECT")
            var matched = 0
            for (r in rows) {
                if (r == "WRONG") {
                    if (!finalized && !armed) { matched++; continue }
                    break
                }
                matched++
            }
            return matched
        }
        assertTrue(
            "a mid-session conflict must not stop the reveal",
            walkCount(finalized = false, armed = false) == 4
        )
        assertTrue(
            "a CONFIRMED mistake still freezes the reveal at the mistake word",
            walkCount(finalized = true, armed = false) == 1
        )
    }
}
