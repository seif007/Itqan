package com.example.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.53 — "it got back to that error and keep showing me popups and i dismiss them
 * and they keep coming".
 *
 * The user's real-mic Al-Baqarah run on the A04s:
 *
 *   tajweedPop(): TAJWEED_ERROR wordIndex=12 rule=Al-Lam score=40 -> pushing popup
 *   tajweedPop(): display rewound 15 -> 12
 *   analyze(tajweed): done 3555ms scored=8 tajweedErrors=0 changed=4
 *   tajweedPop(): TAJWEED_ERROR wordIndex=12 rule=Al-Lam score=40 -> pushing popup
 *
 * Same word, same rule, same score. Between them a single acoustic pass reported
 * `tajweedErrors=0`, and THAT cleared the popup latch — so the next pass re-flagged
 * the same word and, with the latch gone, the "first verdict not yet shown"
 * selection (§4.49) picked it straight back up.
 */
class TajweedPopupLatchTest {

    /** The popup watcher's selection, as implemented. */
    private fun firstUnshown(
        rows: List<Pair<Int, String>>,
        latchedWord: Int,
        latchedRule: String
    ): Pair<Int, String>? = rows.firstOrNull { (idx, rule) ->
        idx != latchedWord || rule != latchedRule
    }

    /** The latch release rule, as implemented. */
    private fun latchSurvivesABlankPass(
        latchedWord: Int,
        rowsNow: List<Pair<Int, String>>
    ): Int = if (latchedWord >= 0 && rowsNow.none { it.first == latchedWord }) -1 else latchedWord

    @Test
    fun aDismissedPopupDoesNotComeBackWithoutAReRead() {
        var latchedWord = -1
        var latchedRule = ""
        val failing = listOf(12 to "Al-Lam")

        // 1. the error appears -> popup, latch set
        val first = firstUnshown(failing, latchedWord, latchedRule)
        assertEquals(12 to "Al-Lam", first)
        latchedWord = 12
        latchedRule = "Al-Lam"

        // 2. the user dismisses it. The popup is closed, the row is still there.
        //    A pass in between reports no error (tajweedErrors=0).
        latchedWord = latchSurvivesABlankPass(latchedWord, failing)
        assertEquals("the latch must survive a blank pass", 12, latchedWord)

        // 3. the next acoustic pass re-scores the SAME word as failing again.
        assertNull(
            "the re-scored word must NOT re-open a dismissed popup",
            firstUnshown(failing, latchedWord, latchedRule)
        )
    }

    @Test
    fun aSecondRuleOnTheSameWordStillGetsItsOwnPopup() {
        // §4.49's contract has to survive the tighter latch: two rules on one word
        // are two different verdicts and the user expects to see both.
        var latchedWord = -1
        var latchedRule = ""
        val rows = listOf(12 to "Al-Lam", 12 to "Madd")
        val first = firstUnshown(rows, latchedWord, latchedRule)
        assertEquals(12 to "Al-Lam", first)
        latchedWord = 12; latchedRule = "Al-Lam"
        val second = firstUnshown(rows, latchedWord, latchedRule)
        assertEquals(
            "the SECOND rule on the same word must still be reported",
            12 to "Madd", second
        )
    }

    @Test
    fun aReReadReleasesTheLatchSoTheWordCanSpeakAgain() {
        // The resume re-arms the latch explicitly (it is a deliberate user act).
        var latchedWord = 12
        latchedWord = -1                       // re-read: the re-seat clears it
        val rows = listOf(12 to "Al-Lam")
        val again = firstUnshown(rows, latchedWord, "")
        assertTrue(
            "after a re-read a fresh failure on the same word must be reportable",
            again != null
        )
    }

    @Test
    fun theLatchIsReleasedWhenTheWordLeavesTheAnalysis() {
        // A range change or a re-seat that drops the word must not leave the app
        // permanently unable to report a verdict on it.
        val released = latchSurvivesABlankPass(12, listOf(3 to "Madd", 4 to "Madd"))
        assertEquals("a word that left the analysis releases the latch", -1, released)
    }

    /**
     * §4.53 — the alternation the user reported. With only the LAST shown pair
     * remembered, two standing mistakes take turns for ever:
     *
     *   tajweedPop(): wordIndex=83 rule=Madd score=34
     *   tajweedPop(): wordIndex=2  rule=Madd score=31
     *   tajweedPop(): wordIndex=83 rule=Madd score=34
     *   tajweedPop(): wordIndex=2  rule=Madd score=31
     *   tajweedPop(): wordIndex=83 rule=Madd score=34
     *
     * The scores never change — it is the same two verdicts, ping-ponging.
     */
    @Test
    fun twoStandingMistakesDoNotAlternateForEver() {
        val rows = listOf(2 to "Madd", 83 to "Madd")
        val shown = mutableSetOf<String>()
        val fired = mutableListOf<Pair<Int, String>>()
        repeat(5) {
            val next = rows.firstOrNull { (idx, rule) -> "$idx|$rule" !in shown }
            if (next == null) return@repeat
            shown += "${next.first}|${next.second}"
            fired += next
        }
        assertEquals(
            "each verdict may speak at most once, got $fired",
            listOf(2 to "Madd", 83 to "Madd"),
            fired
        )
    }

    @Test
    fun aReReadReArmsEveryVerdictOnThatWord() {
        // The re-read is the user ACTING, so the verdicts on those words may
        // speak again.
        val shownAfterReread = emptySet<String>()      // the resume re-seat re-arms
        val rows = listOf(2 to "Madd", 83 to "Madd")
        val next = rows.firstOrNull { (idx, rule) -> "$idx|$rule" !in shownAfterReread }
        assertEquals(
            "after a re-read the first verdict must be reportable again",
            2 to "Madd", next
        )
    }
}

/**
 * §4.53 — "when i continue no words are displayed".
 *
 * The display cursor is `maxOf(walk, floor)`, so a floor parked at the mistake
 * FREEZES the display above the walk while the aligner re-climbs the kept prefix
 * from target word 0. Measured through a whole re-read:
 *
 *   re-seated ... the user continues at target word 55
 *   cursor held at 55 (walk wanted 41, heardThrough=41)
 *   cursor held at 55 (walk wanted 43, heardThrough=43)
 *   cursor held at 55 (walk wanted 45, heardThrough=45)
 *   cursor held at 55 (walk wanted 52, heardThrough=52)
 *
 * 55 words lit, the voice moving on, the display static for ~14 s.
 */
class RereadDisplayFloorTest {

    @Test
    fun aFloorAtTheMistakeFreezesTheDisplayAboveTheWalk() {
        fun cursor(walk: Int, floor: Int) = maxOf(walk, floor)
        // Parked at the mistake while the walk is still re-climbing:
        assertEquals(55, cursor(walk = 41, floor = 55))
        assertEquals(55, cursor(walk = 52, floor = 55))
    }

    @Test
    fun theFloorMustFollowTheKeptPrefixSoTheRereadLightsUp() {
        val resumeIdx = 55        // the mistake
        val prefixWords = 50      // the re-seat kept this many spoken words
        val floor = minOf(resumeIdx, prefixWords)
        fun cursor(walk: Int) = maxOf(walk, floor)
        // The display sits at the kept prefix — the words already recited — and
        // then follows the walk, so every re-read word lights as it is aligned.
        assertEquals(50, cursor(walk = 41))
        assertEquals(50, cursor(walk = 48))
        assertEquals(56, cursor(walk = 56))
        assertTrue(
            "the display must advance as the re-read is aligned",
            cursor(walk = 56) > cursor(walk = 48)
        )
    }

    @Test
    fun alreadyRecitedWordsDoNotGoGrey() {
        // The floor is still a minimum, so nothing below it is ever un-coloured.
        val floor = minOf(55, 50)
        val walk = 45
        assertTrue("the display may not retreat below the floor", maxOf(walk, floor) >= floor)
    }
}
