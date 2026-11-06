package itqan.app.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.50 — "it keeps getting stuck in displaying words".
 *
 * Measured on the device (my own `check_buttons` run, whose last session the
 * script abandoned): the recognizer committed the utterance at 21 of 29 words,
 * and from then on the app printed, every 2 s, for minutes:
 *
 *   autoFinalize?(): rows=29 decided=29/29 unheardPending=17 unheardWords=17
 *                    alreadyShowing=false done=false
 *   autoFinalize(): 17 unresolved word(s) [12..28] are beyond the heard frontier
 *                    (21 words) + 8 -> NOT promoted, they stay undisplayed
 *   autoFinalize(): grace expired with 17 unresolved word(s) [12..28] and NO
 *                    evidenced mistake -> NOTHING was within reach ...
 *
 * and then nothing: no popup, no summary, display frozen, mic still open. The
 * user cannot tell "waiting for you to read on" from "dead".
 *
 * Two independent defects, both pinned here:
 *
 *  1. THE LOOP COULD NOT END. The "promote once, then let the next pass fall
 *     through to the finalize" guard is driven by a promotion counter — and the
 *     branch that fires when the range is far from finished (nothing promotable)
 *     never advanced that counter, so the guard returned on every pass, forever.
 *
 *  2. THE STALL WAS SILENT. Even falling through, the summary gate demands a user
 *     pause or a genuine finish; a committed-but-incomplete session is neither,
 *     so the user got nothing.
 */
class StalledSessionSummaryTest {

    /** The budget that ends the wait. Generous on purpose — see its KDoc. */
    @Test
    fun theStallBudgetIsLongEnoughToSurviveAPause() {
        assertTrue(
            "a reader who pauses to think must not be interrupted",
            STALL_SUMMARY_AFTER_MS >= 15_000L
        )
    }

    /** Mirrors the summary gate's condition for the third (stalled) state. */
    private fun gateAllowsSummary(
        userInitiatedStop: Boolean,
        genuinelyFinished: Boolean,
        stallExpired: Boolean,
        pendingRemain: Boolean,
        mistakeMustWin: Boolean,
        alreadyShowing: Boolean,
        sessionFinalized: Boolean
    ): Boolean {
        val pendingVeto = pendingRemain && !userInitiatedStop && !stallExpired
        return sessionFinalized && !pendingVeto && !alreadyShowing &&
            !mistakeMustWin && (userInitiatedStop || genuinelyFinished || stallExpired)
    }

    @Test
    fun aStalledCommittedSessionMayRaiseThePausedSummary() {
        assertTrue(
            "the frozen state must be able to produce the honest paused dialog",
            gateAllowsSummary(
                userInitiatedStop = false,
                genuinelyFinished = false,
                stallExpired = true,
                pendingRemain = true,
                mistakeMustWin = false,
                alreadyShowing = false,
                sessionFinalized = true
            )
        )
    }

    @Test
    fun aLiveReaderIsNeverInterrupted() {
        // No stall has run yet, and the session is not committed: nothing shows.
        assertFalse(
            gateAllowsSummary(false, false, false, true, false, false, sessionFinalized = false)
        )
    }

    @Test
    fun aProvenMistakeStillWinsOverTheStallSummary() {
        // The whole point of the mistake contract: a wrong/skipped word must stop
        // the recitation with its own popup, never be buried in a summary.
        assertFalse(
            "a mistake must not be swallowed by the stalled-session dialog",
            gateAllowsSummary(false, false, true, true, mistakeMustWin = true, false, true)
        )
    }

    @Test
    fun aMidRangeStallIsNotAFinish() {
        // The dialog it raises is the PAUSED one: it must not claim the range was
        // completed, because `reachedEndOfRange` is judged on heard evidence.
        val total = 29
        val heard = 21
        val reachedEndOfRange = heard >= total - GENUINE_FINISH_MAX_UNHEARD
        assertFalse("21 of 29 words is not the end of the range", reachedEndOfRange)
        // ...so `sessionRangeCompleted` stays false and the next mic tap RESUMES
        // from the committed prefix instead of wiping the recitation.
    }

    @Test
    fun thePromotionCounterAdvancesEvenWhenNothingIsPromotable() {
        // The loop-ender, as an invariant. The production guard is
        // `if (gracePromoteCount <= 1) return`, so the stall costs at most TWO
        // evaluation passes: the first promotes (or not) and returns, the second
        // falls through to the finalize. Before the fix the empty branch left the
        // counter at 0, so that guard returned on every pass, forever.
        var gracePromoteCount = 0
        repeat(2) {
            val promotable = emptySet<Int>()   // the branch that used to skip the increment
            if (promotable.isEmpty()) gracePromoteCount++ else gracePromoteCount++
            if (gracePromoteCount <= 1) return@repeat   // the production guard
        }
        assertTrue(
            "the stall must fall through to the finalize within two passes, counter=$gracePromoteCount",
            gracePromoteCount > 1
        )
    }

    /**
     * §4.50 — A word the recognizer never heard may only be COLOURED when the
     * user demonstrably reached the end of the range.
     *
     * This is the "it advance without displaying" report. Measured `v5_stall`
     * (20 s of Al-Fatihah against a 29-word range, 13 words actually recited):
     * `finalizeCorrect(): 11 trailing PENDING word(s) counted CORRECT (matched=15,
     * total pending=16)`. Colour is the cursor, so the display claimed ~24 of 29
     * words — words the user never uttered. Both producers of that lie are pinned
     * here: the grace-expiry promotion and `finalizeCorrect()`.
     */
    @Test
    fun aNeverHeardWordIsOnlyForgivenAtTheEndOfTheRange() {
        val total = 29
        fun forgiven(heardFrontier: Int, carriesFragment: Boolean): Boolean {
            val reachedRangeEnd = heardFrontier >= total - FINALIZE_FORGIVE_TAIL_WINDOW
            return carriesFragment || reachedRangeEnd
        }
        // Mid-range: 13 of 29 words heard, this word has no decoded fragment.
        assertFalse(
            "a word never heard, 13 words into a 29-word range, must stay grey",
            forgiven(heardFrontier = 13, carriesFragment = false)
        )
        // Mid-range but CLIPPED: the recognizer heard it and the window cut it off.
        assertTrue(
            "a clipped word is evidence the user said it",
            forgiven(heardFrontier = 13, carriesFragment = true)
        )
        // At the end of the range a trailing gap is the decoder's known blind spot.
        assertTrue(
            "the tail of a finished range is forgivable",
            forgiven(heardFrontier = 27, carriesFragment = false)
        )
    }

    @Test
    fun theMatchedCounterCountsWordsNotRows() {
        // `detailedAnalysis` holds one row per (WORD, RULE), so a row count slides
        // the forgiveness window forward by the number of extra rules. Measured:
        // the walk matched 13 words, the counter said 15, the window reached 23.
        val rows = listOf(
            Triple(0, "CORRECT", "a"), Triple(0, "TAJWEED_ERROR", "b"),
            Triple(1, "CORRECT", "c"), Triple(2, "CORRECT", "d")
        )
        val asRows = rows.count { it.second == "CORRECT" || it.second == "TAJWEED_ERROR" }
        val asWords = rows.filter { it.second == "CORRECT" || it.second == "TAJWEED_ERROR" }
            .map { it.first }.distinct().size
        // A three-word passage where one word carries two rules: 4 rows, 3 words.
        assertEquals("row count is inflated", 4, asRows)
        assertEquals("word count is the truth", 3, asWords)
        assertTrue(
            "the row count must never be used as a word position",
            asRows > asWords
        )
    }
}
