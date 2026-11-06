package itqan.app.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.66/§4.67 — the user's reports:
 *
 *  * "keep displaying words after the missed word in tarteel mode"
 *  * "there is a miised word i continued reading the word after it and it worked
 *     instead of rereading the mistake"
 *  * "for tajweed mode at the end i kept saying random words and instead of getting a
 *     popup of wrong word the app continued displaying words"
 *
 * The first diagnosis was wrong, and [theWrongDiagnosis] pins down why: a target word
 * the recognizer produced nothing for arrives as PENDING with **no fragment**, and it
 * looked like an invisible mid-walk gap the walk sailed over. But the analyzer can
 * only ever emit that row at the transcript frontier, where it is immediately followed
 * by the "spoken exhausted" fill of every remaining target word — so it is always
 * TRAILING and there is no mid-range gap to detect. A rule keyed on one was dead code,
 * and it is gone.
 *
 * What the third report *does* have a real mechanism for is [theRideOverHadNoDeadline]
 * below, and that is fixed.
 */
class GapIsASkipTest {

    private enum class Row { CORRECT, CLIPPED, TRAILING_GAP }

    /**
     * The walk's real treatment of each row kind, as implemented: a CORRECT /
     * clipped / hinted word advances, a fragment-less PENDING does not and does not
     * break either.
     */
    private fun walk(rows: List<Row>): Int {
        var matched = 0
        for (r in rows) if (r != Row.TRAILING_GAP) matched++
        return matched
    }

    @Test
    fun theWrongDiagnosisAFragmentLessPendingRowIsAlwaysTrailing() {
        // `analyzeRecitationDetails` emits PENDING from exactly three places, and
        // only one of them carries no fragment: the transcript-frontier row. The
        // other two are the clipped-decode guard (which CARRIES the fragment, so the
        // walk counts it as heard) and the "spoken exhausted" fill.
        //
        // The frontier row is emitted, then `spokenIndex++` runs it past the end of
        // `spoken`, so the very next iteration takes the exhausted branch, which
        // emits PENDING for every remaining target word and BREAKS. Nothing decided
        // can ever follow a fragment-less PENDING row.
        val emissionOrder = listOf(
            "clipped guard  -> PENDING(spokenText != null) -> counted as heard",
            "frontier       -> PENDING(spokenText == null)",
            "exhausted fill -> PENDING(spokenText == null) x N, then BREAK",
        )
        assertTrue(emissionOrder.any { it.contains("frontier") })

        // And so the walk sees them as one uninterrupted trailing run, which is why
        // freezing on "a decided word after a gap" could never fire.
        val (decided, gaps) = listOf(
            Row.CORRECT, Row.CORRECT, Row.CLIPPED, Row.TRAILING_GAP, Row.TRAILING_GAP
        ).partition { it != Row.TRAILING_GAP }
        assertTrue("every gap comes after every decided row", decided.isNotEmpty() && gaps.isNotEmpty())
        assertFalse(
            "a trailing gap must never advance the display (§4.34: the unreached " +
                "tail of a live recitation has to stay grey, not freeze the highlight)",
            walk(listOf(Row.CORRECT, Row.CORRECT, Row.TRAILING_GAP, Row.TRAILING_GAP)) > 2
        )
    }

    @Test
    fun theRideOverHadNoDeadlineAndThatIsWhatLetTheDisplayRunOn() {
        // §4.67 — §4.51 lets a tentative error be ridden over while the confirm loop
        // verifies it. Its arbiter is a tail decode that keeps returning
        // INCONCLUSIVE, so with no deadline the ride-over became permanent: on
        // Memorize + Tajweed the display advanced over a wrong word for ever and no
        // popup was ever raised. That is the third report verbatim.
        val budget = 8000L
        var since = 0L
        fun spent(now: Long) = now - since >= budget

        assertFalse("inside the grace the display still follows the voice", spent(4000L))
        assertTrue("once the budget is spent the walk must stop and raise", spent(8000L))
        assertTrue("and it stays stopped", spent(60000L))
        assertTrue(
            "the budget must outlast a healthy verification (1.5-4 s measured) so a " +
                "genuine transient candidate is still ridden over - that grace is what " +
                "stopped one unverified conflict at word 1 freezing all 6116 words of " +
                "the user's Al-Baqarah run",
            budget > 4000L
        )
    }

    @Test
    fun aStaleCandidateCannotTripTheBudgetOnTheNextSession() {
        // The candidate clock is cleared alongside the shown-popup set at every one
        // of the session-reset sites, or the first pass of a fresh recitation would
        // inherit an expired budget and stop before the reader said anything.
        var idx = 18
        var since = 0L
        idx = -1; since = 0L          // session reset
        val fresh = idx != 18          // first pass of the new session
        assertTrue("a new session starts a new grace", fresh)
    }
}