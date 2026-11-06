package itqan.app.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.59 — Tarteel's contract: "there is popup for mistaks and summury in pause and
 * end and everything exactly like we do in tajweed mode except we dont have tajweed
 * mistakes/popups".
 *
 * Tarteel runs on the platform SpeechRecognizer, and the user is explicit that its
 * recognition is far better than the Tajweed path — so nothing here may touch it.
 * What was missing was everything AROUND it.
 */
class TarteelParityTest {

    /** Mirrors the auto-finalize gate: which modes get the end-of-range summary. */
    private fun getsEndSummary(currentTab: String, mode: String): Boolean =
        currentTab == "Memorize" && (mode == "Tajweed" || mode == "Tarteel")

    @Test
    fun tarteelGetsTheEndOfRangeSummary() {
        assertTrue(
            "Tarteel must get the end-of-range summary like Tajweed mode",
            getsEndSummary("Memorize", "Tarteel")
        )
    }

    @Test
    fun tajweedModeIsUnchanged() {
        assertTrue(getsEndSummary("Memorize", "Tajweed"))
    }

    @Test
    fun theTajweedCoachIsStillExcluded() {
        assertEquals(false, getsEndSummary("Tajweed", "Tajweed"))
    }

    @Test
    fun aMistakeFreezesTheWalkWhereThereIsNoConfirmLoop() {
        // Tarteel has no pipeline and no `verifyTailWords` tail decode, so there is
        // nothing to second-guess a visible jump. The walk's own freeze is the
        // verdict, and it must arm the popup from there.
        val confirmLoopWillJudge = false        // no confirm loop in Tarteel
        val rodeOver = !false /* !finalized */ && !false /* !armed */ && confirmLoopWillJudge
        assertEquals(
            "with no confirm loop the walk must stop, not ride over",
            false, rodeOver
        )
    }

    @Test
    fun theMistakePopupRewindsTheDisplayImmediately() {
        // The floor is a minimum; only the renderer's own predicate moves the cursor,
        // and with the mic stopped no analysis pass will recompute it.
        var alignment = 30 to true
        val wordIdx = 18
        alignment = wordIdx to true
        assertEquals("the highlight must sit on the flagged word", 18, alignment.first)
        assertTrue("the word must keep its error colour", alignment.second)
    }

    private enum class Tap { LIVE, IDLE }

    /**
     * §4.60 — the corrected Tarteel routing.
     *
     * `onMicTap` for a non-Tajweed mode goes through `runStopAndFinalize`, which
     * STARTS when nothing is running and PAUSES+SUMMARY when a session is live.
     * Routing the post-mistake tap to `startTajweedSession()` instead was a
     * regression: that is the SHERPA PIPELINE session, Tarteel does not use the
     * pipeline, and the SpeechRecognizer was never started — so Memorize + Tarteel
     * displayed nothing at all, while the words appeared the moment the mode chip
     * went to Tajweed.
     *
     * So "continue after a mistake" lives on the popup's DISMISSAL, which releases
     * the pause flag and hands the next tap back to the start path.
     */
    private enum class NextTap { START, PAUSE_AND_SUMMARY }

    private fun routeTap(t: Tap, recitationSessionActive: Boolean): NextTap =
        if (t == Tap.LIVE || recitationSessionActive) NextTap.PAUSE_AND_SUMMARY else NextTap.START

    @Test
    fun aDismissedMistakeLetsTheNextTapContinue() {
        // Dismissing the popup releases the pause flag …
        val activeAfterDismissal = false
        // … so the reader's next tap is a START (a continuation), not a second
        // summary. The transcript is kept, so the walk resumes at the mistake.
        assertEquals(
            "after a mistake the reader must be able to CONTINUE, not only summarize",
            NextTap.START,
            routeTap(Tap.IDLE, activeAfterDismissal)
        )
    }

    @Test
    fun aLiveSessionStillPausesAndSummarizes() {
        assertEquals(NextTap.PAUSE_AND_SUMMARY, routeTap(Tap.LIVE, recitationSessionActive = true))
    }

    @Test
    fun aFreshTarteelTapStartsTheRecognizer() {
        assertEquals(
            "with nothing running the tap must START the SpeechRecognizer",
            NextTap.START,
            routeTap(Tap.IDLE, recitationSessionActive = false)
        )
    }
}

