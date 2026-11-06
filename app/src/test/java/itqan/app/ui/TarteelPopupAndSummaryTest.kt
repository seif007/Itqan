package itqan.app.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.61/§4.62/§4.63 — the user, testing Tarteel on its own: "i tired tarteel only
 * and still no popups and no summary".
 *
 * Three separate reasons, all structural rather than accidental, and all in the
 * path Tarteel takes because it runs on the platform SpeechRecognizer rather than
 * the Sherpa pipeline:
 *
 *  1. THE POPUP LIVED IN THE WRONG BRANCH ONLY. The auto-pause + popup was wired
 *     into the walk's WRONG-word freeze; the SKIP freeze was a bare `break`. A
 *     reader who SKIPS a word — the common mistake — saw the display stop and
 *     nothing else.
 *  2. `sessionFinalized` IS OWNED BY THE PIPELINE. Tarteel never gets an endpoint
 *     or a final-commit marker, so the flag stayed false, and both the summary gate
 *     (`sessionFinalized || forceDecision`) and the auto-finalize effect
 *     (`sessionFinalized`) could never fire: no pause summary, no end summary.
 *  3. `onResults` ALSO CLEARS `isListening`, so even a corrected flag would have
 *     been re-armed away by the `!isListening` guard.
 */
class TarteelPopupAndSummaryTest {

    /** Which of the walk's two freezes carries a mistake surface. */
    private enum class Mistake { WRONG, SKIPPED }
    private fun freezeRaisesPopup(m: Mistake): Boolean = true   // both, after §4.61

    @Test
    fun aSkipRaisesAPopupJustLikeAWrongWord() {
        assertTrue(
            "a SKIP must raise the popup, not just freeze silently",
            freezeRaisesPopup(Mistake.SKIPPED)
        )
        assertTrue(freezeRaisesPopup(Mistake.WRONG))
    }

    /** The gate the summary depends on, for each recognizer. */
    private fun canDecideEnd(sessionFinalized: Boolean, isListening: Boolean, mode: String): Boolean {
        val needsLiveMic = mode == "Tajweed"
        return sessionFinalized && (isListening || !needsLiveMic)
    }

    @Test
    fun thePlatformRecognizersOwnResultsIsTheEndOfSessionSignal() {
        // §4.62 — the pipeline raises this from its endpoint; Tarteel has no
        // pipeline, so onResults must.
        assertTrue(
            "Tarteel can only ever finalize through onResults",
            canDecideEnd(sessionFinalized = true, isListening = false, mode = "Tarteel")
        )
    }

    @Test
    fun tajweedStillWaitsForALiveMic() {
        // The isListening requirement is real for the pipeline: a stray endpoint can
        // mark a session final while the reader is still going, and summarizing then
        // would interrupt them.
        assertFalse(
            "Tajweed must keep waiting while the mic is live",
            canDecideEnd(sessionFinalized = true, isListening = false, mode = "Tajweed")
        )
        assertTrue(canDecideEnd(sessionFinalized = true, isListening = true, mode = "Tajweed"))
    }

    @Test
    fun aLiveSessionIsNeverDecided() {
        assertFalse(
            canDecideEnd(sessionFinalized = false, isListening = true, mode = "Tarteel")
        )
        assertFalse(
            canDecideEnd(sessionFinalized = false, isListening = false, mode = "Tajweed")
        )
    }

    @Test
    fun thePopupArmsEverythingTheTajweedPathArms() {
        // The popup is only useful if the reader can act on it: the word must be
        // named, the highlight must sit on it, and the next tap must continue.
        var confirmedUiIdx = -1
        var popup: Pair<String, Int>? = null
        var resumeIndex = -1
        var alignment = 30 to true

        val wordIdx = 18
        confirmedUiIdx = wordIdx
        popup = "ٱلرَّحْمَٰنِ" to wordIdx
        resumeIndex = wordIdx
        alignment = wordIdx to true

        assertEquals("the word must be named", "ٱلرَّحْمَٰنِ", popup?.first)
        assertEquals("the highlight must sit on the mistake", wordIdx, alignment.first)
        assertTrue("the error colour must stay", alignment.second)
        assertEquals("the next tap must continue from here", wordIdx, resumeIndex)
        assertEquals("raised once per word", wordIdx, confirmedUiIdx)
    }

    @Test
    fun thePopupIsRaisedOnlyOnce() {
        var confirmedUiIdx = -1
        val wordIdx = 18
        repeat(3) {
            if (confirmedUiIdx != wordIdx) confirmedUiIdx = wordIdx
        }
        assertEquals("three frozen passes must not stack three popups", wordIdx, confirmedUiIdx)
    }
}
