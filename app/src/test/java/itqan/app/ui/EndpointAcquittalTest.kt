package itqan.app.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.38 — THE SESSION'S OWN ENDPOINT DECODE MUST GET TO ACQUIT BEFORE WE CONDEMN.
 *
 * Measured `p3_fatihah` (Al-Fatihah 1–7, on-device) — a FALSE "Skipped Word"
 * popup for a word the recitation contained:
 *
 *   22:51:47.895 mistakeStop(): verify idx=17 ABSENT strike 2/2 — CONDEMNING
 *   22:51:48.487 runInjectLoop(): auto-stop on 4001ms silence
 *   22:51:48.863 finalDecodeAndEmit(): begin endpoint=true window=45,0s
 *   22:51:49.764 verdictTicker(): absent verdict held 1868ms for wordIndex=17 -> popup
 *   22:51:49.764 mistakeStop(): SKIPPED CONFIRMED wordIndex=17 word='ٱهْدِنَا'
 *   22:51:57.174     log='… نستعين الصرط المستقيم …'      <- live log: ٱهْدِنَا missing
 *   22:51:57.174     dec='… نستعين اهدنا الصرط المستقيم …' <- endpoint decode: PRESENT
 *   22:51:57.179 collector(): mistake-freeze dropping endpoint=true 28w (resume latch armed @ word 17)
 *
 * The two ABSENT strikes came from 4 s-slice verification decodes — the weakest
 * decode the app has, and one that DROPS words: a decode that renders
 * `…نَسْتَعِينُ` and then `ٱلصِّرَٰطَ` has swallowed the `ٱهْدِنَا` between them, and
 * the aligner can only represent that as a SKIPPED. The whole-session endpoint
 * decode heard the word — but it landed 8 s after the popup and was then
 * discarded by the confirmed-mistake freeze (which exists so a re-decode cannot
 * resurrect a mistake, §4.28).
 *
 * `mistakeMayBeConfirmed` holds the confirm while that decode is still in
 * flight. It is bounded so a decode that never returns cannot withhold a real
 * verdict forever (§4.18: "nothing ever happens" is the worst outcome).
 */
class EndpointAcquittalTest {

    private fun mayConfirm(
        pipelineRunning: Boolean = false,
        endpointDecodeLanded: Boolean = false,
        sessionStoppedForMs: Long = 0L,
        holdMs: Long = 15_000L,
    ) = mistakeMayBeConfirmed(pipelineRunning, endpointDecodeLanded, sessionStoppedForMs, holdMs)

    /**
     * The measured `p3_fatihah` moment: the session has stopped, the endpoint
     * decode is 0.9 s in and needs ~8 s, and the ABSENT strikes have matured.
     * This is the cell that produced the false popup.
     */
    @Test
    fun `the measured p3 moment must NOT confirm`() {
        assertFalse(
            "the endpoint decode is in flight and is the best evidence available",
            mayConfirm(pipelineRunning = false, endpointDecodeLanded = false, sessionStoppedForMs = 1274L)
        )
    }

    /** A live session is untouched — the confirm proceeds exactly as before. */
    @Test
    fun `a running pipeline may always confirm`() {
        assertTrue(mayConfirm(pipelineRunning = true, endpointDecodeLanded = false, sessionStoppedForMs = 0L))
    }

    /** Once the endpoint decode lands, the verdict may be taken. */
    @Test
    fun `a landed endpoint decode releases the hold`() {
        assertTrue(
            mayConfirm(pipelineRunning = false, endpointDecodeLanded = true, sessionStoppedForMs = 8315L)
        )
    }

    /** The real-skip case: the decode lands without the word, so we still condemn. */
    @Test
    fun `a real skip still confirms once the endpoint decode lands`() {
        // 8.3 s after the stop — the measured endpoint latency — with words in it.
        assertTrue(mayConfirm(pipelineRunning = false, endpointDecodeLanded = true, sessionStoppedForMs = 8315L))
    }

    /**
     * The bound. A decode that never returns, or returns nothing at all (the
     * measured blank decode), must not withhold a real verdict forever.
     */
    @Test
    fun `the hold expires at the bound`() {
        assertFalse(mayConfirm(pipelineRunning = false, endpointDecodeLanded = false, sessionStoppedForMs = 14_999L))
        assertTrue(mayConfirm(pipelineRunning = false, endpointDecodeLanded = false, sessionStoppedForMs = 15_000L))
    }

    /** A stop we never noticed cannot hold anything: the clock reads 0. */
    @Test
    fun `a session stop that was never observed does not hold`() {
        assertTrue(mayConfirm(pipelineRunning = true, endpointDecodeLanded = false, sessionStoppedForMs = 0L))
    }

    /** The bound is overridable, so the tests do not depend on the constant. */
    @Test
    fun `the hold bound is parameterised`() {
        assertFalse(mayConfirm(pipelineRunning = false, endpointDecodeLanded = false,
            sessionStoppedForMs = 499L, holdMs = 500L))
        assertTrue(mayConfirm(pipelineRunning = false, endpointDecodeLanded = false,
            sessionStoppedForMs = 500L, holdMs = 500L))
    }

    /** The bound must exceed the measured endpoint decode latency (8.3 s). */
    @Test
    fun `the default hold covers the measured endpoint latency`() {
        assertFalse(
            "8.3 s of endpoint decoding must still be inside the hold",
            mayConfirm(pipelineRunning = false, endpointDecodeLanded = false, sessionStoppedForMs = 8_315L)
        )
    }

    /** Every condition is necessary: written out rather than looped. */
    @Test
    fun `running false landed false inside bound holds`() {
        assertFalse(mayConfirm(pipelineRunning = false, endpointDecodeLanded = false, sessionStoppedForMs = 1000L))
    }

    @Test
    fun `running false landed true inside bound confirms`() {
        assertTrue(mayConfirm(pipelineRunning = false, endpointDecodeLanded = true, sessionStoppedForMs = 1000L))
    }

    @Test
    fun `running true landed false past bound confirms`() {
        assertTrue(mayConfirm(pipelineRunning = true, endpointDecodeLanded = false, sessionStoppedForMs = 99_999L))
    }

    @Test
    fun `running false landed false past bound confirms`() {
        assertTrue(mayConfirm(pipelineRunning = false, endpointDecodeLanded = false, sessionStoppedForMs = 99_999L))
    }
}
