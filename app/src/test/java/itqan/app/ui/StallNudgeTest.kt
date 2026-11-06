package itqan.app.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.35 — the stall watchdog must SPEND a decode rather than freeze the walk.
 *
 * The confirm loop that launches a verification decode is EMISSION-driven, and a
 * frozen walk emits nothing. After the FIRST strike nothing ever re-runs it, so
 * the SECOND decode — the one that decides whether to condemn — is never
 * launched. Measured `clean_fatihah_3` (Al-Fatihah 1–7):
 *
 *   mistakeStop(): verify idx=14 WRONG word='نَعْبُدُ' verified=16w found=false
 *   mistakeStop(): verify idx=14 ABSENT strike 1/2 — needs one more decode before condemning
 *   mistakeStop(): candidate row=17 word=14 sightings=2 … sinceVerify=2825ms   <- 175 ms short
 *   …nothing for 9 s…
 *   stallWatchdog(): candidate idx=14 unresolved for 20025ms (attempts=1 strikes=1) -> forgiving
 *
 * The candidate sat WRONG for 20 s, was forgiven UNPROVEN, and the walk was frozen
 * the whole time — `maxMatched=17/29` on audio that reaches 29.
 *
 * `stallShouldNudge()` lets the 2 s watchdog re-open the confirm loop's dwell
 * gate. It is deliberately conservative: it can only make the pipeline finish a
 * judgement it already started, never start a new accusation.
 *
 * `stallForgiveMsFor()` fixes the second half of the same run: idx=15 was
 * forgiven at 4003 ms because its OWN in-flight decode had debited the budget,
 * and that decode returned `found=true` 1,2 s later — the acquittal was thrown away.
 */
class StallNudgeTest {

    private fun nudge(
        isListening: Boolean = true,
        pipelineRunning: Boolean = true,
        jobIdle: Boolean = true,
        cooldownOver: Boolean = true,
        budgetOk: Boolean = true,
        attempts: Int = 1,
        nudges: Int = 0,
        maxAttempts: Int = 3,
    ) = stallShouldNudge(
        isListening, pipelineRunning, jobIdle, cooldownOver, budgetOk, attempts, nudges, maxAttempts
    )

    // ── stallShouldNudge ──────────────────────────────────────────────────

    @Test
    fun `the measured clean_fatihah_3 stall is nudged`() {
        // attempts=1, strikes=1, job idle, cooldown just expired, budget free.
        assertTrue(nudge())
    }

    @Test
    fun `a candidate with no decode spent yet is nudged`() {
        // The confirm loop may never have run at all (fatihah_c30: a row-index
        // flicker reset the 800 ms dwell pre-condition). attempts == 0 must still
        // get a nudge — that is what the removed last-chance block used to do.
        assertTrue(nudge(attempts = 0))
    }

    @Test
    fun `an active verification job is not nudged`() {
        // A decode is already in flight; the watchdog must not pile on.
        assertFalse(nudge(jobIdle = false))
    }

    @Test
    fun `the cooldown is respected`() {
        assertFalse(nudge(cooldownOver = false))
    }

    @Test
    fun `an exhausted budget is not nudged`() {
        // Nothing to spend — the short-circuit below forgives instead.
        assertFalse(nudge(budgetOk = false))
    }

    @Test
    fun `the attempt budget is respected`() {
        assertFalse(nudge(attempts = 3, maxAttempts = 3))
        assertFalse(nudge(attempts = 4, maxAttempts = 3))
        assertTrue(nudge(attempts = 2, maxAttempts = 3))
    }

    @Test
    fun `nudges are bounded - a nudge the confirm loop never honours cannot freeze forever`() {
        // The §4.18 infinite-freeze class: the watchdog re-opens the gate, the
        // confirm loop refuses for some other reason, `attempts` never increments.
        // Without this bound the candidate would be held indefinitely.
        assertTrue(nudge(attempts = 0, nudges = 3, maxAttempts = 3))
        assertFalse(nudge(attempts = 0, nudges = 4, maxAttempts = 3))
        assertFalse(nudge(attempts = 0, nudges = 9, maxAttempts = 3))
    }

    @Test
    fun `a stopped session is never nudged`() {
        assertFalse(nudge(isListening = false))
        assertFalse(nudge(pipelineRunning = false))
    }

    @Test
    fun `every condition is necessary`() {
        // Flip each one at a time from the all-true baseline. Written out
        // explicitly rather than looped over a heterogeneous array — a mixed
        // Array<Any> of Booleans and Ints needs casts and hides which case failed.
        assertTrue(nudge())
        assertFalse("isListening=false", nudge(isListening = false))
        assertFalse("pipelineRunning=false", nudge(pipelineRunning = false))
        assertFalse("jobIdle=false", nudge(jobIdle = false))
        assertFalse("cooldownOver=false", nudge(cooldownOver = false))
        assertFalse("budgetOk=false", nudge(budgetOk = false))
        assertFalse("attempts == max", nudge(attempts = 3, maxAttempts = 3))
        assertFalse("nudges > max", nudge(nudges = 4, maxAttempts = 3))
    }

    // ── stallForgiveMsFor ─────────────────────────────────────────────────

    @Test
    fun `normally the candidate is held 20 s`() {
        assertEquals(20_000L, stallForgiveMsFor(budgetOk = true, attempts = 1, jobIdle = true))
    }

    @Test
    fun `a budget-blocked candidate with a spent decode is forgiven at 3 s`() {
        assertEquals(3_000L, stallForgiveMsFor(budgetOk = false, attempts = 1, jobIdle = true))
    }

    @Test
    fun `a budget-blocked candidate with NO decode spent still gets the full hold`() {
        // Its verification has simply not started; cutting it off early would
        // forgive a word the pipeline never tried to judge.
        assertEquals(20_000L, stallForgiveMsFor(budgetOk = false, attempts = 0, jobIdle = true))
    }

    @Test
    fun `an IN-FLIGHT decode is not cut off - the clean_fatihah_3 idx=15 bug`() {
        // The nominal cost of an active verification is already debited, so
        // `budgetOk` is false for the whole duration of that decode. Forgiving on
        // that basis discarded a verdict still being computed: idx=15 was forgiven
        // at 4003 ms and its decode returned found=true 1,2 s later.
        assertEquals(20_000L, stallForgiveMsFor(budgetOk = false, attempts = 1, jobIdle = false))
    }

    @Test
    fun `the thresholds are overridable and default to the shipped values`() {
        assertEquals(3_000L, stallForgiveMsFor(budgetOk = false, attempts = 1, jobIdle = true))
        assertEquals(20_000L, stallForgiveMsFor(budgetOk = true, attempts = 1, jobIdle = true))
        assertEquals(500L, stallForgiveMsFor(budgetOk = false, attempts = 1, jobIdle = true, blockedMs = 500L))
        assertEquals(9_000L, stallForgiveMsFor(budgetOk = true, attempts = 1, jobIdle = true, normalMs = 9_000L))
    }
}
