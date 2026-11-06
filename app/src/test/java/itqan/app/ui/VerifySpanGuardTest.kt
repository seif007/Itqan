package itqan.app.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the verification guard that stops a false SKIPPED.
 *
 * The user's rule (MEMORY.md): **a false "no mistake" costs nothing, a false
 * mistake popup accuses the user of a recitation error.** On a real device a
 * correctly-recited ٱهْدِنَا was condemned as SKIPPED and auto-paused the session,
 * even though the end-of-session decode plainly heard it.
 *
 * `poolSpansCandidate` is the guard that makes "the word is absent from the
 * decode" meaningful: the decode must have actually COVERED the candidate's
 * position. See its KDoc for the full measured evidence trail.
 */
class VerifySpanGuardTest {

    /** The pool positions from the REAL failing run (fix27_fatihah_1, attempt 1). */
    private val realFailingPool = listOf(
        20, 21, 22, 22, 23, 24, 25, 25, 26, 27, 28, -1
    )

    @Test
    fun `the real false-skipped pool does NOT span word 17`() {
        // The 4 slices decoded words 20-28 only; ٱهْدِنَا's own audio was never in
        // the span, so "absent" said nothing about it. This MUST be rejected.
        assertFalse(
            "the pool that produced the false SKIPPED must not count as spanning it",
            poolSpansCandidate(realFailingPool, 17),
        )
    }

    @Test
    fun `a pool that really covers the candidate DOES span it`() {
        // Words either side of 17, genuinely adjacent in the pool.
        assertTrue(poolSpansCandidate(listOf(13, 14, 15, 16, 18, 19, 20), 17))
        // The candidate's immediate neighbour alone is enough.
        assertTrue(poolSpansCandidate(listOf(16), 17))
        assertTrue(poolSpansCandidate(listOf(18), 17))
        assertTrue(poolSpansCandidate(listOf(-1, 16, -1), 17))
        assertTrue(poolSpansCandidate(listOf(-1, 18, -1), 17))
    }

    @Test
    fun `a pool that only reaches one side does NOT span the candidate`() {
        // All before the candidate, and NOT touching it -> nothing is said about 17.
        assertFalse(poolSpansCandidate(listOf(10, 11, 12, 13, 14, 15), 17))
        // All after the candidate, and NOT touching it -> likewise.
        assertFalse(poolSpansCandidate(listOf(19, 20, 21, 22, 23), 17))
    }

    @Test
    fun `a decoded word immediately beside the candidate IS enough`() {
        // The candidate's own clip may simply be the one adjacent to it, so a
        // pool that reaches word 16 (or 18) has covered the candidate's region.
        assertTrue(poolSpansCandidate(listOf(10, 11, 12, 13, 14, 15, 16), 17))
        assertTrue(poolSpansCandidate(listOf(18, 19, 20, 21), 17))
    }

    @Test
    fun `a pool that JUMPS across the candidate does NOT count as spanning it`() {
        // The old `sawBefore && sawAfter` guard accepted exactly this shape: an
        // early word (9) and a late word (28) with nothing decoded in between.
        assertFalse(
            "a 9 -> 28 jump says nothing about word 17",
            poolSpansCandidate(listOf(9, 28), 17),
        )
        assertFalse(
            "an out-of-order/displaced pool must not straddle",
            poolSpansCandidate(listOf(28, 9), 17),
        )
        // Just past the allowed gap is still a jump.
        assertFalse(poolSpansCandidate(listOf(5, 17), 17, maxGap = 10))
        // Within the gap is a genuine straddle.
        assertTrue(poolSpansCandidate(listOf(16, 17), 17, maxGap = 10))
    }

    @Test
    fun `an all-garbage pool never spans anything`() {
        assertFalse(poolSpansCandidate(listOf(-1, -1, -1, -1), 17))
        assertFalse(poolSpansCandidate(emptyList(), 17))
    }

    @Test
    fun `degenerate indices are handled`() {
        // -1 candidate (no candidate resolved) -> never spans.
        assertFalse(poolSpansCandidate(listOf(0, 1, 2), -1))
        // Word 0 and the last word are edge cases the caller special-cases; the
        // guard itself must still behave sanely.
        assertFalse(poolSpansCandidate(listOf(5, 6, 7), 0))
        assertTrue(poolSpansCandidate(listOf(1), 0))
    }
}
