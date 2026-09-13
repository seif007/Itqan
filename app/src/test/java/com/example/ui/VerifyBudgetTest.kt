package com.example.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.30 — the rolling-window verification budget.
 *
 * Verification decodes share the SINGLE recognizer slot with the live sliding
 * windows, so unbounded verification starves the transcript (measured
 * fix30_verify_1: 4 decodes, ~60 % of wall time, words 16-29 never decoded).
 * The per-candidate attempt cap could not stop it because a long range presents
 * a FRESH candidate every few seconds and each spent its own attempts.
 *
 * These pin the arithmetic of the roll / allow / decline decision. The budget
 * may only ever WITHHOLD a verification — it must never manufacture a verdict,
 * which is why an exhausted budget simply reports `allowed = false`.
 */
class VerifyBudgetTest {

    private val WIN = 20_000L
    private val MAX = 10_000L

    @Test
    fun `a fresh budget allows the first request`() {
        val (start, spent, allowed) = verifyBudgetAfterRequest(
            windowStartMs = 0L, spentMs = 0L, nowMs = 1_000L,
            requestMs = 4_000L, windowMs = WIN, maxMs = MAX,
        )
        assertTrue("first request must be allowed", allowed)
        assertEquals("window must start at now", 1_000L, start)
        assertEquals(4_000L, spent)
    }

    @Test
    fun `allowed spends accumulate inside one window`() {
        // Two 4 s decodes fit in a 10 s budget.
        val (s1, sp1, a1) = verifyBudgetAfterRequest(0L, 0L, 1_000L, 4_000L, WIN, MAX)
        assertTrue(a1)
        val (s2, sp2, a2) = verifyBudgetAfterRequest(s1, sp1, 5_000L, 4_000L, WIN, MAX)
        assertTrue("second decode fits (4+4 <= 10)", a2)
        assertEquals(1_000L, s2)
        assertEquals(8_000L, sp2)
    }

    @Test
    fun `a third decode in the same window is declined`() {
        // 8 s already spent; a third 4 s request would be 12 s > 10 s.
        val (start, spent, allowed) = verifyBudgetAfterRequest(
            windowStartMs = 1_000L, spentMs = 8_000L, nowMs = 9_000L,
            requestMs = 4_000L, windowMs = WIN, maxMs = MAX,
        )
        assertFalse("a third decode must be declined", allowed)
        assertEquals("a declined request must not spend", 8_000L, spent)
        assertEquals("a declined request must not move the window", 1_000L, start)
    }

    @Test
    fun `an expired window rolls and resets the spend`() {
        // Window opened at 1_000; at 21_001 it is 20_001 ms old => expired.
        val (start, spent, allowed) = verifyBudgetAfterRequest(
            windowStartMs = 1_000L, spentMs = 10_000L, nowMs = 21_001L,
            requestMs = 4_000L, windowMs = WIN, maxMs = MAX,
        )
        assertTrue("after the roll the budget is fresh again", allowed)
        assertEquals("the window restarts at now", 21_001L, start)
        assertEquals("spend resets to just this request", 4_000L, spent)
    }

    @Test
    fun `the window does NOT roll one millisecond early`() {
        // 19_999 ms old: still the same window, so a 4 s request over 8 s spent
        // must be declined rather than silently starting a new window.
        val (start, spent, allowed) = verifyBudgetAfterRequest(
            windowStartMs = 1_000L, spentMs = 8_000L, nowMs = 20_999L,
            requestMs = 4_000L, windowMs = WIN, maxMs = MAX,
        )
        assertFalse(allowed)
        assertEquals(1_000L, start)
        assertEquals(8_000L, spent)
    }

    @Test
    fun `a request exactly at the cap is allowed`() {
        // 6 s spent + 4 s = 10 s == MAX, which must be allowed (<=).
        val (_, spent, allowed) = verifyBudgetAfterRequest(
            windowStartMs = 1_000L, spentMs = 6_000L, nowMs = 5_000L,
            requestMs = 4_000L, windowMs = WIN, maxMs = MAX,
        )
        assertTrue("spending exactly up to the cap is allowed", allowed)
        assertEquals(10_000L, spent)
    }

    @Test
    fun `a negative request is clamped to zero and never refunds`() {
        val (_, spent, allowed) = verifyBudgetAfterRequest(
            windowStartMs = 1_000L, spentMs = 10_000L, nowMs = 5_000L,
            requestMs = -5_000L, windowMs = WIN, maxMs = MAX,
        )
        assertTrue("a zero-cost request is always affordable", allowed)
        assertEquals("a negative request must not refund", 10_000L, spent)
    }

    @Test
    fun `the gate probe reserves a nominal cost so back-to-back launches cannot over-commit`() {
        // 8 s spent of a 10 s cap: the probe must already report NO, because
        // reserving the nominal 4 s would exceed the cap.
        assertFalse(
            verifyBudgetAllowsMs(1_000L, 8_000L, 5_000L, WIN, MAX)
        )
        // 4 s spent: reserving 4 s leaves 2 s of headroom => allowed.
        assertTrue(
            verifyBudgetAllowsMs(1_000L, 4_000L, 5_000L, WIN, MAX)
        )
    }

    @Test
    fun `the probe rolls an expired window just like the spend does`() {
        assertTrue(
            "a stale window must not block the next verification forever",
            verifyBudgetAllowsMs(1_000L, 10_000L, 999_999L, WIN, MAX)
        )
    }
}
