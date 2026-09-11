package com.example.sherpa

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * Regression tests for the verification-decode geometry used by
 * [SherpaPipeline.verifyTailWords].
 *
 * WHY THIS EXISTS — "verifyTailWords starvation" (baqarah_c2, 2026-09-11): the
 * verifier used to hand the recognizer ONE long buffer (10-27 s) of the session
 * tail. This model decodes a ~4 s slice cleanly but hallucinates on long
 * buffers, so the verify decode came back as 4-7 words of filler for spans that
 * should have held 20+ words:
 *
 *   verify idx=1 SKIPPED word='ذَٰلِكَ' verified=6w found=false   (span 10 s)
 *   verify idx=1 INCONCLUSIVE — tail decode (6w, need >=8) saneHits=0
 *   verify idx=1 INCONCLUSIVE — tail decode (7w, need >=8) saneHits=0   (15 s)
 *   verify idx=1 INCONCLUSIVE — tail decode (4w, need >=8) saneHits=0   (21 s)
 *
 * ...while the LIVE 4 s window decoded real words ('فِيهِ هُدًى لِلْمُتَّق') out
 * of the SAME audio. A doubt therefore burned ~20 s and was only ever resolved
 * by the end-of-session repair swap.
 *
 * The fix slices the span into 4 s windows and pools them. These tests pin the
 * two properties that make the pooling correct: the slice plan must always
 * reach the newest audio, and seam-overlapping words must not be double-counted
 * (an inflated count would corrupt the caller's `saneHits`/`need` guards).
 *
 * Robolectric (not a bare JVM runner) for consistency with the neighbouring
 * sherpa tests, which need the Android runtime for `android.util.Log`.
 */
@RunWith(RobolectricTestRunner::class)
class VerifySlicePlanTest {

    // Production geometry: 16 kHz, 4 s slices, 3 s stride (1 s overlap), cap 6.
    private val sr = 16_000
    private val sliceLen = sr * 4
    private val hop = sr * 3
    private val maxSlices = 6

    private fun plan(spanFrames: Int) =
        planVerifySlices(0, spanFrames, sliceLen, hop, maxSlices)

    @Test
    fun `a span shorter than one slice yields a single slice covering all of it`() {
        val p = plan(sr * 2)
        assertEquals(1, p.size)
        assertEquals(0, p[0][0])
        assertEquals(sr * 2, p[0][1])
    }

    @Test
    fun `the last slice always ends exactly at the end of the span`() {
        // The frozen candidate sits in the NEWEST audio, so a plan that stops
        // short of the span end would verify the wrong region entirely.
        for (sec in intArrayOf(4, 5, 10, 15, 21, 22, 27, 30, 45)) {
            val span = sr * sec
            val p = plan(span)
            assertTrue("span ${sec}s produced no slices", p.isNotEmpty())
            assertEquals(
                "span ${sec}s: last slice must end at the span end",
                span, p.last()[1]
            )
        }
    }

    @Test
    fun `a plan capped to a single slice is anchored to the newest audio`() {
        // The cap must never silently discard the region under verification.
        val span = sr * 21
        val p = planVerifySlices(0, span, sliceLen, hop, maxSlices = 1)
        assertEquals(1, p.size)
        assertEquals(span, p[0][1])
        assertEquals(span - sliceLen, p[0][0])
    }

    @Test
    fun `the slice count never exceeds the cap`() {
        for (sec in intArrayOf(4, 10, 21, 45)) {
            assertTrue(
                "span ${sec}s exceeded the cap",
                plan(sr * sec).size <= maxSlices
            )
        }
    }

    @Test
    fun `every slice lies inside the span and is non-empty`() {
        for (sec in intArrayOf(1, 4, 10, 21, 45)) {
            val span = sr * sec
            for (s in plan(span)) {
                assertTrue("slice start ${s[0]} < 0 for span ${sec}s", s[0] >= 0)
                assertTrue("slice end ${s[1]} > $span for span ${sec}s", s[1] <= span)
                assertTrue("empty slice for span ${sec}s", s[1] > s[0])
            }
        }
    }

    @Test
    fun `consecutive slices abut or overlap so no audio is left undecoded`() {
        // A gap between slices would leave words undecoded, and an undecoded
        // word looks ABSENT — false evidence for a condemnation.
        for (sec in intArrayOf(10, 21, 30, 45)) {
            val p = plan(sr * sec)
            assertTrue("span ${sec}s produced < 2 slices", p.size >= 2)
            for (i in 0 until p.size - 1) {
                assertTrue(
                    "span ${sec}s: gap between slice $i (ends ${p[i][1]}) " +
                        "and slice ${i + 1} (starts ${p[i + 1][0]})",
                    p[i + 1][0] <= p[i][1]
                )
            }
        }
    }

    @Test
    fun `an empty or inverted span yields no slices`() {
        assertTrue(plan(0).isEmpty())
        assertTrue(planVerifySlices(500, 400, sliceLen, hop, maxSlices).isEmpty())
    }

    // ── SEAM POOLING ────────────────────────────────────────────────────────

    @Test
    fun `a word repeated at a seam is not counted twice`() {
        val pool = ArrayList(listOf("الم", "ذلك", "الكتاب"))
        appendWithoutOverlap(pool, listOf("الكتاب", "لا", "ريب"))
        assertEquals(listOf("الم", "ذلك", "الكتاب", "لا", "ريب"), pool)
    }

    @Test
    fun `seam dedupe tolerates the diacritic drift between two decodes`() {
        // The two slices decode the same word with different tashkeel and with
        // the alef wasla spelled either way; an exact compare would leave the
        // duplicate in and inflate the count the caller's sanity guard reads.
        val pool = ArrayList(listOf("الم", "ٱلْكِتَابُ"))
        appendWithoutOverlap(pool, listOf("الكتاب", "لَا"))
        assertEquals(3, pool.size)
    }

    @Test
    fun `a genuinely repeated word inside the recitation is preserved`() {
        // الرحمن الرحيم twice: only the seam copy is dropped.
        val pool = ArrayList(listOf("الرحمن", "الرحيم"))
        appendWithoutOverlap(pool, listOf("الرحمن", "الرحيم", "الرحمن", "الرحيم"))
        assertEquals(listOf("الرحمن", "الرحيم", "الرحمن", "الرحيم"), pool)
    }

    @Test
    fun `a non-overlapping slice is appended whole`() {
        val pool = ArrayList(listOf("الم"))
        appendWithoutOverlap(pool, listOf("ذلك", "الكتاب"))
        assertEquals(listOf("الم", "ذلك", "الكتاب"), pool)
    }

    @Test
    fun `an empty slice leaves the pool untouched`() {
        val pool = ArrayList(listOf("الم"))
        appendWithoutOverlap(pool, emptyList())
        assertEquals(listOf("الم"), pool)
    }
}
