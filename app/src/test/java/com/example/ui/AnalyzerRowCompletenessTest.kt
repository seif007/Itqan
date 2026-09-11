package com.example.ui

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * Row-completeness invariant of [analyzeRecitationDetails].
 *
 * `detailedAnalysis` is consumed as ONE ROW PER (WORD, RULE), and every
 * downstream decision counts DISTINCT `wordIndex` values: `decided`/`unheard`
 * in the end-of-range gate, `finalizeCorrect()`'s trailing-PENDING forgiveness,
 * and the UI's per-word colouring. A target word with NO row at all is
 * therefore invisible to all of them at once.
 *
 * That was the on-device report "the last word is not displayed and yet I have
 * no mistake popup but a summary saying there are no mistakes" (fatihah_c33):
 * the walk ran out of spoken words one short, `break`ed, and left the last
 * target word row-less — `rows=32` for 29 targets (28 distinct indices),
 * `decided=28/29 unheardPending=0`, no `finalizeCorrect()` line, no popup, and
 * `endOfRange(): wrong=0 skipped=0` while the word rendered at the 25 %-opacity
 * "not recited" fallback.
 *
 * A word the walk never reached is UNHEARD: PENDING with `spokenText == null`.
 * It must exist as a row, and it must NOT advance `matched`.
 */
@RunWith(RobolectricTestRunner::class)
class AnalyzerRowCompletenessTest {

    private val riwayah = "Hafs"
    private val target = listOf(
        "بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ", "ٱلرَّحِيمِ", "ٱلْحَمْدُ", "لِلَّهِ", "رَبِّ"
    )

    private fun norm(w: String) = normalizeForComparison(w, riwayah)

    private fun analyze(spoken: List<String>) = runBlocking {
        analyzeRecitationDetails(
            spoken = spoken,
            targetWordsOriginal = target,
            targetWordsNormalized = target.map { norm(it) },
            riwayah = riwayah,
            context = null,
            isTajweedMode = true,
            scoreTajweed = false
        )
    }

    @Test
    fun everyTargetWordHasARow_evenWhenTheTranscriptIsShort() {
        // The transcript ran out 3 words early (the c33 shape).
        val analysis = analyze(target.take(4).map { norm(it) })

        val idxs = analysis.map { it.wordIndex }.toSortedSet().toList()
        assertEquals(
            "every target word must have a row — a row-less word is invisible to " +
                "decided/unheard/finalizeCorrect and renders as 'not recited'",
            (0 until target.size).toList(), idxs
        )

        // The reached prefix is CORRECT; the unreached tail is UNHEARD
        // (PENDING with spokenText == null) so it cannot advance `matched`.
        for (k in 0 until 4) {
            val rows = analysis.filter { it.wordIndex == k }
            assertTrue("word $k should be CORRECT: $rows", rows.all { it.status == RecitationStatus.CORRECT })
        }
        for (k in 4 until target.size) {
            val rows = analysis.filter { it.wordIndex == k }
            assertTrue("word $k must be PENDING: $rows", rows.all { it.status == RecitationStatus.PENDING })
            assertTrue("word $k must be unheard (no spokenText): $rows", rows.all { it.spokenText == null })
        }
    }

    @Test
    fun anEmptyTranscriptStillYieldsEveryRow() {
        val analysis = analyze(emptyList())
        assertEquals(
            (0 until target.size).toList(),
            analysis.map { it.wordIndex }.toSortedSet().toList()
        )
        assertTrue(analysis.all { it.status == RecitationStatus.PENDING && it.spokenText == null })
    }

    @Test
    fun aFullyRecitedTargetHasNoPendingRows() {
        val analysis = analyze(target.map { norm(it) })
        assertTrue(
            "a complete recitation must have no PENDING row: " +
                analysis.filter { it.status == RecitationStatus.PENDING },
            analysis.none { it.status == RecitationStatus.PENDING }
        )
    }
}
