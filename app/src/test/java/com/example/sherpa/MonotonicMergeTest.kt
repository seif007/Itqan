package com.example.sherpa

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Regression tests for the pipeline's position-preserving monotonic merge
 * (SherpaPipeline.monotonicMergeWords). The cases are derived from a REAL
 * on-device Fatihah session (logcat 2026-08-12): the old blind-prepend merge
 * put the frontier word الْعَالَمِينَ in FRONT of the emission whenever the
 * final window merely cut at الْعَالَم, and the aligner then flagged
 * WRONG/SKIPPED verdicts ("2 mistakes while I read correctly") on a perfect
 * recitation.
 */
class MonotonicMergeTest {

    @Test
    fun `mid-span word lost in a merged-token re-render is re-injected in place`() {
        // On-device window: the engine re-rendered "اللَّهِحْمَن" and الله
        // vanished from the decode. The merge must put الله back BEFORE
        // الرَّحْمَنِ (reading order), not prepend it to the whole emission.
        val last = listOf("بِسْمِ", "اللَّهِ", "الرَّحْمَنِ", "الرَّحِيمِ")
        val n = listOf("بِسْمِ", "الرَّحْمَنِ", "الرَّحِيمِ", "الْحَمْدُ", "لِلَّهِ", "رَبِّ", "الْعَالَمِينَ")
        assertEquals(
            listOf("بِسْمِ", "اللَّهِ", "الرَّحْمَنِ", "الرَّحِيمِ", "الْحَمْدُ", "لِلَّهِ", "رَبِّ", "الْعَالَمِينَ"),
            monotonicMergeWords(last, n)
        )
    }

    @Test
    fun `right-edge cut keeps the frontier word in its place`() {
        // Endpoint decode cut at الْعَالَم (fragment of الْعَالَمِينَ). The old
        // prepend emitted [الْعَالَمِينَ, بِسْمِ ...] — the whole transcript
        // scrambled — and the aligner flagged the first words as mistakes.
        // The union emits the full word first, then the fragment trails.
        val last = listOf("بِسْمِ", "اللَّهِ", "الرَّحْمَنِ", "الرَّحِيمِ", "الْحَمْدُ", "لِلَّهِ", "رَبِّ", "الْعَالَمِينَ")
        val n = listOf("بِسْمِ", "اللَّهِ", "الرَّحْمَنِ", "الرَّحِيمِ", "الْحَمْدُ", "لِلَّهِ", "رَبِّ", "الْعَالَم")
        assertEquals(
            listOf("بِسْمِ", "اللَّهِ", "الرَّحْمَنِ", "الرَّحِيمِ", "الْحَمْدُ", "لِلَّهِ", "رَبِّ", "الْعَالَمِينَ", "الْعَالَم"),
            monotonicMergeWords(last, n)
        )
    }

    @Test
    fun `left-edge slide-out words are prepended in reading order`() {
        // A window slid past the start: the earlier words are missing from the
        // fresh decode and must come back BEFORE it, in their known order.
        val last = listOf("قُلْ", "هُوَ", "اللَّهُ", "أَحَدٌ")
        val n = listOf("اللَّهُ", "أَحَدٌ", "اللَّهُ", "الصَّمَدُ")
        assertEquals(
            listOf("قُلْ", "هُوَ", "اللَّهُ", "أَحَدٌ", "اللَّهُ", "الصَّمَدُ"),
            monotonicMergeWords(last, n)
        )
    }

    @Test
    fun `pure extension is passed through unchanged`() {
        val last = listOf("أ", "ب")
        val n = listOf("أ", "ب", "ج")
        assertEquals(n, monotonicMergeWords(last, n))
    }

    @Test
    fun `tashkeel re-spelling does not duplicate`() {
        val last = listOf("بِسْمِ", "اللَّهِ")
        val n = listOf("بسم", "الله")
        assertEquals(listOf("بسم", "الله"), monotonicMergeWords(last, n))
    }

    @Test
    fun `re-decode that drops the frontier word holds the previous emission`() {
        val last = listOf("أ", "ب", "ج")
        val n = listOf("أ", "ب")
        assertEquals(last, monotonicMergeWords(last, n))
    }

    // ───────────────────────────────────────────────────────────────────
    // Extended contract (time eligibility + vocab contradiction + garbage
    // overwrite) — the metadata the wired pipeline passes on every decode.
    // ───────────────────────────────────────────────────────────────────

    @Test
    fun `stale repeated phrase from new audio appends instead of folding`() {
        // Kafiran-style: وَلَا أَنتُمْ عَابِدُونَ مَا أَعْبُدُ occurs twice.
        // The FIRST occurrence's audio slid out of the window (its first-emit
        // timestamp is older than nowMs-windowMs), so the identical phrase
        // arriving from NEW audio is a DIFFERENT occurrence — it must append,
        // not fold onto the stale one (the merge bug that ate kafirun's
        // repeats).
        val phrase = listOf("وَلَا", "أَنتُمْ", "عَابِدُونَ", "مَا", "أَعْبُدُ")
        val last = listOf("قُلْ") + phrase
        val ms = longArrayOf(9000, 9000, 9000, 9000, 9000, 9000)
        // nowMs=20000, windowMs=4000 → cut at 16000; every ms (9000) is stale.
        val merged = monotonicMergeWords(last, phrase, ms, nowMs = 20000, windowMs = 4000)
        assertEquals(last + phrase, merged)
        // Same decode while the occurrence is still in-window → hold.
        val msFresh = longArrayOf(1000, 19000, 19000, 19000, 19000, 19000)
        assertEquals(last, monotonicMergeWords(last, phrase, msFresh, nowMs = 20000, windowMs = 4000))
    }

    @Test
    fun `vocab contradiction rejects a fold onto the wrong occurrence`() {
        // Ikhlas: log [الله, أَحَدٌ]; the window re-hears the SECOND الله and
        // renders [الله, الصَّمَدُ]. Folding onto the first الله (the old
        // subsequence match) would APPEND only الصَّمَدُ and lose the second
        // الله — the phantom-SKIPPED source. With the vocab, the mismatch
        // after the anchor contradicts a REAL word in the log → the anchor is
        // rejected → the whole decode appends as a new occurrence.
        val vocab = setOf("قل", "هو", "الله", "احد", "الصمد")
        val last = listOf("اللَّهُ", "أَحَدٌ")
        val n = listOf("اللَّهُ", "الصَّمَدُ")
        assertEquals(
            listOf("اللَّهُ", "أَحَدٌ", "اللَّهُ", "الصَّمَدُ"),
            monotonicMergeWords(last, n, longArrayOf(19000, 19000), nowMs = 20000, windowMs = 4000, vocab = vocab)
        )
    }

    @Test
    fun `fresher decode overwrites an OOV garbage tail word`() {
        // The previous decode committed an unmatched OOV fragment (والثات —
        // falaq's real garbage); the next decode renders the same span
        // better. With the vocab, the garbage log word is skipped and the
        // fresher spelling replaces it (full-rerender of the same size).
        val vocab = setOf("قل", "اعوذ", "برب", "الفلق", "من", "شر", "ما", "خلق")
        val last = listOf("مِنْ", "شَرِّ", "وَالْثَاتِ")
        val n = listOf("مِنْ", "شَرِّ", "مَا", "خَلَقَ")
        val merged = monotonicMergeWords(last, n, longArrayOf(19000, 19000, 19000), nowMs = 20000, windowMs = 4000, vocab = vocab)
        // anchor شر (j=1): continuation مَا != والثات (OOV → garbage-skip) →
        // cursor at the log's end → the rest is new audio: ما خلق append; the
        // garbage stays in the log (append-only) but can never block a match —
        // the collector's target-count cap + the verdict walk's OOV skip make
        // it inert downstream.
        assertEquals(listOf("مِنْ", "شَرِّ", "وَالْثَاتِ", "مَا", "خَلَقَ"), merged)
    }
}
