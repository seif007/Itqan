package com.example.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Deterministic JVM harness for the Memorize collector's transcript logic in
 * RecitationApp.kt (the `pipe.partials.collect` block). There is no connected
 * phone and no audio assets on this machine, so instead of feeding real
 * audio through the Sherpa decoder we replay the DECODER'S EMISSION STREAM --
 * the exact input the collector consumes -- and assert the transcript
 * invariants the mistake-stop / resume-from-mistake feature depends on:
 *
 *  1. cumulative emissions never duplicate the committed transcript,
 *  2. a WRONG/SKIPPED auto-pause snapshots the live transcript, and the next
 *     fresh session re-seats the transcript to the words BEFORE the mistake
 *     (lastCommittedWords = that SAME prefix, so re-read words pass through),
 *  3. after a resume the pipeline has reset its emission log — emissions
 *     contain ONLY new audio; the count filter suppresses any replay copies
 *     of committed words while genuinely re-read words pass through
 *     ("I repeat the word and nothing happens" must not regress),
 *  4. the final transcript after a resume contains each word exactly once,
 *  5. repeated words within an ayah survive recommit across emissions,
 *  6. a mic re-tap without a mistake CONTINUES from the committed frontier
 *     (no jump back to the start, no wipe of a partially-read range).
 *
 * The harness below is a VERBATIM MIRROR of the collector: filterMemorizeWords
 * (committed-count suppression + target-count cap) and the freshSession
 * conditional wipe + re-seat. If you change the collector, update this
 * harness and re-run `gradlew testDebugUnitTest`.
 */
class MemorizeTranscriptTrackerTest {

    private val riwayah = "Hafs"

    // Al-Fatiha-like target: note الرَّحْمَٰنِ/الرَّحِيمِ repeated across the
    // first two positions of the second emission in some scenarios.
    private val target = listOf(
        "بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ", "ٱلرَّحِيمِ", "ٱلْحَمْدُ", "لِلَّهِ", "رَبِّ", "ٱلْعَٰلَمِينَ"
    )

    private fun norm(w: String) = normalizeForComparison(w, riwayah)

    private fun assertWordsOnce(all: List<String>) {
        val norms = all.map { norm(it) }
        val distinct = norms.toSet()
        // The phrase "لِلَّهِ" only appears once in Al-Fatiha word lists here;
        // assert no norm is duplicated anywhere in the combined transcript.
        assertEquals("duplicate norm in transcript: $all", distinct.size, norms.size)
    }

    // ── Test 1: clean cumulative session, mid-session endpoint, no duplicates ──
    @Test
    fun cleanSession_neverDuplicatesAndFinalMatches() {
        val sim = CollectorSim(riwayah, target)
        // growth before any endpoint
        sim.onEmission(listOf("بِسْمِ"), endpoint = false, freshSession = true)
        assertEquals(listOf("بِسْمِ"), sim.spokenText)
        sim.onEmission(listOf("بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ"), endpoint = false, freshSession = false)
        assertEquals(listOf("بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ"), sim.spokenText)
        // real sherpa endpoint: commits 0..3, keeps the cursor
        sim.onEmission(listOf("بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ", "ٱلرَّحِيمِ"), endpoint = true, freshSession = false)
        assertEquals(4, sim.committedWordCount)
        assertTrue(sim.sessionFinalized)
        // post-endpoint cumulative emission still shows only the NEW tail
        sim.onEmission(
            listOf("بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ", "ٱلرَّحِيمِ", "ٱلْحَمْدُ", "لِلَّهِ"),
            endpoint = false,
            freshSession = false
        )
        assertEquals(listOf("بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ", "ٱلرَّحِيمِ", "ٱلْحَمْدُ", "لِلَّهِ"), sim.spokenText)
        // the final endpoint commits the still-audible tail (realistic here:
        // the session's last ~10 s contain only the uncommitted words), and
        // the endpoint keeps ALL of it in order (repeats survive)
        sim.onEmission(
            listOf("ٱلْحَمْدُ", "لِلَّهِ", "رَبِّ", "ٱلْعَٰلَمِينَ"),
            endpoint = true,
            freshSession = false
        )
        assertEquals(target.map { norm(it) }, sim.spokenText.map { norm(it) })
        assertWordsOnce(sim.spokenText)
    }

    // ── Test 2: the resume-from-mistake flow (the feature this session) ──
    @Test
    fun mistakeResume_repeatsTheWrongWordAndFinishesClean() {
        val sim = CollectorSim(riwayah, target)
        // Phase 1: recite words 0..4, the STT frontier outruns the user and the
        // analysis marks word 5 (الرَّحِيمِ) WRONG -> auto-pause.
        sim.onEmission(listOf("بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ", "ٱلرَّحِيمِ", "ٱلْحَمْدُ", "لِلَّهِ", "رَبِّ"),
            endpoint = false, freshSession = true)
        sim.onEmission(listOf("بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ", "ٱلرَّحِيمِ", "ٱلْحَمْدُ", "لِلَّهِ", "رَبِّ", "ٱلْعَٰلَمِينَ"),
            endpoint = true, freshSession = false)
        assertEquals(8, sim.committedWordCount)
        // mistakeStop confirmed at wordIndex 5: snapshot live = final + partial
        val live = sim.spokenText
        assertEquals(8, live.size)
        sim.mistakeResumeLiveText = live.joinToString(" ")
        sim.mistakeResumeIndex = 5
        // Phase 2: "continue" -> fresh session -> re-seat to prefix BEFORE word 5.
        // lastCommittedWords becomes the PREFIX itself (not the whole pre-pause
        // transcript): the count filter then suppresses only the prefix words'
        // replay copies, and the re-read mistake words pass through.
        sim.onEmission(emptyList(), endpoint = false, freshSession = true)
        assertEquals(5, sim.spokenText.size)
        assertEquals(target.subList(0, 5).map { norm(it) }, sim.spokenText.map { norm(it) })
        assertEquals(5, sim.lastCommittedWords.size)
        assertEquals(0, sim.committedWordCount)
        // Phase 3: resume() reset the pipeline's emission log — the emission
        // carries ONLY the new audio. The user re-reads the mistake word and
        // the next one; both are NEW relative to the prefix and must extend
        // the transcript (the old whole-transcript cursor ate these).
        sim.onEmission(listOf("لِلَّهِ", "رَبِّ"), endpoint = false, freshSession = false)
        assertEquals(7, sim.spokenText.size)
        // Phase 4: cumulative growth inside the new utterance re-includes the
        // just-spoken words — the committed-prefix suppression must not eat
        // them and the live display stays monotone.
        sim.onEmission(listOf("لِلَّهِ", "رَبِّ", "ٱلْعَٰلَمِينَ"), endpoint = false, freshSession = false)
        assertEquals(8, sim.spokenText.size)
        // Phase 5: the final endpoint decodes the new utterance — the re-read
        // audio only (لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ) — and commits it once.
        sim.onEmission(
            listOf("لِلَّهِ", "رَبِّ", "ٱلْعَٰلَمِينَ"),
            endpoint = true, freshSession = false
        )
        assertEquals(8, sim.spokenText.size)
        assertTrue(sim.sessionFinalized)
        `assertNoDuplicatesAndPositionsMatch`(sim, target)
    }

    // ── Test 3: repeat of word 5 ends the transcript exactly once ──
    @Test
    fun mistakeResume_finalTranscriptContainsEachWordOnce() {
        val sim = CollectorSim(riwayah, target)
        sim.onEmission(target.take(6), endpoint = false, freshSession = true)
        sim.onEmission(target, endpoint = true, freshSession = false)
        val live = sim.spokenText
        sim.mistakeResumeLiveText = live.joinToString(" ")
        sim.mistakeResumeIndex = 3
        sim.onEmission(emptyList(), endpoint = false, freshSession = true)
        assertEquals(3, sim.spokenText.size)
        // Re-read from word 3 to the end: the new utterance's emissions carry
        // only the re-read audio; the live partial extends the prefix...
        sim.onEmission(target.subList(3, target.size), endpoint = false, freshSession = false)
        val reRead = sim.spokenText
        assertEquals("re-read extends the prefix transcript", target.size, reRead.size)
        // ...and the final endpoint decodes the re-read audio ONLY (words 3..7)
        sim.onEmission(target.subList(3, target.size), endpoint = true, freshSession = false)
        assertWordsOnce(sim.spokenText)
        assertEquals(target.size, sim.spokenText.size)
    }

    // ── Test 4: repeated word across emissions survives recommit ──
    @Test
    fun repeatedWordInAyah_isNotDroppedAfterCommit() {
        // "كَلَّا" style repetition: the target contains رَبِّ twice (0 and 4)
        val t = listOf("رَبِّ", "ٱلْحَمْدُ", "لِلَّهِ", "رَبِّ", "ٱلْعَٰلَمِينَ", "رَبِّ")
        val sim = CollectorSim(riwayah, t)
        // first emission commits words 0..2 (incl. the FIRST رَبِّ)
        sim.onEmission(t.subList(0, 3), endpoint = false, freshSession = true)
        sim.onEmission(t.subList(0, 3), endpoint = true, freshSession = false)
        // second emission replays 0..2 and adds the SECOND رَبِّ (index 3)
        sim.onEmission(t.subList(0, 4), endpoint = false, freshSession = false)
        val spoken = sim.spokenText.map { norm(it) }
        assertEquals("second occurrence of رَبِّ survives the count-based dedup",
            listOf(0, 1, 2, 3).map { t[it] }.map { norm(it) }, spoken)
    }

    // ── Test 5: explicit restart (markers cleared) continues from the frontier ──
    @Test
    fun explicitRestart_clearedMarkersDoNotReseat() {
        val sim = CollectorSim(riwayah, target)
        sim.onEmission(target.take(4), endpoint = false, freshSession = true)
        sim.onEmission(target.take(4), endpoint = true, freshSession = false)
        sim.mistakeResumeLiveText = sim.spokenText.joinToString(" ")
        sim.mistakeResumeIndex = 2
        // the user pressed "Restart" / changed range -> markers cleared
        sim.mistakeResumeIndex = -1
        sim.mistakeResumeLiveText = ""
        // A fresh session of a PARTIALLY-read range does NOT wipe: the display
        // continues from the committed frontier (the old unconditional wipe
        // made every mic re-tap "jump back to the start").
        sim.onEmission(listOf("ٱلْحَمْدُ"), endpoint = false, freshSession = true)
        assertEquals(5, sim.spokenText.size)
        assertEquals(target.take(5).map { norm(it) }, sim.spokenText.map { norm(it) })
    }

    // ── Test 6: pause/resume WITHOUT a mistake still shows the re-read words ──
    @Test
    fun normalPauseResume_doesNotLookDeaf() {
        val sim = CollectorSim(riwayah, target)
        sim.onEmission(target.take(4), endpoint = false, freshSession = true)
        sim.onEmission(target.take(5), endpoint = true, freshSession = false)
        assertEquals(target.take(5).map { norm(it) }, sim.spokenText.map { norm(it) })
        // user pauses, taps the mic again: no wipe of the partial range —
        // the committed frontier stays visible.
        sim.onEmission(emptyList(), endpoint = false, freshSession = true)
        assertEquals(5, sim.spokenText.size)
        // ...and continues reciting the NEXT word; it must pass the filter
        // (it is not in the committed transcript) and extend the display.
        sim.onEmission(listOf("لِلَّهِ"), endpoint = false, freshSession = false)
        assertEquals(target.take(6).map { norm(it) }, sim.spokenText.map { norm(it) })
    }

    private fun `assertNoDuplicatesAndPositionsMatch`(sim: CollectorSim, expected: List<String>) {
        val combined = sim.spokenText.map { norm(it) }
        val expectedNorms = expected.map { norm(it) }
        assertWordsOnce(sim.spokenText)
        // words 0..4 (prefix) are intact, then the re-read tail must align
        // positionally with the expected target (norm equality, our matcher
        // strips diacritics so this holds for exact re-reads).
        val prefixEnd = 5
        assertEquals(expectedNorms.subList(0, prefixEnd), combined.subList(0, prefixEnd))
        assertEquals(expectedNorms, combined)
    }
}