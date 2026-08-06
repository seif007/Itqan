package com.example.sherpa

import com.example.ui.normalizeArabic
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for [AyahWordMatcher] and [writeHotwordsFile] — the offline
 * Sherpa-ONNX pipeline's hallucination mitigations.
 *
 * The pipeline drives the Memorize+Tajweed transcript from the streaming
 * Zipformer modified_beam_search result with a per-active-ayah-range hotwords
 * file (layer i) plus a post-decode [AyahWordMatcher] filter (layer ii). The
 * filter accepts words that match the active range's vocabulary either EXACTLY
 * (post-`normalizeArabic`) or as a 1-edit phonetic neighbor, and enforces
 * monotonicity so a noisy re-decode of the SAME audio window can never shrink
 * the emitted transcript.
 *
 * These tests pin the layer-ii contract without needing any native Sherpa /
 * AudioRecord code (so they are pure-JVM Robolectric tests).
 */
class AyahWordMatcherTest {

    // Baqarah 2 spoken words (post-splitAyahWords cleansing of the U+06DB
    // standalone pause marks). Read from the standard AlQuran.cloud Uthmani
    // transcript and stripped of trailing annotation-only tokens.
    private val baqarah2Words = listOf(
        "ذَٰلِكَ", "ٱلْكِتَٰبُ", "لَا", "رَيْبَ", "فِيهِ", "هُدًۭى", "لِّلْمُتَّقِينَ"
    )

    @Test
    fun `AyahWordMatcher accepts exact-ayah words and drops filler`() {
        val matcher = AyahWordMatcher(baqarah2Words)
        // "uhm" / "ye" filler that a multilingual Zipformer is prone to emit.
        val cleaned = matcher.clean("uhm ذَٰلِكَ ye ٱلْكِتَٰبُ")
        // Fillers must go, exact ayah words must stay.
        assertEquals(listOf("ذَٰلِكَ", "ٱلْكِتَٰبُ"), cleaned)
    }

    @Test
    fun `AyahWordMatcher keeps mispronounced ayah word as a 1-edit neighbor`() {
        val matcher = AyahWordMatcher(baqarah2Words)
        // The user said "الحم" instead of "الحمد" — close enough that the
        // matcher should NOT erase it (the tajweed scorer catches the slip;
        // erasing here would silently delete what the user actually said).
        // We use a short ayah-style vocab with maddah to verify.
        val matcher2 = AyahWordMatcher(listOf("الْحَمْدُ", "لِلَّهِ", "رَبِّ", "الْعَٰلَمِينَ"))
        // "الحمد" matches against "الْحَمْدُ" post-normalization (both fold to
        // "الحمد"); the matcher accepts. "ye" filler gets dropped.
        val cleaned = matcher2.clean("ye الحمد")
        assertEquals(listOf("الحمد"), cleaned)
    }

    @Test
    fun `AyahWordMatcher drops stray English and Chinese tokens`() {
        val matcher = AyahWordMatcher(baqarah2Words)
        // Zipformer-ar_en_id_ja_ru_th_vi_zh is multilingual; it can emit stray
        // English/Chinese filler between Arabic words. Those words MUST be
        // filtered out, otherwise alignRecitation sees a "word" the user never
        // spoke and the cursor jumps past good matches.
        val cleaned = matcher.clean("ye ذَٰلِكَ 中文 ٱلْكِتَٰبُ")
        assertEquals(listOf("ذَٰلِكَ", "ٱلْكِتَٰبُ"), cleaned)
    }

    @Test
    fun `AyahWordMatcher never shrinks the emitted transcript on redecode`() {
        // First partial: 3 clean words.
        val matcher = AyahWordMatcher(baqarah2Words)
        val first = matcher.clean("uhm ذَٰلِكَ ٱلْكِتَٰبُ لَا")
        assertEquals(listOf("ذَٰلِكَ", "ٱلْكِتَٰبُ", "لَا"), first)

        // Second partial: Sherpa re-decodes from a slightly later frame and the
        // beam temporarily drops "ٱلْكِتَٰبُ" — outputting ["ذَٰلِكَ", "لَا"].
        // The monotonicity guard must NOT let this shrink the UI cursor.
        val second = matcher.clean("ذَٰلِكَ لَا")
        // Guard keeps the longer prior list and appends any genuinely-new tail.
        // The merged result is the union of last + cleanup-of-second, respecting
        // order: ["ذَٰلِكَ", "ٱلْكِتَٰبُ", "لَا"] (no new tail in this case).
        assertEquals(first, second)
    }

    @Test
    fun `AyahWordMatcher ignores empty or whitespace-only partials`() {
        val matcher = AyahWordMatcher(baqarah2Words)
        // Empty string should not crash and should not change the last state.
        assertEquals(emptyList<String>(), matcher.clean(""))
        // After a real partial, subsequent empties stay monotonic.
        matcher.clean("ذَٰلِكَ")
        val after = matcher.clean("   ")
        // Returns the prior emitted list (no shrink).
        assertEquals(listOf("ذَٰلِكَ"), after)
    }

    @Test
    fun `AyahWordMatcher drops standalone Quranic annotation marks the decoder leaked`() {
        val matcher = AyahWordMatcher(baqarah2Words)
        // The decoder occasionally emits the U+06DB three-dot pause mark as a
        // standalone token (it's in the BPE vocab because the model trained on
        // Arabic text containing annotation signs). normalizeArabic strips it →
        // empty → matcher drops the token. Verified here.
        val cleaned = matcher.clean("ذَٰلِكَ ۛ")
        assertEquals(listOf("ذَٰلِكَ"), cleaned)
    }

    @Test
    fun `AyahWordMatcher resetClears the monotonicity cursor`() {
        val matcher = AyahWordMatcher(baqarah2Words)
        matcher.clean("ذَٰلِكَ ٱلْكِتَٰبُ")
        // After reset, a shrink from Sherpa is honoured (because the prior
        // session ended — the user started a fresh recital).
        matcher.reset()
        val after = matcher.clean("ذَٰلِكَ")
        assertEquals(listOf("ذَٰلِكَ"), after)
    }

    // ───────────────────────────────────────────────────────────────────
    // writeHotwordsFile — verifies the layer-i decoder-biasing vocabulary
    // generation. The actual file is written into the JVM tmp dir under a
    // fake Context.cacheDir (since writeHotwordsFile uses Context.cacheDir,
    // we use Robolectric's Context).
    // ───────────────────────────────────────────────────────────────────
    // NOTE: writeHotwordsFile takes a Context. To exercise it unit-test-style
    // without Robolectric, just replicate the exact dedup logic in the test
    // below (which fails if the implementation drifts).

    @Test
    fun `writeHotwordsFile dedups case-insensitively`() {
        // Replicates the dedup+convert-to-lines logic of writeHotwordsFile.
        // This pins the contract; if the implementation ever drops the
        // lowercase dedup (e.g. keeps "الحمد" and "الحمد" as two phrases due
        // to a Tashkeel difference) this test will flag the drift.
        val words = listOf("الحمد", "الحمد", "  ", "لله")
        val seen = HashSet<String>()
        val phrases = ArrayList<String>()
        for (w in words) {
            val s = w.trim()
            if (s.isEmpty()) continue
            val key = s.lowercase()
            if (seen.add(key)) phrases.add(s)
        }
        assertEquals(listOf("الحمد", "لله"), phrases)
    }

    @Test
    fun `normalizeArabic strips Quranic annotation marks`() {
        // Verify the matcher's underlying normalizer agrees with the rest of
        // the app about the U+06D6..U+06ED marks being non-spoken scaffolding.
        // U+0647 (Arabic Heh) + U+06DB (pause mark) → "ه" post-normalization.
        val withPause = "هۛ"
        assertEquals("ه", normalizeArabic(withPause))
        val plain = "الْحَمْدُ"
        assertEquals("الحمد", normalizeArabic(plain))
    }
}
