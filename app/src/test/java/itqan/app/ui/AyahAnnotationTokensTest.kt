package itqan.app.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for [splitAyahWords] and [isAyahAnnotationOnlyToken].
 *
 * The AlQuran.cloud Uthmani text (and the curated Qalun words_uthmani arrays) interleaves
 * spoken words with freestanding Quranic annotation signs (U+06D6..U+06ED), e.g. the
 * "three-dot" stop mark U+06DB in Surah Al-Baqarah 2:
 *   `ذَٰلِكَ ٱلْكِتَٰبُ لَا رَيْبَ ۛ فِيهِ ۛ هُدًۭى لِّلْمُتَّقِينَ`
 *
 * These marks are not recited -- they are structural pause cues. But `ayah.text.split("\\s+")`
 * kept them as their own whitespace-separated token, and `normalizeArabic` then stripped every
 * character leaving an empty string. The empty target slot could never be matched by speech, so
 * the recitation tracker flagged a WRONG/SKIPPED mistake on something the user was never supposed
 * to say ("the app wants me to read it or it gives me a mistake"). This is the bug the user hit
 * on Al-Baqarah 2; the fix drops those tokens at tokenization time and applies wherever the
 * same marks appear (a scan of Surah Al-Baqarah alone surfaced 490 standalone annotation tokens
 * across many ayahs).
 */
class AyahAnnotationTokensTest {

    // Baqarah 2 verbatim from AlQuran.cloud Uthmani -- contains two standalone ۖۛ (U+06DB)
    // three-dot pause marks between words. After splitAyahWords these MUST NOT appear as words.
    private val baqarahTwo =
        "ذَٰلِكَ ٱلْكِتَٰبُ لَا رَيْبَ ۛ فِيهِ ۛ هُدًۭى لِّلْمُتَّقِينَ"

    @Test
    fun `isAyahAnnotationOnlyToken recognises standalone pause marks`() {
        // Pause / stop signs in the U+06D6..U+06ED Quranic-annotations block.
        assertTrue(isAyahAnnotationOnlyToken("ۛ"))   // U+06DB three-dot
        assertTrue(isAyahAnnotationOnlyToken("ۖ"))   // U+06D6
        assertTrue(isAyahAnnotationOnlyToken("ۗ"))   // U+06D7
        assertTrue(isAyahAnnotationOnlyToken("ۘ"))   // U+06D8
        assertTrue(isAyahAnnotationOnlyToken("ۙ"))   // U+06D9
        assertTrue(isAyahAnnotationOnlyToken("ۚ"))   // U+06DA
        assertTrue(isAyahAnnotationOnlyToken("ۜ"))   // U+06DC
        assertTrue(isAyahAnnotationOnlyToken("۞"))   // U+06DE rub-el-hizb
        assertTrue(isAyahAnnotationOnlyToken("۝"))   // U+06DD end-of-ayah
        assertTrue(isAyahAnnotationOnlyToken("ۭ"))   // U+06ED empty-centre-nun
        // Empty / whitespace-only tokens are also considered "annotation-only".
        assertTrue(isAyahAnnotationOnlyToken(""))
        assertTrue(isAyahAnnotationOnlyToken("   "))
    }

    @Test
    fun `isAyahAnnotationOnlyToken keeps spoken words even when they carry annotation marks inline`() {
        // A word that contains a real spoken letter (even WITH a trailing/prefixed annotation
        // mark) must be KEPT -- the mark is positional, the letter underneath is spoken.
        assertFalse(isAyahAnnotationOnlyToken("هۛ"))   // ه is a real letter + ا mark
        assertFalse(isAyahAnnotationOnlyToken("ۛه"))
        assertFalse(isAyahAnnotationOnlyToken("مۘ"))
        assertFalse(isAyahAnnotationOnlyToken("فِيهِ"))   // the actual word before a ۛ in Baqarah 2
        assertFalse(isAyahAnnotationOnlyToken("ذَٰلِكَ"))
    }

    @Test
    fun `splitAyahWords drops standalone pause marks from Baqarah 2`() {
        val words = splitAyahWords(baqarahTwo)
        // 9 raw whitespace tokens, 2 of which are standalone ێۛ (U+06DB) -> 7 spoken words.
        assertEquals(7, words.size)
        assertEquals("ذَٰلِكَ", words[0])
        assertEquals("ٱلْكِتَٰبُ", words[1])
        assertEquals("لَا", words[2])
        assertEquals("رَيْبَ", words[3])
        assertEquals("فِيهِ", words[4])     // the ۛ after this word was dropped
        assertEquals("هُدًۭى", words[5])     // the ۛ after this word was dropped
        assertEquals("لِّلْمُتَّقِينَ", words[6])
    }

    @Test
    fun `splitAyahWords strips every spoken word normalises to non-empty`() {
        // Every kept word must normalise to a NON-empty string. Otherwise the multi-form
        // candidate lookup / isWordMatch would still choke on an empty target slot downstream,
        // and the recitation tracker would freeze waiting for an impossible utterance.
        val words = splitAyahWords(baqarahTwo)
        for (w in words) {
            val n = normalizeArabic(w)
            assertTrue("kept word \"$w\" should normalise non-empty, got \"$n\"", n.isNotEmpty())
        }
    }

    @Test
    fun `splitAyahWords handles an ayah with no annotation marks unchanged`() {
        // Al-Fatihah 1 has no standalone annotation tokens, so tokenization must produce the
        // exact spoken-word sequence. (Regression guard against an over-eager filter.)
        val fatihahOne = "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ"
        val words = splitAyahWords(fatihahOne)
        assertEquals(listOf("بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ", "ٱلرَّحِيمِ"), words)
    }

    @Test
    fun `splitAyahWords handles strings that are only annotation marks`() {
        // Pure-annotation-token strings (which can occur as standalone entries in a curated
        // JSON array like the Qalun words_uthmani) must collapse to the empty list -- never
        // yielding a target slot.
        assertTrue(splitAyahWords("ۛ ۖ ۗ ۘ ۚ ۜ").isEmpty())
        assertTrue(splitAyahWords("   ").isEmpty())
        assertTrue(splitAyahWords("").isEmpty())
    }
}
