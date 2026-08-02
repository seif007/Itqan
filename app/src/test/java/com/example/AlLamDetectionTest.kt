package com.example

import com.example.ui.alifLamStartIndex
import com.example.ui.firstBaseIndexFrom
import com.example.ui.firstBaseLetterFrom
import com.example.ui.getTajweedRuleForWord
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Regression tests for the Al-Lam (Solar / Lunar) detector at
 * [com.example.ui.getTajweedRuleForWord] and the colorizer helpers
 * ([alifLamStartIndex] / [firstBaseLetterFrom]).
 *
 * Before the fix, both the verdict detector and [com.example.ui.buildTajweedAnnotatedString] read
 * the letter after the Lam with `word[lamIndex + 2]`. In voweled Quran text the Lam of an article is
 * usually silent (لْ), so that index lands on the Sukoon combining mark — not the next base letter —
 * and the whole word fell through to other detectors (e.g. ٱلْحَمْدُ was classified as "Meem
 * Sakinah", ٱلْقَمَرُ's Al-Lam was left uncolored). Same bug affected prefixed articles that carry a
 * vowel mark on the prefix (وَٱلسَّمَآءِ), because `word.startsWith("وال")` only matches the
 * unvoweled-string prefix.
 *
 * These tests pin the new diacritic-skipping + prefixed-article behavior so it cannot regress.
 */
class AlLamDetectionTest {

    // ----- firstBaseLetterFrom / firstBaseIndexFrom -----

    @Test fun firstBaseLetter_skipsHarakatToFindNextLetter() {
        // ٱ + ل + ْ (sukoon) + ح ; first base letter after the lam should be ح, not the sukoon.
        assertEquals('ح', firstBaseLetterFrom("ٱلْحَمْدُ", 2))
        // The solar char of ٱلشَّمْسُ after the lam (there is no sukoon here) should be ش.
        assertEquals('ش', firstBaseLetterFrom("ٱلشَّمْسُ", 2))
        // If only diacritics remain, there is no base letter.
        assertNull(firstBaseLetterFrom("ٱلْ", 2))
    }

    @Test fun firstBaseIndex_returnsCharIndexOfNextLetter() {
        // ٱلْحَمْدُ: starting at index 2 (the sukoon), the next base letter ح is at index 3.
        assertEquals(3, firstBaseIndexFrom("ٱلْحَمْدُ", 2))
        // ٱلشَّمْسُ: starting at index 2, no diacritic to skip, so the next base letter ش is at 2.
        assertEquals(2, firstBaseIndexFrom("ٱلشَّمْسُ", 2))
        // No base letter at all -> -1.
        assertEquals(-1, firstBaseIndexFrom("ٱلْ", 2))
    }

    // ----- alifLamStartIndex: bare + prefixed forms -----

    @Test fun alifLamStartIndex_bareArticle() {
        assertEquals(0, alifLamStartIndex("ٱلْقَمَرُ"))
        assertEquals(0, alifLamStartIndex("ٱلشَّمْسُ"))
        assertEquals(0, alifLamStartIndex("ٱللَّهِ"))
        assertEquals(0, alifLamStartIndex("ٱلنَّاسِ"))
        // Plain ا (not wasla) at the start of an article should also be detected.
        assertEquals(0, alifLamStartIndex("اللَّهِ"))
    }

    @Test fun alifLamStartIndex_prefixedArticle_withVowelOnPrefix() {
        // Real voweled forms: prefix (ب/و/ف/ك) + harakah + (ٱ)ل + base letter.
        // وَٱلسَّمَآءِ (the Madd-rule example that previously failed Al-Lam detection).
        assertEquals(2, alifLamStartIndex("وَٱلسَّمَآءِ"))
        assertEquals(2, alifLamStartIndex("وَٱلْقَمَرُ"))
        assertEquals(2, alifLamStartIndex("بِٱلْكِتَابِ"))
        assertEquals(2, alifLamStartIndex("فَٱلْعَالَمِينَ"))
        assertEquals(2, alifLamStartIndex("كَٱللَّهِ"))
    }

    @Test fun alifLamStartIndex_returnsNullForNonArticle() {
        // وَنَصْرُ has waw + fatha but does NOT continue with (ٱ)ل.
        assertNull(alifLamStartIndex("وَنَصْرُ"))
        // نَصْرُ does not start with an article.
        assertNull(alifLamStartIndex("نَصْرُ"))
        // Short word — can't contain a 3-char article + base letter.
        assertNull(alifLamStartIndex("ٱل"))
    }

    // ----- getTajweedRuleForWord: end-to-end verdict -----

    @Test fun getTajweedRuleForWord_lunarAlLamWithSukoonOnLam_isDetected() {
        // Regression: previously these returned "Meem Sakinah (Labial Rules)" or null because the
        // Sukoon on the Lam was read as the "letter after Lam".
        assertEquals("Al-Lam", getTajweedRuleForWord("ٱلْقَمَرُ", null, "Hafs"))
        assertEquals("Al-Lam", getTajweedRuleForWord("ٱلْحَمْدُ", null, "Hafs"))
        assertEquals("Al-Lam", getTajweedRuleForWord("ٱلْكِتَابُ", null, "Hafs"))
        assertEquals("Al-Lam", getTajweedRuleForWord("ٱلرَّحْمَٰنِ", null, "Hafs"))
        assertEquals("Al-Lam", getTajweedRuleForWord("ٱلْبُرُوجِ", null, "Hafs"))
    }

    @Test fun getTajweedRuleForWord_solarAlLam_isDetected() {
        // Solar: the letter after the Lam carries a shaddah; should remain detected.
        assertEquals("Al-Lam", getTajweedRuleForWord("ٱلشَّمْسُ", null, "Hafs"))
        assertEquals("Al-Lam", getTajweedRuleForWord("ٱلنَّاسِ", null, "Hafs"))
        assertEquals("Al-Lam", getTajweedRuleForWord("ٱللَّهِ", null, "Hafs"))
    }

    @Test fun getTajweedRuleForWord_prefixedAlLam_isDetected() {
        // Previously these returned null (Madd verdict path caught them) because the
        // prefix-with-vowel case didn't match `startsWith("وال")`.
        assertEquals("Al-Lam", getTajweedRuleForWord("وَٱلسَّمَآءِ", null, "Hafs"))
        assertEquals("Al-Lam", getTajweedRuleForWord("وَٱلْقَمَرُ", null, "Hafs"))
        assertEquals("Al-Lam", getTajweedRuleForWord("بِٱلْكِتَابِ", null, "Hafs"))
    }

    @Test fun getTajweedRuleForWord_nonAlLamWords_notMisclassifiedAsAlLam() {
        // Sanity: words that are NOT articles must not return Al-Lam.
        assertNull(getTajweedRuleForWord("نَصْرُ", null, "Hafs"))
        assertNull(getTajweedRuleForWord("ذَاتِ", null, "Hafs"))
        assertNull(getTajweedRuleForWord("أَعْطَيْنَٰكَ", null, "Hafs"))
        // A Noon-sakinah word with no following article should still be Ikhfa, not Al-Lam.
        assertEquals("Iqlab & Ikhfa", getTajweedRuleForWord("أُنزِلَ", null, "Hafs"))
        // A Meem-sakinah word should still be detected as Meem Sakinah, not Al-Lam (independent of
        // the Al-Lam fix).
        assertEquals("Meem Sakinah (Labial Rules)", getTajweedRuleForWord("أَمْ", "لَمْ", "Hafs"))
    }

    @Test fun getTajweedRuleForWord_maddAndOtherRulesUnaffected() {
        // The Madd example token جَآءَ must still return Madd Muttasil (Al-Lam fix didn't touch that
        // branch).
        assertEquals("Madd Muttasil", getTajweedRuleForWord("جَآءَ", null, "Hafs"))
        assertEquals("Madd (Vowel Extension)", getTajweedRuleForWord("بِمَآ", "أُنزِلَ", "Hafs"))
        // Idgham on tanween + following ي.
        assertEquals("Idgham (Merging)", getTajweedRuleForWord("وُجُوهٌ", "يَوْمَئِذٍ", "Hafs"))
        // Izhar (throat letter) on noon-sakinah.
        assertEquals("Izhar (Clarity)", getTajweedRuleForWord("مِنْ", "عَلَقٍ", "Hafs"))
    }
}
