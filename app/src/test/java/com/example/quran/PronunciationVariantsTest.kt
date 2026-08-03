package com.example.quran

import com.example.ui.isWordMatch
import com.example.ui.isWordMatchAny
import com.example.ui.isWordMatchPlural
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for [PronunciationVariants.formGroupsFor]. The tables receive the
 * *display* (Uthmani) word exactly as the AlQuran.cloud API delivers it (with
 * diacritics, dagger alif, hamzat al-wasl) and must emit candidate pronunciations
 * in BASE-LETTER form (what the app's normalizer produces from recognizer output).
 */
class PronunciationVariantsTest {

    // The real spoken-token sequence `tokens` must match ONE of the returned groups.
    // A group List<String> matches a single token; a group List<List<String>> matches
    // len(tokens) consecutive tokens, one per slot.
    private fun spokenMatches(word: String, tokens: List<String>): Boolean {
        for (group in formGroupsFor(word)) {
            @Suppress("UNCHECKED_CAST")
            when {
                group is List<*> && group.isNotEmpty() && group[0] is List<*> -> {
                    if (isWordMatchPlural(tokens, group as List<List<String>>)) return true
                }
                group is List<*> && group.isNotEmpty() && group[0] is String -> {
                    if (tokens.size == 1 && isWordMatchAny(tokens[0], group as List<String>)) return true
                }
            }
        }
        return false
    }

    // -------------------------------------------------------------------
    // Muqatta'at (disjointed letters)
    // -------------------------------------------------------------------

    @Test
    fun `الم expands to segmented letter names`() {
        assertTrue(spokenMatches("الم", listOf("الف", "لام", "ميم")))
        assertTrue(spokenMatches("الم", listOf("الف", "لم", "ميم")))
    }

    @Test
    fun `الم expands to concatenated single token`() {
        // Recognizer sometimes runs the letter names together.
        assertTrue(spokenMatches("الم", listOf("الفلامميم")))
        assertTrue(spokenMatches("الم", listOf("الفلمميم")))
    }

    @Test
    fun `المر expands to four syllables`() {
        assertTrue(spokenMatches("المر", listOf("الف", "لام", "ميم", "را")))
        assertTrue(spokenMatches("المر", listOf("الف", "لام", "ميم", "راء")))
    }

    @Test
    fun `يس accepts both yaseen and ya-seen`() {
        assertTrue(spokenMatches("يس", listOf("ياسين")))
        assertTrue(spokenMatches("يس", listOf("يا", "سين")))
        assertTrue(spokenMatches("يس", listOf("ياء", "سين")))
    }

    @Test
    fun `طسم expands to three syllables`() {
        assertTrue(spokenMatches("طسم", listOf("طا", "سين", "ميم")))
        assertTrue(spokenMatches("طسم", listOf("طاء", "سين", "ميم")))
    }

    @Test
    fun `حم expands to two syllables`() {
        assertTrue(spokenMatches("حم", listOf("حا", "ميم")))
        assertTrue(spokenMatches("حم", listOf("حاء", "ميم")))
    }

    @Test
    fun `كهيعص expands to five syllables`() {
        assertTrue(spokenMatches("كهيعص", listOf("كاف", "ها", "يا", "عين", "صاد")))
        assertTrue(spokenMatches("كهيعص", listOf("كاف", "هاء", "ياء", "عين", "صاد")))
    }

    @Test
    fun `المص expands to four syllables`() {
        assertTrue(spokenMatches("المص", listOf("الف", "لام", "ميم", "صاد")))
    }

    @Test
    fun `ordinary words do not expand`() {
        assertFalse(hasMuqattaatExpansion("الله"))
        assertFalse(hasMuqattaatExpansion("الحمد"))
        assertFalse(hasMuqattaatExpansion("الرحمن"))
        assertFalse(hasMuqattaatExpansion("قل"))
        assertTrue(hasMuqattaatExpansion("الم"))
        assertTrue(hasMuqattaatExpansion("يس"))
        assertTrue(hasMuqattaatExpansion("عسق"))
    }

    // -------------------------------------------------------------------
    // Dagger-alif family (الصلاة, الزكاة, الحياة, الرِّبا, هٰذا, أولٰئك, مئة, سلوة)
    // -------------------------------------------------------------------

    @Test
    fun `الصلاة accepts واو pronunciation`() {
        assertTrue(spokenMatches("ٱلصَّلَوٰةِ", listOf("الصلوه")))
        assertTrue(spokenMatches("ٱلصَّلَوٰةِ", listOf("الصلوة")))
        assertTrue(spokenMatches("ٱلصَّلَوٰةِ", listOf("الصلوا")))
    }

    @Test
    fun `الزكاة accepts واو pronunciation`() {
        assertTrue(spokenMatches("ٱلزَّكَوٰةِ", listOf("الزكوه")))
    }

    @Test
    fun `الحيوة accepts واو pronunciation`() {
        assertTrue(spokenMatches("ٱلْحَيَوٰةِ", listOf("الحيوه")))
    }

    @Test
    fun `الربوا accepts plain alif pronunciation`() {
        assertTrue(spokenMatches("ٱلرِّبَوٰا", listOf("الربا")))
    }

    @Test
    fun `سلوة accepts واو pronunciation`() {
        assertTrue(spokenMatches("سَلَوٰةٍ", listOf("سلوه")))
    }

    @Test
    fun `هذا and اولئك accept recited variants`() {
        assertTrue(spokenMatches("هَٰذَا", listOf("هذا")))
        assertTrue(spokenMatches("أُولَٰئِكَ", listOf("اوليك")))
    }

    @Test
    fun `مئة (dagger alif) accepts ميه spelling`() {
        assertTrue(spokenMatches("مِائَةِ", listOf("ميه")))
    }
}
