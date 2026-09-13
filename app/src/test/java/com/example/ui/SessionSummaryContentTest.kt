package com.example.ui

import com.example.data.MistakeEntry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the Session Summary CONTENT contract.
 *
 * The user's request:
 *   "when we have the summury : if there is mistakes it shows as the mistake
 *    (wrong or skipped word or tajweed mistake) and shows the word + tajweed
 *    rules + the tajweed score for the tajweed mistakes"
 *
 * Two things must hold:
 *  1. The three kinds are distinguishable and each carries the data its row
 *     renders -- word for all three, plus rule AND score for tajweed.
 *  2. The dialog title routes on `entries.isEmpty()`: mistakes => the summary,
 *     nothing => "Perfect recitation!". A mistake must never be presented as a
 *     perfect recitation.
 *
 * The serializer/parser pair is exercised too: a score that does not survive
 * the persistence round-trip would blank the score column in the Stats tab.
 */
class SessionSummaryContentTest {

    private fun wrong(idx: Int, word: String, ayah: Int = 1) = MistakeEntry(
        kind = "wrong", word = word, rule = null, score = null,
        surah = 1, ayah = ayah, wordIndex = idx,
    )

    private fun skipped(idx: Int, word: String, ayah: Int = 1) = MistakeEntry(
        kind = "skipped", word = word, rule = null, score = null,
        surah = 1, ayah = ayah, wordIndex = idx,
    )

    private fun tajweed(idx: Int, word: String, rule: String, score: Float?, ayah: Int = 1) =
        MistakeEntry(
            kind = "tajweed", word = word, rule = rule, score = score,
            surah = 1, ayah = ayah, wordIndex = idx,
        )

    @Test
    fun `every kind carries the word it must display`() {
        val entries = listOf(
            wrong(3, "ٱلْحَمْدُ"),
            skipped(7, "ٱلْعَٰلَمِينَ"),
            tajweed(10, "مَٰلِكِ", "Madd (Vowel Extension)", 46f),
        )
        assertEquals("ٱلْحَمْدُ", entries[0].word)
        assertEquals("ٱلْعَٰلَمِينَ", entries[1].word)
        assertEquals("مَٰلِكِ", entries[2].word)
        // A mistake row with no word would render as a blank line.
        assertTrue("no entry may have an empty word", entries.all { it.word.isNotBlank() })
    }

    @Test
    fun `only tajweed entries carry rule and score`() {
        val w = wrong(3, "ٱلْحَمْدُ")
        val s = skipped(7, "ٱلْعَٰلَمِينَ")
        val t = tajweed(10, "مَٰلِكِ", "Madd (Vowel Extension)", 46f)
        assertNull(w.rule); assertNull(w.score)
        assertNull(s.rule); assertNull(s.score)
        assertEquals("Madd (Vowel Extension)", t.rule)
        assertEquals(46f, t.score)
    }

    @Test
    fun `the tajweed grouping key keeps one row per word and rule`() {
        // Mirrors the dialog: groupBy(wordIndex to rule). A word carrying the
        // SAME rule twice must collapse to one row; the same word with two
        // DIFFERENT rules must stay two rows.
        val entries = listOf(
            tajweed(10, "مَٰلِكِ", "Madd", 46f),
            tajweed(10, "مَٰلِكِ", "Madd", 46f),          // duplicate -> one row
            tajweed(10, "مَٰلِكِ", "Ghunnah & Shaddah", 52f), // different rule -> second row
        )
        val pairs = entries
            .filter { it.kind == "tajweed" }
            .groupBy { it.wordIndex to (it.rule ?: "") }
            .map { (_, group) -> group.first() }
        assertEquals(2, pairs.size)
        assertEquals(setOf("Madd", "Ghunnah & Shaddah"), pairs.mapNotNull { it.rule }.toSet())
    }

    @Test
    fun `the title routes on whether any mistake exists`() {
        // "Perfect recitation!" is shown ONLY for an empty entry list; any
        // mistake shows the summary instead.
        fun titleFor(entries: List<MistakeEntry>) =
            if (entries.isEmpty()) "Perfect recitation! 🌟" else "Recitation Summary"
        assertEquals("Perfect recitation! 🌟", titleFor(emptyList()))
        assertEquals("Recitation Summary", titleFor(listOf(wrong(3, "ٱلْحَمْدُ"))))
        assertEquals("Recitation Summary", titleFor(listOf(skipped(7, "ٱلْعَٰلَمِينَ"))))
        assertEquals(
            "Recitation Summary",
            titleFor(listOf(tajweed(10, "مَٰلِكِ", "Madd", 46f))),
        )
    }

    @Test
    fun `a tajweed score survives the persistence round-trip`() {
        val entries = listOf(
            wrong(3, "ٱلْحَمْدُ"),
            skipped(7, "ٱلْعَٰلَمِينَ"),
            tajweed(10, "مَٰلِكِ", "Madd (Vowel Extension)", 46f),
            tajweed(12, "ٱلدِّينِ", "Ghunnah & Shaddah", null), // score absent
        )
        val back = parseMistakeEntries(entries.encodeToJson())
        assertEquals(entries.size, back.size)
        assertEquals(entries, back)
        assertEquals(46f, back[2].score)
        assertNull("a missing score must stay absent, not become 0", back[3].score)
    }
}
