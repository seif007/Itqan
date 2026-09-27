package com.example.ui

import androidx.compose.ui.graphics.Color
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.49 — "check the rules colors match what we have in memorize + tajweed, not
 * giving me other colors or mingling the colours of the rules (for example we
 * have madd always red)".
 *
 * The contract the user stated is a per-RULE colour, not a per-surface one: Madd
 * is red in the letters, in the rule card, in the score row and in the summary,
 * and no rule ever borrows another's colour. This pins the letter-level half of
 * that contract against [tajweedHeadColor], the single palette the cards, the
 * legend and the score rows all read.
 *
 * The bug this locks down was real: the summary's tajweed rows were hardcoded to
 * Color(0xFF7E57C2) — the Meem-Sakinah head colour — so a MADD mistake came out
 * purple, and the "Skipped Words"/"Tajweed Mistakes" section icons borrowed
 * Idgham orange and Meem-Sakinah purple as generic chrome.
 */
class TajweedRuleColorTest {

    private fun spansOf(word: String, next: String? = null, riwayah: String = "Hafs"): List<Color> =
        buildTajweedAnnotatedString(word, next, riwayah)
            .spanStyles
            .mapNotNull { it.item.color }

    private fun assertWordUsesRuleColor(word: String, head: String, next: String? = null) {
        val expected = tajweedHeadColor(head)
        assertTrue("no palette entry for $head", expected != null)
        val seen = spansOf(word, next)
        assertTrue("no coloured span in '$word' (spans=$seen)", seen.isNotEmpty())
        assertTrue(
            "'$word' should carry $head's colour ${expected!!.toArgbHex()} but painted ${seen.map { it.toArgbHex() }}",
            seen.contains(expected)
        )
    }

    private fun Color.toArgbHex(): String = String.format("#%08X", this.value.toLong())

    @Test
    fun maddIsAlwaysRed() {
        // The user's explicit example. Explicit maddah marks (آ / U+0653 / U+06E4).
        assertWordUsesRuleColor("سَمَآءِ", "MADD")
        assertWordUsesRuleColor("ٱلْجَآوِ", "MADD")
        assertEquals(
            "Madd's palette entry must stay RED (0xFFE53935)",
            Color(0xFFE53935),
            tajweedHeadColor("MADD")
        )
    }

    @Test
    fun everyRuleKeepsItsOwnColourInTheText() {
        // Called the way the UI calls it: ONE token plus its neighbour, which is
        // how the colouriser receives a word inside a sentence.
        assertWordUsesRuleColor("إِنَّ", "GHUNNAH")
        assertWordUsesRuleColor("قَدْ", "QALQALAH")
        assertWordUsesRuleColor("مَن", "IDGHAM", "يَقُولُ")
        assertWordUsesRuleColor("مِن", "IKHFA", "بَعْدِ")
        // Solar Al-Lam: the letter after the lam is ر, which is in تثدذرزسشصضطظلن.
        assertWordUsesRuleColor("ٱلرَّحْمَٰنِ", "ALLAM")
    }

    @Test
    fun lunarLamIsNotIkhfasCyan() {
        // §4.49 — the Al-Lam family's second shade used to BE the Ikhfa colour, so
        // a teal `ٱل` was indistinguishable from an Ikhfa noon. ح is a lunar letter,
        // so ٱلْحَمْدُ takes the lunar branch.
        val spans = spansOf("ٱلْحَمْدُ")
        assertTrue("no coloured span in ٱلْحَمْدُ (spans=$spans)", spans.isNotEmpty())
        assertTrue(
            "the lunar lam must not wear Ikhfa's cyan: ${spans.map { it.toArgbHex() }}",
            !spans.contains(tajweedHeadColor("IKHFA"))
        )
        assertTrue(
            "the lunar lam must wear TAJWEED_LUNAR_LAM: ${spans.map { it.toArgbHex() }}",
            spans.contains(TAJWEED_LUNAR_LAM)
        )
        // ...and it must still differ from the solar lam, or the two sub-types
        // would be indistinguishable.
        assertTrue(
            "lunar and solar Al-Lam must stay distinguishable",
            TAJWEED_LUNAR_LAM != tajweedHeadColor("ALLAM")
        )
    }

    @Test
    fun noTwoRulesShareTheSamePaletteColour() {
        // "Mingling the colours of the rules" is exactly two heads painting the
        // same letters in the same colour. Silah deliberately shares IZHAR (it
        // has no head of its own and is judged by the IZHAR classifier), so it is
        // excluded here and asserted separately.
        val heads = listOf(
            "ALLAM", "MADD", "QALQALAH", "IDGHAM",
            "IKHFA", "IZHAR", "GHUNNAH", "MEEM_SAKINAH"
        )
        val seen = mutableMapOf<Color, String>()
        for (h in heads) {
            val c = tajweedHeadColor(h)!!
            val prev = seen.put(c, h)
            assertTrue("heads '$prev' and '$h' both paint ${c.toArgbHex()}", prev == null)
        }
    }

    @Test
    fun silahSharesTheIzharHeadDeliberately() {
        // Documented, not accidental: Silah is Qalun's mim silah and is scored by
        // the IZHAR classifier, so it resolves to the IZHAR colour everywhere.
        assertEquals("IZHAR", ruleLabelToHead("Silah"))
        assertEquals(tajweedHeadColor("IZHAR"), tajweedHeadColor(ruleLabelToHead("Silah")))
    }
}

