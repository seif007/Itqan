package itqan.app.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.49 — a card's colour IS its head's colour, by construction.
 *
 * `TajweedCardExampleAuditTest` already pins the bundled example pools and the
 * per-head display resolution. This pins the thing that was still unpinned and
 * that the user reported directly: "check the rules colors match what we have in
 * memorize + tajweed, not giving me other colors or mingling the colours of the
 * rules (for example we have madd always red)".
 *
 * The nine card colours used to be nine hand-typed literals. Nothing connected
 * them to [tajweedHeadColor], so the surfaces could — and did — drift: the
 * summary painted EVERY tajweed rule in Color(0xFF7E57C2) (the Meem-Sakinah
 * head colour, so Madd mistakes came out purple), and the "Skipped Words" /
 * "Tajweed Mistakes" section icons borrowed Idgham orange and Meem-Sakinah
 * purple as generic chrome. The card colours now READ the palette
 * (`color = tajweedHeadColor("MADD")!!`), and this test keeps them honest.
 */
class TajweedCardColorBindingTest {

    @Test
    fun everyCardWearsItsHeadPaletteColour() {
        val problems = mutableListOf<String>()
        for (card in tajweedRuleCards) {
            val head = ruleLabelToHead(card.name)
            if (head == null) {
                problems.add("${card.name}: resolves to no model head")
                continue
            }
            val canonical = tajweedHeadColor(head)
            if (canonical == null) {
                problems.add("$head: no palette entry, so the card colour has nothing to match")
                continue
            }
            if (canonical != card.color) {
                problems.add(
                    "${card.name} (head=$head): the card colour does not equal the palette entry"
                )
            }
        }
        assertTrue(
            "${problems.size} card/head colour mismatch(es):\n" + problems.joinToString("\n"),
            problems.isEmpty()
        )
    }

    @Test
    fun maddIsRed() {
        // The user's own example, pinned by value.
        assertEquals(
            "Madd must be red 0xFFE53935",
            androidx.compose.ui.graphics.Color(0xFFE53935),
            tajweedHeadColor("MADD")
        )
    }

    @Test
    fun everyShippedExamplePoolIsUsable() {
        for (card in tajweedRuleCards) {
            assertTrue("card '${card.name}' ships no examples", card.examples.isNotEmpty())
            card.examples.forEach { ex ->
                assertTrue("card '${card.name}' has an example with no Arabic text", ex.arabic.isNotBlank())
            }
        }
    }
}
