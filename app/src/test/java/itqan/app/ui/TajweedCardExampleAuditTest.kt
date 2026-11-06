package itqan.app.ui

import itqan.app.ml.TajweedRules
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * Audits the SHIPPED Tajweed-coach card data (top-level [tajweedRuleCards]).
 *
 * WHY: the user asked to be sure of two things, and both are silent failures on
 * device — nothing crashes, the card just teaches the wrong thing:
 *
 *  1. "if I want to train for Madd it always gives me an example containing
 *     madd" — every bundled example of a card must ACTUALLY exhibit that card's
 *     rule under the same detector chain the generator uses
 *     (`getTajweedRuleForWord` -> `ruleLabelToHead`). The ONLINE generator
 *     guarantees this by construction (it scans for a matching token), but the
 *     BUNDLED fallback pool is hand-written and was never machine-checked —
 *     several entries even carry self-doubting comments ("actually ... has ج in
 *     the middle, but OK. Let me check.").
 *
 *  2. "if there is other rules in the example I get the score for each rule" —
 *     the result card shows one row per rule, built from the PRIMARY per-word
 *     pass plus a SUPPLEMENTAL pass driven by [TajweedRules.detectRulesForWord].
 *     Those are two independent detectors, so an example can contain a rule that
 *     neither pass reports. This test pins the display surface.
 *
 * The test is deliberately DATA-driven off `tajweedRuleCards`, so fixing an
 * example string is the only way to make it pass.
 *
 * Robolectric for consistency with the neighbouring UI tests (Compose Color).
 */
@RunWith(RobolectricTestRunner::class)
class TajweedCardExampleAuditTest {

    /** Every head the trained model can score, i.e. every head a row can show. */
    private val scoreableHeads = listOf(
        "GHUNNAH", "MADD", "QALQALAH", "IDGHAM", "IKHFA", "IZHAR", "MEEM_SAKINAH", "ALLAM"
    )

    private fun tokens(s: String) = s.split(Regex("\\s+")).filter { it.isNotBlank() }

    /**
     * Heads the EXAMPLE-GENERATION chain sees. Mirrors
     * `generateRandomRecitationExample` exactly: per token, the FIRST matching rule
     * from `getTajweedRuleForWord`, mapped through `ruleLabelToHead`.
     */
    private fun primaryHeads(arabic: String, riwayah: String = "Hafs"): List<String> {
        val t = tokens(arabic)
        return t.mapIndexedNotNull { i, w ->
            getTajweedRuleForWord(w, t.getOrNull(i + 1), riwayah)?.let { ruleLabelToHead(it) }
        }
    }

    /**
     * Heads the MULTI-RULE COVERAGE path sees — every rule per word, from the
     * ml-side table the model was calibrated with.
     */
    private fun allDetectedHeads(arabic: String): List<String> {
        val t = tokens(arabic)
        return t.flatMapIndexed { i, w ->
            TajweedRules.detectRulesForWord(w, t.getOrNull(i + 1))
        }.filter { it != "NONE" }.distinct()
    }

    /**
     * Heads the COLOURISER paints — exactly what the user sees highlighted, and
     * (after the fix) the ONLY source of the coach's score rows. This is the
     * ground truth the row count must match.
     */
    private fun paintedHeads(arabic: String, riwayah: String = "Hafs"): List<String> =
        tajweedHeadsInPhrase(tokens(arabic), riwayah).keys.toList()

    /**
     * The coach's OWN head -> friendly-name lookup (RecitationApp.kt, supplemental
     * scoring): `rulesList.firstOrNull { ruleLabelToHead(it.name) == head }?.name`.
     * Kept as a copy so the test fails if either side of that contract drifts.
     */
    private fun cardLabelForHead(head: String): String? =
        tajweedRuleCards.firstOrNull { ruleLabelToHead(it.name) == head }?.name

    /** Every card must resolve to a real model head, else it can never be scored. */
    @Test
    fun `every card maps to a trained model head`() {
        val bad = tajweedRuleCards
            .filter { ruleLabelToHead(it.name) == null }
            .map { it.name }
        assertTrue(
            "Cards whose name does not resolve through ruleLabelToHead. Such a card can " +
                "never be generated for -- the expand effect does " +
                "`ruleLabelToHead(rule.name) ?: return`, so the card freezes on bundled " +
                "example #1 and never fetches a random one -- and its score rows fall back " +
                "to the untranslated raw head string: $bad",
            bad.isEmpty()
        )
    }

    /** The user's question 1: a card's example must contain that card's rule. */
    @Test
    fun `every bundled example exhibits its own card's rule`() {
        val violations = mutableListOf<String>()
        for (card in tajweedRuleCards) {
            val head = ruleLabelToHead(card.name) ?: continue
            card.examples.forEachIndexed { i, ex ->
                val seen = primaryHeads(ex.arabic)
                if (head !in seen) {
                    violations.add(
                        "[${card.name}] example #${i + 1} '${ex.arabic}' " +
                            "-> primary heads $seen (expected $head)"
                    )
                }
            }
        }
        assertTrue(
            "${violations.size} bundled example(s) do NOT contain their card's rule:\n" +
                violations.joinToString("\n"),
            violations.isEmpty()
        )
    }

    /**
     * The user's question 1, at the DISPLAY level: the rule a card teaches must be
     * VISIBLY PAINTED in its example, because the score rows are now drawn from
     * the painted set. If a card's head is detected but not painted, its example
     * can carry no row for the rule it is teaching.
     */
    @Test
    fun `every bundled example paints its own card's rule`() {
        val violations = mutableListOf<String>()
        for (card in tajweedRuleCards) {
            val head = ruleLabelToHead(card.name) ?: continue
            card.examples.forEachIndexed { i, ex ->
                val painted = paintedHeads(ex.arabic)
                if (head !in painted) {
                    violations.add(
                        "[${card.name}] example #${i + 1} '${ex.arabic}' -> painted $painted"
                    )
                }
            }
        }
        assertTrue(
            "${violations.size} bundled example(s) do NOT paint their card's rule:\n" +
                violations.joinToString("\n"),
            violations.isEmpty()
        )
    }

    /**
     * The alignment that keeps the primary score rows alive: every head the
     * primary pass scores must also be PAINTED, otherwise the row filter in
     * `ruleAgg` (painted heads only) would drop a rule the user does see.
     */
    @Test
    fun `primary heads are a subset of the painted heads`() {
        val violations = mutableListOf<String>()
        for (card in tajweedRuleCards) {
            card.examples.forEachIndexed { i, ex ->
                val orphans = primaryHeads(ex.arabic).toSet() - paintedHeads(ex.arabic).toSet()
                if (orphans.isNotEmpty()) {
                    violations.add("[${card.name}] #${i + 1} '${ex.arabic}' -> primary-only $orphans")
                }
            }
        }
        assertTrue(
            "Heads scored by the primary pass but NOT painted would be dropped from the " +
                "score rows:\n" + violations.joinToString("\n"),
            violations.isEmpty()
        )
    }

    /**
     * The user's question 2, made NON-VACUOUS.
     *
     * The row set itself is primaryHeads ∪ supplementalHeads, and supplementalHeads
     * is BY CONSTRUCTION (allDetected - primaryHeads) — so "does every detected head
     * get a row" is a tautology that can never fail. What CAN fail, and what the user
     * actually sees, is how each row RESOLVES for display: the coach maps the head to
     * a friendly card name, then feeds that name to getAppString (Arabic label),
     * tajweedHeadColor (rule colour) and decisionThresholdForHead (verdict bar). When
     * the card lookup misses, the row renders as the raw head — "MADD", untranslated,
     * in the theme colour. That is exactly what a missing `"Madd" -> "MADD"` case
     * caused, so this is the invariant worth pinning.
     */
    @Test
    fun `every scoreable head is fully resolvable for display`() {
        val problems = mutableListOf<String>()
        for (head in scoreableHeads) {
            val label = cardLabelForHead(head)
            if (label == null) {
                problems.add("$head: no card label (row would show the raw head string)")
            } else if (getAppString(true, label) == label) {
                problems.add("$head: card label '$label' has no Arabic translation")
            }
            if (tajweedHeadColor(head) == null) {
                problems.add("$head: no rule colour (row falls back to the theme colour)")
            }
        }
        assertTrue(
            "${problems.size} display-resolution problem(s) for scoreable heads:\n" +
                problems.joinToString("\n"),
            problems.isEmpty()
        )
    }

    /**
     * `analyzeRecitationDetails` labels every multi-rule row by composing
     * `headToFriendlyLabel` then `ruleLabelToHead` (RecitationApp.kt ~2753). If the
     * pair does not round-trip, those rows silently lose their head — and with it
     * their rule colour and calibrated verdict threshold.
     */
    @Test
    fun `head to friendly label round-trips through ruleLabelToHead`() {
        val broken = scoreableHeads.filter { h ->
            val friendly = headToFriendlyLabel(h)
            friendly == null || ruleLabelToHead(friendly) != h
        }
        assertTrue(
            "Heads whose friendly label does not map back to the same head (multi-rule rows " +
                "would lose their head, colour and threshold): $broken",
            broken.isEmpty()
        )
    }

    /**
     * Not an assertion — a full dump of what each card's pool actually contains,
     * so a human can see which rules each example teaches. Written to the test
     * report's stdout.
     */
    @Test
    fun `report the rules each bundled example actually contains`() {
        val sb = StringBuilder()
        sb.appendLine("=== Tajweed card example audit ===")
        for (card in tajweedRuleCards) {
            val head = ruleLabelToHead(card.name)
            sb.appendLine()
            sb.appendLine("## ${card.name}  (head=$head, ${card.examples.size} examples)")
            card.examples.forEachIndexed { i, ex ->
                val primary = primaryHeads(ex.arabic)
                val painted = paintedHeads(ex.arabic)
                val detected = allDetectedHeads(ex.arabic)
                sb.appendLine(
                    "  #%-2d %-42s painted=%-34s primary=%-26s detected=%s".format(
                        i + 1, ex.arabic, painted, primary, detected
                    )
                )
                sb.appendLine(
                    "        rows(painted)=%s  rows(old: primary+detected)=%s".format(
                        painted.size, (primary.toSet() + detected).size
                    )
                )
            }
        }
        println(sb)
        assertTrue(true)
    }
}
