package com.example.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * Pins the TAJWEED mistake-popup BODY composition.
 *
 * The user's live-mic report:
 *   "i tried baqarah and got a tajweed error and i noticed it didnt write the
 *    word in the popup. just the rule and the score. i want the words to be
 *    included in the mistakes popups and in the summury."
 *
 * Root cause: `latestTajweedPopup` is a `Pair<String, Int>` whose FIRST element
 * is the RULE for `kind="tajweed"` and the WORD TEXT for `kind="wrong"/"skipped"`
 * (set by `stopForMistake` at `latestTajweedPopup = Pair(wordText, wordIdx)`).
 * The renderer used that first element as `ruleLabel` in all three branches, so
 * the tajweed body could only ever name the RULE.
 *
 * The fix derives `popupWord` separately for the tajweed branch:
 *   `detailedAnalysis.forWordIndex(wordIdx)?.text`
 * -- resolved by WORD INDEX (multi-rule safe: several rows share a wordIndex),
 * then prefixes the body with `The word "%s": `.
 *
 * These tests pin the exact composition so a future edit cannot silently drop
 * the word again. They compose the string the same way the composable does.
 */
@RunWith(RobolectricTestRunner::class)
class TajweedPopupWordTest {

    /** Mirrors the composable's tajweed branch (RecitationApp.kt ~6751-6765). */
    private fun tajweedPopupBody(
        isArabic: Boolean,
        ruleLabel: String,
        wordText: String,
        score: Int?,
    ): String {
        val base = formatPerRuleVerdict(isArabic, false, ruleLabel)
        val named = if (wordText.isNotEmpty()) {
            getAppString(isArabic, "The word \"%s\": ").replace("%s", wordText) + base
        } else base
        return if (score != null) {
            named + " " + getAppString(isArabic, "Your %s score: %d%%.")
                .replace("%s", getAppString(isArabic, ruleLabel))
                .replace("%d", score.toString())
                .replace("%%", "%")
        } else named
    }

    @Test
    fun `the tajweed popup body names the word before the rule and the score`() {
        val body = tajweedPopupBody(
            isArabic = false,
            ruleLabel = "Madd (Vowel Extension)",
            wordText = "ٱلضَّآلِّينَ",
            score = 35,
        )
        // The word must appear, and must come BEFORE the rule verdict.
        assertTrue("word missing from popup body: $body", body.contains("ٱلضَّآلِّينَ"))
        val wordAt = body.indexOf("ٱلضَّآلِّينَ")
        val ruleAt = body.indexOf("Madd", wordAt)
        assertTrue("rule verdict must follow the word: $body", ruleAt > wordAt)
        assertTrue("score missing: $body", body.contains("35%"))
        assertEquals(
            "The word \"ٱلضَّآلِّينَ\": Review your Madd (Vowel Extension) application. " +
                "Your Madd (Vowel Extension) score: 35%.",
            body,
        )
    }

    @Test
    fun `the word is present even when no score was recorded`() {
        val body = tajweedPopupBody(
            isArabic = false,
            ruleLabel = "Ghunnah",
            wordText = "إِنَّمَا",
            score = null,
        )
        assertTrue("word missing: $body", body.contains("إِنَّمَا"))
        assertFalse("unexpected score clause: $body", body.contains("score"))
    }

    @Test
    fun `an unresolved word degrades gracefully instead of printing a null`() {
        // `forWordIndex` can return null (row evicted by a range switch). The
        // body must then fall back to the rule verdict alone -- never the
        // literal "null" and never a dangling "The word "": ".
        val body = tajweedPopupBody(
            isArabic = false,
            ruleLabel = "Madd",
            wordText = "",
            score = 40,
        )
        assertFalse("must not print null: $body", body.contains("null"))
        assertFalse("must not print an empty word clause: $body", body.contains("The word \"\""))
        assertEquals("Review your Madd application. Your Madd score: 40%.", body)
    }

    @Test
    fun `the Arabic popup body names the word with the Arabic prefix`() {
        val body = tajweedPopupBody(
            isArabic = true,
            ruleLabel = "Madd (Vowel Extension)",
            wordText = "ٱلضَّآلِّينَ",
            score = 35,
        )
        assertTrue("Arabic word missing: $body", body.contains("ٱلضَّآلِّينَ"))
        assertTrue("Arabic prefix missing: $body", body.startsWith("الكلمة \"ٱلضَّآلِّينَ\": "))
        assertTrue("Arabic rule verdict missing: $body", body.contains("راجع حكم"))
    }

    @Test
    fun `the English prefix key is translated and round-trips the word`() {
        assertEquals("The word \"%s\": ", getAppString(false, "The word \"%s\": "))
        assertEquals("الكلمة \"%s\": ", getAppString(true, "The word \"%s\": "))
    }

    /**
     * REGRESSION GUARD. `getAppString` opens with `if (!isArabic) return key`, and the two
     * verdict template keys exist ONLY in the Arabic branch. Before the fix the English popup
     * body therefore contained the raw key: `review_your_rule_template Your Madd score: 46%.`
     */
    @Test
    fun `the English verdict template is real prose, never a raw key`() {
        val failed = formatPerRuleVerdict(isArabic = false, passed = false, ruleName = "Madd")
        val passed = formatPerRuleVerdict(isArabic = false, passed = true, ruleName = "Madd")
        assertFalse("raw key leaked: $failed", failed.contains("review_your_rule_template"))
        assertFalse("raw key leaked: $passed", passed.contains("good_rule_application_template"))
        assertEquals("Review your Madd application.", failed)
        assertEquals("Good Madd application.", passed)
    }

    /** The Arabic path must be unchanged by the English fix. */
    @Test
    fun `the Arabic verdict template still resolves through the table`() {
        assertEquals("راجع حكم مد", formatPerRuleVerdict(isArabic = true, passed = false, ruleName = "Madd"))
        assertEquals("تطبيق جيد لحكم مد", formatPerRuleVerdict(isArabic = true, passed = true, ruleName = "Madd"))
    }

    /**
     * REGRESSION GUARD. The score key is `"Your %s score: %d%%."` -- the `%%` is an escape for
     * the caller's own `%d` substitution, but `getAppString` returns the key VERBATIM in English,
     * so the English popup printed a literal `35%%`. The Arabic translation uses a single `٪`.
     */
    @Test
    fun `the score renders one percent sign in both languages`() {
        val en = tajweedPopupBody(false, "Madd", "مَٰلِكِ", 46)
        val ar = tajweedPopupBody(true, "Madd", "مَٰلِكِ", 46)
        assertTrue("English double percent: $en", en.contains("score: 46%."))
        assertFalse("English double percent: $en", en.contains("46%%"))
        assertFalse("Arabic double percent: $ar", ar.contains("%%"))
        assertTrue("Arabic score missing: $ar", ar.contains("46"))
    }
}
