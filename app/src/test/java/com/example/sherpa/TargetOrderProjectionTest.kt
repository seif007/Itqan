package com.example.sherpa

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * §4.41 — THE HEARD WORDS ARE ORDERED BY THE TARGET, NOT BY THE DECODE.
 *
 * `monotonicMergeWords`'s superset repair adopts the DECODE's reading order on
 * the premise that the session-covering decode heard the log's words "in the
 * right order". Measured on Al-Fatihah 1–7, four consecutive runs of the SAME
 * injected audio, the premise fails exactly when the repair fires:
 *
 *   p1_fatihah   repair fired 0x  -> maxMatched=29  CLEAN
 *   p2_fatihah   repair fired 0x  -> maxMatched=29  CLEAN
 *   p3_fatihah   repair fired 1x  -> maxMatched=17  NOT CLEAN
 *   p4_fatihah   repair fired 1x  -> maxMatched=17  NOT CLEAN
 *
 * Both failing decodes rotate the HEAD — الْحَمْدُ emitted before
 * الرَّحْمَٰنِ الرَّحِيمِ — while the live LOG keeps the correct head. Neither
 * source is trustworthy on ORDER; the target always is.
 *
 * `projectOntoTargetOrder` reorders the union of the two sources into the
 * target's order. It can only REORDER, never INVENT, and these tests pin that:
 * a word heard in neither source stays absent, a word the user never said never
 * appears, and the target's multiplicity caps a log bloated by duplicated
 * window-edge re-appends.
 *
 * The fixtures are the NORMALIZED forms the device actually printed
 * (`mergeDiag log=… / dec=…`), so `normalizeArabic` is idempotent on them —
 * `normalizeArabic` drops the dagger alif U+0670, which is why the Uthmani
 * `ٱلْعَٰلَمِينَ` reaches the merge as `العلمين`.
 */
class TargetOrderProjectionTest {

    /**
     * Al-Fatihah 1–7, 29 target words, in target order (normalized).
     *
     * NOTE `ملك` (11th word), not `مالك`: the Uthmani مَٰلِكِ spells the alif as a
     * DAGGER ALIF (U+0670), and `normalizeArabic` skips U+0670 — so the
     * normalized target form the merge scores against is `ملك`. Writing the
     * plain-alif `مالك` here makes the fixture disagree with the device (it did,
     * and these two tests caught it).
     */
    private val fatihah = listOf(
        "بسم", "الله", "الرحمن", "الرحيم", "الحمد", "لله", "رب", "العلمين",
        "الرحمن", "الرحيم", "ملك", "يوم", "الدين", "اياك", "نعبد", "واياك",
        "نستعين", "اهدنا", "الصرط", "المستقيم", "صرط", "الذين", "انعمت",
        "عليهم", "غير", "المغضوب", "عليهم", "ولا", "الضالين",
    )

    /**
     * The measured `p4_fatihah` pair, verbatim from
     * `mergeDiag L=23 n=24 … log='…' / dec='…'`:
     *
     *   log  بسم الله الرحمن الرحيم الحمد لله رب العلمين الرحيم يوم الدين ملك
     *        نعبد واياك نستعين المستقيم الذين انعمت عليهم غير المغضوب عليهم ولا
     *   dec  م الله الحمد رب العلمين الرحمن الرحيم يوم الدين اياك نعبد واياك
     *        نستعين اهدنا الصرط المستتق صرط الذين انعمت عليهم المغضوب عليهم ولا
     *        الضالين
     *
     * The decode ROTATED the head (`الحمد` third instead of `الرحمن`), which is
     * what collapsed the walk to `matched=4`.
     */
    private val p4Log = listOf(
        "بسم", "الله", "الرحمن", "الرحيم", "الحمد", "لله", "رب", "العلمين",
        "الرحيم", "يوم", "الدين", "ملك", "نعبد", "واياك", "نستعين", "المستقيم",
        "الذين", "انعمت", "عليهم", "غير", "المغضوب", "عليهم", "ولا",
    )
    private val p4Dec = listOf(
        "م", "الله", "الحمد", "رب", "العلمين", "الرحمن", "الرحيم", "يوم", "الدين",
        "اياك", "نعبد", "واياك", "نستعين", "اهدنا", "الصرط", "المستتق", "صرط",
        "الذين", "انعمت", "عليهم", "المغضوب", "عليهم", "ولا", "الضالين",
    )

    /**
     * The measured `p3_fatihah` pair (`mergeDiag L=22 n=25 …`). It is the
     * SECOND independent reproduction of the same head rotation:
     *
     *   log  بسم الله الرحمن الرحيم الحمد رب العلمين غير الرحيم الرحمن ملك يوم
     *        الدين اياك نعبد واياك نستعين الصرط المستقيم المغضوب عليهم ولا
     *   dec  بسم الله الحمد رب العلمين الرحمن الرحيم يوم الدين اياك نعبد واياك
     *        نستعين اهدنا الصرط المستقيم صرط الذين انعمت عليهم غير المغضوب
     *        عليهم ولا الضالين
     *
     * NOTE: neither source contains `لله` — the decode rendered it as the
     * one-letter fragment `لِ`, which normalizes to `ل`. The projection must
     * therefore return 28 words, NOT 29: it may reorder, never invent.
     */
    private val p3Log = listOf(
        "بسم", "الله", "الرحمن", "الرحيم", "الحمد", "رب", "العلمين", "غير",
        "الرحيم", "الرحمن", "ملك", "يوم", "الدين", "اياك", "نعبد", "واياك",
        "نستعين", "الصرط", "المستقيم", "المغضوب", "عليهم", "ولا",
    )
    private val p3Dec = listOf(
        "بسم", "الله", "الحمد", "رب", "العلمين", "الرحمن", "الرحيم", "يوم",
        "الدين", "اياك", "نعبد", "واياك", "نستعين", "اهدنا", "الصرط",
        "المستقيم", "صرط", "الذين", "انعمت", "عليهم", "غير", "المغضوب",
        "عليهم", "ولا", "الضالين",
    )

    // ─────────────────────────────────────────────────────────────────────
    // The measured runs
    // ─────────────────────────────────────────────────────────────────────

    /** The p4 union contains all 29 target words, so it must project to all 29. */
    @Test
    fun `the p4 union projects to the whole 29-word target in target order`() {
        val projected = projectOntoTargetOrder(p4Log + p4Dec, fatihah)
        assertEquals(fatihah, projected)
        assertEquals(29, projected.size)
    }

    /** The p4 projection must fix the rotated head the decode introduced. */
    @Test
    fun `the p4 projection restores the head the decode rotated`() {
        val projected = projectOntoTargetOrder(p4Log + p4Dec, fatihah)
        // Target head: بسم الله الرحمن الرحيم الحمد …
        assertEquals(listOf("بسم", "الله", "الرحمن", "الرحيم", "الحمد"), projected.take(5))
        // The decode had put الحمد third; it must no longer be third.
        assertFalse(projected[2] == "الحمد")
    }

    /**
     * p3's union is one word short: `لله` is in neither source. The projection
     * must NOT fill the hole — a word nobody heard must stay unheard, or the
     * app would forgive a genuine skip.
     */
    @Test
    fun `the p3 union projects to 28 words because neither source heard لله`() {
        val projected = projectOntoTargetOrder(p3Log + p3Dec, fatihah)
        assertEquals(28, projected.size)
        assertFalse("a word heard in neither source must never be invented", projected.contains("لله"))
        // Everything else is present, in the target's order.
        assertEquals(fatihah.filter { it != "لله" }, projected)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Reorder-only: never invent, never drop a heard target word
    // ─────────────────────────────────────────────────────────────────────

    /** A word the user never said cannot appear just because the target has it. */
    @Test
    fun `a target word absent from both sources is not invented`() {
        val heard = listOf("بسم", "الله", "الرحيم", "الحمد")   // الرحمن never heard
        val projected = projectOntoTargetOrder(heard, fatihah)
        assertFalse(projected.contains("الرحمن"))
        assertEquals(listOf("بسم", "الله", "الرحيم", "الحمد"), projected)
    }

    /** Order is the target's even when the heard list arrives fully reversed. */
    @Test
    fun `a reversed heard list is reordered into the target's order`() {
        val heard = listOf("الضالين", "ولا", "عليهم", "بسم")
        assertEquals(
            listOf("بسم", "عليهم", "ولا", "الضالين"),
            projectOntoTargetOrder(heard, fatihah)
        )
    }

    /** A heard word that is not in the target (a stray/OOV fragment) is dropped. */
    @Test
    fun `a heard word outside the target vocabulary is dropped`() {
        val heard = listOf("بسم", "م", "الله", "تفسير")
        assertEquals(
            listOf("بسم", "الله"),
            projectOntoTargetOrder(heard, fatihah)
        )
    }

    /** A hearer who said a word twice yields it twice when the target has it twice. */
    @Test
    fun `repeats survive up to the target's own multiplicity`() {
        // الرحمن and الرحيم each occur twice in Al-Fatihah (words 3/4 and 9/10).
        val heard = listOf("الرحمن", "الرحيم", "الرحمن", "الرحيم")
        assertEquals(
            listOf("الرحمن", "الرحيم", "الرحمن", "الرحيم"),
            projectOntoTargetOrder(heard, fatihah)
        )
    }

    /**
     * …and NOT beyond it. A log bloated by duplicated window-edge re-appends
     * (measured elsewhere: Falaq's شَرِّ committed ten times) must be trimmed to
     * the target's count, or every later word shifts one slot and the aligner
     * charges a false SKIPPED for each.
     */
    @Test
    fun `a log bloated beyond the target's multiplicity is trimmed`() {
        val heard = List(10) { "الرحمن" } + listOf("بسم", "الله")
        val projected = projectOntoTargetOrder(heard, fatihah)
        assertEquals(2, projected.count { it == "الرحمن" })
        assertEquals(listOf("بسم", "الله", "الرحمن", "الرحمن"), projected)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Spelling, and the degenerate inputs
    // ─────────────────────────────────────────────────────────────────────

    /**
     * The LOG's spelling wins for display — it is the already-committed
     * rendering. p4's decode rendered the word as `المستتق` while the log had
     * `المستقيم`; the emitted word must be the log's, and the decode's mangled
     * form must not survive as a second copy.
     */
    @Test
    fun `the log's spelling wins over the decode's for the same word`() {
        val projected = projectOntoTargetOrder(p4Log + p4Dec, fatihah)
        assertEquals(1, projected.count { it == "المستقيم" })
        assertFalse(projected.contains("المستتق"))
    }

    /** An empty heard list returns empty; an empty target returns the input. */
    @Test
    fun `degenerate inputs are returned unchanged`() {
        assertTrue(projectOntoTargetOrder(emptyList(), fatihah).isEmpty())
        val heard = listOf("بسم", "الله")
        assertEquals(heard, projectOntoTargetOrder(heard, emptyList()))
    }

    /**
     * The projection never emits a word the target does not contain, so the
     * caller can safely hand its result straight to the aligner.
     */
    @Test
    fun `every emitted word is in the target`() {
        val projected = projectOntoTargetOrder(p4Log + p4Dec + listOf("تفسير", "xx"), fatihah)
        assertTrue(projected.all { it in fatihah })
    }
}
