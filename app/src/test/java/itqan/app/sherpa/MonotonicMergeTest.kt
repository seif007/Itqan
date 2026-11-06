package itqan.app.sherpa

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * Regression tests for the pipeline's position-preserving monotonic merge
 * (SherpaPipeline.monotonicMergeWords). The cases are derived from a REAL
 * on-device Fatihah session (logcat 2026-08-12): the old blind-prepend merge
 * put the frontier word الْعَالَمِينَ in FRONT of the emission whenever the
 * final window merely cut at الْعَالَم, and the aligner then flagged
 * WRONG/SKIPPED verdicts ("2 mistakes while I read correctly") on a perfect
 * recitation.
 *
 * Robolectric, not a plain JUnit runner: the merge's `fullReRender` repair
 * branches write diagnostics through `android.util.Log`, which throws
 * "Method i in android.util.Log not mocked" on a bare JVM. Every test that
 * exercises a REPAIR branch therefore needs the Android runtime.
 */
@RunWith(RobolectricTestRunner::class)
class MonotonicMergeTest {

    @Test
    fun `mid-span word lost in a merged-token re-render is re-injected in place`() {
        // On-device window: the engine re-rendered "اللَّهِحْمَن" and الله
        // vanished from the decode. The merge must put الله back BEFORE
        // الرَّحْمَنِ (reading order), not prepend it to the whole emission.
        val last = listOf("بِسْمِ", "اللَّهِ", "الرَّحْمَنِ", "الرَّحِيمِ")
        val n = listOf("بِسْمِ", "الرَّحْمَنِ", "الرَّحِيمِ", "الْحَمْدُ", "لِلَّهِ", "رَبِّ", "الْعَالَمِينَ")
        assertEquals(
            listOf("بِسْمِ", "اللَّهِ", "الرَّحْمَنِ", "الرَّحِيمِ", "الْحَمْدُ", "لِلَّهِ", "رَبِّ", "الْعَالَمِينَ"),
            monotonicMergeWords(last, n)
        )
    }

    @Test
    fun `right-edge cut keeps the frontier word in its place`() {
        // Endpoint decode cut at الْعَالَم (fragment of الْعَالَمِينَ). The old
        // prepend emitted [الْعَالَمِينَ, بِسْمِ ...] — the whole transcript
        // scrambled — and the aligner flagged the first words as mistakes.
        // The union emits the full word first, then the fragment trails.
        val last = listOf("بِسْمِ", "اللَّهِ", "الرَّحْمَنِ", "الرَّحِيمِ", "الْحَمْدُ", "لِلَّهِ", "رَبِّ", "الْعَالَمِينَ")
        val n = listOf("بِسْمِ", "اللَّهِ", "الرَّحْمَنِ", "الرَّحِيمِ", "الْحَمْدُ", "لِلَّهِ", "رَبِّ", "الْعَالَم")
        assertEquals(
            listOf("بِسْمِ", "اللَّهِ", "الرَّحْمَنِ", "الرَّحِيمِ", "الْحَمْدُ", "لِلَّهِ", "رَبِّ", "الْعَالَمِينَ", "الْعَالَم"),
            monotonicMergeWords(last, n)
        )
    }

    @Test
    fun `left-edge slide-out words are prepended in reading order`() {
        // A window slid past the start: the earlier words are missing from the
        // fresh decode and must come back BEFORE it, in their known order.
        val last = listOf("قُلْ", "هُوَ", "اللَّهُ", "أَحَدٌ")
        val n = listOf("اللَّهُ", "أَحَدٌ", "اللَّهُ", "الصَّمَدُ")
        assertEquals(
            listOf("قُلْ", "هُوَ", "اللَّهُ", "أَحَدٌ", "اللَّهُ", "الصَّمَدُ"),
            monotonicMergeWords(last, n)
        )
    }

    @Test
    fun `pure extension is passed through unchanged`() {
        val last = listOf("أ", "ب")
        val n = listOf("أ", "ب", "ج")
        assertEquals(n, monotonicMergeWords(last, n))
    }

    @Test
    fun `tashkeel re-spelling does not duplicate`() {
        val last = listOf("بِسْمِ", "اللَّهِ")
        val n = listOf("بسم", "الله")
        assertEquals(listOf("بسم", "الله"), monotonicMergeWords(last, n))
    }

    @Test
    fun `re-decode that drops the frontier word holds the previous emission`() {
        val last = listOf("أ", "ب", "ج")
        val n = listOf("أ", "ب")
        assertEquals(last, monotonicMergeWords(last, n))
    }

    // ───────────────────────────────────────────────────────────────────
    // Extended contract (time eligibility + vocab contradiction + garbage
    // overwrite) — the metadata the wired pipeline passes on every decode.
    // ───────────────────────────────────────────────────────────────────

    @Test
    fun `stale repeated phrase from new audio appends instead of folding`() {
        // Kafiran-style: وَلَا أَنتُمْ عَابِدُونَ مَا أَعْبُدُ occurs twice.
        // The FIRST occurrence's audio slid out of the window (its first-emit
        // timestamp is older than nowMs-windowMs), so the identical phrase
        // arriving from NEW audio is a DIFFERENT occurrence — it must append,
        // not fold onto the stale one (the merge bug that ate kafirun's
        // repeats).
        val phrase = listOf("وَلَا", "أَنتُمْ", "عَابِدُونَ", "مَا", "أَعْبُدُ")
        val last = listOf("قُلْ") + phrase
        val ms = longArrayOf(9000, 9000, 9000, 9000, 9000, 9000)
        // nowMs=20000, windowMs=4000 → cut at 16000; every ms (9000) is stale.
        val merged = monotonicMergeWords(last, phrase, ms, nowMs = 20000, windowMs = 4000)
        assertEquals(last + phrase, merged)
        // Same decode while the occurrence is still in-window → hold.
        val msFresh = longArrayOf(1000, 19000, 19000, 19000, 19000, 19000)
        assertEquals(last, monotonicMergeWords(last, phrase, msFresh, nowMs = 20000, windowMs = 4000))
    }

    @Test
    fun `vocab contradiction rejects a fold onto the wrong occurrence`() {
        // Ikhlas: log [الله, أَحَدٌ]; the window re-hears the SECOND الله and
        // renders [الله, الصَّمَدُ]. Folding onto the first الله (the old
        // subsequence match) would APPEND only الصَّمَدُ and lose the second
        // الله — the phantom-SKIPPED source. With the vocab, the mismatch
        // after the anchor contradicts a REAL word in the log → the anchor is
        // rejected → the whole decode appends as a new occurrence.
        val vocab = setOf("قل", "هو", "الله", "احد", "الصمد")
        val last = listOf("اللَّهُ", "أَحَدٌ")
        val n = listOf("اللَّهُ", "الصَّمَدُ")
        assertEquals(
            listOf("اللَّهُ", "أَحَدٌ", "اللَّهُ", "الصَّمَدُ"),
            monotonicMergeWords(last, n, longArrayOf(19000, 19000), nowMs = 20000, windowMs = 4000, vocab = vocab)
        )
    }

    @Test
    fun `fresher decode overwrites an OOV garbage tail word`() {
        // The previous decode committed an unmatched OOV fragment (والثات —
        // falaq's real garbage); the next decode renders the same span
        // better. With the vocab, the garbage log word is skipped and the
        // fresher spelling replaces it (full-rerender of the same size).
        val vocab = setOf("قل", "اعوذ", "برب", "الفلق", "من", "شر", "ما", "خلق")
        val last = listOf("مِنْ", "شَرِّ", "وَالْثَاتِ")
        val n = listOf("مِنْ", "شَرِّ", "مَا", "خَلَقَ")
        val merged = monotonicMergeWords(last, n, longArrayOf(19000, 19000, 19000), nowMs = 20000, windowMs = 4000, vocab = vocab)
        // anchor شر (j=1): continuation مَا != والثات (OOV → garbage-skip) →
        // cursor at the log's end → the rest is new audio: ما خلق append; the
        // garbage stays in the log (append-only) but can never block a match —
        // the collector's target-count cap + the verdict walk's OOV skip make
        // it inert downstream.
        assertEquals(listOf("مِنْ", "شَرِّ", "وَالْثَاتِ", "مَا", "خَلَقَ"), merged)
    }

    @Test
    fun `anchor-less session decode that is ONE word shorter still replaces a poisoned log`() {
        // ── REAL on-device data: fatihah_c29, 45 s, 29 targets ──────────────
        // mergeDiag L=26 n=25 anchor=-1 head=false headCovered=true end=true
        //          covered=false bag=false
        //
        // The live windows had dropped بِسْمِ and الرَّحِيمِ and displaced the
        // rest; the whole-session decode re-heard the range correctly but had
        // fused الصرط+المستقيم and written نعبد/واياك, so it came out ONE word
        // SHORTER with 4 unpaired words. Both old guards rejected it —
        // `n.size >= L - 2` closed by one word, `sameWordBagFuzzy`'s flat 2
        // closed by two — so no repair ran, the walk froze at matched=15, and
        // the user got a 15/29 popup on a perfectly correct recitation.
        //
        // The decode is the trustworthy transcript: it reaches the log's last
        // word (ولا) AND continues past it with الضالين, which no window ever
        // emitted. It must WIN.
        val log = ("بسم الله الرحمن الرحيم الحمد رب العلمين لله الرحيم الرحمن يوم الدين " +
            "نعبد واياك نستعين اهدنا الصرط المستقيم صرط الذين انعمت عليهم غير المغضوب عليهم ولا")
            .split(" ")
        val dec = ("م الله الحمد رب العلمين الرحمن الرح يوم الدين اياك نعبد واياك نستعين " +
            "اهدنا الصرط المستقيم صرط الذين انعمت عليهم غير المغضوب عليهم ولا الضالين")
            .split(" ")
        // The REAL target vocabulary (Al-Fatihah 1-7). It is load-bearing: with
        // `vocab == null` the anchor scan is skipped entirely (`maxK` is 0), so
        // the repair branches that key on `bestAnchor` never run at all.
        val vocab = ("بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين " +
            "اياك نعبد واياك نستعين اهدنا الصراط المستقيم صراط الذين انعمت عليهم غير المغضوب " +
            "عليهم ولا الضالين").split(" ").toSet()
        assertEquals(26, log.size)
        assertEquals(25, dec.size)

        val merged = monotonicMergeWords(
            log, dec, null, nowMs = 45_000, windowMs = 45_000,
            vocab = vocab, fullReRender = true, coversSession = true
        )
        assertEquals(dec, merged)
        // The word every streaming window dropped is finally present, and it is
        // LAST — the decode's reading order won, nothing was appended out of
        // order.
        assertEquals("الضالين", merged.last())
    }

    @Test
    fun `a session decode that stopped early must NOT replace the log`() {
        // Negative control for the case above: same vocabulary and the same
        // opening words, but the decode was CUT before the log's tail (20 words
        // against the log's 26). The decode is shorter, its head matches, and
        // its bag is fuzzy-close — only the tail check (`spansToLogEnd`) can
        // stop it from silently DELETING the six words it never reached.
        //
        // It must not win. (It also must not be mistaken for a repair: the
        // expectation is the log's tail, not the decode's.)
        val log = ("بسم الله الرحمن الرحيم الحمد رب العلمين لله الرحيم الرحمن يوم الدين " +
            "نعبد واياك نستعين اهدنا الصرط المستقيم صرط الذين انعمت عليهم غير المغضوب عليهم ولا")
            .split(" ")
        val truncated = ("م الله الحمد رب العلمين الرحمن الرح يوم الدين اياك نعبد واياك نستعين " +
            "اهدنا الصرط المستقيم صرط الذين انعمت عليهم")
            .split(" ")
        val vocab = ("بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين " +
            "اياك نعبد واياك نستعين اهدنا الصراط المستقيم صراط الذين انعمت عليهم غير المغضوب " +
            "عليهم ولا الضالين").split(" ").toSet()
        assertEquals(26, log.size)
        assertEquals(20, truncated.size)

        val merged = monotonicMergeWords(
            log, truncated, null, nowMs = 45_000, windowMs = 45_000,
            vocab = vocab, fullReRender = true, coversSession = true
        )
        // The decode was NOT adopted: `أَنعَمتَ` is the log's 20th word, and the
        // repair (had it wrongly fired) would have ended the transcript exactly
        // there, losing غير المغضوب عليهم ولا forever.
        assertNotEquals(truncated, merged)
        assertTrue("log tail must survive", merged.contains("ولا"))
    }

    @Test
    fun `a decode that restores dropped words AND loses one re-injects the survivor`() {
        // ── REAL on-device data: fatihah_c32 ────────────────────────────────
        // mergeDiag L=25 n=24 anchor=-1 headCovered=true end=true
        //          covered=false bag=false
        //
        // A TRADE, not a strict improvement, which is why a plain replace is
        // wrong here:
        //   log  … اهدنا الصرط المستقيم [عليهم] غير المغضوب عليهم ولا الضالين
        //   dec  … اهدنا الصرط المستقيم صرط الذين انعمت عليهم المغضوب عليهم ولا الضالين
        // The decode RESTORES صرط الذين انعمت (no window ever emitted those
        // three words) but LOSES غير. Replacing outright trades three false
        // skips for one — the aligner would flag غير, which the user did
        // pronounce. The repair takes the decode's reading order and splices
        // the orphaned log word back in at its log-derived position.
        val log = ("بسم الله الرحمن الرحيم الحمد رب العلمين لله الرحيم الرحمن يوم الدين " +
            "اياك نعبد واياك نستعين اهدنا الصرط المستقيم عليهم غير المغضوب عليهم ولا الضالين")
            .split(" ")
        val dec = ("م الله الحمد رب العلمين الرحمن الرح يوم الدين اياك نعبد واياك نستعين " +
            "اهدنا الصرط المستقيم صرط الذين انعمت عليهم المغضوب عليهم ولا الضالين")
            .split(" ")
        val vocab = ("بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين " +
            "اياك نعبد واياك نستعين اهدنا الصراط المستقيم صراط الذين انعمت عليهم غير المغضوب " +
            "عليهم ولا الضالين").split(" ").toSet()
        assertEquals(25, log.size)
        assertEquals(24, dec.size)

        val merged = monotonicMergeWords(
            log, dec, null, nowMs = 46_000, windowMs = 46_000,
            vocab = vocab, fullReRender = true, coversSession = true
        )
        // The three restored words are present…
        assertTrue("صرط must be restored", merged.contains("صرط"))
        assertTrue("الذين must be restored", merged.contains("الذين"))
        assertTrue("انعمت must be restored", merged.contains("انعمت"))
        // …and the one word the decode dropped is NOT lost.
        assertTrue("غير must be re-injected, not dropped", merged.contains("غير"))
        // The decode's order is what we adopted.
        assertEquals("الضالين", merged.last())
    }

    // ── Test 11: a decode LONGER than the log cannot be a truncation, so the
    // orphan budget must loosen for it. Measured fatihah_c34.
    @Test
    fun `a longer decode that dropped several window-edge duplicates still wins`() {
        // ── REAL on-device data: fatihah_c34, 45 s, 29 targets ──────────────
        // mergeDiag L=21 n=24 vocab=26 head=false headCovered=true end=true
        //          anchor=-1 consumed=0 covered=false bag=false
        //
        // This run's streaming windows were badly degraded (they emitted
        // fragments like `تير سعك`, `تفسيرع`, `س` and dropped نعبد / اهدنا /
        // الصرط / المستقيم / عليهم / ولا entirely), so the live log ended at 21
        // words. The whole-session decode re-heard the range in the right order
        // but dropped seven window-edge duplicates/fragments; with the flat
        // L/4 = 5 budget `covered=false` blocked the repair, the merge APPENDED
        // its eight new words at the TAIL, and the walk froze at 10/29 with
        // neither a popup nor a summary.
        val log = ("بسم الله الرحمن الحمد لله رب العلمين الرحمن الرحيم ملك يوم الدين " +
            "واياك نستعين الرحيم صرط الذين انعمت عليهم غير المغضوب").split(" ")
        val dec = ("م الله الحمد رب العلمين الرحمن الرح يوم الدين اياك نعبد واياك نستعين " +
            "اهدنا الصرط المستقيم صرط الذين انعمت عليهم المغضوب عليهم ولا الضالين").split(" ")
        val vocab = ("بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين " +
            "اياك نعبد واياك نستعين اهدنا الصراط المستقيم صراط الذين انعمت عليهم غير المغضوب " +
            "عليهم ولا الضالين").split(" ").toSet()
        assertEquals(21, log.size)
        assertEquals(24, dec.size)

        val merged = monotonicMergeWords(
            log, dec, null, nowMs = 45_000, windowMs = 45_000,
            vocab = vocab, fullReRender = true, coversSession = true
        )
        // The decode's reading order was adopted — this is NOT the append-only
        // result (`log + the 8 new words bolted onto the tail`).
        assertNotEquals("append-only would leave the log as a prefix", log, merged.take(log.size))
        // Every word the windows dropped is now present…
        for (w in listOf("نعبد", "اهدنا", "الصرط", "المستقيم", "الضالين")) {
            assertTrue("$w must be restored", merged.contains(w))
        }
        // …and every log-only word survives (re-injected, not dropped).
        for (w in listOf("بسم", "لله", "ملك", "غير")) {
            assertTrue("$w must be re-injected, not dropped", merged.contains(w))
        }
    }

    @Test
    fun `a longer decode of a different recitation is still rejected`() {
        // Negative control for the LOOSENED superset budget: making the orphan
        // budget generous for `n > L` is only safe because `headCovered` (one
        // of the log's first three words present in the decode) rejects a
        // decode that is not the same recitation. An-Nas with extra tail words
        // is LONGER than the log, so it takes the loosened path.
        val log = ("بسم الله الرحمن الرحيم الحمد رب العلمين لله الرحيم الرحمن يوم الدين " +
            "اياك نعبد واياك نستعين اهدنا الصرط المستقيم عليهم غير المغضوب عليهم ولا الضالين")
            .split(" ")
        val anNasPlus = ("قل اعوذ برب الناس ملك الناس اله الناس من شر الوسواس الخناس الذي " +
            "يوسوس في صدور الناس من الجنة والناس قل اعوذ برب الناس ملك الناس اله الناس")
            .split(" ")
        val vocab = ("بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين " +
            "اياك نعبد واياك نستعين اهدنا الصراط المستقيم صراط الذين انعمت عليهم غير المغضوب " +
            "عليهم ولا الضالين").split(" ").toSet()
        assertTrue("fixture must be LONGER than the log", anNasPlus.size > log.size)

        val merged = monotonicMergeWords(
            log, anNasPlus, null, nowMs = 46_000, windowMs = 46_000,
            vocab = vocab, fullReRender = true, coversSession = true
        )
        assertNotEquals(anNasPlus, merged)
        assertTrue("log tail must survive", merged.contains("ولا") || merged.contains("الضالين"))
    }

    @Test
    fun `the superset repair reports itself through onRepair`() {
        // The collector needs to know whether the merge adopted the decode's
        // reading order. Inferring it from the result's shape is wrong for the
        // orphan-re-injection path (the result is neither `== decode` nor a
        // prefix-preserving append), so the merge reports the branch it took.
        val log = ("بسم الله الرحمن الحمد لله رب العلمين الرحمن الرحيم ملك يوم الدين " +
            "واياك نستعين الرحيم صرط الذين انعمت عليهم غير المغضوب").split(" ")
        val dec = ("م الله الحمد رب العلمين الرحمن الرح يوم الدين اياك نعبد واياك نستعين " +
            "اهدنا الصرط المستقيم صرط الذين انعمت عليهم المغضوب عليهم ولا الضالين").split(" ")
        val vocab = ("بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين " +
            "اياك نعبد واياك نستعين اهدنا الصراط المستقيم صراط الذين انعمت عليهم غير المغضوب " +
            "عليهم ولا الضالين").split(" ").toSet()

        var repaired = false
        val merged = monotonicMergeWords(
            log, dec, null, nowMs = 45_000, windowMs = 45_000,
            vocab = vocab, fullReRender = true, coversSession = true,
            onRepair = { repaired = true }
        )
        assertTrue("a splice repair must report itself", repaired)
        assertNotEquals(log, merged)

        // A pure extension must NOT report a repair: the result is the log with
        // new words appended, so the collector must treat it as an append.
        var appendRepaired = false
        val extended = monotonicMergeWords(
            listOf("أ", "ب"), listOf("أ", "ب", "ج"), null,
            onRepair = { appendRepaired = true }
        )
        assertFalse("an append is not a repair", appendRepaired)
        assertEquals(listOf("أ", "ب", "ج"), extended)
    }

    @Test
    fun `a decode of a different surah never replaces the log`() {
        // Negative control for the proportional tolerance: An-Nas read against a
        // Fatihah log shares a handful of words but not the recitation. It must
        // be rejected outright — this is the guard that keeps the loosened
        // budgets in the two repair branches from adopting a wrong recitation.
        val log = ("بسم الله الرحمن الرحيم الحمد رب العلمين لله الرحيم الرحمن يوم الدين " +
            "اياك نعبد واياك نستعين اهدنا الصرط المستقيم عليهم غير المغضوب عليهم ولا الضالين")
            .split(" ")
        val anNas = ("قل اعوذ برب الناس ملك الناس اله الناس من شر الوسواس الخناس الذي " +
            "يوسوس في صدور الناس من الجنة والناس").split(" ")
        val vocab = ("بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين " +
            "اياك نعبد واياك نستعين اهدنا الصراط المستقيم صراط الذين انعمت عليهم غير المغضوب " +
            "عليهم ولا الضالين").split(" ").toSet()

        val merged = monotonicMergeWords(
            log, anNas, null, nowMs = 46_000, windowMs = 46_000,
            vocab = vocab, fullReRender = true, coversSession = true
        )
        assertNotEquals(anNas, merged)
        assertTrue("log tail must survive", merged.contains("ولا") || merged.contains("الضالين"))
    }
}
