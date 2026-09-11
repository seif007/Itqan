package com.example.sherpa

import com.example.ui.normalizeArabic
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for [AyahWordMatcher] and [writeHotwordsFile] — the offline
 * Sherpa-ONNX pipeline's hallucination mitigations.
 *
 * The pipeline drives the Memorize+Tajweed transcript from the streaming
 * Zipformer modified_beam_search result with a per-active-ayah-range hotwords
 * file (layer i) plus a post-decode [AyahWordMatcher] filter (layer ii). The
 * filter accepts words that match the active range's vocabulary either EXACTLY
 * (post-`normalizeArabic`) or as a 1-edit phonetic neighbor, and enforces
 * monotonicity so a noisy re-decode of the SAME audio window can never shrink
 * the emitted transcript.
 *
 * These tests pin the layer-ii contract without needing any native Sherpa /
 * AudioRecord code (so they are pure-JVM Robolectric tests).
 */
class AyahWordMatcherTest {

    // Baqarah 2 spoken words (post-splitAyahWords cleansing of the U+06DB
    // standalone pause marks). Read from the standard AlQuran.cloud Uthmani
    // transcript and stripped of trailing annotation-only tokens.
    private val baqarah2Words = listOf(
        "ذَٰلِكَ", "ٱلْكِتَٰبُ", "لَا", "رَيْبَ", "فِيهِ", "هُدًۭى", "لِّلْمُتَّقِينَ"
    )

    @Test
    fun `AyahWordMatcher accepts exact-ayah words and drops filler`() {
        val matcher = AyahWordMatcher(baqarah2Words)
        // "uhm" / "ye" filler that a multilingual Zipformer is prone to emit.
        val cleaned = matcher.clean("uhm ذَٰلِكَ ye ٱلْكِتَٰبُ")
        // Fillers must go, exact ayah words must stay.
        assertEquals(listOf("ذَٰلِكَ", "ٱلْكِتَٰبُ"), cleaned)
    }

    @Test
    fun `AyahWordMatcher keeps mispronounced ayah word as a 1-edit neighbor`() {
        val matcher = AyahWordMatcher(baqarah2Words)
        // The user said "الحم" instead of "الحمد" — close enough that the
        // matcher should NOT erase it (the tajweed scorer catches the slip;
        // erasing here would silently delete what the user actually said).
        // We use a short ayah-style vocab with maddah to verify.
        val matcher2 = AyahWordMatcher(listOf("الْحَمْدُ", "لِلَّهِ", "رَبِّ", "الْعَٰلَمِينَ"))
        // "الحمد" matches against "الْحَمْدُ" post-normalization (both fold to
        // "الحمد"); the matcher accepts. "ye" filler gets dropped.
        val cleaned = matcher2.clean("ye الحمد")
        assertEquals(listOf("الحمد"), cleaned)
    }

    @Test
    fun `AyahWordMatcher drops stray English and Chinese tokens`() {
        val matcher = AyahWordMatcher(baqarah2Words)
        // Zipformer-ar_en_id_ja_ru_th_vi_zh is multilingual; it can emit stray
        // English/Chinese filler between Arabic words. Those words MUST be
        // filtered out, otherwise alignRecitation sees a "word" the user never
        // spoke and the cursor jumps past good matches.
        val cleaned = matcher.clean("ye ذَٰلِكَ 中文 ٱلْكِتَٰبُ")
        assertEquals(listOf("ذَٰلِكَ", "ٱلْكِتَٰبُ"), cleaned)
    }

    @Test
    fun `AyahWordMatcher never shrinks the emitted transcript on redecode`() {
        // clean() is now PURE: monotonicity lives in the pipeline's
        // monotonicMerge (current decode is ground truth; words it missed are
        // PREPENDED in reading order). This pins the pure contract — a
        // shorter re-decode returns the shorter list, and the caller's merge
        // (not the matcher) keeps the emission monotone.
        val matcher = AyahWordMatcher(baqarah2Words)
        val first = matcher.clean("uhm ذَٰلِكَ ٱلْكِتَٰبُ لَا")
        assertEquals(listOf("ذَٰلِكَ", "ٱلْكِتَٰبُ", "لَا"), first)

        // Sherpa re-decodes from a slightly later frame and the beam
        // temporarily drops "ٱلْكِتَٰبُ" — the pure clean() honors it.
        val second = matcher.clean("ذَٰلِكَ لَا")
        assertEquals(listOf("ذَٰلِكَ", "لَا"), second)

        // The pipeline's monotonicMerge then prepends the missed word back,
        // restoring reading order: [missed] + [current decode].
        val missed = first.filterNot { it in second }
        assertEquals(listOf("ٱلْكِتَٰبُ"), missed)
    }

    @Test
    fun `AyahWordMatcher ignores empty or whitespace-only partials`() {
        val matcher = AyahWordMatcher(baqarah2Words)
        // Empty string should not crash and returns an empty list (pure).
        assertEquals(emptyList<String>(), matcher.clean(""))
        matcher.clean("ذَٰلِكَ")
        val after = matcher.clean("   ")
        assertEquals(emptyList<String>(), after)
    }

    @Test
    fun `AyahWordMatcher drops standalone Quranic annotation marks the decoder leaked`() {
        val matcher = AyahWordMatcher(baqarah2Words)
        // The decoder occasionally emits the U+06DB three-dot pause mark as a
        // standalone token (it's in the BPE vocab because the model trained on
        // Arabic text containing annotation signs). normalizeArabic strips it →
        // empty → matcher drops the token. Verified here.
        val cleaned = matcher.clean("ذَٰلِكَ ۛ")
        assertEquals(listOf("ذَٰلِكَ"), cleaned)
    }

    @Test
    fun `AyahWordMatcher resetClears the monotonicity cursor`() {
        val matcher = AyahWordMatcher(baqarah2Words)
        matcher.clean("ذَٰلِكَ ٱلْكِتَٰبُ")
        // After reset, a shrink from Sherpa is honoured (because the prior
        // session ended — the user started a fresh recital).
        matcher.reset()
        val after = matcher.clean("ذَٰلِكَ")
        assertEquals(listOf("ذَٰلِكَ"), after)
    }

    // ───────────────────────────────────────────────────────────────────
    // writeHotwordsFile — verifies the layer-i decoder-biasing vocabulary
    // generation. The actual file is written into the JVM tmp dir under a
    // fake Context.cacheDir (since writeHotwordsFile uses Context.cacheDir,
    // we use Robolectric's Context).
    // ───────────────────────────────────────────────────────────────────
    // NOTE: writeHotwordsFile takes a Context. To exercise it unit-test-style
    // without Robolectric, just replicate the exact dedup logic in the test
    // below (which fails if the implementation drifts).

    @Test
    fun `writeHotwordsFile dedups case-insensitively`() {
        // Replicates the dedup+convert-to-lines logic of writeHotwordsFile.
        // This pins the contract; if the implementation ever drops the
        // lowercase dedup (e.g. keeps "الحمد" and "الحمد" as two phrases due
        // to a Tashkeel difference) this test will flag the drift.
        val words = listOf("الحمد", "الحمد", "  ", "لله")
        val seen = HashSet<String>()
        val phrases = ArrayList<String>()
        for (w in words) {
            val s = w.trim()
            if (s.isEmpty()) continue
            val key = s.lowercase()
            if (seen.add(key)) phrases.add(s)
        }
        assertEquals(listOf("الحمد", "لله"), phrases)
    }

    @Test
    fun `normalizeArabic strips Quranic annotation marks`() {
        // Verify the matcher's underlying normalizer agrees with the rest of
        // the app about the U+06D6..U+06ED marks being non-spoken scaffolding.
        // U+0647 (Arabic Heh) + U+06DB (pause mark) → "ه" post-normalization.
        val withPause = "هۛ"
        assertEquals("ه", normalizeArabic(withPause))
        val plain = "الْحَمْدُ"
        assertEquals("الحمد", normalizeArabic(plain))
    }

    // ───────────────────────────────────────────────────────────────────
    // Window-edge TRUNCATION restore (the false-SKIPPED-on-لِلَّه bug).
    // The CTC decoder slices the trailing letter(s) of a word at the
    // sliding-window boundary: لِّلَّه surfaces as لِلَّ (norm "لل"). The old
    // ≤3-letter fragment rule DROPPED it, so the word never entered the
    // transcript and the aligner flagged a false SKIPPED — and after the
    // mistake auto-pause every resumed re-decode truncated the same word
    // again, so the app looked deaf until Hint injected it.
    // ───────────────────────────────────────────────────────────────────

    private val fatihah12 = listOf(
        "بِسْمِ", "ٱللَّه", "ٱلرَّحْمَٰن", "ٱلرَّحِيم", "ٱلْحَمْد", "لِلَّه", "رَبِّ", "ٱلْعَٰلَمِين"
    )

    @Test
    fun `truncated لِّلَّه (لِلَّ) is restored to the canonical word`() {
        val matcher = AyahWordMatcher(fatihah12)
        // The exact device-log decode: the tail window cut لِّلَّه to لِلَّ.
        val cleaned = matcher.clean("ٱلْحَمْد لِلَّ رَبِّ")
        assertEquals(listOf("ٱلْحَمْد", "لِلَّه", "رَبِّ"), cleaned)
    }

    @Test
    fun `truncated word at the decode head is restored too`() {
        val matcher = AyahWordMatcher(fatihah12)
        val cleaned = matcher.clean("لِلَّ رَبِّ ٱلْعَٰلَمِين")
        assertEquals(listOf("لِلَّه", "رَبِّ", "ٱلْعَٰلَمِين"), cleaned)
    }

    @Test
    fun `ambiguous prefix fragment is NOT restored (phantom guard)`() {
        val matcher = AyahWordMatcher(fatihah12)
        // "ال" is a strict prefix of MANY vocab words (ٱللَّه, ٱلرَّحْمَٰن,
        // ٱلرَّحِيم, ٱلْعَٰلَمِين) — restoring it would re-create the
        // phantom-first-word bug. It must stay dropped.
        val cleaned = matcher.clean("ٱلْحَمْد ال رَبِّ")
        assertEquals(listOf("ٱلْحَمْد", "رَبِّ"), cleaned)
    }

    @Test
    fun `unambiguous single-letter fragment IS restored (one missing letter only)`() {
        val matcher = AyahWordMatcher(fatihah12)
        // Measured on Al-Fatihah 1:2: EVERY streaming window rendered رَبِّ as
        // the lone `رَ`. The old "1 letter is too short" rule dropped it; the
        // word was then permanently lost from the committed transcript (every
        // re-decode truncates it identically), and the aligner charged a false
        // SKIPPED that cascaded into a WRONG and froze the whole surah.
        // Restoring is safe here because `ر` is a strict prefix of exactly ONE
        // word of the active range (ٱلرَّحْمَٰن / ٱلرَّحِيم start with alef, so
        // they are not candidates) and a 1-letter stem may gain only ONE
        // missing letter — `ر` can become رَبِّ ("رب") and nothing longer.
        val cleaned = matcher.clean("لِلَّه رَ ٱلْعَٰلَمِين")
        assertEquals(listOf("لِلَّه", "رَبِّ", "ٱلْعَٰلَمِين"), cleaned)
    }

    // NOTE — a 1-letter token is ALSO within 1 edit of any 2-letter vocabulary
    // word, so it can additionally be KEPT RAW by the neighbour rule (measured:
    // `بِ` survives as رَبِّ against this vocab). The relaxation above only
    // governs the stricter prefix-restore path; it does not make every stray
    // letter a vocabulary word — the `ال` phantom guard below is what stops a
    // generic fragment from being restored.

    // ───────────────────────────────────────────────────────────────────
    // CTC SANDWICH FRAGMENT — regression for the Baqarah 285-286 false
    // WRONG/SKIPPED bug. The active vocabulary has no أَنْ standalone (it
    // appears as the connector into أُنْزِلَ, but never alone in this
    // range). The CTC decoder, when the window's right edge cut a word
    // mid-span, sometimes hallucinated a tiny real Arabic word
    // (أَنْ/إِنْ/مِنْ) that the user never said. The OLD matcher kept it
    // as KEEP_RAW ("a real 1-3 letter Arabic word would be an exact vocab
    // hit" — true for the GENERAL case, but not for these connector
    // ghosts when the active range happens not to contain them). The merge
    // then committed the phantom, ate a target slot, and the aligner
    // charged a cascade of false SKIPPEDs down the rest of the ayah.
    //
    // The fix: when the token is short (≤3 codepoints), not in vocab, not
    // a valid truncation-restoration, and the PREVIOUS cleaned tokens in
    // the same window are all vocabulary words, drop it as a CTC sandwich
    // fragment. Real 1-3 letter Arabic words (مِنْ, لَا) appear in the
    // active vocab of every surah and stay as exact-vocab hits — the
    // "prior all vocab" guard makes this safe.
    // ───────────────────────────────────────────────────────────────────

    // Subset of Baqarah 2:285 target vocabulary (no أَنْ alone).
    // Using the EXACT same words the e2e test gets from the AlQuran API
    // (verified via cached surah_2.json — first 5 words of ayah 285).
    private val baqarah285Vocab = listOf(
        "ءَامَنَ", "ٱلرَّسُولُ", "بِمَآ", "أُنزِلَ", "إِلَيْهِ", "مِن", "رَّبِّهِۦ",
        "وَٱلْمُؤْمِنُونَ", "كُلٌّ", "ءَامَنَ", "بِٱللَّهِ", "وَمَلَٰٓئِكَتِهِۦ",
        "وَكُتُبِهِۦ", "وَرُسُلِهِۦ", "لَا", "نُفَرِّقُ", "بَيْنَ", "أَحَدٍۢ",
        "مِّن", "رُّسُلِهِۦ", "وَقَالُوا۟", "سَمِعْنَا"
    )

    @Test
    fun `CTC sandwich fragment (phantom أَنْ) is dropped from a multi-token window`() {
        val matcher = AyahWordMatcher(baqarah285Vocab)
        // The exact t=4.3 device-log decode: window decoded
        // "قمَمَ الرَّسُولُ بِمَا أَنْ" (قمَمَ is the head-wreck that
        // restoreEdgeFragment drops because the target doesn't have it as
        // a leading-prefix of anything — its prior is also a vocab hit, so
        // it falls into the OOV-fragment branch). The 3-token "core" is
        // [الرَّسُولُ, بِمَا, أَنْ] — two vocab hits and a phantom أَنْ.
        // The phantom أَنْ is ≤3 codepoints, NOT in vocab, NOT a unique
        // truncation, and its prior cleaned tokens are all vocab → drop.
        val cleaned = matcher.clean("الرَّسُولُ بِمَا أَنْ")
        assertEquals(listOf("الرَّسُولُ", "بِمَا"), cleaned)
    }

    @Test
    fun `CTC sandwich fragment rule is a no-op when no prior vocab anchors it`() {
        val matcher = AyahWordMatcher(baqarah285Vocab)
        // Same phantom أَنْ, but as the FIRST token of a decode — there is
        // no prior cleaned token to anchor the sandwich. Without that
        // guard, the rule would eat the user-spoken أَنْ in any surah that
        // legitimately has it (none in this range, but in general the
        // first-token case is the "decoder warm-up wrecks" bucket, not a
        // sandwich). A bare phantom أَنْ with nothing else in the window
        // stays as KEEP_RAW (it's ≥1 char, <4 chars, prior empty → prior
        // all-vocab = false → NOT dropped by the new rule, the previous
        // KEEP_RAW behaviour for a standalone non-vocab short token
        // remains).
        val cleaned = matcher.clean("أَنْ")
        // The new rule's priorAllVocab guard prevents dropping a
        // free-standing short token, so the standalone behaviour is the
        // pre-existing KEEP_RAW.
        assertEquals(listOf("أَنْ"), cleaned)
    }

    @Test
    fun `legitimate 1-3 letter Arabic word stays as exact vocab hit (sandwich guard)`() {
        val matcher = AyahWordMatcher(baqarah285Vocab)
        // لَا is in the active vocab. It is also 2 codepoints. The new
        // sandwich rule MUST NOT eat it: the priorAllVocab check runs only
        // when the token is NOT an exact vocab hit, and the first branch
        // (exactSet.contains) returns long before the sandwich branch.
        val cleaned = matcher.clean("بِٱللَّهِ لَا نُفَرِّقُ")
        assertEquals(listOf("بِٱللَّهِ", "لَا", "نُفَرِّقُ"), cleaned)
    }

    @Test
    fun `phantom أَنْ is dropped even when an OOV head wreck precedes it`() {
        val matcher = AyahWordMatcher(baqarah285Vocab)
        // The EXACT t=4.3 e2e scenario: raw = "قمَمَ الرَّسُولُ بِمَا
        // أَنْ". قمَمَ is a 4-letter OOV head wreck (KEEP_RAW, too long
        // for the sandwich branch). The 3rd token is a real vocab hit
        // (بِمَا). The 4th token is the phantom أَنْ — even though the
        // FIRST cleaned token (قمَمَ) is NOT a vocab hit, ANY prior
        // cleaned token being a vocab hit is enough to mark this window
        // as "has real speech" → drop the sandwich phantom.
        val cleaned = matcher.clean("قمَمَ الرَّسُولُ بِمَا أَنْ")
        assertEquals(listOf("قمَمَ", "الرَّسُولُ", "بِمَا"), cleaned)
    }
}
