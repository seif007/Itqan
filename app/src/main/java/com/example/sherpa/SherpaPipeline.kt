package com.example.sherpa

import android.content.Context
import android.util.Log
import com.example.ui.normalizeArabic as normalizeArabicUi
import kotlin.math.max
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

// ─────────────────────────────────────────────────────────────────────────────
// Itqān — Quran-STT FastConformer (CTC) offline recognition pipeline.
//
// This file implements the **single-AudioRecord shared pipeline** for the
// Memorize+Tajweed mode: exactly one `AudioRecord` instance feeds BOTH
// (a) the ONNX Runtime engine in QuranSttEngine.kt, running the NeMo
// **FastConformer large CTC head** model from `Muno459/fastconformer-quran`
// (fine-tuned on EveryAyah + Tlog + Muaalem-annotated Quran recitation,
// int8-quantized, NPL-1.1 non-commercial) and (b) the existing TajweedNet
// acoustic scorer (which receives the same PCM samples via a `FloatArray`
// sink). Using one capture session — instead of the platform
// `SpeechRecognizer` in parallel with our own `AudioRecord` — preserves the
// single-mic invariant that previously broke recitation on the Samsung A05s
// (which starves partials as soon as two HAL clients compete for the mic).
//
// This branch replaced the previous sherpa-onnx Tarteel-Whisper recognizer
// (whisper-tiny-ar-quran) — a full-context attention decoder that won Arabic
// accuracy but reads the whole window each time. The CTC engine is sparse and
// lock-free in comparison; it decodes with plain greedy argmax + blank
// collapse on every window.
//
// The model is NOT a streaming engine: it transcribes a WHOLE audio window.
// To get near-real-time display we emulate streaming with **sliding-window
// re-decode**: every ~1.5 s the pipeline decodes the trailing ~4 s window,
// diffs the result against what has already been emitted (the collector's
// committed-word cursor dedups the overlap — see RecitationApp), and emits
// only the NEW words. A final decode of the session's tail (up to 30 s) runs
// on stop()/pause() and is emitted as an `endpoint=true` event so the
// transcript commits everything.
//
// Misrecognition mitigation is layered:
//   (i)   the speech gate below never decodes silence windows (and CTC does
//         not hallucinate filler out of silence the way attention decoders
//         do).
//   (ii)  the `AyahWordMatcher` post-filter (unchanged) drops tokens that
//         match no ayah word and fuzzy-corrects near-misses to the canonical
//         spelling.
//   (iii) the collector's committed-count dedup means a re-decode of an
//         already-seen window can never duplicate or retract words.
//
// Model weight layout: the two model files are bundled as APK assets
// (`assets/quran_stt/`) — a ~125 MB int8 ONNX graph (CTC head export of
// Muno459/fastconformer-quran) plus the SentencePiece token map. On first
// launch `SherpaModelDownloader` copies them into `filesDir/sherpa_model/`
// (it keeps its name/API so the Compose download dialog flow works
// unchanged). The model is NPL-1.1 licensed (non-commercial).
// ─────────────────────────────────────────────────────────────────────────────

private const val TAG = "ItqanSherpa"

/** Asset subfolder + model files (sizes = exact bytes of the exported files). */
private const val ASSET_SUBDIR = "quran_stt"

private data class ModelFile(val name: String, val size: Long)

private val MODEL_FILES = listOf(
    ModelFile("model.q8.onnx", 131_801_646L),
    ModelFile("tokens.txt",       12_847L),
)
private val MODEL_TOTAL_BYTES = MODEL_FILES.sumOf { it.size }

/** Absolute on-disk model dir under `filesDir`. */
fun sherpaModelDir(context: Context): File =
    File(context.filesDir, "sherpa_model").apply { mkdirs() }

/** True iff every model file is present at its known size. */
fun isSherpaModelReady(context: Context): Boolean {
    val dir = sherpaModelDir(context)
    return MODEL_FILES.all { f ->
        val file = File(dir, f.name)
        file.exists() && file.length() >= f.size
    }
}

/**
 * Thin Arabic normalizer shim. Delegates to [com.example.ui.normalizeArabic]
 * so the matcher uses EXACTLY the same folding rules (أ/إ/آ/ء→ا, ة→ه, ى/ئ→ي,
 * ؤ→و + strip tashkeel/maddah/Quranic annotation signs U+06D6..U+06ED) that
 * the memorize-aligner uses for `targetNorm`. Keeping one source of truth
 * avoids drift between "what the matcher thinks is matching" and "what the
 * aligner thinks is matching" — a token the matcher accepts must also survive
 * alignment, or it would pop a phantom wrong-word mistake.
 */
private fun normalizeArabicPub(text: String): String = normalizeArabicUi(text)

/**
 * Copies the bundled Quran-STT model assets into `filesDir/sherpa_model/` if
 * not already present. Reports per-file and overall progress via [progress].
 *
 * Keeps the exact downloader API (Progress flow + [isComplete]) so the
 * Compose model-ready dialog in RecitationApp works unchanged — it only ever
 * "downloads" from the APK's own assets now, which makes first launch
 * instant (a copy, not a 300 MB network transfer).
 */
class SherpaModelDownloader(private val context: Context) {

    sealed interface Progress {
        /** No copy has been attempted yet. */
        data object Idle : Progress
        /** Per-file + overall byte progress. `receivedTotal` / `totalBytes`. */
        data class Downloading(
            val currentFile: String,
            val fileIndex: Int,
            val fileCount: Int,
            val receivedFileBytes: Long,
            val fileBytes: Long,
            val receivedTotal: Long,
            val totalBytes: Long
        ) : Progress
        /** All files have landed on disk and passed the size check. */
        data object Done : Progress
        /** A file failed to copy. `partialFile` removed. */
        data class Failed(val message: String) : Progress
    }

    private val _progress = MutableStateFlow<Progress>(Progress.Idle)
    val progress: StateFlow<Progress> = _progress.asStateFlow()

    /**
     * Concurrency guard so the Compose side can fire `downloadIfNeeded()`
     * from MULTIPLE `LaunchedEffect(Unit)` blocks without duplicate parallel
     * work. Whichever caller wins the `compareAndSet` schedules the work.
     */
    private val inflight = java.util.concurrent.atomic.AtomicBoolean(false)

    /** True iff the model is already on disk (no work to do). */
    fun isComplete(): Boolean = isSherpaModelReady(context)

    /** Idempotent asset extraction. Safe to call from multiple effects. */
    suspend fun downloadIfNeeded() = withContext(Dispatchers.IO) {
        // Fast path: already complete. Flip straight to Done so the UI dialog
        // collapses without ever showing a spurious "Downloading" frame.
        if (isComplete()) {
            _progress.value = Progress.Done
            return@withContext
        }
        if (!inflight.compareAndSet(false, true)) return@withContext
        try {
            val dir = sherpaModelDir(context)
            val totalFiles = MODEL_FILES.size
            var receivedTotal = 0L

            MODEL_FILES.forEachIndexed { idx, mf ->
                val target = File(dir, mf.name)
                if (target.exists() && target.length() >= mf.size) {
                    receivedTotal += mf.size
                    _progress.value = Progress.Downloading(
                        currentFile = mf.name, fileIndex = idx + 1, fileCount = totalFiles,
                        receivedFileBytes = mf.size, fileBytes = mf.size,
                        receivedTotal = receivedTotal, totalBytes = MODEL_TOTAL_BYTES
                    )
                    return@forEachIndexed
                }
                _progress.value = Progress.Downloading(
                    currentFile = mf.name, fileIndex = idx + 1, fileCount = totalFiles,
                    receivedFileBytes = 0L, fileBytes = mf.size,
                    receivedTotal = receivedTotal, totalBytes = MODEL_TOTAL_BYTES
                )
                try {
                    val tmp = File(dir, "${mf.name}.tmp")
                    tmp.delete()
                    val bytes = context.assets.open("$ASSET_SUBDIR/${mf.name}").use { input ->
                        val buf = ByteArray(64 * 1024)
                        var received = 0L
                        FileOutputStream(tmp).use { sink ->
                            while (true) {
                                val n = input.read(buf)
                                if (n <= 0) break
                                sink.write(buf, 0, n)
                                received += n
                                _progress.value = Progress.Downloading(
                                    currentFile = mf.name, fileIndex = idx + 1, fileCount = totalFiles,
                                    receivedFileBytes = received, fileBytes = mf.size,
                                    receivedTotal = receivedTotal + received,
                                    totalBytes = MODEL_TOTAL_BYTES
                                )
                            }
                        }
                        received
                    }
                    if (bytes < mf.size) {
                        tmp.delete()
                        throw IOException("Short extract for ${mf.name}: got $bytes / ${mf.size}")
                    }
                    if (!tmp.renameTo(target)) {
                        tmp.copyTo(target, overwrite = true)
                        tmp.delete()
                    }
                    receivedTotal = MODEL_FILES.subList(0, idx + 1).sumOf { it.size }
                } catch (e: Throwable) {
                    Log.e(TAG, "Failed to extract ${mf.name} from assets", e)
                    File(dir, "${mf.name}.tmp").delete()
                    _progress.value = Progress.Failed(
                        message = "${mf.name}: ${e.message ?: e.javaClass.simpleName}"
                    )
                    throw IOException("Asset extract failed for ${mf.name}", e)
                }
            }
            if (!isSherpaModelReady(context)) {
                _progress.value = Progress.Failed("Model files are incomplete")
                throw IOException("Model files are incomplete after extraction")
            }
            _progress.value = Progress.Done
        } finally {
            // Release the single-flight latch whether we succeeded or failed so
            // a Tap-Retry works without needing an Activity restart.
            inflight.set(false)
        }
    }
}

/**
 * Factory for the [QuranSttEngine] given the on-disk model files.
 *
 * The FastConformer graph is fully dynamic on the time axis, so every
 * sliding-window decode runs at the window's native length — a ~4 s window
 * costs a 4 s forward pass (RTF ~0.04 on a phone CPU), which is what makes
 * near-real-time display possible.
 */
fun createQuranSttEngine(context: Context): QuranSttEngine {
    val dir = sherpaModelDir(context)
    return QuranSttEngine.create(
        modelFile = File(dir, MODEL_FILES[0].name),
        tokensFile = File(dir, MODEL_FILES[1].name),
        threads = 4,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// AyahWordMatcher — hallucination guard (post-filter layer).
//
// Sherpa's offline Whisper output is noisy relative to the ayah text: tiny
// models transpose letters, add stray words, and can hallucinate short filler
// on near-silence. We counteract that by:
//   (a) keeping a sliding-window view of the recognized transcript,
//   (b) matching each emitted word against the active ayah range vocabulary
//       (case-insensitive, Arabic-normalized so that "الْحَمْدُ" and "الحمد"
//        are the same token),
//   (c) dropping unmatched words UNLESS they are plausible phonetic neighbors
//       of an ayah word — see [matchesAnyAyahWord] — which preserves the user's
//       occasional mispronunciation for the tajweed scorer to detect instead of
//       silently erasing it,
//   (d) enforcing monotonicity — once word #i in the filtered transcript has
//       been committed to the SharedFlow, a later noisy decode can never move
//       the cursor backwards (dedup + monotonicity guard).
// ─────────────────────────────────────────────────────────────────────────────

/** Bounded Levenshtein on normalized strings, early-exit if [ceiling] exceeded. */
private fun editDistanceLe(a: String, b: String, ceiling: Int): Int {
    val m = a.length
    val n = b.length
    if (m == 0) return n
    if (n == 0) return m
    if (kotlin.math.abs(m - n) > ceiling) return ceiling + 1
    var prev = IntArray(n + 1) { it }
    var curr = IntArray(n + 1)
    for (i in 1..m) {
        curr[0] = i
        var minRow = curr[0]
        for (j in 1..n) {
            val cost = if (a[i - 1] == b[j - 1]) 0 else 1
            val v = minOf(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost
            )
            curr[j] = v
            if (v < minRow) minRow = v
        }
        if (minRow > ceiling) return ceiling + 1
        val tmp = prev; prev = curr; curr = tmp
    }
    return prev[n]
}

/**
 * Per-session ayah vocabulary + fuzzy corrector. Construct once per
 * Memorize+Tajweed session, providing the canonical target words (`targetOrig`
 * already cleansed of standalone Quranic annotation marks by `splitAyahWords`
 * upstream) and their normalized form via [normalizeArabic].
 *
 * The matcher is **per-partial**: for each partial transcript it:
 *   1. Splits into whitespace-separated tokens.
 *   2. Normalizes each word.
 *   3. Drops it if the normalized form is empty (typical for stray annotation
 *      signs the model emitted as filler).
 *   4. Otherwise tries, in order:
 *        a. exact vocabulary hit  → keep the token as-is;
 *        b. typo correction: best Levenshtein neighbor in the select range's
 *           vocabulary within a length-scaled edit budget → REPLACE the token
 *           with the CANONICAL ayah word ("instantly correcting typos, returning
 *           the exact text" — the user-facing transcript becomes the ayah's own
 *           spelling with absolute precision);
 *        c. long-common-prefix match vs. a vocabulary word (single-syllable
 *           slip like الحم vs الحمد) → canonical replacement;
 *        d. nothing matched at all, but the token IS Arabic script → keep the
 *          RAW token. Previously this class dropped these tokens, which caused
 *          "I speak and nothing happens, and I don't get a mistake": a
 *          mispronounced non-vocabulary word was silently erased — no display,
 *          no WRONG flag. Keeping it lets the downstream aligner surface the
 *          mistake;
 *        e. nothing matched AND not Arabic (the model's rare non-Quranic
 *          filler) → drop.
 *   5. Returns the cleaned (fuzzy-corrected) transcript.
 *
 * **Monotonicity**: the caller also passes [lastEmittedCount]. The matcher will
 * never return a transcript SHORTER than the previously committed cursor; if a
 * new partial decodes to fewer words it is treated as a re-decode of the same
 * window and the trailing tokens are re-used from the last emitted clean list.
 * This guarantees the UI's "current word highlight" cursor never moves
 * backwards during a single utterance.
 */
class AyahWordMatcher(
    ayahWords: List<String>
) {
    /** Original (canonical) ayah words, same indices as the caller's targetOrig. */
    private val canonTargets: List<String> = ayahWords
    /** Normalized target words, same indices as the caller's targetOrig. */
    private val normTargets: List<String> = ayahWords.map { normalizeArabicPub(it) }
    /** Quick membership set (normalized) for the common exact case. */
    private val exactSet: Set<String> = normTargets.toHashSet()
    /** Cached fuzzy lookups — the dictionary is small (<= a few hundred words per ayah range). */
    private val fuzzyCache = HashMap<String, FuzzyResult>()

    /** Result of a fuzzy lookup: either the canonical replacement or "keep raw". */
    private class FuzzyResult(val canonical: String?) {
        companion object {
            val KEEP_RAW = FuzzyResult(null)
            val DROP = FuzzyResult(DROP_SENTINEL)
            private const val DROP_SENTINEL = "\u0000DROP"
        }
        val isDrop: Boolean get() = canonical == DROP_SENTINEL
    }

    /**
     * True iff the token is Arabic script (U+0600..U+06FF, U+0750..U+077F,
     * U+08A0..U+08FF). Used to decide between "keep raw for the aligner to
     * flag a mistake" vs "drop hallucinated non-Arabic filler".
     */
    private fun isArabicScript(s: String): Boolean = s.any { c ->
        c in '\u0600'..'\u06FF' || c in '\u0750'..'\u077F' || c in '\u08A0'..'\u08FF'
    }

    /**
     * Clean + fuzzy-correct a partial transcript. Returns the corrected token
     * list. PURE (no session state): the caller's monotonicMerge owns
     * monotonicity and reading order across sliding-window re-decodes.
     */
    /**
     * Position-aware restore of a window-EDGE fragment in the ≤3-letter zone.
     *
     * The sliding-window decoder cuts words at the window edges: the FIRST
     * token of a MULTI-token decode can lose its LEADING letter(s) (بسم الله
     * decodes as سم الله) — a target that ENDS WITH the token restores the
     * canonical spelling.
     *
     * A SINGLE-token short decode is NOT restored: the lone fragment (a م cut
     * at the window's right edge) carries no context, and promoting it to a
     * full word commits that word BEFORE its real neighbors exist in the
     * log. The merge then cannot re-fold it once its audio slides out of the
     * window — the next full decode re-appends the word and the premature
     * copy stays as a phantom (kafirun's lone م → مَا inserted between
     * الْكَافِرُونَ and لَا, scrambling every verdict). Dropping the fragment
     * is safe: the next window re-decodes the completed word whole.
     *
     * A short token anywhere else — in particular a MULTI-token decode's LAST
     * word — is a window-edge fragment, not a dropped letter: falaq's decode
     * [مِنْ شَرِّ م] promoted the trailing م to مِنْ and the merge then
     * doubled من شر.
     */
    private fun restoreEdgeFragment(norm: String, isFirst: Boolean, isSingle: Boolean, isLast: Boolean = false): String? {
        if (isSingle) return null
        val cp = norm.codePointCount(0, norm.length)
        if (cp > 3 && !(isLast && cp >= 4)) return null
        for (i in normTargets.indices) {
            val t = normTargets[i]
            if (isFirst) {
                val d = editDistanceLe(norm, t, ceiling = 1)
                if (d != 1) continue
                // Restore only when the target is STRICTLY LONGER than the token:
                // the window cut off the token's tail. The reversed direction
                // (token ends with the target) replaces a COMPLETE word with a
                // shorter/different one — kafirun's وَلَا was swapped to لَآ.
                if (t.endsWith(norm) && t.length > norm.length) return canonTargets[i]
            }
            // Trailing-edge cut (ma'un's final الْمَاع -> الْمَاعون): the LAST
            // token of a multi-token decode that is an EXACT leading prefix of
            // a target word was cut at the window's right edge — restore it.
            // Guards: the token must be a real word stem (>=4 letters — a 1-3
            // letter stub like falaq's trailing م stays dropped, that restore
            // doubled من شر), and the cut tail is <=2 letters, so a complete
            // shorter word can never be promoted into a different longer one
            // (exact vocab hits never reach here anyway).
            if (isLast && !isFirst && cp >= 4 && t.startsWith(norm)) {
                val ext = t.length - norm.length
                if (ext in 1..2) return canonTargets[i]
            }
        }
        return null
    }

    /**
     * Restore a window-edge TRUNCATION of a vocabulary word that landed where
     * [restoreEdgeFragment]'s position gates don't reach (a MIDDLE token, or a
     * first token whose trailing letters were cut). The CTC decoder slices the
     * trailing letter(s) of a word at the sliding-window boundary: لِّلَّه
     * surfaces as لِلَّ (norm "لل"), which the ≤3-letter fragment rule below
     * would otherwise DROP — permanently losing the word, so the aligner flags
     * a false SKIPPED and, after the mistake auto-pause, the resumed session
     * looks "deaf" because every re-decode truncates the same word again.
     *
     * Restore is conservative: the fragment must be a STRICT prefix of EXACTLY
     * ONE distinct vocabulary word with 1-2 letters missing. Uniqueness is the
     * guard against the phantom-first-word bug — a generic fragment like "ال"
     * prefixes many words (ٱللَّه, ٱلرَّحْمَٰن, ٱلْعَٰلَمِين…) and must stay
     * dropped, while "لل" prefixes only لِّلَّه in the active range and is safe
     * to restore.
     */
    private fun restoreTruncatedWord(norm: String): String? {
        val cp = norm.codePointCount(0, norm.length)
        if (cp < 2) return null
        // A 2-letter fragment may only restore a 1-letter cut (لل → لِلَّه):
        // allowing 2 missing letters from a 2-letter stem (ال → ٱللَّه) would
        // re-create the phantom-first-word bug. Longer stems (≥3 letters)
        // carry enough context to allow a 2-letter cut.
        val maxMissing = if (cp >= 3) 2 else 1
        var matchIdx = -1
        var matchCount = 0
        val seen = HashSet<String>()
        for (i in normTargets.indices) {
            val t = normTargets[i]
            if (t.length > norm.length && t.startsWith(norm)) {
                val missing = t.length - norm.length
                if (missing in 1..maxMissing && seen.add(t)) {
                    matchCount++
                    matchIdx = i
                    if (matchCount > 1) return null   // ambiguous — leave to the drop rule
                }
            }
        }
        return if (matchCount == 1) canonTargets[matchIdx] else null
    }

    @Synchronized
    fun clean(partial: String): List<String> {
        val rawTokens = partial.split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (rawTokens.isEmpty()) return emptyList()
        val cleaned = ArrayList<String>(rawTokens.size)
        val single = rawTokens.size == 1
        for ((tokIdx, tok) in rawTokens.withIndex()) {
            val norm = normalizeArabicPub(tok)
            if (norm.isEmpty()) continue // stray annotation / zero-content token
            // Exact vocabulary hit FIRST — a complete ayah word must never be
            // "restored" into a DIFFERENT word that merely contains it (the
            // kafirun swap: complete لَا edge-restored to وَلَآ because وَلَآ
            // ends with لَا, scrambling every لا/ولا verdict).
            if (exactSet.contains(norm)) {
                cleaned.add(tok)
                continue
            }
            // Window-edge fragment restore (position-gated, see kdoc).
            val restored = restoreEdgeFragment(
                norm,
                isFirst = tokIdx == 0,
                isSingle = single,
                isLast = tokIdx == rawTokens.lastIndex
            )
            if (restored != null) {
                cleaned.add(restored)
                continue
            }
            when (val hit = fuzzyLookup(norm)) {
                null -> {
                    // Exact vocabulary hit — keep the token as spoken (it IS
                    // one of the ayah's words).
                    cleaned.add(tok)
                }
                else -> {
                    if (hit.isDrop) {
                        // Non-Arabic hallucinated filler — drop.
                        continue
                    }
                    val canonical = hit.canonical
                    if (canonical != null) {
                        // Typo-corrected: emit the canonical ayah spelling so
                        // the transcript + alignment use the EXACT target text.
                        cleaned.add(canonical)
                    } else if (isArabicScript(tok)) {
                        // CTC-era fragment rule: the sliding-window decoder cuts
                        // words at every window edge, and a cut token surfaces
                        // as a short Arabic fragment (الر, لِ, الْعَََم) that
                        // matches no vocabulary word. Keeping it (the old
                        // Whisper policy, so the aligner could flag a real
                        // misreading) made these fragments enter the committed
                        // transcript: the aligner flagged a "false mistake" and
                        // re-reciting the same words then looked like a repeat
                        // of the committed fragments — "the app isn't
                        // listening". Unmatched tokens of ≤3 letters are ALWAYS
                        // fragments here (a real 1-3 letter Arabic word would
                        // be an exact vocabulary hit).
                        //
                        // BEFORE dropping, try to restore a window-edge
                        // TRUNCATION of a unique vocabulary word (لِلَّه → لِلَّ):
                        // the old "drop it, the next window re-decodes it whole"
                        // assumption FAILS when the same word is truncated on
                        // every re-decode — the word is permanently lost, the
                        // aligner flags a false SKIPPED, and the resumed session
                        // looks deaf. Restoring the canonical word fixes both.
                        val truncRestored = restoreTruncatedWord(norm)
                        if (truncRestored != null) {
                            cleaned.add(truncRestored)
                        } else {
                            val cp = norm.codePointCount(0, norm.length)
                            // CTC SANDWICH FRAGMENT: a tiny Arabic word (1-3
                            // letters) that the matcher could not resolve to
                            // ANY vocabulary word, and arrives AFTER at
                            // least one real vocabulary word has already been
                            // matched in this same window, is a CTC
                            // hallucination from a sub-word spike (Baqarah
                            // 285's t=4.3 "قمَمَ الرَّسُولُ بِمَا أَنْ"
                            // cleanly produced [قمَمَ, الرَّسُولُ, بِمَا,
                            // أَنْ] — أَنْ is a real 1-3 letter Arabic word
                            // but NOT in the active 285-286 vocabulary; the
                            // merge then committed it as if it were speech,
                            // the live log acquired a phantom 4th word, and
                            // the aligner eventually charged WRONG/SKIPPED
                            // against a perfect recitation for the next ~70
                            // words). Drop here at the source: it's not in
                            // the vocab and a real 1-3 letter Arabic word
                            // in this position would be an exact vocab hit.
                            // ≥4-letter KEEP_RAW remains — those signal a
                            // genuine misreading and the aligner should
                            // flag them. Bare non-vocab short tokens with
                            // no prior vocab anchor (the very first token
                            // of a window or the only token in a window)
                            // keep the historic KEEP_RAW behaviour so the
                            // aligner can still flag a genuinely-spoken
                            // short word it doesn't recognize.
                            //
                            // Note the guard is "AT LEAST ONE PRIOR VOCAB
                            // HIT", not "ALL prior cleaned are vocab" — the
                            // first token of a decode may be a 4-letter OOV
                            // head wreck (Baqarah's قمَمَ, restored by
                            // KEEP_RAW because cp=4 is too long to call a
                            // sandwich) and we must still drop the tiny
                            // phantom that comes after the next vocab hit.
                            val priorHasVocab = cleaned.any {
                                normalizeArabicPub(it) in exactSet
                            }
                            if (cp <= 3 && priorHasVocab) {
                                // drop
                            } else {
                                // Terminal mismatch in Arabic script: keep
                                // the RAW token so the downstream aligner
                                // flags WRONG/SKIPPED instead of the user
                                // hearing nothing happen. Covers ≥4-letter
                                // tokens AND standalone 1-3 letter tokens.
                                cleaned.add(tok)
                            }
                        }
                    }
                }
            }
        }
        return cleaned
    }

    /**
     * Core fuzzy decision for one (normalized) spoken token.
     *
     * Returns null for an EXACT vocabulary hit; otherwise a [FuzzyResult]:
     * canonical replacement when the token is within [typoBudget] edits of a
     * vocabulary word (or shares its long common prefix), KEEP-RAW for Arabic
     * tokens that match nothing, or DROP for non-Arabic filler.
     */
    private fun fuzzyLookup(norm: String): FuzzyResult? {
        if (norm.isEmpty()) return FuzzyResult.DROP
        if (exactSet.contains(norm)) return null
        val cached = fuzzyCache[norm]
        if (cached != null) return cached

        val res: FuzzyResult
        if (!isArabicScript(norm)) {
            // Non-Arabic token never belongs to the ayah → the model is
            // hallucinating filler: drop it silently instead of showing garbage.
            res = FuzzyResult.DROP
        } else {
            // Levenshtein budget scales with word length: ~1 edit for short
            // words, up to 3 for long ones — generous enough to absorb
            // decoding typos while still rejecting real misreadings.
            // Zero-budget (≤3 letter) tokens never fuzzy-match HERE: the only
            // legitimate short-token restore is a window-edge cut, which
            // clean() handles position-aware via restoreEdgeFragment before
            // this lookup runs.
            val budget = lengthBudget(norm)
            var bestIdx = -1
            var bestDist = Int.MAX_VALUE
            if (budget > 0) {
                for (i in normTargets.indices) {
                    val t = normTargets[i]
                    val d = editDistanceLe(norm, t, ceiling = budget)
                    if (d <= budget && d < bestDist) {
                        bestDist = d
                        bestIdx = i
                    }
                }
            }
            if (bestIdx == -1) {
                // Anagram / transposed-letter restore: the CTC decoder
                // occasionally swaps letters (kafirun's يَٰٓأَيُّهَا decoded
                // as أيهاي). Same length + same letter multiset maps the
                // scramble back onto the canonical word — the single highest-
                // confidence correction for an otherwise OOV token that kept
                // cascading the whole alignment (a phantom anyhaa ate a slot
                // and every following word flipped WRONG).
                if (norm.length >= 4) {
                    for (i in normTargets.indices) {
                        val t = normTargets[i]
                        if (t.length == norm.length && sameLetterBag(norm, t)) {
                            bestIdx = i
                            break
                        }
                    }
                }
            }
            if (bestIdx == -1) {
                // Long-common-prefix fallback: a one-syllable slip (الحم vs
                // الحمد) can exceed the raw edit budget when the stems differ
                // in length; the prefix rule still anchors it to the target.
                // Strictness: the common prefix must be ≥ 3 letters AND cover
                // all-but-one of the TARGET's letters, so a mid-word fragment
                // like "الع" (2/4 letters of "ٱللَّهِ") can NEVER be promoted
                // to a canonical word — that was the phantom-first-word bug
                // (a 0.6 s cut window decoded "الع", the fallback "corrected"
                // it to "ٱللَّهِ", and the aligner flagged the true first
                // word as WRONG).
                var prefIdx = -1
                for (i in normTargets.indices) {
                    val t = normTargets[i]
                    val tLen = t.codePointCount(0, t.length)
                    val common = prefixLen(norm, t)
                    val minLen = minOf(norm.codePointCount(0, norm.length), tLen)
                    if (common >= minLen - 1 && common >= 3 && common >= tLen - 1) { prefIdx = i; break }
                }
                res = if (prefIdx == -1) {
                    // Really doesn't match anything: keep the raw Arabic token
                    // so the aligner can flag it (no more silent "nothing").
                    FuzzyResult.KEEP_RAW
                } else {
                    FuzzyResult(canonTargets[prefIdx])
                }
            } else {
                res = FuzzyResult(canonTargets[bestIdx])
            }
        }
        fuzzyCache[norm] = res
        return res
    }

    /** Edit-distance budget scaled by the spoken token's codepoint length. */
    private fun lengthBudget(norm: String): Int {
        val cp = norm.codePointCount(0, norm.length)
        return when {
            cp <= 3 -> 0        // 1-2 letter noise isn't a word yet
            cp <= 5 -> 1
            cp <= 8 -> 2
            else -> 3
        }
    }

    private fun prefixLen(a: String, b: String): Int {
        var n = 0
        val lim = minOf(a.length, b.length)
        while (n < lim && a[n] == b[n]) n++
        return n
    }

    /** True iff [a] and [b] hold the exact same letter multiset (an anagram). */
    private fun sameLetterBag(a: String, b: String): Boolean {
        if (a.length != b.length) return false
        val bag = HashMap<Char, Int>(a.length)
        for (c in a) bag[c] = (bag[c] ?: 0) + 1
        for (c in b) {
            val left = (bag[c] ?: 0) - 1
            if (left < 0) return false
            bag[c] = left
        }
        return true
    }

    /** Reset between sessions (e.g. on ayah range change). */
    @Synchronized
    fun reset() {
        fuzzyCache.clear()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SingleAudioPipeline — one AudioRecord → Tarteel Whisper + TajweedNet.
//
// On the Samsung A05s the platform SpeechRecognizer and a parallel AudioRecord
// starve each other's partials when run together (regression the user
// reported). The fix: drive BOTH consumers from ONE AudioRecord session using
// sherpa's offline Whisper as the transcript source. The TajweedNet acoustic
// scorer keeps receiving the same PCM samples via [fullAudioSink], which the
// caller harvests on stop.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Emitted by the pipeline's decode worker. `partial` is the cleaned
 * transcript (post-`AyahWordMatcher`); `rawPartial` the raw Whisper text
 * (the Tajweed Rules Coach tab shows raw words since its example sentences
 * are not part of the Memorize vocabulary). `endpoint` true marks a session
 * final commit (emitted once, after the final full-window decode on
 * stop()/pause()).
 */
data class SherpaPartialEvent(
    val partial: List<String>,
    val rawPartial: String,
    val endpoint: Boolean,
    /**
     * True when the pipeline just STARTED a fresh utterance (start()/resume()
     * after a pause). The collector uses this to reset its committed-word
     * cursor: without it, words a user re-recites after a pause are filtered
     * out as "already committed" and the app looks deaf ("I repeat the word
     * and nothing happens"). The event carries no words itself.
     */
    val freshSession: Boolean = false,
    /**
     * True when the final tail decode REPLACED the emission log (merge
     * repair): the carried [partial] is the authoritative full-session
     * transcript in reading order and must OVERWRITE the collector's
     * committed transcript, not append to it (the live log it supersedes
     * was scrambled by window-edge losses).
     */
    val repaired: Boolean = false
)

class SingleAudioPipeline(
    private val context: Context,
    /** Caller-supplied vocab for the active ayah range (display words). */
    private val ayahWordsProvider: () -> List<String>,
    /** Reserved for the (now hotword-less) Whisper engine; kept for API parity. */
    @Suppress("UNUSED_PARAMETER")
    private val extraPhrasesProvider: () -> List<String> = { emptyList() }
) {
    /** Debug identity so we can tell apart stale vs. current pipeline instances. */
    val pipelineId: Int = System.identityHashCode(this)
    /** Debug accessor for the teardown-in-flight latch. */
    val isStoppingLog: Boolean get() = isStopping
    /**
     * Coarse lifecycle state of the pipeline, exposed for the Compose UI so it
     * can show a "Warming up…" affordance while the ONNX encoder is loading.
     *
     * The recognizer is NOT ready to capture until state flips to
     * [State.Running]. Mirrors the [running] boolean but with an explicit
     * "Loading" transition so the UI can distinguish "starting (loading model
     * on IO)" from "started".
     */
    sealed interface State {
        data object Idle : State
        data object Loading : State
        data object Running : State
        data class Failed(val message: String) : State
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    /** True iff the pipeline is fully up (AudioRecord + recognizer live). */
    @Volatile var running: Boolean = false
        private set

    /**
     * Monotone capture-session epoch. Bumped by [pause]/[stop] and by every
     * (re)start of the read worker. The read loop captures the epoch at launch
     * and exits the moment it changes — this is the ONLY reliable way to make a
     * worker that is blocked inside a blocking `AudioRecord.read()` (returns
     * every ~250 ms) actually stop when the user pauses and then resumes within
     * that 250 ms window. Without it the old worker sees `running` flipped back
     * to true, keeps reading from the OLD AudioRecord, holds the single-mic HAL,
     * and the NEW AudioRecord can never initialize → total deafness after a
     * quick pause→resume (the "app doesn't listen after pause+restart" bug).
     */
    @Volatile private var captureEpoch = 0

    private val sampleRate = 16_000

    /** Sliding-window decode geometry (tune by latency on-device). */
    private val windowSeconds = 4.0f                 // trailing audio per decode
    private val decodeIntervalMs = 1500L             // min gap between starts
    private val speechFrameThr = 0.012f              // ≈ -38 dBFS per-frame speech gate
    private val audioFloorRms = 0.01f                // ≈ -40 dB RMS chunk gate (see runReadLoop)
    private val noSpeechStopMs = 4000L               // stop decoding 4 s after last speech
    // Auto-stop the mic after this much wall-clock silence. Distinct from
    // [noSpeechStopMs] (which only halts the decode worker); the user contract
    // is that a long pause hands the mic back without an explicit tap. Matches
    // a normal breath between ayahs in casual recitation. The final-commit
    // decode runs on auto-stop just like a manual pause()/stop().
    private val autoStopSilenceMs = 4000L
    private val minDecodeFrames = sampleRate * 3 / 2 // never decode < 1.5 s (mid-word slices)

    /** Wall-clock time of the most recent chunk that had ANY speech. -1 until
     *  the first speech frame arrives. Used by both capture paths (mic + inject)
     *  to drive the auto-stop mic timer. */
    @Volatile private var lastSpeechWallMs: Long = -1L

    /**
     * True when the auto-stop silence timer has expired (no speech for at
     * least [autoStopSilenceMs]). Cleared on every speech chunk; the mic loop
     * uses this to call [stop] gracefully.
     */
    @Volatile private var autoStopFired: Boolean = false

    // NN engine lifecycle. There is NO persistent state: the FastConformer
    // decodes a whole window, so every decode builds a fresh feature matrix.
    @Volatile private var recognizer: QuranSttEngine? = null

    // Background workers.
    private var readJob: Job? = null
    private var decodeJob: Job? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    // Streaming partials flow — UI subscribes and updates word highlight.
    // DROP_OLDEST + 64-cap means a burst of partials never backpressures the
    // workers (which would stall the AudioRecord reads and starve the
    // TajweedNet sink); the UI only needs the LATEST partial anyway.
    private val _partials = MutableSharedFlow<SherpaPartialEvent>(
        replay = 0,
        extraBufferCapacity = 64,
        onBufferOverflow = kotlinx.coroutines.channels.BufferOverflow.DROP_OLDEST
    )
    val partials: SharedFlow<SherpaPartialEvent> = _partials.asSharedFlow()

    // Audio accumulator that mirrors MicRecorderController's behaviour so the
    // existing TajweedNet scorer (which consumes a FloatArray of [-1,1] PCM on
    // stop) works unchanged. The pipeline never shrinks this buffer mid-flight;
    // callers harvest it via [stop] as a single contiguous FloatArray. The
    // decode windows are snapshots of its TRAIL, so growth is unbounded but
    // every decode only copies the trailing `windowSeconds`.
    private var audio = FloatArray(sampleRate * 10)
    private var audioSize = 0
    private val audioSync = Any()

    // UN-GATED twin of [audio] for the TajweedNet scorer. The gated buffer
    // drops silent 250 ms chunks, which (a) glues words together with no
    // inter-word silence — out-of-distribution for a model trained on natural
    // clips where trimSilence isolates a word — and (b) COMPRESSES the
    // timeline, so wall-clock word timings map onto the wrong audio. The
    // scorer needs real-time audio WITH silence (exactly what the old
    // MicRecorderController path fed it). STT keeps using the gated buffer.
    // Cost: 64 KB/s (a 1-minute session ≈ 3.8 MB).
    private var rawAudio = FloatArray(sampleRate * 10)
    private var rawAudioSize = 0

    // Speech tracking (absolute frame indices into `audio`).
    // sessionSpeechStart = first audio frame that looked like speech this
    // session (-1 until then); lastSpeechFrame = last frame above threshold.
    @Volatile private var sessionSpeechStart = -1
    @Volatile private var lastSpeechFrame = -1L

    @Volatile private var matcher: AyahWordMatcher? = null

    // Sliding-window monotonicity log: the window re-decodes the SAME trailing
    // audio every `decodeIntervalMs`, so Whisper can re-spell or drop words
    // that were already committed to the UI. These two lists (cleaned + raw
    // word streams) are the pipeline's own "already shown" log: every emitted
    // event must be a strict monotonic EXTENSION of them, otherwise the
    // collector's committed-word cursor would commit the same word twice and
    // the aligner would flag phantom WRONG words. Only touched from the
    // decodeBusy-serialized decode paths (single caller at a time), so no
    // extra locking is needed.
    private var lastEmitClean: List<String> = emptyList()
    private var lastEmitRaw: List<String> = emptyList()

    // First-emission timestamp of each log word, parallel to the two logs
    // above. Lets [monotonicMergeWords] time-eligibility-check anchors: a word
    // first emitted before `nowMs - windowMs` had its audio LEFT of the
    // decoded window and can never be re-heard, so an identical phrase from
    // NEW audio is appended (a new occurrence) instead of folding onto the old
    // one (falaq's مِنْ شَرِّ repeats, kafirun's repeated phrases).
    private var lastEmitCleanMs: LongArray = LongArray(0)
    private var lastEmitRawMs: LongArray = LongArray(0)

    // Normalized active-range vocabulary — the contradiction-rejection input
    // of [monotonicMergeWords]: a log word NOT in this set is treated as an
    // OOV decode artifact the next decode may legally overwrite.
    private var vocabNorm: Set<String> = emptySet()
    // MULTISET of the same vocabulary (normalized, WITH duplicates) — the echo
    // guard counts target multiplicity against it (a Set would flatten
    // الله x2 to x1 and drop the real second occurrence).
    private var vocabMultiset: List<String> = emptyList()

    private fun resetEmitLogs() {
        lastEmitClean = emptyList()
        lastEmitRaw = emptyList()
        lastEmitCleanMs = LongArray(0)
        lastEmitRawMs = LongArray(0)
        // Bump the utterance generation: any ASYNC final-decode captured from a
        // PREVIOUS pause() (see pause()'s scope.launch) must recognize that the
        // utterance has since restarted and skip its emit — otherwise it would
        // merge the OLD session's words into the NEW session's freshly-reset log
        // and emit them as an endpoint, corrupting the resumed transcript and
        // prematurely setting sessionFinalized (the "deaf after resume" bug).
        utteranceGen++
    }

    /** Monotone utterance generation, bumped by [resetEmitLogs]. Volatile:
     *  read by the async final-decode (IO scope) and written on the main thread. */
    @Volatile private var utteranceGen = 0

    private fun rebuildVocab(ayahWords: List<String>) {
        vocabNorm = ayahWords.map { normalizeArabicPub(it) }
            .filter { it.isNotEmpty() }
            .toSet()
        vocabMultiset = ayahWords.map { normalizeArabicPub(it) }
            .filter { it.isNotEmpty() }
    }

    /** Grow/shrink the per-word timestamp log to match a merge result. */
    private fun updatedEmitMs(old: LongArray, oldSize: Int, mergedSize: Int, nowMs: Long): LongArray {
        if (mergedSize == oldSize) return old // hold OR same-size full re-render
        if (mergedSize < oldSize) return LongArray(mergedSize) { nowMs } // replacement (final-decode repair)
        val out = LongArray(mergedSize)
        val keep = minOf(old.size, mergedSize)
        if (keep > 0) old.copyInto(out, 0, 0, keep)
        for (i in oldSize until mergedSize) out[i] = nowMs
        return out
    }

    /**
     * Time+vocab-aware merge of the fresh window decode [cleaned] onto the
     * clean log. Updates [lastEmitCleanMs] to stay parallel to the log the
     * caller is about to store. [windowMs] is the audio span the decode could
     * have re-heard (the trailing window, or the final tail's length).
     */
    private fun mergeClean(cleaned: List<String>, windowMs: Long, nowMs: Long, fullReRender: Boolean = false, coversSession: Boolean = false): List<String> {
        val merged = monotonicMergeWords(lastEmitClean, cleaned, lastEmitCleanMs, nowMs, windowMs, vocabNorm, vocabMultiset, fullReRender, coversSession)
        lastEmitCleanMs = updatedEmitMs(lastEmitCleanMs, lastEmitClean.size, merged.size, nowMs)
        return merged
    }

    /** Raw-log merge: same time eligibility, no vocab (raw words are display-only). */
    private fun mergeRaw(rawWords: List<String>, windowMs: Long, nowMs: Long): List<String> {
        val merged = monotonicMergeWords(lastEmitRaw, rawWords, lastEmitRawMs, nowMs, windowMs, null)
        lastEmitRawMs = updatedEmitMs(lastEmitRawMs, lastEmitRaw.size, merged.size, nowMs)
        return merged
    }

    // Serializes decode work: only one native decode at a time (window decodes
    // from the decode worker AND the final commit decode on stop/pause).
    private val decodeBusy = java.util.concurrent.atomic.AtomicBoolean(false)
    @Volatile private var lastDecodeStartMs = 0L
    @Volatile private var lastSnapSkipLogMs = 0L

    /**
     * Cancel-during-load latch set by [stop]/[release] so an in-flight
     * [start] IO job — still busy constructing the ONNX recognizer — knows
     * NOT to flip `running=true` and tear itself down instead.
     */
    @Volatile private var cancelledDuringLoad: Boolean = false
    @Volatile private var startRequestedWhileLoading: Boolean = false

    /**
     * "Teardown in flight" latch — set true by [stop]/[release] at the moment
     * they enqueue the **asynchronous** worker join + teardown on the IO
     * scope, and cleared in that same IO job after teardown() has actually
     * released the [QuranSttEngine] handle.
     *
     * [start] checks this latch FIRST; [resume] too. See the streaming-era
     * docs for the SIGSEGV race this closes (a stale worker dereferencing a
     * freed recognizer while a fast re-start resurrects `running`).
     */
    @Volatile private var isStopping: Boolean = false

    /**
     * Kicks off asynchronous recognizer + AudioRecord startup. Returns
     * immediately — the heavy work `createWhisperRecognizer(...)` (which loads
     * the int8 ONNX pair into ONNX Runtime, ~1-3 s on the Samsung A05s) runs
     * on [Dispatchers.IO], NEVER on the Compose main thread, which would
     * otherwise ANR the app.
     *
     * Lifecycle outcome is observable via [state]:
     *  - `Idle`     : pipeline was never started,
     *  - `Loading`  : recognizer is being constructed on IO,
     *  - `Running`  : AudioRecord is live, partials are flowing,
     *  - `Failed`   : construction failed (mic busy / model corrupted).
     *
     * Returns `false` synchronously ONLY when the model isn't on disk yet
     * (caller must run the sherpa asset-extraction dialog first).
     */
    @Synchronized
    fun start(): Boolean {
        Log.i(TAG, "start(): id=$pipelineId running=$running state=${_state.value}")
        if (running || _state.value is State.Loading) {
            // A load (usually prewarm()) is already in flight. If it is the
            // prewarm path it will NOT start the read loop when it finishes,
            // so remember the user's tap and continue into a live session the
            // moment the recognizer lands.
            startRequestedWhileLoading = true
            return true
        }
        if (isStopping) {
            Log.w(TAG, "start(): previous teardown still in flight — refusing to start a second pipeline")
            _state.value = State.Failed("Recognizer is still shutting down — try again in a moment")
            return false
        }
        if (!isSherpaModelReady(context)) {
            Log.w(TAG, "start(): model not extracted yet")
            return false
        }
        cancelledDuringLoad = false
        val prewarmed = recognizer != null
        _state.value = State.Loading
        jobCreateAndRun(prewarmed)
        return true
    }

    private fun jobCreateAndRun(prewarmed: Boolean) {
        readJob = scope.launch {
            try {
                val ayahWords = ayahWordsProvider()
                matcher = AyahWordMatcher(ayahWords)
                rebuildVocab(ayahWords)
                resetEmitLogs()
                val rec = if (prewarmed) {
                    Log.i(TAG, "start(): reusing prewarmed recognizer id=$pipelineId")
                    recognizer!!
                } else {
                    createQuranSttEngine(context).also {
                        Log.i(TAG, "start(): QuranSttEngine created OK")
                    }
                }
                if (cancelledDuringLoad) {
                    try { rec.close() } catch (_: Throwable) {}
                    _state.value = State.Idle
                    return@launch
                }
                recognizer = rec
                startRequestedWhileLoading = false
                running = true
                _state.value = State.Running
                Log.i(TAG, "start(): session speech reset id=$pipelineId")
                sessionSpeechStart = -1
                lastSpeechFrame = -1L
                // Reset the auto-stop mic timer (lastSpeechWallMs is the wall
                // clock of the most recent speech chunk; autoStopFired is the
                // latch that prevents multiple fires during one silence gap).
                lastSpeechWallMs = -1L
                autoStopFired = false
                startReadLoop()
                notifyFreshSession()
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to start Sherpa pipeline", t)
                teardown()
                _state.value = State.Failed(
                    t.message ?: t.javaClass.simpleName
                )
                running = false
            }
        }
    }

    /**
     * Warm-up pass that pre-loads the ONNX recognizer in the background
     * WITHOUT touching the microphone, so the user's FIRST mic tap starts
     * immediately instead of stalling while the model loads.
     *
     * Idempotent + race-safe: no-ops when the recognizer is already built,
     * already loading, or when a teardown is in flight. [start] checks
     * `recognizer != null` and reuses this warm handle.
     */
    @Synchronized
    fun prewarm() {
        if (recognizer != null || running || _state.value is State.Loading || isStopping) return
        if (!isSherpaModelReady(context)) return
        _state.value = State.Loading
        scope.launch {
            try {
                val ayahWords = ayahWordsProvider()
                matcher = AyahWordMatcher(ayahWords)
                rebuildVocab(ayahWords)
                resetEmitLogs()
                val rec = createQuranSttEngine(context)
                Log.i(TAG, "prewarm(): QuranSttEngine created OK id=$pipelineId")
                if (cancelledDuringLoad || isStopping) {
                    try { rec.close() } catch (_: Throwable) {}
                    startRequestedWhileLoading = false
                    _state.value = State.Idle
                    return@launch
                }
                recognizer = rec
                if (startRequestedWhileLoading && !cancelledDuringLoad && !isStopping) {
                    // The user tapped the mic while this prewarm was loading.
                    // Continuation: go straight into a live session.
                    startRequestedWhileLoading = false
                    running = true
                    _state.value = State.Running
                    sessionSpeechStart = -1
                    lastSpeechFrame = -1L
                    startReadLoop()
                    notifyFreshSession()
                } else {
                    startRequestedWhileLoading = false
                    _state.value = State.Idle
                }
            } catch (t: Throwable) {
                Log.e(TAG, "prewarm(): failed to warm recognizer", t)
                startRequestedWhileLoading = false
                try { recognizer?.close() } catch (_: Throwable) {}
                recognizer = null
                _state.value = State.Idle
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Pause / Resume — the lightweight alternative to stop()/start() that the
    // Memorize+Tajweed FAB needs for a pause-and-continue flow.
    //
    // Whisper has no in-flight utterance to preserve — the accumulate-decoded
    // transcript lives in the collector (committedWordCount), so pause() stops
    // the AudioRecord + decode workers, runs ONE final decode to commit the
    // tail, and resume() re-spawns the workers with a fresh speech window. No
    // native handles are freed, so resume costs no model reload and cannot
    // race a teardown.
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Emits the freshSession marker the collector listens for. Called right
     * after the workers (re)start so the transcript cursor resets in sync with
     * the pipeline's own utterance state.
     */
    private fun notifyFreshSession() {
        _partials.tryEmit(SherpaPartialEvent(
            partial = emptyList(),
            rawPartial = "",
            endpoint = true,
            freshSession = true
        ))
    }

    /**
     * Writes the given PCM as a 16-bit mono 16 kHz WAV under filesDir/sherpa_debug/ for offline analysis (debug tooling — lets the PC-side sherpa/whisper decode EXACTLY what the device heard). */
    private fun writeDebugWav(samples: FloatArray, prefix: String) {
        try {
            val dir = File(context.filesDir, "sherpa_debug")
            if (!dir.exists()) dir.mkdirs()
            val file = File(dir, "${prefix}_${System.currentTimeMillis()}.wav")
            val data = ByteArray(samples.size * 2)
            for (i in samples.indices) {
                val v = (samples[i] * 32767f).toInt().coerceIn(-32768, 32767)
                data[i * 2] = (v and 0xFF).toByte()
                data[i * 2 + 1] = ((v shr 8) and 0xFF).toByte()
            }
            val byteRate = sampleRate * 2
            val out = java.io.ByteArrayOutputStream()
            fun w32(x: Int) {
                out.write(x and 0xFF); out.write((x shr 8) and 0xFF)
                out.write((x shr 16) and 0xFF); out.write((x shr 24) and 0xFF)
            }
            fun w16(x: Int) {
                out.write(x and 0xFF); out.write((x shr 8) and 0xFF)
            }
            w32(0x46464952) // "RIFF"
            w32(36 + data.size)
            w32(0x45564157) // "WAVE"
            w32(0x20746D66) // "fmt "
            w32(16)
            w16(1); w16(1)
            w32(sampleRate); w32(byteRate); w16(2); w16(16)
            w32(0x61746164) // "data"
            w32(data.size)
            out.write(data)
            file.writeBytes(out.toByteArray())
            Log.i(TAG, "wav dumped ${"%.2f".format(samples.size / 16000f)}s -> ${file.absolutePath}")
        } catch (t: Throwable) {
            Log.w(TAG, "wav dump failed", t)
        }
    }

    /**
     * "Soft stop" for a pause-and-resume flow: stops the workers and runs a
     * final commit decode, leaving the [QuranSttEngine] alive so the next
     * [resume] re-spawns capture in milliseconds.
     *
     * Returns the UN-GATED audio captured since the last resume in 16 kHz
     * mono [-1,1] — real-time PCM including the inter-word silence the
     * TajweedNet scorer was trained on. The STT final decode still consumes
     * the gated buffer internally.
     *
     * Idempotent: calling pause() when already paused is a no-op (returns an
     * empty FloatArray).
     */
    @Synchronized
    fun pause(): FloatArray {
        Log.i(TAG, "pause(): id=$pipelineId running=$running paused=$paused isStopping=$isStopping")
        if (!running) return FloatArray(0)
        running = false
        cancelledDuringLoad = true
        // Bump the capture epoch so the read worker exits even if a fast
        // resume() flips `running` back to true before its next loop check
        // (the worker is blocked inside AudioRecord.read() for up to ~250 ms).
        captureEpoch++
        // The AudioRecord handle will be released by the worker's `finally`
        // block once it observes running=false and exits its loop. [resume]
        // waits on that old job before creating the new AudioRecord (the A05s
        // single-mic HAL invariant).
        //
        // Two snapshots: `out` is the GATED buffer (speech-only) that the STT
        // final decode consumes; `rawOut` is the UN-GATED real-time buffer
        // (speech + inter-word silence) that the TajweedNet scorer needs — it
        // was trained on clips with natural silence, and its word-timing clock
        // (tajweedRecordingStartMs wall clock) matches the un-gated timeline,
        // not the gate-compressed one.
        val out: FloatArray
        val rawOut: FloatArray
        synchronized(audioSync) {
            out = audio.copyOf(audioSize)
            rawOut = rawAudio.copyOf(rawAudioSize)
            audio = FloatArray(sampleRate * 10)
            audioSize = 0
            rawAudio = FloatArray(sampleRate * 10)
            rawAudioSize = 0
        }
        // Final commit decode once the workers have drained (they see
        // running=false and exit promptly). No teardown — recognizer stays.
        // Pass the SNAPSHOT we just took: the accumulator was reset to an
        // empty buffer for resume, so the final decode must consume `out`
        // (the old code read the emptied accumulator and silently never
        // committed anything).
        val oldRead = readJob
        val oldDecode = decodeJob
        // Capture the utterance generation NOW: if the user resumes before this
        // async decode fires, resume() bumps the generation and the decode must
        // skip its emit (it belongs to the OLD utterance — emitting it would
        // merge stale words into the new session's reset log).
        val gen = utteranceGen
        scope.launch {
            try { oldRead?.join() } catch (_: Throwable) {}
            try {
                kotlinx.coroutines.withTimeoutOrNull(3000L) { oldDecode?.join() }
            } catch (_: Throwable) {}
            writeDebugWav(out, "pause")
            finalDecodeAndEmit(endpoint = true, sessionAudio = out, expectedGen = gen)
        }
        _partials.tryEmit(SherpaPartialEvent(
            partial = emptyList(),
            rawPartial = "",
            endpoint = true
        ))
        paused = true
        _state.value = State.Idle
        return rawOut
    }

    /**
     * True while the debug injection file is present — the pipeline is being
     * driven by the PC test harness rather than the real microphone. The UI
     * uses this to relax tajweed-popup auto-stops during automated runs
     * (wrong/skipped words still stop the session).
     */
    fun isInjecting(): Boolean =
        java.io.File(context.filesDir, "inject.pcm").exists()

    /**
     * Invalidates any IN-FLIGHT final commit decode launched by an earlier
     * [pause]/[stop]. Their async job captured the utterance generation at
     * call time; bumping it now makes [finalDecodeAndEmit]'s generation guard
     * silently drop the emit instead of merging words into whatever session
     * is active when the decode finally finishes (~3-4 s later on slow
     * phones). Called by RecitationApp when the ayah range / surah changes
     * right after a manual stop: the decoded words belong to the OLD target
     * and must never surface against the new one ("popup before I even
     * recited").
     */
    fun invalidateInFlightCommit() {
        utteranceGen++
        Log.i(TAG, "invalidateInFlightCommit(): utteranceGen bumped -> $utteranceGen")
    }

    /**
     * True iff the pipeline has been paused via [pause] (the recognizer is
     * still live and can be resumed). Distinguishes "never started" from
     * "started and paused". Cleared by [resume], [stop] (via teardown), and
     * [release].
     */
    @Volatile var paused: Boolean = false
        private set

    /**
     * Live copy of the CURRENT utterance's GATED PCM (16 kHz mono, [-1,1]) —
     * speech chunks only, silence dropped. This is the STT decode buffer; the
     * TajweedNet scorer must use [snapshotRawAudio] instead (the gate glues
     * words and compresses the timeline, which is out-of-distribution for the
     * model and misaligns wall-clock word timings). The accumulator is wiped
     * by pause()/stop(), so the snapshot is always exactly the audio captured
     * since the last (re)start — matching the collector's
     * `tajweedRecordingStartMs` clock base.
     */
    fun snapshotAudio(): FloatArray = synchronized(audioSync) { audio.copyOf(audioSize) }

    /**
     * Live copy of the CURRENT utterance's UN-GATED PCM (16 kHz mono, [-1,1])
     * — every captured chunk including silence, in real time. This is the
     * buffer the TajweedNet scorer must use: it has the inter-word silence the
     * model was trained on (so trimSilence can isolate a word) and its
     * timeline matches the wall-clock word timings. See [rawAudio].
     */
    fun snapshotRawAudio(): FloatArray = synchronized(audioSync) { rawAudio.copyOf(rawAudioSize) }

    /**
     * One-off high-quality re-decode of the session tail, for MISTAKE
     * VERIFICATION / TRANSCRIPT RECONCILIATION. Runs the recognizer over up to
     * [trailingSeconds] of the SAME gated speech-only buffer the live sliding
     * windows consume — but as a single full-context offline pass (exactly what
     * [finalDecodeAndEmit] does on pause), which recovers words a 4 s window
     * cut at its edge or folded mid-span.
     *
     * Read-only by design: unlike every other decode path it does NOT touch
     * the monotonic emit logs ([lastEmitClean]/[lastEmitRaw]) and does NOT
     * emit a partials event — the caller only wants the word list.
     *
     * SERIALIZATION — bounded wait, not instant bail-out: the live decode loop
     * re-decodes every ~1.5 s and each decode takes 1-2.5 s on slow phones, so
     * an unconditional [decodeBusy] grab used to lose nearly every race
     * (measured: two consecutive verifications returned EMPTY while the user
     * waited stuck at one word). We now wait up to ~2.5 s for a free slot.
     * Returns an EMPTY list when still inconclusive (not running, no
     * recognizer, slot never freed, too little audio, blank result).
     *
     * Blocking (~0.5-3 s per call on-device); invoke from Dispatchers.IO.
     */
    fun verifyTailWords(trailingSeconds: Int = 10): List<String> {
        val rec = recognizer ?: return emptyList()
        if (!running) return emptyList()
        var waitedMs = 0
        while (!decodeBusy.compareAndSet(false, true)) {
            if (!running) return emptyList()
            if (waitedMs >= 2500) {
                Log.w(TAG, "verifyTailWords(): decode slot never freed in ${waitedMs}ms — inconclusive")
                return emptyList()
            }
            Thread.sleep(100)
            waitedMs += 100
        }
        val t0 = System.currentTimeMillis()
        try {
            val spanSec = trailingSeconds.coerceIn(4, 45)
            val src = synchronized(audioSync) {
                if (audioSize <= 0 || lastSpeechFrame < 0) return@synchronized null
                var fromIdx = maxOf(0, audioSize - sampleRate * spanSec)
                // Trim leading silence before the session's first speech frame,
                // same rule as snapshotWindow().
                if (sessionSpeechStart >= fromIdx && sessionSpeechStart < audioSize) {
                    fromIdx = sessionSpeechStart
                }
                val weight = audioSize - fromIdx
                // Never decode a sub-word window (< minDecodeFrames ≈ 1.5 s):
                // a fragment slice produces bogus words (see snapshotWindow).
                if (weight < minDecodeFrames) return@synchronized null
                FloatArray(weight).also { System.arraycopy(audio, fromIdx, it, 0, weight) }
            } ?: return emptyList()
            Log.i(TAG, "verifyTailWords(): begin window=${"%.1f".format(src.size / 16000f)}s")
            val rawText = rec.transcribe(src)
            if (rawText.isBlank()) {
                Log.w(TAG, "verifyTailWords(): EMPTY after ${System.currentTimeMillis() - t0}ms — inconclusive")
                return emptyList()
            }
            val words = matcher?.clean(rawText)
                ?: rawText.split(Regex("\\s+")).filter { it.isNotEmpty() }
            Log.i(TAG, "verifyTailWords(): done ${System.currentTimeMillis() - t0}ms ${words.size}w '${rawText.take(80)}'")
            return words
        } catch (t: Throwable) {
            Log.w(TAG, "verifyTailWords(): error after ${System.currentTimeMillis() - t0}ms", t)
            return emptyList()
        } finally {
            decodeBusy.set(false)
        }
    }

    /**
     * Resumes the pipeline from a [pause]-flipped idle state by re-spawning
     * the AudioRecord + decode workers on the EXISTING [QuranSttEngine].
     * No model reload, no chance of racing a previous teardown because pause()
     * did NOT free the natives.
     *
     * CRITICAL — the A05s single-mic HAL race: we MUST wait for the OLD
     * AudioRecord (captured inside the paused read worker's closure) to be
     * released by its `finally` block BEFORE we create a new one. We do this
     * by capturing the OLD `job` reference and having the new worker
     * `job.join()` it (suspended on the IO scope, NOT main).
     *
     * Returns false if the pipeline was never started in this session (caller
     * should fall through to [start]) OR if a teardown is in flight (caller
     * should retry shortly). Returns true on a successful resume launch.
     */
    @Synchronized
    fun resume(): Boolean {
        Log.i(TAG, "resume(): id=$pipelineId running=$running paused=$paused isStopping=$isStopping recognizer=${recognizer != null}")
        if (running) return true
        if (!paused) return false           // not paused → caller falls through to start()
        if (isStopping) {                  // defensive: a teardown IS pending
            Log.w(TAG, "resume(): teardown in flight — caller should retry")
            return false
        }
        val rec = recognizer ?: return false
        val oldRead = readJob
        running = true
        cancelledDuringLoad = false
        paused = false
        _state.value = State.Running
        sessionSpeechStart = -1
        lastSpeechFrame = -1L
        // The utterance restarts: the previous emission log and the matcher's
        // monotonic state belong to the OLD utterance. Without this reset the
        // FIRST re-recitation merges against the old cumulative log and the
        // pipeline can hold/no-op on re-spoken words (the user re-recites up
        // to the mistake and "nothing happens"). The collector sees the
        // freshSession event below and resets its committed cursor in sync.
        resetEmitLogs()
        matcher?.reset()
        // INJECTION MODE: resume continues from the CURRENT audio position —
        // right after the word that triggered the pause (the user asked for
        // "continue from the next ayah", not a re-read of the mistake word).
        notifyFreshSession()
        // Re-spawn BOTH workers on the IO scope: read (but FIRST join the old
        // read worker so the A05s mic HAL is free for the new AudioRecord) and
        // decode. FIX: resume() used to relaunch ONLY the read worker, so a
        // pause→resume left the mic capturing while the decode worker stayed
        // dead — zero decodes on the retry (the "I recite again, nothing
        // happens" bug).
        readJob = scope.launch {
            try { oldRead?.join() } catch (_: Throwable) {}
            if (!running || recognizer == null) {
                _state.value = State.Idle
                return@launch
            }
            runReadLoop()
        }
        val oldDecode = decodeJob
        decodeJob = scope.launch {
            try { oldDecode?.join() } catch (_: Throwable) {}
            if (!running || recognizer == null) {
                _state.value = State.Idle
                return@launch
            }
            decodeLoop()
        }
        return true
    }

    /**
     * Notifies the pipeline that the active ayah range changed (e.g. user
     * picked a different surah/ayah). Resets the [AyahWordMatcher]'s
     * monotonicity cursor so the next partial decodes cleanly against the new
     * vocab. (Whisper has no hotword mechanism — the unused hotwords plumbing
     * of the streaming era is gone; this call only refreshes the post-filter.)
     */
    @Synchronized
    fun updateHotwords() {
        val ayahWords = ayahWordsProvider()
        matcher = AyahWordMatcher(ayahWords)
        rebuildVocab(ayahWords)
        resetEmitLogs()
    }

    /**
     * Stops the workers, runs a final commit decode, releases the recognizer
     * asynchronously, and returns the accumulated full recitation PCM
     * (16 kHz mono, [-1,1]).
     *
     * CRITICAL: this MUST be safe to call from the Compose main thread, AND
     * MUST NOT race the IO workers. The previous implementation did
     * `runBlocking { job?.join() }` which blocked main until the worker
     * drained — enough to trip the 5 s ANR watchdog. Teardown is scheduled on
     * the IO scope: join read + decode workers FIRST (so no one is inside a
     * native call), run the final commit decode, then free the native handle,
     * and only then clear `isStopping`.
     */
    @Synchronized
    fun stop(): FloatArray {
        Log.i(TAG, "stop(): id=$pipelineId running=$running state=${_state.value}")
        if (!running) {
            // Even when not running, we may be MID-LOAD (state == Loading).
            // Flip the cancel latch so the start IO job tears down the
            // recognizer it is currently constructing rather than letting it
            // go live after the user already gave up on this session.
            cancelledDuringLoad = true
            return FloatArray(0)
        }
        running = false
        cancelledDuringLoad = true
        isStopping = true
        // Bump the capture epoch so the read worker exits even if it is blocked
        // inside AudioRecord.read() when `running` flips (see pause()).
        captureEpoch++
        // Snapshot the audio accumulators synchronously. The workers only write
        // to them while running=true; setting running=false above guarantees no
        // further concurrent writes once we enter this @Synchronized block.
        // `out` = gated buffer for the STT final decode; `rawOut` = un-gated
        // real-time buffer returned to the TajweedNet scorer (see pause()).
        val out: FloatArray
        val rawOut: FloatArray
        synchronized(audioSync) {
            out = audio.copyOf(audioSize)
            rawOut = rawAudio.copyOf(rawAudioSize)
            audio = FloatArray(sampleRate * 10)
            audioSize = 0
            rawAudio = FloatArray(sampleRate * 10)
            rawAudioSize = 0
        }
        val oldRead = readJob
        val oldDecode = decodeJob
        // Schedule teardown AFTER the workers have actually exited, so the
        // native recognizer handle can't be freed mid-decode; run the final
        // whisper decode on the SESSION PCM `out` (the accumulator is already
        // reset for the next session) BEFORE freeing it so the transcript
        // commits the tail.
        scope.launch {
            try { oldRead?.join() } catch (_: Throwable) {}
            try {
                kotlinx.coroutines.withTimeoutOrNull(3000L) { oldDecode?.join() }
            } catch (_: Throwable) {}
            writeDebugWav(out, "stop")
            finalDecodeAndEmit(endpoint = true, sessionAudio = out)
            teardown()
            // ONLY now — after teardown() has actually freed the native
            // recognizer handle — clear the gate so a follow-up [start] can
            // proceed safely.
            isStopping = false
            _state.value = State.Idle
        }
        // Synthetic endpoint so the partial collector commits whatever the
        // last window already emitted, even if the final decode yields nothing
        // new (the collector's `ifEmpty { livePartial }` fallback).
        _partials.tryEmit(SherpaPartialEvent(
            partial = emptyList(),
            rawPartial = "",
            endpoint = true
        ))
        return rawOut
    }

    private fun teardown() {
        try { recognizer?.close() } catch (_: Throwable) {}
        recognizer = null
        matcher = null
        // Also clear the pause/resume latch so a follow-up [resume] on this
        // pipeline doesn't try to re-spawn the workers against freed handles.
        paused = false
    }

    /**
     * Releases native resources unconditionally (lifecycle teardown).
     *
     * Invoked by the Compose `LaunchedEffect(targetOrig)` when the ayah range
     * changes. CRITICAL: this is called from the Compose main thread while
     * the IO workers may be mid-decode. We flip the latches (workers stop at
     * their next loop iteration) and SCHEDULE join + final decode + teardown
     * on the IO scope — no synchronous native call from this method, no
     * join-from-main, no ANR risk.
     */
    fun release() {
        running = false
        cancelledDuringLoad = true
        isStopping = true
        val out: FloatArray
        synchronized(audioSync) {
            out = audio.copyOf(audioSize)
            audio = FloatArray(sampleRate * 10)
            audioSize = 0
            rawAudio = FloatArray(sampleRate * 10)
            rawAudioSize = 0
        }
        val oldRead = readJob
        val oldDecode = decodeJob
        scope.launch {
            try { oldRead?.join() } catch (_: Throwable) {}
            try {
                kotlinx.coroutines.withTimeoutOrNull(3000L) { oldDecode?.join() }
            } catch (_: Throwable) {}
            finalDecodeAndEmit(endpoint = true, sessionAudio = out)
            teardown()
            isStopping = false
            _state.value = State.Idle
        }
    }

    /**
     * Launches the AudioRecord worker + the sliding-window decode worker.
     * The read worker captures and updates the speech tracking; the decode
     * worker periodically decodes the trailing window.
     */
    private fun startReadLoop() {
        readJob = scope.launch { runReadLoop() }
        decodeJob = scope.launch { decodeLoop() }
    }

    /**
     * The AudioRecord worker body. Runs inside whichever coroutine it is
     * invoked from (the fresh-start launcher, or the pause/resume launcher
     * that first joins the OLD worker so the single-mic HAL is free).
     *
     * Body sketch: getMinBufferSize → make AudioRecord → startRecording →
     * `while (running) { read shortBuf; convert; append to audioSync sink;
     * update speech window tracking }` → finally stop+release the AudioRecord.
     * No native calls at all in this worker — whisper decoding happens in
     * [decodeLoop]/[finalDecodeAndEmit].
     */
    private suspend fun runReadLoop() {
        // Capture the epoch this worker was launched under. If pause()/stop()
        // bumps it (even while we are blocked inside read()), we must exit —
        // checking `running` alone is NOT enough: a fast resume() flips
        // `running` back to true before our next check and we would keep
        // reading from the OLD AudioRecord, holding the single-mic HAL and
        // starving the new session's AudioRecord (total deafness).
        val myEpoch = captureEpoch
        // ── DEBUG AUDIO INJECTION BYPASS ── When a PCM file exists in the
        // app's private storage, feed IT through the exact mic chunk path
        // (same RMS gate, same sinks, same 250 ms cadence) instead of opening
        // the microphone. Lets the PC drive a deterministic "recitation" into
        // the app for reproducible on-device testing. Delete the file (or use
        // the app normally without pushing it) for real-mic behaviour.
        val inject = java.io.File(context.filesDir, "inject.pcm")
            .takeIf { it.exists() && it.length() > 4096 }
        if (inject != null) {
            runInjectLoop(inject, myEpoch)
            return
        }
        val minBuf = android.media.AudioRecord.getMinBufferSize(
            sampleRate,
            android.media.AudioFormat.CHANNEL_IN_MONO,
            android.media.AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBuf <= 0) {
            Log.e(TAG, "getMinBufferSize failed: $minBuf")
            running = false
            _state.value = State.Failed("AudioRecord.getMinBufferSize failed")
            return
        }
        // Read chunk: ~250 ms of frames so the decode windows move promptly.
        val bufFrames = maxOf(minBuf / 4, sampleRate / 4)
        // RETRY LOOP: on the A05s single-mic HAL the microphone is often still
        // held by the PREVIOUS AudioRecord for a few hundred ms after
        // pause→resume (its release() returns before the hardware frees the
        // stream). Constructing the new AudioRecord in that window fails to
        // initialize — and the old code then set running=false and returned,
        // killing the session PERMANENTLY: the user recited and saw no words
        // and no mistake popup ("after pause+restart the app doesn't listen").
        // Retry the construction every 250 ms (up to ~2 s) while the session
        // is still supposed to run; only give up after every attempt failed.
        var audioRecord: android.media.AudioRecord? = null
        var attempt = 0
        while (running && myEpoch == captureEpoch && attempt < 8) {
            attempt++
            val ar = android.media.AudioRecord(
                android.media.MediaRecorder.AudioSource.MIC,
                sampleRate,
                android.media.AudioFormat.CHANNEL_IN_MONO,
                android.media.AudioFormat.ENCODING_PCM_16BIT,
                maxOf(bufFrames * 2, minBuf)
            )
            var started = false
            if (ar.state == android.media.AudioRecord.STATE_INITIALIZED) {
                try {
                    ar.startRecording()
                    started = ar.recordingState == android.media.AudioRecord.RECORDSTATE_RECORDING
                } catch (t: Throwable) {
                    Log.w(TAG, "runReadLoop(): startRecording attempt $attempt threw", t)
                }
            }
            if (started) {
                audioRecord = ar
                break
            }
            Log.w(TAG, "runReadLoop(): AudioRecord attempt $attempt not ready (state=${ar.state}) — retrying in 250ms")
            try { ar.stop() } catch (_: Throwable) {}
            try { ar.release() } catch (_: Throwable) {}
            kotlinx.coroutines.delay(250)
        }
        if (audioRecord == null) {
            if (myEpoch != captureEpoch) {
                // The session was paused/stopped while we were retrying — exit
                // quietly WITHOUT touching `running` (a new session may own it).
                Log.i(TAG, "runReadLoop(): epoch changed during AudioRecord retry — exiting")
                return
            }
            Log.e(TAG, "AudioRecord not initialised after $attempt attempts")
            running = false
            _state.value = State.Failed("Microphone could not be started — please try again")
            return
        }
        Log.i(TAG, "runReadLoop(): AudioRecord state=${audioRecord.state} (attempt $attempt)")
        val shortBuf = ShortArray(bufFrames)
        // RMS gate: a read chunk whose RMS stays below [audioFloorRms]
        // (~-40 dBFS) is treated as silence and NEVER enters the accumulator
        // — so Whisper can never decode a silence-padded window (the
        // "mistake before I even start reading" hallucination) and the
        // TajweedNet scorer only sees real speech. Speech tracking
        // (sessionSpeechStart/lastSpeechFrame) is driven only by chunks that
        // pass the gate, keeping the gate and the decode eligibility
        // consistent.
        var rmsSum = 0.0
        var levelSum = 0.0
        var levelCount = 0L
        var lastLevelLogMs = System.currentTimeMillis()
        var levelPeak = -180.0
        var droppedChunks = 0L
        var droppedSum = 0L
        try {
            // startRecording() already succeeded inside the retry loop above.
            Log.i(TAG, "runReadLoop(): recording started")
            while (running && myEpoch == captureEpoch) {
                val n = audioRecord.read(shortBuf, 0, shortBuf.size)
                if (n > 0) {
                    if (!running || myEpoch != captureEpoch) break
                    // Convert once; the UN-GATED scorer sink keeps every chunk
                    // (including silence, real time) while the gated STT sink
                    // below keeps only speech chunks.
                    val floats = FloatArray(n)
                    var chunkEnergy = 0.0
                    var chunkAbs = 0.0
                    for (i in 0 until n) {
                        val v = shortBuf[i] / 32768f
                        floats[i] = v
                        chunkEnergy += v * v
                        chunkAbs += if (v < 0f) -v.toDouble() else v.toDouble()
                    }
                    // 0) TajweedNet UN-GATED sink: append BEFORE the silence
                    // gate so the scorer gets real-time audio with the
                    // inter-word silence the model was trained on (the gated
                    // buffer glues words and compresses the timeline — see
                    // [rawAudio]).
                    synchronized(audioSync) {
                        ensureRawAudioCapacity(rawAudioSize + n)
                        System.arraycopy(floats, 0, rawAudio, rawAudioSize, n)
                        rawAudioSize += n
                    }
                    // Per-chunk RMS: silence floor gate BEFORE anything else.
                    val chunkRms = Math.sqrt(chunkEnergy / n)
                    if (chunkRms < audioFloorRms) {
                        droppedChunks++
                        droppedSum += n
                        // Still feed the 1 s level probe so the log shows the
                        // room's noise floor separately from speech.
                        levelSum += chunkAbs
                        levelCount += n
                        val nowMs2 = System.currentTimeMillis()
                        if (nowMs2 - lastLevelLogMs >= 1000L) {
                            levelLog(levelSum, levelCount, levelPeak, nowMs2, droppedChunks, droppedSum)
                            levelSum = 0.0
                            levelCount = 0L
                            lastLevelLogMs = nowMs2
                            levelPeak = -180.0
                            droppedChunks = 0L
                            droppedSum = 0L
                        }
                        // Auto-stop: if the mic has been silent for
                        // [autoStopSilenceMs] AFTER at least one speech frame,
                        // hand control back to the user. We check inside the
                        // silence branch so the timer only advances when the
                        // app is actually quiet (not when the user is mid
                        // recitation).
                        if (lastSpeechWallMs > 0L &&
                            !autoStopFired &&
                            nowMs2 - lastSpeechWallMs >= autoStopSilenceMs) {
                            autoStopFired = true
                            Log.i(
                                TAG,
                                "runReadLoop(): auto-stop on ${nowMs2 - lastSpeechWallMs}ms silence"
                            )
                            // Hop to the IO scope to call stop() off the read
                            // loop (stop is suspend; doing it from the audio
                            // thread would deadlock with the decode worker).
                            scope.launch { stop() }
                        }
                        continue
                    }
                    var speechFrames = 0
                    for (i in 0 until n) {
                        val v = floats[i]
                        val abs = if (v < 0f) -v.toDouble() else v.toDouble()
                        if (abs >= speechFrameThr) {
                            speechFrames++
                            if (sessionSpeechStart < 0) {
                                sessionSpeechStart = audioSize + i
                            }
                        }
                        levelSum += abs
                        if (abs > levelPeak) levelPeak = abs
                    }
                    levelCount += n
                    val nowMs = System.currentTimeMillis()
                    if (nowMs - lastLevelLogMs >= 1000L) {
                        levelLog(levelSum, levelCount, levelPeak, nowMs, droppedChunks, droppedSum)
                        levelSum = 0.0
                        levelCount = 0L
                        lastLevelLogMs = nowMs
                        levelPeak = -180.0
                        droppedChunks = 0L
                        droppedSum = 0L
                    }
                    // 1) TajweedNet sink: append to the full-audio accumulator.
                    synchronized(audioSync) {
                        ensureAudioCapacity(audioSize + n)
                        for (i in 0 until n) audio[audioSize + i] = floats[i]
                        audioSize += n
                    }
                    // 2) Speech tracker (after the sink so indices are stable).
                    if (speechFrames > 0) {
                        lastSpeechFrame = (audioSize - 1).toLong()
                        // Wall-clock tracker for the auto-stop silence timer.
                        // Reset [autoStopFired] the moment new speech arrives,
                        // so a future second silence window can fire again
                        // after a resume (the user can re-tap the mic to
                        // start another session, which resets the field).
                        lastSpeechWallMs = System.currentTimeMillis()
                        autoStopFired = false
                    }
                } else if (n == android.media.AudioRecord.ERROR ||
                    n == android.media.AudioRecord.ERROR_INVALID_OPERATION) {
                    break
                }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "AudioRecord read loop crashed", t)
        } finally {
            Log.i(TAG, "runReadLoop(): exiting at running=$running epoch=$myEpoch/$captureEpoch")
            try { audioRecord.stop() } catch (_: Throwable) {}
            try { audioRecord.release() } catch (_: Throwable) {}
            Log.i(TAG, "runReadLoop(): AudioRecord released")
        }
    }


    /**
     * DEBUG: feed [f] (16 kHz mono 16-bit LE PCM) through the EXACT mic chunk
     * path - same RMS floor gate, same un-gated TajweedNet sink, same gated
     * STT sink, same speech tracking, same 250 ms real-time cadence - without
     * opening the microphone. After EOF, silence is fed so the session stays
     * live for a natural finalize-on-stop. [injectPos] survives pause/resume
     * so a resumed session continues where it left off.
     */
    @Volatile private var injectPos: Long = 0L

    private fun runInjectLoop(f: java.io.File, myEpoch: Int) {
        Log.i(TAG, "runInjectLoop(): INJECT MODE file=${f.name} bytes=${f.length()} startPos=$injectPos")
        val chunkSamples = sampleRate / 4
        val buf = ByteArray(chunkSamples * 2)
        var rmsSum = 0.0
        var levelSum = 0.0
        var levelCount = 0L
        var lastLevelLogMs = System.currentTimeMillis()
        var levelPeak = -180.0
        var droppedChunks = 0L
        var droppedSum = 0L
        var stream: java.io.InputStream? = null
        var nextChunkNs = System.nanoTime()
        var eof = injectPos >= f.length()
        try {
            while (running && myEpoch == captureEpoch) {
                if (!eof && stream == null) {
                    stream = java.io.BufferedInputStream(f.inputStream(), 1 shl 16)
                    var toSkip = injectPos
                    while (toSkip > 0) {
                        val s = stream!!.skip(toSkip)
                        if (s <= 0) break
                        toSkip -= s
                    }
                    if (toSkip > 0) { eof = true; injectPos = f.length() }
                }
                var off = 0
                if (!eof) {
                    while (off < buf.size) {
                        val r = stream!!.read(buf, off, buf.size - off)
                        if (r <= 0) { eof = true; break }
                        off += r
                    }
                    if (eof) {
                        try { stream!!.close() } catch (_: Throwable) {}
                        stream = null
                        injectPos = f.length()
                        Log.i(TAG, "runInjectLoop(): file EOF - feeding silence from here")
                    }
                }
                val samples = off / 2
                val floats = FloatArray(chunkSamples)
                var chunkEnergy = 0.0
                for (i in 0 until samples) {
                    val lo = buf[i * 2].toInt() and 0xFF
                    val hi = buf[i * 2 + 1].toInt()
                    val v = ((hi shl 8) or lo) / 32768f
                    floats[i] = v
                    chunkEnergy += v * v
                }
                for (i in samples until chunkSamples) floats[i] = 0f // silence pad at EOF
                injectPos += off
                // UN-GATED scorer sink: every chunk, real time (same as mic).
                synchronized(audioSync) {
                    ensureRawAudioCapacity(rawAudioSize + chunkSamples)
                    System.arraycopy(floats, 0, rawAudio, rawAudioSize, chunkSamples)
                    rawAudioSize += chunkSamples
                }
                val chunkRms = Math.sqrt(chunkEnergy / chunkSamples.coerceAtLeast(1))
                if (chunkRms >= audioFloorRms) {
                    var speechFrames = 0
                    for (i in 0 until chunkSamples) {
                        val v = floats[i]
                        val abs = if (v < 0f) -v.toDouble() else v.toDouble()
                        if (abs >= speechFrameThr) {
                            speechFrames++
                            if (sessionSpeechStart < 0) sessionSpeechStart = audioSize + i
                        }
                        levelSum += abs
                        if (abs > levelPeak) levelPeak = abs
                    }
                    levelCount += chunkSamples
                    synchronized(audioSync) {
                        ensureAudioCapacity(audioSize + chunkSamples)
                        for (i in 0 until chunkSamples) audio[audioSize + i] = floats[i]
                        audioSize += chunkSamples
                    }
                    if (speechFrames > 0) {
                        lastSpeechFrame = (audioSize - 1).toLong()
                        // Wall-clock tracker for the auto-stop silence timer
                        // (mirrors the mic loop). Reset [autoStopFired] so a
                        // second silence window after a resume can refire.
                        lastSpeechWallMs = System.currentTimeMillis()
                        autoStopFired = false
                    }
                } else {
                    droppedChunks++
                    droppedSum += chunkSamples
                    // Silence chunk: same auto-stop check as the mic path.
                    if (lastSpeechWallMs > 0L &&
                        !autoStopFired &&
                        System.currentTimeMillis() - lastSpeechWallMs >= autoStopSilenceMs) {
                        autoStopFired = true
                        Log.i(
                            TAG,
                            "runInjectLoop(): auto-stop on ${System.currentTimeMillis() - lastSpeechWallMs}ms silence"
                        )
                        scope.launch { stop() }
                    }
                }
                val nowMs = System.currentTimeMillis()
                if (nowMs - lastLevelLogMs >= 1000L) {
                    levelLog(levelSum, levelCount, levelPeak, nowMs, droppedChunks, droppedSum)
                    levelSum = 0.0; levelCount = 0L; lastLevelLogMs = nowMs; levelPeak = -180.0
                    droppedChunks = 0L; droppedSum = 0L
                }
                // Real-time cadence: one 250 ms chunk per 250 ms, like the mic.
                nextChunkNs += 250_000_000L
                val now = System.nanoTime()
                if (nextChunkNs > now) Thread.sleep((nextChunkNs - now) / 1_000_000L)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "runInjectLoop(): crashed", t)
        } finally {
            try { stream?.close() } catch (_: Throwable) {}
            Log.i(TAG, "runInjectLoop(): exiting at running=$running epoch=$myEpoch/$captureEpoch pos=$injectPos")
        }
    }
    /** 1 s-interval mic level probe. Silent chunks (below the RMS floor) are counted separately so a quiet room is distinguishable from a dead mic. */
    private fun levelLog(
        levelSum: Double,
        levelCount: Long,
        levelPeak: Double,
        nowMs: Long,
        droppedChunks: Long,
        droppedSum: Long
    ) {
        val avg = levelSum / levelCount
        val avgDb = if (avg > 0) 20.0 * Math.log10(avg) else -180.0
        val peakDb = if (levelPeak > 0) 20.0 * Math.log10(levelPeak) else -180.0
        Log.i(
            TAG,
            "runReadLoop(): lvl avgDb=${"%.1f".format(avgDb)} peakDb=${"%.1f".format(peakDb)} silenceDropped=${"%.1f".format(droppedSum / 16000f)}s x$droppedChunks"
        )
    }

    /**
     * Sliding-window decode worker. While the pipeline is running, decodes the
     * trailing [windowSeconds] of audio every [decodeIntervalMs] — but ONLY
     * when the window actually contains recent speech:
     *   - < [noSpeechStopMs] since the last speech frame → user stopped
     *     talking; further decodes would only re-emit the same words on a
     *     silence-padded window (or make the engine hallucinate filler on
     *     pure silence).
     *   - < [windowSeconds]/2 of speech total → nothing worth decoding yet.
     */
    private suspend fun decodeLoop() {
        // Capture the epoch this worker was launched under (see runReadLoop for
        // the same rationale): a fast pause→resume flips `running` back to true
        // before this loop wakes from its delay, so checking `running` alone
        // would let the OLD decode loop keep running AND make the NEW decode
        // coroutine's `oldDecode.join()` hang forever (the new loop never
        // starts → no decodes → deafness). Exit the moment the epoch changes.
        val myEpoch = captureEpoch
        var lastHeartMs = 0L
        while (running && myEpoch == captureEpoch) {
            delay(150)
            val nowMs = System.currentTimeMillis()
            // 2 s heartbeat: proves the decode worker is alive and reveals
            // whether a decode is stuck (busy=true for ages) or starved.
            if (nowMs - lastHeartMs >= 2000L) {
                lastHeartMs = nowMs
                Log.i(TAG, "decodeLoop(): alive busy=${decodeBusy.get()} lastDecodeAge=${nowMs - lastDecodeStartMs}ms running=$running audio=${"%.1f".format(audioSize / 16000f)}s")
            }
            if (nowMs - lastDecodeStartMs < decodeIntervalMs) continue
            if (decodeBusy.get()) {
                // Watchdog: a wedged native decode must not permanently blind
                // the pipeline (observed on-device: a silence-padded window
                // hung inside ONNX Runtime and every later decode bailed on
                // decodeBusy — the "no words after restart" bug).
                if (nowMs - lastDecodeStartMs > 12_000L) {
                    Log.w(TAG, "decodeLoop(): decode stuck ${nowMs - lastDecodeStartMs}ms — releasing decodeBusy to unwedge")
                    decodeBusy.set(false)
                }
                continue
            }
            decodeWindow()
        }
    }

    /** Snapshot the decode window: trailing [windowSeconds], silence-trimmed. */
    private fun snapshotWindow(): FloatArray? {
        synchronized(audioSync) {
            if (audioSize <= 0) return null
            val toIdx = audioSize
            // NEVER decode before the first real speech frame: pure-silence
            // windows made Whisper hallucinate a phantom word ("فَالطِّئِين")
            // and the aligner flagged a "mistake before I even start reading".
            if (lastSpeechFrame < 0) return null
            if (lastSpeechFrame < toIdx - sampleRate * (noSpeechStopMs / 1000) - 1L) {
                // Nothing spoken recently — decoding would only re-emit the
                // same words (or let tiny Whisper hallucinate on silence).
                throttleSnapLog("skip (no speech in last ${noSpeechStopMs}ms)")
                return null
            }
            var fromIdx = maxOf(0, toIdx - (windowSeconds * sampleRate).toInt())
            // Trim silence before the session's first speech frame.
            if (sessionSpeechStart >= fromIdx && sessionSpeechStart < toIdx) {
                fromIdx = sessionSpeechStart
            }
            val weight = toIdx - fromIdx
            // MINIMUM WINDOW: ~1.5 s. A shorter slice starts (or ends) mid-word,
            // and Whisper turns a cut fragment into a bogus word — the very
            // "الْع"→"ٱللَّهِ" phantom that flagged a WRONG on the user's first
            // word. Never decode a sub-word window; wait for the next cadence.
            if (weight < minDecodeFrames) {
                throttleSnapLog("skip (window too short ${"%.1f".format(weight / 16000f)}s < ${"%.1f".format(minDecodeFrames / 16000f)}s)")
                return null
            }
            // Speech-ratio gate: only decode speech-rich windows. A window
            // padded with seconds of silence is exactly what made Whisper
            // generate long hallucinated filler and wedge the decode.
            var speechFrames = 0
            for (i in fromIdx until toIdx) {
                val v = audio[i]
                val abs = if (v < 0f) -v else v
                if (abs >= speechFrameThr) speechFrames++
            }
            if (speechFrames * 100L < weight * 35L) {
                throttleSnapLog("skip (speech ${(speechFrames * 100L) / weight}% < 35% of window)")
                return null
            }
            val win = FloatArray(weight)
            System.arraycopy(audio, fromIdx, win, 0, weight)
            return win
        }
    }

    /** 2 s-throttled log for [snapshotWindow] skip reasons (kept out of the hot path). */
    private fun throttleSnapLog(msg: String) {
        val nowMs = System.currentTimeMillis()
        if (nowMs - lastSnapSkipLogMs >= 2000L) {
            lastSnapSkipLogMs = nowMs
            Log.i(TAG, "snapshotWindow(): $msg")
        }
    }

    /**
     * Decode one sliding window and emit the NEW words. The full window text
     * is emitted (cumulative); the collector's committed-word cursor dedups
     * everything already on screen, so this function is idempotent in the UI.
     */
    private fun decodeWindow() {
        if (!running) return
        val rec = recognizer ?: return
        val win = snapshotWindow() ?: return
        if (!decodeBusy.compareAndSet(false, true)) return
        val t0 = System.currentTimeMillis()
        lastDecodeStartMs = t0
        try {
            Log.i(TAG, "decodeWindow(): begin window=${"%.1f".format(win.size / 16000f)}s")
            val rawText = rec.transcribe(win)
            if (rawText.isBlank()) {
                Log.w(TAG, "decodeWindow(): EMPTY after ${System.currentTimeMillis() - t0}ms — engine decoded nothing for ${"%.1f".format(win.size / 16000f)}s")
                return
            }
            val cleaned = matcher?.clean(rawText) ?: rawText.split(Regex("\\s+"))
                .filter { it.isNotEmpty() }
            val rawWords = rawText.split(Regex("\\s+")).filter { it.isNotEmpty() }
            // Time+vocab-aware merge: the window covers the trailing
            // `win.size` samples of audio ending at t0 — a log word first
            // emitted before that span cannot be re-heard, so an identical
            // phrase from NEW audio appends instead of folding (repeated
            // phrases across ayahs), and an OOV garbage word in the log is
            // overwritable by a better re-render.
            val windowMs = (win.size * 1000L) / sampleRate
            val mergedClean = mergeClean(cleaned, windowMs, t0)
            val mergedRaw = mergeRaw(rawWords, windowMs, t0)
            if (mergedClean == lastEmitClean && mergedRaw == lastEmitRaw) {
                // Re-decode of the same window with no new words — nothing to
                // emit (identical emissions would just be re-shown partials).
                Log.d(TAG, "decodeWindow(): hold — no new words. lastClean=${lastEmitClean.size} clean=${cleaned.size} lastRaw=${lastEmitRaw.size} raw=${rawWords.size} rawText='${rawText.take(60)}'")
                return
            }
            val newCleanCount = mergedClean.size - lastEmitClean.size
            lastEmitClean = mergedClean
            lastEmitRaw = mergedRaw
            Log.i(TAG, "decodeWindow(): done ${System.currentTimeMillis() - t0}ms new=$newCleanCount w emitted=${mergedClean.size}w raw='${rawText.take(80)}'")
            _partials.tryEmit(SherpaPartialEvent(
                partial = mergedClean,
                rawPartial = mergedRaw.joinToString(" "),
                endpoint = false
            ))
        } catch (t: Throwable) {
            Log.w(TAG, "decodeWindow(): error after ${System.currentTimeMillis() - t0}ms", t)
        } finally {
            decodeBusy.set(false)
        }
    }

    /**
     * Final session decode: transcribes the session's tail audio and emits it
     * as an [endpoint] event so the collector commits the tail. [sessionAudio]
     * is the PCM snapshot stop()/pause() took when they reset the accumulator
     * (the last ~10 s covers every recitation tail; cap keeps the on-device
     * decode bounded). Called once from the stop()/pause()/release() IO jobs
     * AFTER the workers have joined, and BEFORE teardown() frees the
     * recognizer.
     */
    private fun finalDecodeAndEmit(endpoint: Boolean, sessionAudio: FloatArray?, expectedGen: Int = -1) {
        val rec = recognizer ?: return
        if (decodeBusy.get()) return
        // Utterance-generation guard: this async decode was captured from a
        // PREVIOUS pause(); if the utterance has since restarted (resume()
        // bumped the generation), its words belong to the OLD session and must
        // NOT be merged into the new session's freshly-reset emit log.
        if (expectedGen >= 0 && expectedGen != utteranceGen) {
            Log.i(TAG, "finalDecodeAndEmit(): skip — utterance restarted (gen $expectedGen -> $utteranceGen)")
            return
        }
        val src = sessionAudio ?: return
        if (src.size <= 0) return
        var fromIdx = maxOf(0, src.size - 10 * sampleRate)
        if (src.size - fromIdx < sampleRate) {
            fromIdx = 0
        }
        if (src.size - fromIdx < sampleRate / 2) return
        val win = FloatArray(src.size - fromIdx)
        System.arraycopy(src, fromIdx, win, 0, win.size)
        if (!decodeBusy.compareAndSet(false, true)) return
        val t0 = System.currentTimeMillis()
        try {
            Log.i(TAG, "finalDecodeAndEmit(): begin endpoint=$endpoint window=${"%.1f".format(win.size / 16000f)}s")
            val rawText = rec.transcribe(win)
            if (rawText.isBlank()) {
                Log.w(TAG, "finalDecodeAndEmit(): EMPTY after ${System.currentTimeMillis() - t0}ms — engine decoded nothing for ${"%.1f".format(win.size / 16000f)}s")
                return
            }
            val cleaned = matcher?.clean(rawText) ?: rawText.split(Regex("\\s+"))
                .filter { it.isNotEmpty() }
            val rawWords = rawText.split(Regex("\\s+")).filter { it.isNotEmpty() }
            // The tail decode re-hears the last `win.size` samples of the
            // session; merge with that span as the eligibility window so the
            // (usually cleaner) final render folds onto the live log without
            // re-appending words the log already holds outside the tail.
            val windowMs = (win.size * 1000L) / sampleRate
            val prevCleanSize = lastEmitClean.size
            val mergedClean = mergeClean(cleaned, windowMs, t0, fullReRender = true, coversSession = fromIdx == 0)
            val mergedRaw = mergeRaw(rawWords, windowMs, t0)
            if (mergedClean == lastEmitClean && mergedRaw == lastEmitRaw) {
                Log.d(TAG, "finalDecodeAndEmit(): hold — no new words. lastClean=${lastEmitClean.size} clean=${cleaned.size} lastRaw=${lastEmitRaw.size} raw=${rawWords.size} rawText='${rawText.take(60)}'")
                return
            }
            // The append-only merge only GROWS the log — a same-size or
            // smaller result means the final decode REPLACED it (repair).
            val repaired = mergedClean != lastEmitClean && mergedClean.size <= prevCleanSize
            val newCleanCount = mergedClean.size - lastEmitClean.size
            lastEmitClean = mergedClean
            lastEmitRaw = mergedRaw
            Log.i(TAG, "finalDecodeAndEmit(): done ${System.currentTimeMillis() - t0}ms new=$newCleanCount w emitted=${mergedClean.size}w raw='${rawText.take(80)}'")
            _partials.tryEmit(SherpaPartialEvent(
                partial = mergedClean,
                rawPartial = mergedRaw.joinToString(" "),
                endpoint = endpoint,
                repaired = repaired
            ))
        } catch (t: Throwable) {
            Log.w(TAG, "finalDecodeAndEmit(): error after ${System.currentTimeMillis() - t0}ms", t)
        } finally {
            decodeBusy.set(false)
        }
    }

    /**
     * Doubles the audio buffer's capacity (encoded as FloatArray) to [needed]. */
    private fun ensureAudioCapacity(needed: Int) {
        if (needed <= audio.size) return
        var cap = audio.size
        while (cap < needed) cap *= 2
        val next = FloatArray(cap)
        audio.copyInto(next, 0, 0, audioSize)
        audio = next
    }

    /**
     * Doubles the UN-GATED scorer buffer's capacity to [needed]. Same growth
     * policy as [ensureAudioCapacity]; caller must hold [audioSync]. */
    private fun ensureRawAudioCapacity(needed: Int) {
        if (needed <= rawAudio.size) return
        var cap = rawAudio.size
        while (cap < needed) cap *= 2
        val next = FloatArray(cap)
        rawAudio.copyInto(next, 0, 0, rawAudioSize)
        rawAudio = next
    }
}

/**
 * Append-only monotonic union of the session's already-emitted word log
 * [last] and a fresh window decode [n]. Returns the list to EMIT next.
 *
 * Sliding-window reality (quran_stt engine): every window re-renders the
 * trailing ~4 s and returns that span's words, so a re-decode can:
 *   - DROP words that slid out of the window's LEFT edge (the user said them
 *     earlier — they belong BEFORE the current decode),
 *   - miss a word mid-span (a re-spelling merged two words, e.g. the on-device
 *     "اللَّهِحْمَن" for "اللَّهِ الرَّحْمَنِ" — الله vanished from the
 *     render), and
 *   - CUT a word at the window's RIGHT edge (the decode fragment العالم for
 *     الْعَالَمِينَ), losing the one that came before.
 *
 * The merge explains the LONGEST PREFIX of [n] as a re-render of [last]'s
 * recent tail and APPENDS the rest (the genuinely new words). A word of [n]
 * is accepted as a re-render of `last[j]` only when THREE conditions hold:
 *
 *  1. TIME-ELIGIBILITY — the occurrence's audio could still be inside the
 *     decoded window. A word first emitted at [lastEmitMs]`[j]` had its audio
 *     at or before that emission, so once `lastEmitMs[j] < nowMs - windowMs`
 *     its audio is definitely LEFT of the window and the occurrence can never
 *     be re-heard: it is ineligible (falaq's ayah-3 من شر can no longer be
 *     re-rendered once the window has slid into ayah-4, so the identical
 *     phrase arriving from NEW audio is appended, not folded).
 *
 *  2. CONTIGUOUS CONTINUATION — if `n[i]` re-renders `last[j]`, the NEXT
 *     decode word must continue at `last[j+1]` (or `last[j+2]`, since a
 *     re-render can drop a word mid-span — the "اللَّهِحْمَن" case). The old
 *     any-position subsequence match folded al-ikhlas' SECOND الله into the
 *     first (الله الصمد re-rendered as if it were الله احد) and the aligner
 *     charged a phantom SKIPPED; a contiguous continuation can no longer
 *     bridge two different occurrences of a repeated word.
 *
 *  3. CONTRADICTION REJECTION — when the continuation mismatches the log,
 *     the log word AFTER the anchor decides: is it a REAL vocabulary word
 *     ([vocab] hit), the audio after the anchor is known and the decode heard
 *     something else → the anchor is a DIFFERENT occurrence → reject it and
 *     append (the al-ikhlas الله case). Is it an unmatched OOV fragment
 *     (garbage the previous decode committed), the new decode may simply be a
 *     better rendering of the same span → skip the garbage and keep matching
 *     (the fatihah حْمَحِي case).
 *
 * When timing metadata is absent ([lastEmitMs] == null) every occurrence is
 * eligible, and with [vocab] == null a broken chain keeps the legacy
 * "fold the consistent prefix, append the rest" behaviour — plain 2-arg calls
 * (unit tests, the raw-word merge) stay on the historic contract.
 *
 * [last] is never reordered, shrunk or rewritten — only the not-yet-shown
 * tail of [n] is appended, in decode order. Therefore the "hold" case
 * (nothing new) is simply: result == [last]. TWO exceptions: when [n] is a
 * FULL re-render of [last] (same size, anchor at 0), the FRESHER decode's
 * spelling wins (a tashkeel re-spelling replaces the stale rendering); and
 * the session's FINAL tail decode ([fullReRender] = true), which re-hears
 * the whole recitation in one pass — when nothing of it can anchor on the
 * log yet it covers the log closely (the log is the same words scrambled /
 * poisoned by window-edge losses, e.g. falaq's dropped وَمِن), the fresh
 * decode REPLACES the log so the commit preserves reading order.
 */
internal fun monotonicMergeWords(
    last: List<String>,
    n: List<String>,
    lastEmitMs: LongArray? = null,
    nowMs: Long = Long.MAX_VALUE,
    windowMs: Long = Long.MAX_VALUE,
    vocab: Set<String>? = null,
    vocabMultiset: List<String>? = null,
    fullReRender: Boolean = false,
    coversSession: Boolean = false
): List<String> {
    if (last.isEmpty()) return n
    if (n.isEmpty()) return last
    val L = last.size
    val lastNorm = last.map { normalizeArabicPub(it) }
    val nNorm = n.map { normalizeArabicPub(it) }
    val eligibilityCut = nowMs - windowMs

    // Try to explain n[i..) as the continuation of a re-render currently
    // anchored at last[cursor], starting at decode word i0. Returns the index
    // of the first NOT-consumed n-word, or null when the continuation
    // CONTRADICTS the log (the anchor must be a different occurrence of the
    // word).
    fun extend(cursor0: Int, i0: Int): Int? {
        var cursor = cursor0
        var i = i0 + 1
        var garbageSkips = 0
        while (i < n.size) {
            if (cursor >= L - 1) return i      // re-render reached the log's end: the rest is new audio
            val want = nNorm[i]
            when {
                lastNorm[cursor + 1] == want -> { cursor++; i++ }
                cursor + 2 < L && lastNorm[cursor + 2] == want -> { cursor += 2; i++ } // re-render dropped one log word
                else -> {
                    val nextLog = lastNorm[cursor + 1]
                    if (vocab != null && nextLog.isNotEmpty() && nextLog !in vocab && garbageSkips < 2) {
                        // The log's next word is an unmatched OOV fragment —
                        // this decode may be a better rendering of the same
                        // span: step over the garbage, keep the chain alive.
                        cursor++; garbageSkips++
                    } else if (vocab == null) {
                        return i               // legacy: fold the consistent prefix, append the rest
                    } else {
                        return null            // real word in the log after the anchor → contradiction
                    }
                }
            }
        }
        return n.size
    }

    // Candidate anchors: occurrences of the decode's leading word, scanned
    // newest-first. With a vocab we ALSO try anchoring on the 2nd/3rd decode
    // word when the leading word(s) found no anchor — a decode can start with
    // a cut/glued garbage token (asr's fused إِنَّنْسَانَ, ikhlas' cut لَّهُۥ
    // before ٱللَّهِ) that is really the window-edge wreck of already-logged
    // audio; anchoring past it explains the rest as a re-render instead of
    // appending a phantom copy (the merge-log poisoning that duplicated whole
    // ayah spans). Keep the anchor that explains the MOST of n (maximal
    // re-render); ties stay with the earliest prefix / newest occurrence.
    var bestConsumed = 0
    var bestAnchor = -1
    var bestK = 0
    val maxK = if (vocab == null) 0 else minOf(2, n.size - 1)
    for (k in 0..maxK) {
        // A skipped prefix word must be pure garbage (OOV fragment, fused
        // wreck). The moment the prefix word is a REAL vocab word it is
        // genuine new speech (fatihah ayah 3's الرحمن arriving after the
        // left edge dropped الرحمن الرحيم) and must be APPENDED, never eaten
        // to anchor deeper — break, since every larger k would swallow it.
        if (k > 0 && nNorm[k - 1] in vocab!!) break
        for (j in L - 1 downTo 0) {
            if (lastNorm[j] != nNorm[k]) continue
            if (lastEmitMs != null && lastEmitMs[j] < eligibilityCut) continue
            val consumed = extend(j, k) ?: continue
            if (consumed > bestConsumed) { bestConsumed = consumed; bestAnchor = j; bestK = k }
        }
    }
    // Full re-render of the whole log (no skipped prefix) → the fresher
    // decode's spelling wins.
    if (bestConsumed == n.size && bestAnchor == 0 && bestK == 0 && n.size == L) return n
    // FINAL-decode repair: the tail decode re-heard the whole recitation in
    // one pass and nothing of it could anchor on the log — the log was
    // poisoned by window-edge losses (words displaced or dropped and
    // re-appended at the tail). When the decode is the SAME BAG of words as
    // the log (each log word pairs with a decode word, ≤1 edit apart for
    // spelling variants like اعوذ/أَعُوذ or غاسق/غَاسقٍ), trust the fresher,
    // correctly-ordered decode and replace the log with it.
    // ikhlas-style head re-render: the decode explained ITSELF ENTIRELY as
    // a re-render anchored at the log's head, skipping garbage log words via
    // the +2 jump (live-window garbage لَّهُۥ ٱللَّهِ in place of اللَّهُ).
    // The size==L case already returned above; here the decode is SHORTER
    // than the poisoned log. It must still span to the log's LAST word —
    // a decode that stops mid-log is a truncation, not a repair.
    val spansToLogEnd = lastNorm.isNotEmpty() && nNorm.any {
        it == lastNorm.last() || editDistanceLe(it, lastNorm.last(), ceiling = 1) <= 1
    }
    // The decode must ALSO cover the log's head: a head-truncated decode of
    // long audio (humazah: CTC dropped the first ayahs) passes the bag
    // check anyway — leftover log words are unconstrained — and would
    // replace the transcript, LOSING words it never heard.
    val spansLogHead = lastNorm.isNotEmpty() && nNorm.any {
        it == lastNorm.first() || editDistanceLe(it, lastNorm.first(), ceiling = 1) <= 1
    }
    val headAnchoredPoison = bestAnchor == 0 && bestK == 0 && bestConsumed == n.size &&
        n.size < L && spansToLogEnd
    if (fullReRender && vocab != null && (bestAnchor == -1 || headAnchoredPoison) &&
        n.size <= L + 2 && spansLogHead) {
        // A session-covering decode may legitimately be much SHORTER than a
        // log bloated by duplicated window-edge re-appends (falaq's شَرِّ x10):
        // there the bag check, not the size, is the validator. A partial
        // decode must still stay within 2 words of the log.
        val sizeOk = if (coversSession) n.size * 2 >= L else n.size >= L - 2
        if (sizeOk && sameWordBagFuzzy(lastNorm, nNorm)) return n
    }
    // HEAD-ANCHORED LOG POISON: the final decode anchored at the log's head
    // (bestAnchor close to 0) — the live log's leading words are a clean
    // re-render — yet the final decode DOES NOT INCLUDE one or two words
    // the live log committed mid-log. Those words are window-edge phantoms
    // the live windows admitted but the session-covering tail decode
    // correctly refused to re-hear (Baqarah 285's t=4.3 phantom أَنْ: the
    // live log committed it after بِمَا, the final decode correctly
    // rendered بِمَا أُنزِلَ without it, but the merge's append-only
    // anchor path kept the phantom — every subsequent word then shifted
    // by one slot and the aligner charged 6 false SKIPPED + 1 false WRONG
    // on a perfect recitation). When the decode is a session-covering
    // full re-render, the bag of words is similar (the live log has the
    // phantom, the decode doesn't — sameWordBagFuzzy allows ≤2 unpaired),
    // and the decode size is within 2 of the log, trust the fresher
    // decode's reading order and drop the live log's extra words.
    if (fullReRender && vocab != null && coversSession && bestAnchor in 0..1 &&
        !headAnchoredPoison && spansLogHead && spansToLogEnd &&
        n.size <= L && n.size >= L - 2 && sameWordBagFuzzy(lastNorm, nNorm)) {
        return n
    }
    // FINAL-decode prefix trim: the decode explained ITSELF ENTIRELY as a
    // re-render of the log's tail (anchor mid-log, everything consumed) —
    // the log words BEFORE the anchor are not in the decode. When their
    // audio was inside the decoded window the decode SHOULD have heard
    // them: they are premature phantoms (kawthar's lone إِنَّ committed
    // before إِنَّا) — drop them and take the decode. Words whose audio is
    // LEFT of the window are legitimately absent and keep the log intact.
    if (fullReRender && vocab != null && bestConsumed == n.size && bestAnchor in 1..2) {
        var allInWindow = true
        if (lastEmitMs != null) {
            for (j in 0 until bestAnchor) {
                if (lastEmitMs[j] < eligibilityCut) { allInWindow = false; break }
            }
        }
        if (allInWindow) return n
    }
    var tail = n.subList(bestConsumed, n.size)
    if (vocab != null) {
        // OOV tail-word filter (active whether or not an anchor was found —
        // a window can re-render the log's tail AND still surface a fresh
        // OOV phantom AFTER the re-render block: ma'un's t=22.3
        // "تفسيرع الرحيم" had the OOV تف سيرع anchored on the log AND
        // smuggled a real Arabic word (الرحيم) past the anchor into the new
        // tail, where it had no target slot (surah 107:1-7's vocabulary
        // never contains الرحيم) and the aligner later charged a SKIPPED
        // mistake when it could not consume the real upcoming هُمْ). The
        // collector drops OOV words anyway (target-count cap 0), so never
        // let them into the emission log.
        tail = tail.filter { normalizeArabicPub(it).let { nn -> nn.isNotEmpty() && nn in vocab } }
    }
    if (vocab != null && vocabMultiset != null && tail.isNotEmpty()) {
        // ECHO GUARD (target-count): a tail word whose vocab count is ALREADY
        // fully satisfied by the log is a re-render echo of SPENT audio, not
        // new speech (measured on Māʿūn: a 4th أَلَا appended after the target's
        // three; on Fātiḥah: الرَّحْمَٰنِ echoing after الْحَمْدُ). Dropping it at
        // the source stops the cascade where the echo steals the alignment
        // slot of the real upcoming occurrence and the collector then strips
        // the REAL word as over-count. Legitimate repeats (target count not
        // yet spent) pass through untouched.
        // Counts come from vocabMultiset — the MULTISET of the normalized
        // target. With only a Set (legacy callers) the guard is DISABLED: a
        // set flattens قُلْ ×2 to ×1 and would drop the REAL second قُلْ of
        // الإخلاص (measured).
        val logCounts = HashMap<String, Int>()
        for (w in lastNorm) if (w.isNotEmpty()) logCounts[w] = (logCounts[w] ?: 0) + 1
        val vocabCounts = HashMap<String, Int>()
        for (v in vocabMultiset) {
            val nn = normalizeArabicPub(v)
            if (nn.isNotEmpty()) vocabCounts[nn] = (vocabCounts[nn] ?: 0) + 1
        }
        tail = tail.filter { w ->
            val nn = normalizeArabicPub(w)
            if (nn.isEmpty()) return@filter false
            val vc = vocabCounts[nn] ?: return@filter false
            val lc = logCounts[nn] ?: 0
            if (lc >= vc) false else { logCounts[nn] = lc + 1; true }
        }
    }
    if (bestAnchor == -1 && tail.isEmpty()) return last
    val out = ArrayList<String>(last.size + tail.size)
    out.addAll(last)
    out.addAll(tail)
    return out
}

/**
 * True when all but ≤2 words of [decode] pair with a DISTINCT word of
 * [log] (exact normalized equality or ≤1 edit — a spelling variant like
 * اعوذ/أَعُوذ or غَاسق/غاسق). The 2 unpaired allowance covers words the
 * live windows missed entirely and the final decode RESTORES (nasr's
 * النَّاسَ). Order-blind on purpose: the final decode's order is the
 * trustworthy one; this only checks it is the same recitation. Leftover
 * log words (phantom duplicates, garbage) are unconstrained — the caller's
 * size guard already ensures the decode spans the whole session.
 */
private fun sameWordBagFuzzy(log: List<String>, decode: List<String>): Boolean {
    val used = BooleanArray(log.size)
    var unpaired = 0
    for (wd in decode) {
        if (wd.isEmpty()) continue
        var matched = -1
        for (j in log.indices) {
            if (used[j]) continue
            val wl = log[j]
            if (wl.isEmpty()) continue
            if (wl == wd || editDistanceLe(wl, wd, ceiling = 1) <= 1) { matched = j; break }
        }
        if (matched == -1) {
            if (++unpaired > 2) return false
        } else {
            used[matched] = true
        }
    }
    return true
}