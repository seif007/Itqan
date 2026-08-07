package com.example.sherpa

import android.content.Context
import android.util.Log
import com.example.ui.normalizeArabic as normalizeArabicUi
import com.k2fsa.sherpa.onnx.EndpointConfig
import com.k2fsa.sherpa.onnx.EndpointRule
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineRecognizer
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineStream
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit

// ─────────────────────────────────────────────────────────────────────────────
// Itqān — offline streaming speech recognizer pipeline.
//
// This file implements the **single-AudioRecord shared pipeline** that the user
// signed off on for the Memorize+Tajweed mode: exactly one `AudioRecord` instance
// feeds BOTH (a) the Sherpa-ONNX `OnlineRecognizer` (for word-by-word transcript
// partials that drive the Memorize alignment / word highlight) AND (b) the
// existing TajweedNet acoustic scorer (which receives the same PCM samples via a
// `FloatArray` sink). Using one capture session — instead of running the
// platform `SpeechRecognizer` in parallel with our own `AudioRecord` — preserves
// the single-mic invariant that previously broke recitation on the Samsung A05s
// (which starves partials as soon as two HAL clients compete for the mic).
//
// The Sherpa-ONNX JNI bits + the trained streaming Zipformer weights are NOT
// bundled in the APK. The JNI bridge ships inside the vendored
// `app/libs/sherpa-onnx-1.13.4.aar` (4 ABIs: arm64-v8a / armeabi-v7a / x86 /
// x86_64). The ~323 MB streaming Zipformer model
// (`csukuangfj/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10`)
// is downloaded on first launch into `context.filesDir/sherpa_model/` with a
// Compose-observable progress dialog, so the APK stays ~30 MB and nothing is
// checked into source control. The entire stack is Apache-2.0 — free for both
// the developer and the end user (no per-call cloud APIs, no non-commercial
// license such as Moonshine Community License, which was the reason Moonshine
// Arabic was rejected during the research phase).
//
// Hallucination mitigation (4 layers, see `SingleAudioPipeline.start()`):
//   (i)  `hotwordsFile` — decoder-level vocabulary bias on the active ayah
//        range's spoken-form words. Updated each session via `updateHotwords()`
//        before a new `OnlineStream` is created.
//   (ii) post-filter on each decoded partial: drop recognized words that do not
//        plausibly match any word in the active ayah range (see
//        `AyahWordMatcher`). This is the"`only keep what the user was supposed
//        to say"`guard the user asked for.
//   (iii) endpoint-driven emission: Sherpa's `isEndpoint()` triggers a "final"
//        segment commit; partials are emitted continuously between endpoints.
//   (iv) dedup + monotonicity guard inside `AyahWordMatcher` so noisy
//        re-decodings of the same audio window can never shrink the transcript.
// ─────────────────────────────────────────────────────────────────────────────

private const val TAG = "ItqanSherpa"

/** HuggingFace repo for the streaming Zipformer (Apache-2.0). */
private const val MODEL_REPO =
    "csukuangfj/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10"

/** Concrete files & sizes (verified against HF API bytes). */
private data class ModelFile(val name: String, val size: Long)

private val MODEL_FILES = listOf(
    ModelFile("encoder-epoch-75-avg-11-chunk-16-left-128.int8.onnx", 296_583_597L),
    ModelFile("decoder-epoch-75-avg-11-chunk-16-left-128.onnx",       33_837_085L),
    ModelFile("joiner-epoch-75-avg-11-chunk-16-left-128.int8.onnx",    8_257_421L),
    ModelFile("tokens.txt",                                              195_244L),
    // bpe.model is used by some Zipformer variants for LM vocab; harmless to ship.
    ModelFile("bpe.model",                                               476_049L),
)
private val MODEL_TOTAL_BYTES = MODEL_FILES.sumOf { it.size }

/** Absolute on-disk model dir under `filesDir`. */
fun sherpaModelDir(context: Context): File =
    File(context.filesDir, "sherpa_model").apply { mkdirs() }

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

/** True iff every model file is present at its known size. */
fun isSherpaModelReady(context: Context): Boolean {
    val dir = sherpaModelDir(context)
    return MODEL_FILES.all { f ->
        val file = File(dir, f.name)
        file.exists() && file.length() >= f.size
    }
}

/**
 * Downloads the streaming Zipformer weights into `filesDir/sherpa_model/` if not
 * already present. Reports per-file and overall progress via [progress].
 *
 * Failure semantics: a quick HTTP HEAD is performed first so we don't allocate
 * diskspace for a 323 MB transfer we cannot actually fetch. Any IOException or
 * short-read deletes the partial file and rethrows; the caller surfaces the
 * error in the Compose download dialog and lets the user retry.
 */
class SherpaModelDownloader(private val context: Context) {

    sealed interface Progress {
        /** Download has not started yet (e.g. locally complete / waiting). */
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
        /** A file failed to download. `partialFile` removed. */
        data class Failed(val message: String) : Progress
    }

    private val _progress = MutableStateFlow<Progress>(Progress.Idle)
    val progress: StateFlow<Progress> = _progress.asStateFlow()

    private val client by lazy {
        OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .retryOnConnectionFailure(true)
            .build()
    }

    /**
     * Concurrency guard so the Compose side can fire `downloadIfNeeded()`
     * from MULTIPLE `LaunchedEffect(Unit)` blocks (Memorize+Tajweed FAB AND
     * the Tajweed Rules Coach landing) without spawning duplicate parallel
     * downloads. Whichever caller wins the `compareAndSet` schedules the work;
     * subsequent callers see an already-active job and return immediately.
     */
    private val inflight = java.util.concurrent.atomic.AtomicBoolean(false)

    /** True iff download is already complete (no work to do). */
    fun isComplete(): Boolean = isSherpaModelReady(context)

    /**
     * Idempotent model download. Safe to call from multiple Compose effects
     * and safe to re-call after a failure (the [inflight] guard prevents
     * concurrent duplicates; per-file size-checks make a re-run after a partial
     * download a resume, not a restart). Returns immediately if the model is
     * already on disk.
     */
    suspend fun downloadIfNeeded() = withContext(Dispatchers.IO) {
        // Fast path: already complete. Flip straight to Done so the UI dialog
        // collapses without ever showing a spurious "Downloading" frame.
        if (isComplete()) {
            _progress.value = Progress.Done
            return@withContext
        }
        // Single-flight: another caller is already driving the download.
        if (!inflight.compareAndSet(false, true)) return@withContext
        try {
            val dir = sherpaModelDir(context)
            val totalFiles = MODEL_FILES.size
            var receivedTotal = 0L

            MODEL_FILES.forEachIndexed { idx, mf ->
                val target = File(dir, mf.name)
                // Resume support: if the existing partial is at least mf.size, skip.
                if (target.exists() && target.length() >= mf.size) {
                    receivedTotal += mf.size
                    _progress.value = Progress.Downloading(
                        currentFile = mf.name, fileIndex = idx + 1, fileCount = totalFiles,
                        receivedFileBytes = mf.size, fileBytes = mf.size,
                        receivedTotal = receivedTotal, totalBytes = MODEL_TOTAL_BYTES
                    )
                    return@forEachIndexed
                }
                // Streams via HF CDN; follows HF's 307 to the Xet CDN behind the scenes
                // because OkHttp auto-follows redirects.
                val url = "https://huggingface.co/$MODEL_REPO/resolve/main/${mf.name}?download=true"
                _progress.value = Progress.Downloading(
                    currentFile = mf.name, fileIndex = idx + 1, fileCount = totalFiles,
                    receivedFileBytes = 0L, fileBytes = mf.size,
                    receivedTotal = receivedTotal, totalBytes = MODEL_TOTAL_BYTES
                )
                try {
                    downloadFile(url, target, mf, idx, totalFiles)
                    receivedTotal = MODEL_FILES.subList(0, idx + 1).sumOf { it.size }
                } catch (e: Throwable) {
                    Log.e(TAG, "Failed to download ${mf.name}", e)
                    target.delete()
                    _progress.value = Progress.Failed(
                        message = "${mf.name}: ${e.message ?: e.javaClass.simpleName}"
                    )
                    throw IOException("Download failed for ${mf.name}", e)
                }
            }
            // Final integrity check — refuse to declare success with a truncated file.
            if (!isSherpaModelReady(context)) {
                _progress.value = Progress.Failed("Model files are incomplete")
                throw IOException("Model files are incomplete after download")
            }
            _progress.value = Progress.Done
        } finally {
            // Release the single-flight latch whether we succeeded or failed so
            // a Tap-Retry works without needing an Activity restart.
            inflight.set(false)
        }
    }

    /**
     * Streams `url` into `target`. Per-chunk progress is posted to [_progress]
     * (overall bytes received = sum of completed prior files + bytes read into
     * this file so far). The temp file is written to `target.tmp` first then
     * renamed, so a partial `.tmp` leftover from a previous failed run can never
     * masquerade as a complete model file (the integrity check ignores `.tmp`
     * fragments — it only looks for `MODEL_FILES` exact names).
     */
    private fun downloadFile(
        url: String,
        target: File,
        mf: ModelFile,
        fileIndex: Int,
        fileCount: Int
    ) {
        val req = Request.Builder().url(url)
            .header("User-Agent", "Itqan-ModelDownloader/1.0")
            .header("Accept", "application/octet-stream")
            .build()
        client.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) {
                throw IOException("HTTP ${resp.code} for ${mf.name}")
            }
            val total = mf.size
            val priorFilesTotal = MODEL_FILES.subList(0, fileIndex).sumOf { it.size }
            val tmp = File(target.parentFile, "${target.name}.tmp")
            tmp.delete()
            var received = 0L
            FileOutputStream(tmp).use { sink ->
                val source = resp.body?.byteStream()
                    ?: throw IOException("Empty response body for ${mf.name}")
                val buf = ByteArray(64 * 1024)
                while (true) {
                    val n = source.read(buf)
                    if (n <= 0) break
                    sink.write(buf, 0, n)
                    received += n
                    _progress.value = Progress.Downloading(
                        currentFile = mf.name, fileIndex = fileIndex + 1, fileCount = fileCount,
                        receivedFileBytes = received, fileBytes = total,
                        receivedTotal = priorFilesTotal + received,
                        totalBytes = MODEL_TOTAL_BYTES
                    )
                }
            }
            if (received < total) {
                tmp.delete()
                throw IOException("Short read for ${mf.name}: got $received / $total")
            }
            // Atomic rename onto the final path. On Android's POSIX FS this is atomic when
            // both paths are on the same filesystem, which they always are here.
            if (!tmp.renameTo(target)) {
                // Fallback: copy then delete.
                tmp.copyTo(target, overwrite = true)
                tmp.delete()
            }
        }
    }
}

/**
 * Factory for the Sherpa [OnlineRecognizer] given the on-disk model files.
 *
 * Reads absolute file paths from [sherpaModelDir]; passes a [hotwordsFile]
 * (an absolute path to a newline-separated list of vocabulary phrases) so the
 * decoder biases its beam toward the active ayah range's words. The hotwords
 * score of 1.5f is the Sherpa-recommended default for transducer models — high
 * enough to sway the beam when the audio is genuinely ambiguous, low enough
 * that a clear "han" cannot force-decode as "fī" if the audio disagrees.
 */
fun createOnlineRecognizer(
    context: Context,
    hotwordsFile: String = ""
): OnlineRecognizer {
    val dir = sherpaModelDir(context)
    val cfg = OnlineRecognizerConfig(
        featConfig = FeatureConfig(sampleRate = 16_000, featureDim = 80),
        modelConfig = OnlineModelConfig(
            transducer = OnlineTransducerModelConfig(
                encoder = File(dir, MODEL_FILES[0].name).absolutePath,
                decoder = File(dir, MODEL_FILES[1].name).absolutePath,
                joiner = File(dir, MODEL_FILES[2].name).absolutePath,
            ),
            tokens = File(dir, "tokens.txt").absolutePath,
            numThreads = 2,
            // The model is multilingual (ar/en/id/ja/ru/th/vi/zh), so tell Sherpa
            // which modeling unit family Zipformer-transducer uses here.
            modelType = "zipformer",
            provider = "cpu",
        ),
        endpointConfig = EndpointConfig(
            // rule1: trailing silence >= 2.4s -> endpoint (long pause == end of utterance).
            rule1 = EndpointRule(false, 2.4f, 0.0f),
            // rule2: utterance has speech, then trailing silence >= 1.4s -> endpoint
            // (a normal inter-ayah pause). This drives our "commit final partial"
            // emission in the pipeline.
            rule2 = EndpointRule(true, 1.4f, 0.0f),
            // rule3: an utterance longer than 20s is force-ended to bound memory.
            rule3 = EndpointRule(false, 0.0f, 20.0f),
        ),
        enableEndpoint = true,
        // CRITICAL: hotwordsFile > 0 ONLY fires when decodingMethod ==
        // "modified_beam_search" (the C++ Validate() aborts otherwise, line 148
        // of online-recognizer.cc). greedy_search silently ignores hotwords,
        // defeating mitigation layer (i). Beam search with 4 active paths is
        // only marginally slower than greedy on the int8 model and gives us the
        // vocabulary bias — the user's primary requested hallucination guard.
        decodingMethod = "modified_beam_search",
        maxActivePaths = 4,
        hotwordsFile = hotwordsFile,
        hotwordsScore = 1.5f,
    )
    return OnlineRecognizer(assetManager = null, config = cfg)
}

/**
 * Writes the active ayah range's spoken-form words into a newline-separated
 * hotwords file under [context]'s cache dir and returns its absolute path.
 *
 * Sherpa hotwords use the model's token vocabulary; for Zipformer-transducer
 * multilingual models with BPE, the recommended form is the SPACE-joined list
 * of word tokens, one phrase per line, e.g.:
 *   ال حمد لله رب العالمين
 * Each line biases the beam toward that phrase. We dedupe case-insensitively and
 * drop words that normalize to empty (annotation-only tokens are already gone
 * from `targetOrig`, but we double-guard).
 *
 * `extraPhrases` is the optional extension point used by the memorize pipeline
 * to inject spoken-form expansions of muqatta'at (e.g. "ألف لام ميم" for الم)
 * — pass them in to widen the bias without rewriting the canonical target list.
 */
fun writeHotwordsFile(
    context: Context,
    words: List<String>,
    extraPhrases: List<String> = emptyList()
): String {
    val seen = HashSet<String>()
    val phrases = ArrayList<String>(words.size + extraPhrases.size)
    for (w in words) {
        val s = w.trim()
        if (s.isEmpty()) continue
        val key = s.lowercase()
        if (seen.add(key)) phrases.add(s)
    }
    for (p in extraPhrases) {
        val s = p.trim()
        if (s.isEmpty()) continue
        val key = s.lowercase()
        if (seen.add(key)) phrases.add(s)
    }
    val file = File(context.cacheDir, "sherpa_hotwords.txt")
    file.writeText(phrases.joinToString("\n"))
    return file.absolutePath
}

// ─────────────────────────────────────────────────────────────────────────────
// AyahWordMatcher — hallucination guard (mitigation layer ii + iv).
//
// Sherpa's streaming Zipformer is broadcast-trained and emits a noisy partial:
// it sometimes splices in filler words ("uhm", "yep", stray English/Chinese
// tokens because the model is multilingual) or re-reads a window of audio and
// drops a word that was actually correct. We counteract that by:
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
//
// The matcher does NOT have to be perfect: this is the *post* layer on top of
// the *pre* decoder-level `hotwordsFile` bias (layer i). Their combination is
// what cancels both the gross hallucinations (decoder beam steered off-target)
// and the residual "soft" ones (decoder steered correctly but the model pads
// the tail of a partial with a stray fill word).
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Phonetic-neighbor matching for Arabic ayah words.
 *
 * Compares two normalized strings (post-`normalizeArabic`) allowing an edit
 * distance of [maxEdits] OR a long common-prefix. The long-common-prefix rule
 * is what makes a one-syllable mispronunciation like "الحم" still count as a
 * match for "الحمد" (so it gets sent to the tajweed scorer, which is where
 * mispronunciations SHOULD be detected — not silently erased here).
 *
 * Returns true iff the two strings are within the phonetic neighborhood.
 */
private fun matchesAyahWord(
    candidate: String,
    ayahNorm: String,
    maxEdits: Int = 1
): Boolean {
    if (candidate == ayahNorm) return true
    // Empty matching against anything is a non-match.
    if (candidate.isEmpty() || ayahNorm.isEmpty()) return false
    if (candidate.length < 2 || ayahNorm.length < 2) {
        return candidate == ayahNorm
    }
    val ce = candidate.codePointCount(0, candidate.length)
    val ae = ayahNorm.codePointCount(0, ayahNorm.length)
    // Long-common-prefix: cheap and handles the typical single-syllable slip
    // ("الحم" vs "الحمد") well.
    val prefixLen = run {
        var n = 0
        val lim = minOf(candidate.length, ayahNorm.length)
        while (n < lim && candidate[n] == ayahNorm[n]) n++
        n
    }
    if (prefixLen >= minOf(ce, ae) - 1 && minOf(ce, ae) >= 3) return true
    // Edit distance for shorter candidates.
    if (kotlin.math.abs(ce - ae) > maxEdits) return false
    return editDistanceLe(candidate, ayahNorm, maxEdits) <= maxEdits
}

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
 * Per-session ayah vocabulary + filter. Construct once per Memorize+Tajweed
 * session, providing the canonical target words (`targetOrig` already cleansed
 * of standalone Quranic annotation marks by `splitAyahWords` upstream) and
 * their normalized form via [normalizeArabic].
 *
 * The matcher is **per-partial**: for each Sherpa partial transcript it:
 *   1. Splits into whitespace-separated tokens.
 *   2. Normalizes each word.
 *   3. Drops it if the normalized form is empty (typical for stray annotation
 *      signs the model emitted as filler) or it does not appear in the active
 *      ayah vocab EVEN as a single-edit neighbor.
 *   4. Returns the cleaned transcript.
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
    /** Normalized target words, same indices as the caller's targetOrig. */
    private val normTargets: List<String> = ayahWords.map { normalizeArabicPub(it) }
    /** Quick membership set (lowercased + normalized) for the common exact case. */
    private val exactSet: Set<String> = normTargets.map { it }.toHashSet()
    /** Cached neighbor lookups — the dictionary is small (<= a few hundred words per ayah range). */
    private val neighborCache = HashMap<String, Boolean>()

    // Track the last successfully returned clean transcript so we can enforce
    // monotonicity (layer iv) across noisy re-decodes.
    private var lastCleanList: List<String> = emptyList()

    /**
     * Cleans a partial transcript. Returns the cleaned token list — never grows
     * shorter than the previously-emitted one (monotonicity guard).
     */
    @Synchronized
    fun clean(partial: String): List<String> {
        val rawTokens = partial.split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (rawTokens.isEmpty()) return lastCleanList
        val cleaned = ArrayList<String>(rawTokens.size)
        for (tok in rawTokens) {
            val norm = normalizeArabicPub(tok)
            if (norm.isEmpty()) continue // stray annotation / zero-content token
            if (exactSet.contains(norm)) {
                cleaned.add(tok); continue
            }
            val ok = neighborCache.getOrPut(norm) {
                // Loop the (usually short) vocab and accept the neighbor.
                var any = false
                for (t in normTargets) {
                    if (matchesAyahWord(norm, t, maxEdits = 1)) { any = true; break }
                }
                any
            }
            if (ok) cleaned.add(tok)
        }
        // Monotonicity: never shrink the emitted list within a session.
        if (lastCleanList.isNotEmpty() && cleaned.size < lastCleanList.size) {
            // The first lastCleanList.size words are kept (anchor), tail tokens
            // from the new partial that fall beyond are appended only if they
            // are genuinely new (dedup against lastCleanList's tail).
            val merged = ArrayList(lastCleanList)
            val seen = lastCleanList.toHashSet()
            for (i in lastCleanList.size until cleaned.size) {
                val w = cleaned[i]
                if (seen.add(w)) merged.add(w)
            }
            lastCleanList = merged
            return merged
        }
        lastCleanList = cleaned
        return cleaned
    }

    /** Reset between sessions (e.g. on ayah range change). */
    @Synchronized
    fun reset() {
        lastCleanList = emptyList()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SingleAudioPipeline — one AudioRecord → Sherpa + TajweedNet.
//
// Why this exists: on the Samsung A05s the platform SpeechRecognizer and a
// parallel AudioRecord starve each other's partials when run together
// (regression the user reported in commit-message-words). The fix the user
// signed off on: drive BOTH consumers from ONE AudioRecord session using the
// offline Sherpa OnlineRecognizer as the transcript source, eliminating the
// second HAL client entirely. The TajweedNet acoustic scorer keeps receiving
// the same PCM samples via [fullAudioSink], which the caller harvests on stop.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Emitted by the pipeline's recogniser thread. `partial` is the cleaned
 * transcript (post-`AyahWordMatcher`); `endpoint` true marks a sentence-final
 * commit (the Sherpa `isEndpoint` flag tripped, the stream was reset, and the
 * caller should treat everything up to now as a "final" segment).
 */
data class SherpaPartialEvent(
    val partial: List<String>,
    val rawPartial: String,
    val endpoint: Boolean
)

class SingleAudioPipeline(
    private val context: Context,
    /** Caller-supplied vocab for the active ayah range (display words). */
    private val ayahWordsProvider: () -> List<String>,
    /** Caller-supplied extra hotword phrases (e.g. muqatta'at expansions). */
    private val extraPhrasesProvider: () -> List<String> = { emptyList() }
) {
    /**
     * Coarse lifecycle state of the pipeline, exposed for the Compose UI so it
     * can show a "Loading recognizer…" affordance instead of ANR-ing when the
     * 296 MB ONNX encoder first lands in the ONNX Runtime.
     *
     * The recognizer is NOT ready to drive AudioRecord through [startReadLoop]
     * until state flips to [State.Running]. Mirrors the [running] boolean but
     * with an explicit "Loading" transition so the UI can distinguish "starting
     * (loading model on IO)" from "started".
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

    private val sampleRate = 16_000

    // Native recognizer lifecycle.
    @Volatile private var recognizer: OnlineRecognizer? = null
    @Volatile private var stream: OnlineStream? = null

    // Background read/decode thread.
    private var job: Job? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    // Streaming partials flow — UI subscribes and updates word highlight.
    // DROP_OLDEST + 64-cap means a burst of partials never backpressures the
    // read/decode loop (which would stall the AudioRecord reads and starve the
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
    // callers harvest it via [stop] as a single contiguous FloatArray.
    //
    // Since the worker thread and the [stop] method can both touch this state,
    // access is guarded by [audioSync] (a plain Object monitor) rather than
    // Mutex — Mutex.withLock is suspend-only and we need synchronous access.
    private var audio = FloatArray(sampleRate * 10)
    private var audioSize = 0
    private val audioSync = Any()
    @Volatile private var matcher: AyahWordMatcher? = null

    /**
     * Cancel-during-load latch set by [stop]/[release] so an in-flight
     * [start] IO job — still busy constructing the 296 MB ONNX recognizer —
     * knows NOT to flip `running=true` and tear itself down instead. Without
     * this, a user who taps the mic, sees `isListening` go true immediately,
     * then taps Stop before the recognizer finishes loading (a 2–5 s window on
     * the A05s) would end up with a fully-loaded recognizer + live AudioRecord
     * running under a deactivated UI, leaking mic + 300 MB RSS until the next
     * pipeline reload.
     *
     * SEE ALSO [isStopping] — the related (but distinct) "teardown in flight"
     * latch, which prevents a NEW [start] from racing the pending teardown
     * that an OLD [stop]/[release] scheduled on the IO scope.
     */
    @Volatile private var cancelledDuringLoad: Boolean = false

    /**
     * "Teardown in flight" latch — set true by [stop]/[release] at the moment
     * they enqueue the **asynchronous** `worker.join() + teardown()` on the IO
     * scope, and cleared in that same IO job after teardown() has actually
     * released the native `OnlineRecognizer`/`OnlineStream` handles.
     *
     * Why this exists: [stop] and [release] are non-blocking by design (the
     * main thread can't safely await a native join). But that means a caller
     * that immediately re-taps the mic — or a `LaunchedEffect(targetOrig)`
     * that re-fires on a range change — can land in [start] while the OLD
     * pipeline's teardown is STILL pending on the IO scope. The old teardown
     * would then either (a) free `stream#A` underneath a still-running worker
     * `W_A` that [start]#2 just resurrected by flipping `running=true` again
     * (native use-after-free → SIGSEGV) — or (b) leave behind a zombie
     * AudioRecord (#A) which \(W_B\)'s new AudioRecord then can't initialize
     * on the Samsung A05s single-mic invariant ("AudioRecord not initialised"
     * → user sees "crash from beginning").
     *
     * Contract:
     *  - [start] checks this latch FIRST. If true → previous teardown pending,
     *    return false synchronously. The caller surfaces a "model still
     *    shutting down, try again in a moment" message; the user can re-tap
     *    once `isStopping` flips false (after teardown() finishes on IO).
     *  - [stop]/[release] set it true the moment they enqueue teardown.
     *  - The teardown IO job clears it after teardown() returns.
     *  - [isSherpaModelReady] is **never** affected by this latch (the model
     *    files stay on disk regardless), so the user still sees the model
     *    as ready and CAN retry; we just block the early re-start that would
     *    race the pending teardown.
     */
    @Volatile private var isStopping: Boolean = false

    /**
     * Kicks off asynchronous recognizer + AudioRecord startup. Returns
     * immediately — the heavy work `createOnlineRecognizer(...)` (which loads
     * the 296 MB int8 encoder ONNX + ORT init, taking 2–5 s on the Samsung
     * A05s) runs on [Dispatchers.IO], NEVER on the Compose main thread which
     * would otherwise ANR the app and Android would auto-kill it.
     *
     * Lifecycle outcome is observable via [state]:
     *  - `Idle`     : pipeline was never started,
     *  - `Loading`  : recognizer is being constructed on IO,
     *  - `Running`  : AudioRecord is live, partials are flowing,
     *  - `Failed`   : construction failed (mic busy / model corrupted).
     *
     * Returns `false` synchronously ONLY when the model isn't on disk yet
     * (caller must launch the sherpa downloader UI first). Every other case
     * returns `true` to indicate "async start accepted", and the UI watches
     * [state] to know when audio is actually live.
     *
     * Safe to call from main thread — does NO heavy work inline.
     */
    @Synchronized
    fun start(): Boolean {
        if (running || _state.value is State.Loading) return true
        // Block an early re-start while a previous teardown is still pending
        // on the IO scope. The user tapped mic → pause → tap mic again FAST;
        // or `LaunchedEffect(targetOrig)` just fired `release()` (which
        // schedules async teardown) and immediately assigned a fresh pipeline
        // that is now being asked to start(). Allowing the new start() to
        // proceed here would race the old teardown: the OLD worker might still
        // be running (because new running=true would resurrect a worker whose
        // `while (running)` would see running=true again), or the OLD AudioRe-
        // cord + OnlineStream might still be half-freed, and we'd get a SIGSEGV
        // under our feet on the A05s single-mic setup. Bail out instead;
        // caller surfaces "model still shutting down, try again in a moment".
        if (isStopping) {
            Log.w(TAG, "start(): previous teardown still in flight — refusing to start a second pipeline")
            _state.value = State.Failed("Recognizer is still shutting down — try again in a moment")
            return false
        }
        if (!isSherpaModelReady(context)) {
            Log.w(TAG, "start(): model not downloaded yet")
            return false
        }
        cancelledDuringLoad = false
        _state.value = State.Loading
        job = scope.launch {
            try {
                // (Re)build the hotwords file from the active ayah range. The
                // hotwords bias (layer i) is applied at recognizer-construction-
                // time per the Sherpa README.
                val ayahWords = ayahWordsProvider()
                matcher = AyahWordMatcher(ayahWords)
                val hotwordsPath = writeHotwordsFile(context, ayahWords, extraPhrasesProvider())
                // Heavy: loads the 296 MB encoder ONNX into ONNX Runtime.
                val rec = createOnlineRecognizer(context, hotwordsPath)
                val st = rec.createStream("")
                // Cancel-during-load race guard: if [stop] was called while we
                // were busy constructing the recognizer above, tear it down
                // instead of flipping running=true. The user already saw
                // isListening go false synchronously so the UI is correct.
                if (cancelledDuringLoad) {
                    try { st.release() } catch (_: Throwable) {}
                    try { rec.release() } catch (_: Throwable) {}
                    _state.value = State.Idle
                    return@launch
                }
                recognizer = rec
                stream = st
                running = true
                _state.value = State.Running
                startReadLoop(st)
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to start Sherpa pipeline", t)
                teardown()
                _state.value = State.Failed(
                    t.message ?: t.javaClass.simpleName
                )
                running = false
            }
        }
        return true
    }

    // ───────────────────────────────────────────────────────────────────────
    // Pause / Resume — the lightweight alternative to stop()/start() that the
    // Memorize+Tajweed FAB needs for a pause-and-continue flow.
    //
    // PROBLEM (the bug the user just reported): pausing then tapping the mic
    // to continue crashed the app, AND every subsequent fresh open also crashed
    // "from the beginning". Root cause was a stop()/start() race: stop()
    // enqueues teardown on the IO scope (wait-for-worker + free OnlineStream +
    // OnlineRecognizer), and a fast follow-up start() — either from a quick
    // resume tap, OR from the LaunchedEffect(targetOrig) range-change that
    // instantly reassigns a new pipeline into the holder — could race that
    // pending teardown. The new start() reads `running=false` and proceeds to
    // flip `running=true`; that RESURRECTS the OLD worker's `while (running)`
    // loop, which still holds stream#A. The OLD teardown then frees stream#A
    // under the resurrected worker's feet -> native use-after-free -> SIGSEGV.
    // Worse, on the Samsung A05s the OLD AudioRecord#A would still be holding
    // the single mic HAL, so AudioRecord#B's `STATE_INITIALIZED` check returned
    // false ("AudioRecord not initialised") and the user saw what felt like a
    // crash from the very first tap on subsequent launches.
    //
    // FIX: pause() / resume() reuse the SAME OnlineRecognizer + OnlineStream
    // across a pause-resume cycle — only the cheap AudioRecord + worker thread
    // are torn down on pause and re-spawned on resume. The 296 MB ONNX stays
    // loaded in memory so there's no 2-5 s reload either. Teardown ONLY
    // happens on the explicit [stop] terminal flow (the "Stop Listening" tap
    // that ends the recitation session) or [release] (range change / app
    // background) — never on the FAB pause-then-continue path.
    // ───────────────────────────────────────────────────────────────────────

    /**
     * "Soft stop" for a pause-and-resume flow: stops the AudioRecord worker and
     * releases ONLY the AudioRecord (cheap Android-resource release), leaving
     * the [OnlineRecognizer] + [OnlineStream] (and the 296 MB ONNX weights
     * loaded in memory) intact so the next [resume] / [start] can re-spawn
     * the worker in milliseconds without reloading the model.
     *
     * Returns the audio captured since the last resume in 16 kHz mono [-1,1],
     * matching [stop]'s shape so the TajweedNet scorer unchanged.
     *
     * Idempotent: calling pause() when already paused is a no-op (returns an
     * empty FloatArray). The `paused` flag distinguishes "never started" from
     * "started and paused", which [resume] uses to decide whether to fresh
     * start (no recognizer yet → fall through to [start]) or resume (re-spawn
     * the worker on the existing stream).
     */
    @Synchronized
    fun pause(): FloatArray {
        if (!running) return FloatArray(0)
        running = false
        cancelledDuringLoad = true
        // The AudioRecord handle will be released by the worker's `finally`
        // block once it observes running=false and exits its loop. We MUST NOT
        // call [resume] until that finally block has actually released the OLD
        // AudioRecord — on the Samsung A05s the mic HAL is a single-instance
        // resource, so a new AudioRecord created before the old one is released
        // will return STATE_UNINITIALIZED and the user sees "the mic crashes
        // from the beginning". [resume] therefore waits on the OLD job's
        // completion (job?.join()) before spawning the new AudioRecord.
        val out: FloatArray
        synchronized(audioSync) {
            out = audio.copyOf(audioSize)
            audio = FloatArray(sampleRate * 10)
            audioSize = 0
        }
        try { stream?.inputFinished() } catch (_: Throwable) {}
        // Tell Sherpa to commit any in-flight partial as a final segment, so
        // the partials collector (in the UI) appends it to the final transcript
        // before the worker goes quiet.
        _partials.tryEmit(SherpaPartialEvent(
            partial = emptyList(),
            rawPartial = "",
            endpoint = true
        ))
        paused = true
        _state.value = State.Idle
        return out
    }

    /**
     * True iff the pipeline has been paused via [pause] (the recognizer/stream
     * are still live and can be resumed). Distinguishes "never started" from
     * "started and paused". Cleared by [resume], [stop] (via teardown), and
     * [release].
     */
    @Volatile var paused: Boolean = false
        private set

    /**
     * Resumes the pipeline from a [pause]-flipped idle state by re-spawning the
     * AudioRecord worker on the EXISTING Sherpa [OnlineRecognizer] +
     * [OnlineStream]. No model reload, no 2-5 s wait, no chance of racing a
     * previous teardown because pause() did NOT free the natives, only the
     * AudioRecord (pending the worker's own finally block on the IO scope).
     *
     * CRITICAL — the A05s single-mic HAL race: we MUST wait for the OLD
     * AudioRecord (captured inside the paused W_A worker's closure) to be
     * released by W_A's `finally` block BEFORE we try to create AudioRecord#B
     * in W_B. We do this by capturing the OLD `job` reference and having W_B
     * `job.join()` it (suspended on the IO scope, NOT main) before spawning
     * the new AudioRecord. This is what makes pause/resume race-free on a
     * single-mic device.
     *
     * Returns false if the pipeline was never started in this session (caller
     * should fall through to [start]) OR if a teardown is in flight (caller
     * should retry shortly). Returns true on a successful resume launch —
     * audio is live when [state] flips back to [State.Running].
     *
     * The worker's read/decode loop is identical to `startReadLoop` from [start]:
     * it grabs a fresh AudioRecord (now that the OLD one is released), pushes
     * the PCM into the SAME stream, and emits partials — exactly as if the user
     * had never paused. The decoder's persistent state between pauses is
     * preserved because we never reset the stream between pause/resume; the
     * AyahWordMatcher's monotonicity cursor is also preserved, matching user
     * expectation that the word-by-word highlight continues from where they
     * left off.
     */
    @Synchronized
    fun resume(): Boolean {
        if (running) return true
        if (!paused) return false           // not paused → caller falls through to start()
        if (isStopping) {                  // defensive: a teardown IS pending
            Log.w(TAG, "resume(): teardown in flight — caller should retry")
            return false
        }
        val rec = recognizer ?: return false
        val st = stream ?: return false
        val oldWorkerToJoin = job          // W_A — captured for join on IO
        running = true
        cancelledDuringLoad = false
        paused = false
        _state.value = State.Running
        // Re-spawn the AudioRecord worker on the IO scope, but FIRST join the
        // old job so the A05s mic HAL is free for the new AudioRecord.
        job = scope.launch {
            try { oldWorkerToJoin?.join() } catch (_: Throwable) {}
            // Defensive: someone could have called stop()/release() while we
            // were waiting for oldWorkerToJoin. Don't resuscitate a torn-down
            // pipeline.
            if (!running || recognizer == null || stream == null) {
                _state.value = State.Idle
                return@launch
            }
            // Re-spawn the AudioRecord + read-loop body against the SAME
            // OnlineStream. The OLD AudioRecord has been released by the OLD
            // worker's finally block (we just joined it), so the single-mic
            // HAL on the A05s is free for the new AudioRecord instance.
            runReadLoop(st)
        }
        return true
    }
    /**
     * Notifies the pipeline that the active ayah range changed (e.g. user
     * picked a different surah/ayah). Resets the [AyahWordMatcher]'s
     * monotonicity cursor so the next partial decodes cleanly against the new
     * vocab.
     *
     * The Sherpa recognizer's `hotwordsFile` bias (layer i) is read at
     * recognizer-construction-time (the C++ Validate() path checks the file
     * exists and the beam search loads it once). Mid-session hot-reload is *not*
     * supported by the Sherpa API for the model-level bias, so changing the ayah
     * range mid-recital keeps the OLD bias. This is fine because the existing
     * Compose UI calls [stop] + [startTajweedSession] on range change (see the
     * `LaunchedEffect(viewModel.ayahs, ...)` reset), which produces a fresh
     * pipeline with a fresh hotwords file matching the new range.
     */
    @Synchronized
    fun updateHotwords() {
        val ayahWords = ayahWordsProvider()
        matcher = AyahWordMatcher(ayahWords)
    }

    /**
     * Removes the (now-stale) duplicate comment block above — the canonical
     * contract is documented in the larger comment that follows.
     */
    /**
     * Stops the worker, releases the recognizer & stream asynchronously, and
     * returns the accumulated full recitation PCM (16 kHz mono, [-1,1]).
     *
     * CRITICAL: this MUST be safe to call from the Compose main thread, AND
     * MUST NOT race the IO worker. The previous implementation did
     * `runBlocking { job?.join() }` which blocks main until the AudioRecord-
     * worker drains its current read buffer — on the A05s this is enough to
     * trip the 5s ANR watchdog when the app is shutting down.
     *
     * An EARLIER fix then made `stop()` schedule the teardown on the IO scope
     * with `_state.value = Idle` synchronously, but it still called
     * `stream?.release()` inside `teardown()` that the IO job would invoke
     * — without FIRST waiting for the AudioRecord worker to actually leave
     * the `acceptWaveform(...)` / `decode(st)` native call it was in the
     * middle of. That was a **native use-after-free**: the freed `st` pointer
     * was still being dereferenced by the worker thread (synchronous C++
     * OnlineStream is NOT thread-safe between two callers), giving SIGSEGV
     * on the user's report "second ayah -> app crashes" — the range change
     * triggers `LaunchedEffect(targetOrig)` -> `release()` -> `teardown()` ->
     * `stream=null`+`stream?.release()` while the worker iterated on `st`.
     *
     * The contract here is now:
     *   1. flip `running = false` AND `isStopping = true` (the worker sees
     *      `running` between iterations and exits; a fast follow-up [start]
     *      sees `isStopping` and refuses to build a new pipeline until
     *      teardown is done),
     *   2. snapshot the audio accumulator synchronously (the worker only writes
     *      while running=true),
     *   3. emit the synthetic endpoint so the partial collector commits the
     *      last segment,
     *   4. SCHEDULE `job.join() + teardown()` on the IO scope and clear
     *      `isStopping` AFTER teardown() returns — this awaits the worker
     *      BEFORE freeing native handles (no concurrent native access), and
     *      guarantees a re-entry into [start] sees `isStopping == false`
     *      only once the OLD recognizer/stream/AudioRecord are fully gone.
     *      The scheduler coroutine waits on the join from the IO scope,
     *      which is allowed to suspend (it is NOT the main thread).
     */
    @Synchronized
    fun stop(): FloatArray {
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
        // Block a fast re-entry into [start] until the teardown we are about to
        // schedule has actually finished freeing the native handles. Without
        // this latch, a user tapping mic → pause → tap mic would race a second
        // `start()` against the OLD pipeline's pending teardown: the new start
        // builds recognizer#B and flips running=true, which can resurrect the
        // OLD worker's `while (running)` loop — that worker then dereferences
        // stream#A which the OLD teardown is about to free, giving SIGSEGV on
        // the A05s single-mic setup. SEE [isStopping] doc for the full race.
        isStopping = true
        // Tell the recognizer the audio stream is done so the worker doesn't
        // push more PCM into a half-torn-down recognizer (best-effort; safe to
        // fail because the worker checks `running` itself).
        try { stream?.inputFinished() } catch (_: Throwable) {}
        // Snapshot the audio accumulator synchronously. The worker only writes
        // to it while running=true; setting running=false above guarantees no
        // further concurrent writes once we enter this @Synchronized block.
        val out: FloatArray
        synchronized(audioSync) {
            out = audio.copyOf(audioSize)
            audio = FloatArray(sampleRate * 10)
            audioSize = 0
        }
        val workerToJoin = job
        // Schedule teardown AFTER the worker has actually exited its `while(running)`
        // loop, so the native OnlineStream handle can't be freed while the worker
        // is mid-`decode(st)`. The IO scope is fine tosuspend on.
        scope.launch {
            try { workerToJoin?.join() } catch (_: Throwable) {}
            teardown()
            // ONLY now — after teardown() has actually freed the native
            // recognizer/stream handles — clear the gate so a follow-up
            // [start] can proceed safely.
            isStopping = false
            _state.value = State.Idle
        }
        // Emit a synthetic endpoint so the partial collector commits the last
        // segment even though the worker hasn't finished draining yet.
        _partials.tryEmit(SherpaPartialEvent(
            partial = emptyList(),
            rawPartial = "",
            endpoint = true
        ))
        return out
    }

    private fun teardown() {
        try { stream?.release() } catch (_: Throwable) {}
        stream = null
        try { recognizer?.release() } catch (_: Throwable) {}
        recognizer = null
        matcher = null
        // Also clear the pause/resume latch so a follow-up [resume] on this
        // pipeline doesn't try to re-spawn the worker against freed handles.
        paused = false
    }

    /**
     * Releases native resources unconditionally (lifecycle teardown).
     *
     * Invoked by the Compose `LaunchedEffect(targetOrig)` when the ayah range
     * changes — the most recent crash site. CRITICAL: this is called from the
     * Compose main thread while the IO AudioRecord worker is potentially mid
     * native call (`st.acceptWaveform` / `rec.decode(st)`). Calling
     * `stream?.release()` / `recognizer?.release()` synchronously here would
     * free the native pointers under the worker's feet -> SIGSEGV (the exact
     * "second ayah crashes" bug the user reported).
     *
     * The new contract: we flip `running=false` (worker checks this on its
     * loop) and `cancelledDuringLoad=true` (start IO job checks this) AND
     * `isStopping=true` (any follow-up [start] refuses to proceed until
     * teardown is done — see [isStopping] doc for the stop/start race it
     * closes), and we SCHEDULE the teardown on the IO scope to run AFTER the
     * worker has actually exited (`job?.join()`). The new pipeline that
     * replaces THIS one runs on its own fresh `CoroutineScope` so it can't
     * collide with this scheduled teardown either — even on a hot range-change
     * burst (Compose cancelling the old LaunchedEffect mid-flight), the
     * `launch` here stays alive until the worker drains because it captures
     * its parent-job reference.
     *
     * No synchronous native call from this method, no join-from-main, no
     * ANR risk.
     */
    fun release() {
        running = false
        cancelledDuringLoad = true
        isStopping = true
        try { stream?.inputFinished() } catch (_: Throwable) {}
        val workerToJoin = job
        // Schedule teardown on the IO scope AFTER the worker has joined. We do
        // NOT cancel this scope from inside the launcher (that would cancel the
        // launcher itself mid-flight); the calling Composable discards its
        // holder reference and replaces it with a fresh pipeline whose own
        // CoroutineScope is independent, so the OLD scope becomes unreachable
        // for new work the moment the holder field is overwritten. The single
        // `launch` here completes its `join()+teardown()` and the scope goes
        // idle. We also clear `isStopping` AFTER teardown() returns — the
        // holder field is overwritten by then so a follow-up [start] on the
        // NEW pipeline sees its own (initially false) isStopping; this clear
        // only matters as a defensive no-op for any lingering reference.
        scope.launch {
            try { workerToJoin?.join() } catch (_: Throwable) {}
            teardown()
            isStopping = false
            _state.value = State.Idle
        }
    }

    /**
     * Launches a fresh AudioRecord worker on the IO scope immediately, capturing
     * it as [job]. Used by the fresh-start path in [start]. The pause/resume
     * path's [resume] does its OWN `scope.launch { oldWorkerToJoin.join();
     * runReadLoop(st) }` so it can ensure the OLD AudioRecord is released before
     * creating the new one on the single-mic A05s.
     */
    private fun startReadLoop(st: OnlineStream) {
        job = scope.launch { runReadLoop(st) }
    }

    /**
     * The AudioRecord + decode worker body. Runs inside whichever coroutine it
     * is invoked from (the fresh-start `startReadLoop` launcher, or the
     * pause/resume `[resume]` launcher that first joins the OLD worker so the
     * single-mic HAL is free before creating the new AudioRecord).
     *
     * Body sketch: getMinBufferSize → make AudioRecord → startRecording →
     * `while (running) { read shortBuf; convert; append to audioSync sink;
     * st.acceptWaveform(floats, sr); decodeAndEmit(st) }` → finally stop+release
     * the AudioRecord. Pure Kotlin once the AudioRecord is created — no native
     * calls outside `st.acceptWaveform` and `decodeAndEmit(st)`.
     *
     * State-writer on failure: if minBuf<=0 OR AudioRecord doesn't INIT,
     * flips `running=false` AND emits [State.Failed("AudioRecord ...")] so the
     * Compose UI surfaces a "mic unavailable" message instead of locking up.
     */
    private suspend fun runReadLoop(st: OnlineStream) {
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
        val bufFrames = maxOf(minBuf, sampleRate * 2) / 2
        val audioRecord = android.media.AudioRecord(
            android.media.MediaRecorder.AudioSource.MIC,
            sampleRate,
            android.media.AudioFormat.CHANNEL_IN_MONO,
            android.media.AudioFormat.ENCODING_PCM_16BIT,
            bufFrames * 2
        )
        if (audioRecord.state != android.media.AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord not initialised")
            audioRecord.release()
            running = false
            _state.value = State.Failed("AudioRecord not initialised")
            return
        }
        val shortBuf = ShortArray(bufFrames)
        try {
            audioRecord.startRecording()
            while (running) {
                val n = audioRecord.read(shortBuf, 0, shortBuf.size)
                if (n > 0) {
                    // Re-check `running` between a possibly-blocking
                    // `audioRecord.read(...)` and the native `acceptWaveform`
                    // — [stop]/[release] may have landed during the read,
                    // in which case a teardown-job is queued on this same
                    // IO scope waiting for the worker to exit; do NOT push
                    // PCM into `st` after running flips false because that
                    // job will free `st` the moment this loop returns.
                    if (!running) break
                    val floats = FloatArray(n)
                    for (i in 0 until n) floats[i] = shortBuf[i] / 32768f
                    // 1) TajweedNet sink: append to the full-audio accumulator.
                    synchronized(audioSync) {
                        ensureAudioCapacity(audioSize + n)
                        for (i in 0 until n) audio[audioSize + i] = floats[i]
                        audioSize += n
                    }
                    // 2) Sherpa OnlineRecognizer: push the same PCM.
                    try {
                        st.acceptWaveform(floats, sampleRate)
                        decodeAndEmit(st)
                    } catch (t: Throwable) {
                        Log.w(TAG, "decode error", t)
                    }
                } else if (n == android.media.AudioRecord.ERROR ||
                    n == android.media.AudioRecord.ERROR_INVALID_OPERATION) {
                    break
                }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "AudioRecord read loop crashed", t)
        } finally {
            try { audioRecord.stop() } catch (_: Throwable) {}
            try { audioRecord.release() } catch (_: Throwable) {}
        }
    }

    /** Doubles the audio buffer's capacity (encoded as FloatArray) to [needed]. */
    private fun ensureAudioCapacity(needed: Int) {
        if (needed <= audio.size) return
        var cap = audio.size
        while (cap < needed) cap *= 2
        val next = FloatArray(cap)
        audio.copyInto(next, 0, 0, audioSize)
        audio = next
    }

    private fun decodeAndEmit(st: OnlineStream) {
        // Defensive: if [stop]/[release] flipped running=false, do NOT touch
        // native pointers — a teardown is scheduled on the IO scope that will
        // free `st` once this worker has exited. Calling into `rec.isReady(st)`
        // here after teardown completed would dereference a freed handle.
        if (!running) return
        val rec = recognizer ?: return
        try {
            while (running && rec.isReady(st)) rec.decode(st)
            if (!running) return
            val result = rec.getResult(st)
            val rawText = result.text ?: ""
            val cleaned = matcher?.clean(rawText) ?: rawText.split(Regex("\\s+"))
                .filter { it.isNotEmpty() }
            val isEndpoint = try { rec.isEndpoint(st) } catch (_: Throwable) { false }
            _partials.tryEmit(SherpaPartialEvent(
                partial = cleaned,
                rawPartial = rawText,
                endpoint = isEndpoint
            ))
            if (isEndpoint) {
                // Commit + reset so the next utterance starts with an empty
                // decoder state. The stream object itself is reused.
                try { rec.reset(st) } catch (_: Throwable) {}
                matcher?.reset()
            }
        } catch (t: Throwable) {
            Log.w(TAG, "decode/emit error", t)
        }
    }
}

