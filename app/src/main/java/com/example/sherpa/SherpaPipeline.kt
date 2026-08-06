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

    /** True iff download is already complete (no work to do). */
    fun isComplete(): Boolean = isSherpaModelReady(context)

    suspend fun downloadIfNeeded() = withContext(Dispatchers.IO) {
        if (isComplete()) {
            _progress.value = Progress.Done
            return@withContext
        }
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
    @Volatile var running: Boolean = false
        private set

    private val sampleRate = 16_000

    // Native recognizer lifecycle.
    private var recognizer: OnlineRecognizer? = null
    private var stream: OnlineStream? = null

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
    private var matcher: AyahWordMatcher? = null

    /**
     * Resolves + opens the recognizer and starts the worker thread.
     *
     * Returns false if either (a) the model files aren't on disk yet (caller
     * must launch the sherpa downloader UI first) or (b) AudioRecord couldn't
     * be initialised (microphone busy / denied). On false, `running` is left
     * false and the caller surfaces an error in Compose state the same way
     * `MicRecorderController.start()` does today.
     */
    @Synchronized
    fun start(): Boolean {
        if (running) return true
        if (!isSherpaModelReady(context)) {
            Log.w(TAG, "start(): model not downloaded yet")
            return false
        }
        // (Re)build the hotwords file from the active ayah range. The hotwords
        // bias (layer i) is applied at recognizer-construction-time per the
        // Sherpa README; for per-stream hotword overrides there is the
        // `OnlineRecognizer#createStream(hotwords: String)` API, which we use
        // below when starting the stream so changing the ayah range mid-recital
        // can be hot-reloaded via [updateHotwords] without rebuilding the
        // recognizer.
        val ayahWords = ayahWordsProvider()
        matcher = AyahWordMatcher(ayahWords)
        val hotwordsPath = writeHotwordsFile(context, ayahWords, extraPhrasesProvider())
        return try {
            recognizer = createOnlineRecognizer(context, hotwordsPath)
            stream = recognizer!!.createStream("")
            startReadLoop()
            running = true
            true
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to start Sherpa pipeline", t)
            teardown()
            false
        }
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
     * Stops the worker, releases the recognizer & stream, and returns the
     * accumulated full recitation PCM (16 kHz mono, [-1,1]). The returned
     * FloatArray is the same shape [MicRecorderController.stop] produced, so
     * the existing TajweedNet feeding path consumes it unchanged.
     */
    @Synchronized
    fun stop(): FloatArray {
        if (!running) return FloatArray(0)
        running = false
        // Signal end-of-stream to the recognizer (lets the decoder flush a
        // final endpoint) before teardown.
        try { stream?.inputFinished() } catch (_: Throwable) {}
        // `Job.join()` and `flushFinal()` are suspend functions; we drive them
        // through runBlocking because this method is non-suspend and is called
        // from the Compose UI thread directly (matching the contract of
        // MicRecorderController.stop()). Both are bounded by the worker's own
        // running=false guard, so blocking here never deadlocks with the worker.
        try {
            kotlinx.coroutines.runBlocking { job?.join(); flushFinal() }
        } catch (_: Throwable) {}
        teardown()
        // The audio accumulator is only touched by the worker thread (and us,
        // here, after the worker has joined). Plain synchronized avoids the
        // suspend-only Mutex.withLock.
        val out: FloatArray
        synchronized(audioSync) {
            out = audio.copyOf(audioSize)
            audio = FloatArray(sampleRate * 10)
            audioSize = 0
        }
        return out
    }

    private fun teardown() {
        try { stream?.release() } catch (_: Throwable) {}
        stream = null
        try { recognizer?.release() } catch (_: Throwable) {}
        recognizer = null
        matcher = null
    }

    /** Releases native resources unconditionally (lifecycle teardown). */
    fun release() {
        if (running) stop()
        scope.cancel()
    }

    private fun startReadLoop() {
        job = scope.launch {
            val minBuf = android.media.AudioRecord.getMinBufferSize(
                sampleRate,
                android.media.AudioFormat.CHANNEL_IN_MONO,
                android.media.AudioFormat.ENCODING_PCM_16BIT
            )
            if (minBuf <= 0) {
                Log.e(TAG, "getMinBufferSize failed: $minBuf")
                running = false
                return@launch
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
                return@launch
            }
            val shortBuf = ShortArray(bufFrames)
            try {
                audioRecord.startRecording()
                while (running) {
                    val n = audioRecord.read(shortBuf, 0, shortBuf.size)
                    if (n > 0) {
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
                            stream?.acceptWaveform(floats, sampleRate)
                            decodeAndEmit()
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

    private fun decodeAndEmit() {
        val rec = recognizer ?: return
        val st = stream ?: return
        try {
            while (rec.isReady(st)) rec.decode(st)
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

    private suspend fun flushFinal() {
        // Nothing further to drain at the audio level — `decodeAndEmit` already
        // drains the decoder after every push. Just emit a synthetic endpoint
        // so the consumer commits the last partial as final.
        _partials.tryEmit(SherpaPartialEvent(
            partial = emptyList(),
            rawPartial = "",
            endpoint = true
        ))
    }
}

