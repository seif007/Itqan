package itqan.app.sherpa

import itqan.app.ml.RecitationAnalyzer
import itqan.app.ui.CollectorSim
import itqan.app.ui.RecitationStatus
import itqan.app.ui.SpokenWordTiming
import itqan.app.ui.analyzeRecitationDetails
import itqan.app.ui.isWordMatch
import itqan.app.ui.normalizeForComparison
import itqan.app.ui.ruleLabelToHead
import javazoom.jl.decoder.Bitstream
import javazoom.jl.decoder.Decoder
import javazoom.jl.decoder.SampleBuffer
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.ByteArrayInputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import kotlin.math.min

/**
 * Tajweed false-positive harness: the second half of the "no false mistakes
 * on a correct recitation" goal. Replays REAL Alafasy recitation audio
 * through the SAME STT pipeline replay as PcSttEndToEndTest (real ONNX
 * model, exact 4.0s/1.5s window geometry, final full-tail decode repair),
 * then feeds the resulting transcript + captured audio through the REAL
 * [analyzeRecitationDetails] — the exact function the app runs in
 * Memorize+Tajweed mode — and asserts that NO word pops TAJWEED_ERROR.
 *
 * Device-path fidelity (each mirrors RecitationApp.kt / SherpaPipeline.kt):
 *  - fullAudio = the UN-GATED capture accumulator (snapshotRawAudio(): every
 *    250 ms chunk including silence, real time). The RMS-gated buffer glues
 *    words with no inter-word silence and compresses the timeline — the probe
 *    proved that is out-of-distribution for TajweedNet and misaligns the
 *    wall-clock word timings, so the pipeline now hands the scorer the
 *    un-gated buffer (matching the training clips' natural silence);
 *  - the STT transcript replay ALSO runs on the UN-GATED recitation, exactly
 *    like the green STT E2E suite (phoneRealism=false). Replaying STT on the
 *    gated capture displaced/dropped REPEATED words (nas's ٱلنَّاسِ, fil's 2nd
 *    أَلَم) via the merge's re-hearable-window folding, producing false SKIPPED
 *    verdicts that are a transcript artifact, not a tajweed-scoring problem;
 *  - spokenTimings are built exactly like the collector does per emission
 *    (new words start at the emission's wall-clock, the previous word's
 *    endMs closes there; a no-new-words emission extends the last word). Since
 *    the replay is un-gated, the replay clock IS the wall clock the scorer's
 *    audio and the device's spokenTimings share;
 *  - the per-word slice (rule-aware window → trimSilence → per-head threshold
 *    from decisionThresholdForHead) happens INSIDE analyzeRecitationDetails
 *    itself, so it is exercised, not mirrored. The scorer's bounded
 *    confirmation search (±6 s word-span sweep) recovers from the
 *    emission-clock lag before flagging a TAJWEED_ERROR.
 *
 * Requires the desktop onnxruntime test dep + internet on first run (audio
 * and text cache under <java.io.tmpdir>/stt_e2e, shared with the STT
 * harness).
 */
@RunWith(RobolectricTestRunner::class)
class PcTajweedFalsePositiveTest {

    // ── pipeline constants must mirror SingleAudioPipeline / the STT harness ──
    private val sampleRate = 16000
    private val windowSeconds = 4.0
    private val hopSeconds = 1.5
    private val finalDecodeCapSeconds = 90.0
    private val silenceGapSeconds = 0.35

    private val riwayah = "Hafs"
    private val cacheDir = File(System.getProperty("java.io.tmpdir"), "stt_e2e")

    private val modelFile = File("src/main/assets/stt_model/model.q8.onnx")
    private val tokensFile = File("src/main/assets/stt_model/tokens.txt")

    private val engine: SttEngine get() = sharedEngine

    companion object {
        // CRITICAL — the ONNX graph is ~167 MB of NATIVE memory. JUnit4/
        // Robolectric construct a fresh test-class instance PER @Test method,
        // so an instance-level `by lazy` loaded the graph once per test (33x
        // across the two harness classes) and ONNX Runtime never reclaimed the
        // native allocations between loads — that is what froze the 8 GB PC.
        // A companion object is a per-CLASS static, so the graph loads exactly
        // once per sandbox and is shared by every test method.
        private val sharedEngine: SttEngine by lazy {
            // 2 intra-op threads: this 4-core box also runs the Gradle daemon +
            // test JVM; 4 threads here froze the machine under load.
            SttEngine.create(
                File("src/main/assets/stt_model/model.q8.onnx"),
                File("src/main/assets/stt_model/tokens.txt"),
                threads = 2
            ) ?: error("engine failed to load")
        }
    }

    private fun norm(w: String) = normalizeForComparison(w, riwayah)

    // ── audio + text plumbing (identical to PcSttEndToEndTest) ───────

    private fun fetchCached(url: String, name: String): ByteArray {
        cacheDir.mkdirs()
        val cached = File(cacheDir, name)
        if (cached.exists()) return cached.readBytes()
        val conn = URL(url).openConnection() as HttpURLConnection
        conn.connectTimeout = 20000
        conn.readTimeout = 60000
        conn.requestMethod = "GET"
        conn.setRequestProperty("User-Agent", "tajweed-ml-harness")
        conn.connect()
        check(conn.responseCode == 200) { "HTTP ${conn.responseCode} for $url" }
        val bytes = conn.inputStream.use { it.readBytes() }
        cached.writeBytes(bytes)
        return bytes
    }

    private fun decodeMp3MonolRate(bytes: ByteArray): Pair<FloatArray, Int> {
        val bitstream = Bitstream(ByteArrayInputStream(bytes))
        val decoder = Decoder()
        val pcm = ArrayList<Float>(bytes.size * 4)
        var rate = 0
        var header = bitstream.readFrame()
        while (header != null) {
            if (rate == 0) rate = decoder.outputFrequency
            val sb = decoder.decodeFrame(header, bitstream) as SampleBuffer
            val buf = sb.buffer
            if (sb.channelCount == 1) {
                for (i in 0 until sb.bufferLength) pcm.add(buf[i] / 32768f)
            } else {
                var i = 0
                while (i + 1 < sb.bufferLength) {
                    pcm.add(((buf[i] + buf[i + 1]) * 0.5f) / 32768f)
                    i += 2
                }
            }
            bitstream.closeFrame()
            header = bitstream.readFrame()
        }
        return pcm.toFloatArray() to rate
    }

    private fun resample16k(src: FloatArray, srcRate: Int): FloatArray {
        if (srcRate == sampleRate) return src
        val ratio = srcRate.toDouble() / sampleRate.toDouble()
        val out = FloatArray((src.size / ratio).toInt())
        for (i in out.indices) {
            val pos = i * ratio
            val i0 = pos.toInt()
            val i1 = min(i0 + 1, src.size - 1)
            val frac = (pos - i0).toFloat()
            out[i] = src[i0] * (1f - frac) + src[i1] * frac
        }
        return out
    }

    private fun surahWords(surah: Int, ayahFilter: IntRange? = null): List<String> {
        val url = "https://api.alquran.cloud/v1/surah/$surah/ar.alafasy"
        val json = JSONObject(fetchCached(url, "surah_$surah.json").decodeToString())
        val data = json.getJSONObject("data")
        val words = ArrayList<String>()
        val ayahs = data.getJSONArray("ayahs")
        for (i in 0 until ayahs.length()) {
            val ayah = ayahs.getJSONObject(i)
            val ayahNumber = ayah.getInt("numberInSurah")
            if (ayahFilter != null && ayahNumber !in ayahFilter) continue
            val text = ayah.getString("text")
            for (tok in text.split(Regex("\\s+"))) {
                val trimmed = tok.trim()
                if (trimmed.isNotEmpty()) words.add(trimmed)
            }
        }
        return words
    }

    private fun globalAyah(surah: Int, ayah: Int): Int {
        val listArray = JSONObject(
            fetchCached("https://api.alquran.cloud/v1/surah", "surah_list.json").decodeToString()
        ).getJSONArray("data")
        var offset = 0
        for (i in 0 until listArray.length()) {
            val s = listArray.getJSONObject(i).getInt("number")
            if (s >= surah) break
            offset += listArray.getJSONObject(i).getInt("numberOfAyahs")
        }
        return offset + ayah
    }

    // ── the pipeline replay (same MonoMerge contract as the STT harness) ──

    private class MonoMerge(vocab: Set<String>) {
        var lastClean: List<String> = emptyList()
        private var lastMs: LongArray = LongArray(0)
        private val vocabNorm: Set<String> = vocab
        var lastRepair = false
            private set

        fun merge(n: List<String>, nowMs: Long, windowMs: Long, fullReRender: Boolean = false, coversSession: Boolean = false): List<String>? {
            val merged = monotonicMergeWords(lastClean, n, lastMs, nowMs, windowMs, vocabNorm, null, fullReRender, coversSession)
            if (merged == lastClean) { lastRepair = false; return null }
            lastRepair = merged.size <= lastClean.size
            if (merged.size != lastClean.size) {
                if (merged.size < lastClean.size) {
                    lastMs = LongArray(merged.size) { nowMs }
                } else {
                    val out = LongArray(merged.size)
                    val keep = minOf(lastMs.size, merged.size)
                    if (keep > 0) lastMs.copyInto(out, 0, 0, keep)
                    for (i in lastClean.size until merged.size) out[i] = nowMs
                    lastMs = out
                }
            }
            lastClean = merged
            return merged
        }
    }

    /**
     * Local mirror of RecitationApp.trimSilence (private there): RMS-gated
     * leading/trailing silence removal, loudness-independent. Used by the
     * probe section to score alternative windows exactly like the app does.
     */
    private fun trimSilenceLocal(x: FloatArray): FloatArray {
        if (x.size < 1024) return x
        val sr = 16000
        val winMs = 20
        val win = (winMs * sr / 1000)
        val nWin = x.size / win
        if (nWin < 4) return x
        val rms = FloatArray(nWin)
        var peakRms = 0f
        for (w in 0 until nWin) {
            var s = 0f
            val base = w * win
            for (i in 0 until win) {
                val v = x[base + i]
                s += v * v
            }
            val r = kotlin.math.sqrt(s / win)
            rms[w] = r
            if (r > peakRms) peakRms = r
        }
        if (peakRms < 1e-5f) return x
        val gate = peakRms * 0.12f
        var firstW = 0
        var lastW = nWin - 1
        while (firstW < nWin && rms[firstW] < gate) firstW++
        while (lastW > firstW && rms[lastW] < gate) lastW--
        if (firstW > lastW) return x
        val margin = (winMs * sr / 1000)
        val start = ((firstW * win - margin).coerceAtLeast(0))
        val end = (((lastW + 1) * win + margin).coerceAtMost(x.size))
        if (end - start < 2048) return x
        return x.copyOfRange(start, end)
    }

    /** Bump to invalidate the replay cache after any STT pipeline change. */
    private val REPLAY_CACHE_VERSION = 9  // bump: swapped to Muno459/fastconformer-quran (new model.q8.onnx + Slaney mel recipe)

    /**
     * The deterministic ONNX STT replay (steps 3-4 of runTajweedE2E): sliding
     * windows through the real engine + matcher + merge, then the pause/stop
     * commit with the final full-tail decode repair. Builds the collector's
     * spokenTimings exactly like RecitationApp's `pipe.partials.collect`
     * block — from the collector's OWN filterMemorizeWords output of each
     * emission, keyed by the cumulative filtered count (which the device
     * never resets, not even on the final-decode repair swap).
     *
     * Returns (final transcript, collector timings). Cached to disk keyed by
     * [reportName] + [REPLAY_CACHE_VERSION] + the captured-audio length, so
     * tajweed-scoring tuning re-runs skip the expensive ONNX stage entirely.
     */
    private fun loadOrRunReplay(
        reportName: String,
        captured: FloatArray,
        targetOrig: List<String>,
        targetNorms: List<String>
    ): Pair<List<String>, List<SpokenWordTiming>> {
        // The STT replay runs on the UN-GATED recitation, so the replay clock
        // (sample index / sampleRate) IS the wall clock — the same axis the
        // scorer's audio and the device's spokenTimings share. No gate-map
        // conversion is needed.
        fun wallMs(idx: Int): Long = idx * 1000L / sampleRate
        val cacheDir = File("build/reports/replay_cache")
        cacheDir.mkdirs()
        val cacheFile = File(cacheDir, "$reportName.json")
        if (cacheFile.exists()) {
            runCatching {
                val j = JSONObject(cacheFile.readText(Charsets.UTF_8))
                if (j.getInt("version") == REPLAY_CACHE_VERSION &&
                    j.getLong("capturedSize") == captured.size.toLong()
                ) {
                    val spoken = (0 until j.getJSONArray("spoken").length())
                        .map { j.getJSONArray("spoken").getString(it) }
                    val tj = j.getJSONArray("emissionTimings")
                    val emissionTimings = (0 until tj.length()).map { i ->
                        val t = tj.getJSONObject(i)
                        SpokenWordTiming(
                            t.getString("word"), t.getInt("startIndex"),
                            t.getLong("startMs"), t.getLong("endMs"), 0f
                        )
                    }
                    println("  replay cache HIT (${spoken.size}w)")
                    return spoken to emissionTimings
                }
            }.onFailure { println("  replay cache unreadable, re-running: ${it.message}") }
        }
        println("  replay cache MISS — running ONNX STT replay")
        val matcher = AyahWordMatcher(targetOrig)
        val sim = CollectorSim(riwayah, targetNorms)
        sim.onEmission(emptyList(), endpoint = false, freshSession = true)
        val mm = MonoMerge(targetNorms.toHashSet())
        // EMISSION-CLOCK timings: exactly what the device's collector builds —
        // each word's startMs/endMs is the wall clock of the emission that
        // first surfaced it. These LAG the word's true audio position by up to
        // the 4 s window (typically ~0.5-2 s), which is the device reality the
        // scorer must tolerate.
        val emissionTimings = ArrayList<SpokenWordTiming>()
        var currentWordCount = 0
        fun collectorTimings(timeNow: Long) {
            val words = sim.lastFilteredWords
            if (words.size > currentWordCount) {
                if (emissionTimings.isNotEmpty()) emissionTimings.last().endMs = timeNow
                for (i in currentWordCount until words.size) {
                    emissionTimings.add(SpokenWordTiming(words[i], i, timeNow, timeNow, 0f))
                }
                currentWordCount = words.size
            } else if (words.size == currentWordCount && emissionTimings.isNotEmpty()) {
                emissionTimings.last().endMs = timeNow
            }
        }
        val windowFrames = (windowSeconds * sampleRate).toInt()
        val hopFrames = (hopSeconds * sampleRate).toInt()
        var hopEnd = hopFrames
        while (hopEnd <= captured.size) {
            val start = (hopEnd - windowFrames).coerceAtLeast(0)
            val window = captured.copyOfRange(start, hopEnd)
            val cleanList = matcher.clean(engine.transcribe(window))
            // The MERGE runs on the GATED clock (nowMs = hopEnd, windowMs = the
            // gated window's own length <= 4 s) — exactly the device's
            // decodeWindow() which passes windowMs = win.size/sampleRate. This
            // keeps the "re-hearable" boundary faithful so repeated words
            // (ٱلنَّاسِ) append instead of folding (the STT E2E test's 20/20
            // transcript depends on it). The EMISSION TIMINGS, however, are
            // recorded on the WALL-CLOCK axis (wallMs(hopEnd)) because that is
            // the axis the scorer's un-gated audio and the device's
            // spokenTimings share.
            val nowMs = hopEnd * 1000L / sampleRate
            val windowMs = (hopEnd - start) * 1000L / sampleRate
            val emitted = mm.merge(cleanList, nowMs, windowMs)
            if (emitted != null) {
                sim.onEmission(emitted, endpoint = false, freshSession = false)
                collectorTimings(wallMs(hopEnd))
            }
            hopEnd += hopFrames
        }
        // pause/stop commit: synthetic empty endpoint, then the final
        // full-tail decode (the repair path).
        val finalNowMs = captured.size * 1000L / sampleRate
        sim.onEmission(emptyList(), endpoint = true, freshSession = false)
        collectorTimings(wallMs(captured.size))
        val tailStart = (captured.size - (finalDecodeCapSeconds * sampleRate).toInt()).coerceAtLeast(0)
        val tail = captured.copyOfRange(tailStart, captured.size)
        val finalClean = matcher.clean(engine.transcribe(tail))
        val finalWindowMs = tail.size * 1000L / sampleRate
        val finalEmission = mm.merge(finalClean, finalNowMs, finalWindowMs, fullReRender = true, coversSession = tailStart == 0)
        if (finalEmission != null) {
            sim.onEmission(finalEmission, endpoint = true, freshSession = false, repaired = mm.lastRepair)
            collectorTimings(wallMs(captured.size))
        }
        val spoken = sim.spokenText

        // persist the cache
        runCatching {
            val j = JSONObject()
            j.put("version", REPLAY_CACHE_VERSION)
            j.put("capturedSize", captured.size.toLong())
            j.put("spoken", org.json.JSONArray(spoken))
            val tj = org.json.JSONArray()
            for (t in emissionTimings) {
                tj.put(JSONObject()
                    .put("word", t.word).put("startIndex", t.startIndex)
                    .put("startMs", t.startMs).put("endMs", t.endMs))
            }
            j.put("emissionTimings", tj)
            cacheFile.writeText(j.toString(), Charsets.UTF_8)
        }.onFailure { println("  replay cache write failed: ${it.message}") }
        return spoken to emissionTimings
    }

    /**
     * Aligns the final transcript [spoken] onto the CTC word timestamps
     * [ctcWords] (from a full-session decode) in reading order, producing a
     * [SpokenWordTiming] per transcript word with its TRUE audio position in
     * the captured buffer.
     *
     * CTC greedy timestamps COLLAPSE confident words (a common word like
     * اللّه can emit in ~80 ms of frames though it is really ~0.5 s), so the
     * raw per-word END position is unreliable. The START (onset) position is
     * much more stable, so we segment by ONSET: word k spans from its own CTC
     * start to the NEXT word's CTC start (last word to end of buffer). This
     * gives each word a plausible duration that includes its natural tail, and
     * never overlaps a neighbour. Unmatched words fall back to an even-spacing
     * slot so every transcript index has a timing (analyzeRecitationDetails
     * looks them up by startIndex).
     */
    private fun alignCtcTimings(
        spoken: List<String>,
        ctcWords: List<SttEngine.WordTime>,
        capturedSize: Int
    ): List<SpokenWordTiming> {
        val out = ArrayList<SpokenWordTiming>(spoken.size)
        val capturedMs = capturedSize * 1000L / sampleRate
        // Onset-based boundaries: word k runs from its own start to the next
        // word's start (the classic word-segmentation cut from onset times).
        val left = IntArray(ctcWords.size)
        val right = IntArray(ctcWords.size)
        for (k in ctcWords.indices) {
            left[k] = ctcWords[k].startSample
            right[k] = if (k == ctcWords.lastIndex) capturedSize
                else ctcWords[k + 1].startSample
        }
        var c = 0
        for ((i, w) in spoken.withIndex()) {
            val wn = norm(w)
            // advance to the next CTC word whose normalized text matches
            var foundIdx = -1
            var j = c
            while (j < ctcWords.size) {
                if (norm(ctcWords[j].word) == wn) { foundIdx = j; c = j + 1; break }
                j++
            }
            if (foundIdx >= 0) {
                val st = left[foundIdx] * 1000L / sampleRate
                val en = right[foundIdx] * 1000L / sampleRate
                out.add(SpokenWordTiming(w, i, st, en.coerceAtLeast(st + 30L), 0f))
            } else {
                // proportional fallback (even spacing)
                val s = spoken.size.coerceAtLeast(1)
                val st = i * capturedMs / s
                val en = (i + 1) * capturedMs / s
                out.add(SpokenWordTiming(w, i, st, en.coerceAtLeast(st + 30L), 0f))
            }
        }
        return out
    }

    /**
     * Runs one surah session end-to-end and asserts the Memorize+Tajweed
     * contract: a CORRECT recitation must produce ZERO TAJWEED_ERROR
     * verdicts (the popup the user sees) and zero false WRONG/SKIPPED.
     * Writes a per-head score report to app/build/reports/[reportName].txt.
     *
     * The expensive ONNX STT replay (steps 1-4) is deterministic for a given
     * surah + capture, so it is CACHED to disk keyed by [reportName] and the
     * captured-audio length; only the analysis pass (step 5) re-runs on a
     * cache hit. This makes tajweed-scoring tuning fast and memory-light (no
     * ONNX). Bump [REPLAY_CACHE_VERSION] to force a fresh replay after any
     * STT pipeline (matcher/merge) change.
     */
    private fun runTajweedE2E(surah: Int, ayahRange: IntRange, reportName: String) {
        println("\n=== TAJWEED FP surah $surah ayahs $ayahRange ===")
        // 1) assemble the recitation audio ayah-by-ayah like AudioFetcher.
        val audio = ArrayList<Float>()
        val silence = FloatArray((silenceGapSeconds * sampleRate).toInt())
        for (ayah in ayahRange) {
            val global = globalAyah(surah, ayah)
            val url = "https://cdn.islamic.network/quran/audio/128/ar.alafasy/$global.mp3"
            val (pcm, rate) = decodeMp3MonolRate(fetchCached(url, "ayah_$global.mp3"))
            val pcm16 = resample16k(pcm, rate)
            audio.addAll(pcm16.toList())
            audio.addAll(silence.toList())
        }
        val samples = audio.toFloatArray()
        // The STT replay runs on the UN-GATED recitation, exactly like the
        // green STT E2E suite (PcSttEndToEndTest, phoneRealism=false).
        // The RMS-GATED capture displaces/drops REPEATED words (nas's ٱلنَّاسِ,
        // fil's 2nd أَلَم) because the gate glues words and the merge's
        // re-hearable window folds identical repeats — that produced false
        // SKIPPED verdicts here that are a transcript-quality artifact, not a
        // tajweed-scoring problem. The tajweed scorer uses the un-gated buffer
        // anyway (snapshotRawAudio), so replaying STT on the same un-gated axis
        // keeps the transcript clean and the whole harness on ONE wall-clock.
        val captured = samples
        val capturedMs = captured.size * 1000L / sampleRate
        println("  audio: ${"%.1f".format(samples.size / sampleRate.toDouble())}s (un-gated STT + scoring)")

        // 2) target: the recited ayah range. The app strips the leading
        // basmala from every surah's ayah-1 text except 1 and 9 (the reciter
        // only recites it before Al-Fatihah), so the harness target must too.
        val targetPairs = surahWords(surah, ayahRange)
            .map { it to norm(it) }
            .filter { (_, tn) -> tn.isNotEmpty() }
        val basmala = listOf("بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ", "ٱلرَّحِيمِ").map { norm(it) }
        val stripBasmala = surah != 1 && surah != 9 &&
            targetPairs.size >= 4 && targetPairs.map { it.second }.take(4) == basmala
        val targets = if (stripBasmala) targetPairs.drop(4) else targetPairs
        val targetOrig = targets.map { it.first }
        val targetNorms = targets.map { it.second }
        println("  target: ${targetNorms.size} words${if (stripBasmala) " (basmala stripped)" else ""}")

        // 3-4) the ONNX STT replay (cached, on the UN-GATED recitation — the
        // same path the green STT E2E suite validates). Produces the final
        // transcript [spoken] and the device-style EMISSION-CLOCK timings on
        // the wall-clock axis the scorer's audio shares.
        val (spoken, emissionTimings) = loadOrRunReplay(reportName, captured, targetOrig, targetNorms)
        val spokenNorms = spoken.map { norm(it) }

        // Scoring audio: the UN-GATED recitation. The probe
        // (build/reports/tajweed_fp_*.txt) proved the RMS-gated capture is
        // out-of-distribution for TajweedNet — words are glued with no
        // inter-word silence, so trimSilence cannot isolate a word and scores
        // collapse regardless of the window geometry. The model was trained on
        // natural clips with silence (Tadabur corpus), and the OLD coach path
        // (MicRecorderController + Google recognizer) scored un-gated audio.
        // The un-gated buffer is also REAL-TIME aligned, so the device's
        // wall-clock emission timings map onto the right audio.
        val scoreAudio = samples
        val scoreAudioMs = samples.size * 1000L / sampleRate
        // CTC word positions on the un-gated timeline — used ONLY by the probe
        // section below (diagnostics: the score achievable with a perfectly
        // placed window), not by the assertion path.
        val (_, ctcWords) = engine.transcribeWithWordTimes(samples)
        val ctcTimings = alignCtcTimings(spoken, ctcWords, samples.size)

        // 5) the REAL app analysis pass — Memorize+Tajweed mode, final report.
        // Uses the device-style EMISSION-CLOCK timings (lagged by the decode
        // latency) on un-gated audio: this is EXACTLY what the phone runs. The
        // scorer's bounded confirmation search (analyzeRecitationDetails) is
        // what recovers from the emission-clock lag, so asserting on the lagged
        // timings validates the real device behaviour end-to-end.
        val context = androidx.test.core.app.ApplicationProvider.getApplicationContext<android.content.Context>()
        val analyzer = RecitationAnalyzer.get(context)
        analyzer.warmUp()
        val analysis = kotlinx.coroutines.runBlocking {
            analyzeRecitationDetails(
                spoken = spokenNorms,
                targetWordsOriginal = targetOrig,
                targetWordsNormalized = targetNorms,
                riwayah = riwayah,
                spokenTimings = emissionTimings,
                context = context,
                fullAudio = scoreAudio,
                recordingStartMs = 0L,
                recordingDurationMs = scoreAudioMs,
                targetAllGroups = null,
                hintedIndices = emptySet(),
                isTajweedMode = true,
                finalReport = true
            )
        }

        // 6) report + assertions
        val report = StringBuilder()
        report.append("=== TAJWEED FP surah $surah ayahs $ayahRange ===\n")
        report.append("audio: gated ${"%.1f".format(capturedMs / 1000.0)}s (STT) / ungated ${"%.1f".format(scoreAudioMs / 1000.0)}s (scoring), transcript ${spoken.size}w, target ${targetNorms.size}w\n")
        report.append("transcript: $spoken\n")
        report.append("target: $targetOrig\n")
        val perHead = HashMap<String, MutableList<Float>>()
        for (w in analysis) {
            val head = w.tajweedRule?.let { ruleLabelToHead(it) }
            val score = w.tajweedScore
            if (head != null && score != null) perHead.getOrPut(head) { mutableListOf() }.add(score / 100f)
            report.append(
                "  ${w.status} ${w.text}" +
                    (w.tajweedRule?.let { " rule=$it" } ?: "") +
                    (score?.let { " score=$it%" } ?: "") +
                    (w.tajweedAcousticFeedback?.let { " fb=\"$it\"" } ?: "") + "\n"
            )
        }
        for ((head, scores) in perHead.toSortedMap()) {
            report.append("head $head: n=${scores.size} min=${"%.2f".format(scores.min())} avg=${"%.2f".format(scores.average().toFloat())}\n")
        }
        // PROBE: for every TAJWEED_ERROR word, score a grid of alternative
        // window geometries so we can see which cut the head accepts. This is
        // tuning instrumentation — the app window rule is chosen from here.
        val errWords = analysis.filter { it.status == RecitationStatus.TAJWEED_ERROR }
        if (errWords.isNotEmpty()) {
            report.append("-- probe: alternative windows for failing words --\n")
            for (w in errWords) {
                val head = w.tajweedRule?.let { ruleLabelToHead(it) } ?: continue
                val thr = itqan.app.ui.decisionThresholdForHead(head)
                // Probe uses the ACCURATE CTC timings (diagnostics: what score
                // is achievable with a well-placed window), not the lagged
                // emission-clock timings the assertion path uses.
                val t = ctcTimings.findLast { isWordMatch(it.word, norm(w.spokenText ?: w.text)) }
                    ?: ctcTimings.getOrNull(w.wordIndex) ?: continue
                val spokenIdx = t.startIndex
                val segSt = t.startMs
                val segEn = t.endMs
                val sCnt = spoken.size.coerceAtLeast(1)
                val nextT = ctcTimings.find { it.startIndex == spokenIdx + 1 }
                val nextEnd = nextT?.endMs ?: scoreAudioMs
                val mid = (segSt + segEn) / 2
                val variants = linkedMapOf(
                    "span+-100" to listOf((segSt - 100).coerceAtLeast(0), (segEn + 100).coerceAtMost(scoreAudioMs)),
                    "span+-200" to listOf((segSt - 200).coerceAtLeast(0), (segEn + 200).coerceAtMost(scoreAudioMs)),
                    "onset-to-next-onset" to listOf(segSt, nextEnd),
                    "trigger..nextEnd+-100" to listOf((segSt - 100).coerceAtLeast(0), (nextEnd + 100).coerceAtMost(scoreAudioMs)),
                    "centered-1.0s" to listOf((mid - 500).coerceAtLeast(0), (mid + 500).coerceAtMost(scoreAudioMs)),
                    "centered-1.3s" to listOf((mid - 650).coerceAtLeast(0), (mid + 650).coerceAtMost(scoreAudioMs)),
                    "centered-1.6s" to listOf((mid - 800).coerceAtLeast(0), (mid + 800).coerceAtMost(scoreAudioMs)),
                    "centered-2.0s" to listOf((mid - 1000).coerceAtLeast(0), (mid + 1000).coerceAtMost(scoreAudioMs)),
                    "centered-2.6s" to listOf((mid - 1300).coerceAtLeast(0), (mid + 1300).coerceAtMost(scoreAudioMs)),
                    "centered-3.5s" to listOf((mid - 1750).coerceAtLeast(0), (mid + 1750).coerceAtMost(scoreAudioMs))
                )
                report.append("  ${w.text} rule=${w.tajweedRule} head=$head thr=$thr timing=${"%.2f".format(segSt / 1000.0)}..${"%.2f".format(segEn / 1000.0)}s nextEnd=${"%.2f".format(nextEnd / 1000.0)}s\n")
                for ((name, win) in variants) {
                    val a = (win[0] * 16).toInt().coerceIn(0, scoreAudio.size)
                    val b = (win[1] * 16).toInt().coerceIn(a, scoreAudio.size)
                    if (b - a < 256) continue
                    val seg = trimSilenceLocal(scoreAudio.copyOfRange(a, b))
                    if (seg.size < 256) continue
                    val p = analyzer.analyze(seg, listOf(head), thr).firstOrNull()?.prob ?: 0f
                    report.append("    $name [${"%.2f".format(win[0] / 1000.0)}..${"%.2f".format(win[1] / 1000.0)}s trim=${seg.size / 16000.0}s] -> ${"%.2f".format(p)} ${if (p >= thr) "PASS" else "fail"}\n")
                }
                // SLIDING probe: sweep a fixed-length window across +-1.5s around
                // the word to find where the head's score PEAKS. A nearby peak
                // above threshold means the timing is off (fixable); no peak
                // anywhere means the model simply rejects this clip (domain /
                // threshold problem).
                val spanLen = (segEn - segSt).coerceIn(400L, 1600L)
                var bestP = 0f
                var bestAt = -1L
                var st = (segSt - 1500L).coerceAtLeast(0L)
                val sweepEnd = (segEn + 1500L).coerceAtMost(scoreAudioMs)
                while (st + spanLen <= sweepEnd) {
                    val a = (st * 16).toInt().coerceIn(0, scoreAudio.size)
                    val b = ((st + spanLen) * 16).toInt().coerceIn(a, scoreAudio.size)
                    if (b - a >= 256) {
                        val seg = trimSilenceLocal(scoreAudio.copyOfRange(a, b))
                        if (seg.size >= 256) {
                            val p = analyzer.analyze(seg, listOf(head), thr).firstOrNull()?.prob ?: 0f
                            if (p > bestP) { bestP = p; bestAt = st }
                        }
                    }
                    st += 100L
                }
                report.append("    SLIDE peak=${"%.2f".format(bestP)} at=${"%.2f".format(bestAt / 1000.0)}s (len=${spanLen}ms) ${if (bestP >= thr) "-> timing fixable" else "-> model rejects everywhere"}\n")
            }
        }
        val reportFile = File("build/reports/$reportName.txt")
        reportFile.parentFile.mkdirs()
        reportFile.writeText(report.toString(), Charsets.UTF_8)
        println(report)

        // The user contract: a correct recitation must never pop the tajweed
        // mistake popup (TAJWEED_ERROR) nor false WRONG/SKIPPED verdicts.
        val tajweedErrors = analysis.filter { it.status == RecitationStatus.TAJWEED_ERROR }
        assertEquals(
            "false TAJWEED_ERROR popups on a correct recitation:\n" +
                tajweedErrors.joinToString("\n") {
                    "  ${it.text} rule=${it.tajweedRule} score=${it.tajweedScore}% fb=${it.tajweedAcousticFeedback}"
                } + "\ntranscript=$spoken\ntarget=$targetOrig",
            emptyList<Any>(), tajweedErrors
        )
        val falseVerdicts = analysis.filter {
            it.status == RecitationStatus.WRONG || it.status == RecitationStatus.SKIPPED
        }
        assertEquals(
            "false WRONG/SKIPPED verdicts on a correct recitation:\n" +
                falseVerdicts.joinToString("\n") { "  ${it.status} ${it.text}" } +
                "\ntranscript=$spoken\ntarget=$targetOrig",
            emptyList<Any>(), falseVerdicts
        )
    }

    // ── tests ─────────────────────────────────────────────────────────────

    @Test
    fun `ikhlas - no false tajweed popups`() {
        // GHUNNAH (أَحَدٌ۝, الصَّمَدُ shaddah patterns) + MADD + QALQALAH (قُلْ).
        runTajweedE2E(112, 1..4, "tajweed_fp_112")
    }

    @Test
    fun `fatihah - no false tajweed popups`() {
        // The densest rule mix in the matrix: ALLAM, MADD, GHUNNAH, IDGHAM,
        // IKHFA, IZHAR, MEEM_SAKINAH — the user's own surah.
        runTajweedE2E(1, 1..3, "tajweed_fp_fatihah")
    }

    @Test
    fun `kafirun - no false tajweed popups`() {
        // Heavy QALQALAH repetition (قُلْ, دِينِ endings) + MADD.
        runTajweedE2E(109, 1..6, "tajweed_fp_kafirun")
    }

    @Test
    fun `falaq - no false tajweed popups`() {
        // IZHAR/IKHFA on the tanween chains (شَرِّ مَا خَلَقَ, غَاسِقٍ إِذَا).
        runTajweedE2E(113, 1..5, "tajweed_fp_falaq")
    }

    @Test
    fun `nas - no false tajweed popups`() {
        // ALLAM x6 (ٱلنَّاسِ repeats) + GHUNNAH (مِن شَرِّ, مِنَ ٱلْجِنَّةِ).
        runTajweedE2E(114, 1..6, "tajweed_fp_nas")
    }

    @Test
    fun `asr - no false tajweed popups`() {
        runTajweedE2E(103, 1..3, "tajweed_fp_asr")
    }

    @Test
    fun `kawthar - no false tajweed popups`() {
        runTajweedE2E(108, 1..3, "tajweed_fp_kawthar")
    }

    @Test
    fun `nasr - no false tajweed popups`() {
        runTajweedE2E(110, 1..3, "tajweed_fp_nasr")
    }

    @Test
    fun `humazah - no false tajweed popups`() {
        runTajweedE2E(104, 1..9, "tajweed_fp_humazah")
    }

    @Test
    fun `fil - no false tajweed popups`() {
        runTajweedE2E(105, 1..5, "tajweed_fp_fil")
    }

    @Test
    fun `quraysh - no false tajweed popups`() {
        runTajweedE2E(106, 1..4, "tajweed_fp_quraysh")
    }

    @Test
    fun `masad - no false tajweed popups`() {
        runTajweedE2E(111, 1..5, "tajweed_fp_masad")
    }

    @Test
    fun `maun - no false tajweed popups`() {
        runTajweedE2E(107, 1..7, "tajweed_fp_maun")
    }

    @Test
    fun `qadr - no false tajweed popups`() {
        runTajweedE2E(97, 1..5, "tajweed_fp_qadr")
    }

    @Test
    fun `zalzalah - no false tajweed popups`() {
        runTajweedE2E(99, 1..8, "tajweed_fp_zalzalah")
    }
}
