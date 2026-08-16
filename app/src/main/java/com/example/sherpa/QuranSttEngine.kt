package com.example.sherpa

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sin

// ─────────────────────────────────────────────────────────────────────────────
// Quran STT engine — Saboorhsn/quran-stt-onnx (Apache-2.0).
//
// A NeMo FastConformer large CTC model fine-tuned on EveryAyah + Tlog for
// Quranic recitation with Tajweed diacritics, exported to a single unified
// ONNX graph:
//
//   Audio (16 kHz mono)
//     → log-mel [80 bands, n_fft=512, win 400, hop 160, Hanning]
//     → per-band mean/std normalization
//     → FastConformer encoder (8× temporal subsampling) + linear head
//     → CTC log-softmax over 1025 classes (1024 SentencePiece BPE + blank)
//     → greedy argmax + blank/repeat collapse → piece string → text
//
// The feature recipe below mirrors the exporter's reference implementation
// (tajweed/aligner.py + the README's log_mel_extract): mel filterbank as
// linear-triangular bins over the [0, 8 kHz] mel scale, log with a 2^-24
// floor, then per-row z-scoring. The SentencePiece surface pieces are read
// from the bundled `tokens.txt` ("<piece> <id>" lines) so no SentencePiece
// dependency is needed at runtime; "▁" marks a word boundary.
//
// This class REPLACES the previous sherpa-onnx Whisper recognizer at the
// decode seams of SingleAudioPipeline (window decode + final commit decode).
// The ONNX graph is fully dynamic on the time axis, so a 4 s window costs
// roughly 4 s worth of compute (RTF ~0.04 on CPU).
// ─────────────────────────────────────────────────────────────────────────────

private const val STT_TAG = "ItqanStt"

class QuranSttEngine private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val pieces: Array<String>,
) : AutoCloseable {

    companion object {
        const val N_MELS = 80
        const val N_FFT = 512
        const val WIN_LEN = 400
        const val HOP = 160
        const val N_BINS = N_FFT / 2 + 1 // 257
        const val BLANK_ID = 1024
        const val N_VOCAB = 1025
        /** FastConformer temporal subsampling (3× stride-2 convs): each CTC
         *  output frame spans [SUBSAMPLE] log-mel input frames. */
        const val SUBSAMPLE = 8
        private const val MEL_FLOOR = 1.0 / (1L shl 24) // 2^-24 (float32 eps-ish floor)
        private const val NORM_EPS = 1e-5f

        /**
         * Loads the ONNX model + token map. Call on a background thread: the
         * 167 MB int8 graph takes ~1-3 s to initialize on a phone CPU.
         */
        fun create(modelFile: File, tokensFile: File, threads: Int = 4): QuranSttEngine {
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(threads) }
            val session = env.createSession(modelFile.absolutePath, opts)
            val pieces = readPieces(tokensFile)
            Log.i(STT_TAG, "QuranSttEngine loaded: ${modelFile.name} (${modelFile.length()} bytes)")
            return QuranSttEngine(env, session, pieces)
        }

        private fun readPieces(file: File): Array<String> {
            val out = arrayOfNulls<String>(N_VOCAB)
            file.forEachLine { line ->
                val s = line.trim()
                if (s.isEmpty()) return@forEachLine
                val sp = s.lastIndexOf(' ')
                if (sp <= 0) return@forEachLine
                val piece = s.substring(0, sp)
                val id = s.substring(sp + 1).trim().toIntOrNull() ?: return@forEachLine
                if (id >= 0 && id < N_VOCAB) out[id] = piece
            }
            return Array(N_VOCAB) { out[it] ?: "" }
        }
    }

    // Precomputed per-frame Hanning window and the mel filterbank matrix.
    private val hanning = FloatArray(WIN_LEN) { i ->
        (0.5f - 0.5f * cos(2.0 * PI * i / (WIN_LEN - 1))).toFloat()
    }
    private val melFilterbank: FloatArray by lazy { buildMelFilterbank() }

    private fun buildMelFilterbank(): FloatArray {
        // Exactly the exporter's recipe: bins = floor((n_fft+1) * hz_pts / sr)
        // with linear triangular slopes on the mel scale (0..8 kHz).
        val fb = FloatArray(N_MELS * N_BINS)
        val hzPts = DoubleArray(N_MELS + 2)
        for (m in 0..N_MELS + 1) {
            val mel = m.toDouble() * (2595.0 * log10(1.0 + 8000.0 / 700.0)) / (N_MELS + 1)
            hzPts[m] = 700.0 * (10.0.pow(mel / 2595.0) - 1.0)
        }
        val bins = IntArray(N_MELS + 2) { m ->
            ((N_FFT + 1) * hzPts[m] / 16000.0).toInt()
        }
        for (m in 1..N_MELS) {
            for (k in bins[m - 1] until bins[m]) {
                fb[(m - 1) * N_BINS + k] =
                    ((k - bins[m - 1]).toFloat() / (bins[m] - bins[m - 1]).toFloat())
            }
            for (k in bins[m] until bins[m + 1]) {
                fb[(m - 1) * N_BINS + k] =
                    ((bins[m + 1] - k).toFloat() / (bins[m + 1] - bins[m]).toFloat())
            }
        }
        return fb
    }

    /**
     * Transcribes a 16 kHz float PCM buffer ([-1,1]) to text. Blocking; call
     * on the pipeline's IO dispatcher.
     */
    fun transcribe(samples: FloatArray): String {
        if (samples.isEmpty()) return ""
        val nFrames = 1 + (samples.size - 1) / HOP
        if (nFrames < 2) return ""
        val t0 = System.currentTimeMillis()
        val mel = logMel(samples, nFrames)
        val tMel = System.currentTimeMillis()
        val lengthTensor = OnnxTensor.createTensor(
            env, LongBuffer.wrap(longArrayOf(nFrames.toLong())), longArrayOf(1)
        )
        val audioTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(mel), longArrayOf(1L, N_MELS.toLong(), nFrames.toLong())
        )
        val result = try {
            session.run(mapOf("audio_signal" to audioTensor, "length" to lengthTensor))
        } finally {
            lengthTensor.close()
            audioTensor.close()
        }
        val tRun = System.currentTimeMillis()
        return try {
            val out = result.get(0) as OnnxTensor
            val buf = out.floatBuffer
            buf.position(0)
            val lp = FloatArray(buf.remaining())
            buf.get(lp)
            val tOut = lp.size / N_VOCAB
            val text = ctcDecode(lp)
            Log.i(
                STT_TAG,
                "transcribe(): %.2fs in, ${nFrames}f, mel=${tMel - t0}ms run=${tRun - tMel}ms tOut=$tOut raw='${text.take(60)}'"
            )
            text
        } finally {
            result.close()
        }
    }

    /**
     * A decoded word with its acoustic span in INPUT SAMPLES (relative to the
     * buffer passed to [transcribeWithWordTimes]). Derived from the CTC output
     * frames: the FastConformer subsamples time 8x and the log-mel hop is
     * [HOP] samples, so output frame t ≈ sample t * 8 * HOP.
     */
    data class WordTime(val word: String, val startSample: Int, val endSample: Int)

    /**
     * Transcribes [samples] AND returns per-word sample ranges. The text equals
     * [transcribe]; the timestamps let callers slice the audio back out per word
     * (tajweed scoring needs the word's real position, not an even-spacing guess).
     */
    fun transcribeWithWordTimes(samples: FloatArray): Pair<String, List<WordTime>> {
        if (samples.isEmpty()) return "" to emptyList()
        val nFrames = 1 + (samples.size - 1) / HOP
        if (nFrames < 2) return "" to emptyList()
        val mel = logMel(samples, nFrames)
        val lengthTensor = OnnxTensor.createTensor(
            env, LongBuffer.wrap(longArrayOf(nFrames.toLong())), longArrayOf(1)
        )
        val audioTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(mel), longArrayOf(1L, N_MELS.toLong(), nFrames.toLong())
        )
        val result = try {
            session.run(mapOf("audio_signal" to audioTensor, "length" to lengthTensor))
        } finally {
            lengthTensor.close()
            audioTensor.close()
        }
        return try {
            val out = result.get(0) as OnnxTensor
            val buf = out.floatBuffer
            buf.position(0)
            val lp = FloatArray(buf.remaining())
            buf.get(lp)
            ctcDecodeWithTimes(lp, samples.size)
        } finally {
            result.close()
        }
    }

    /** Greedy CTC decode that also records each word's first/last output frame. */
    private fun ctcDecodeWithTimes(lp: FloatArray, totalSamples: Int): Pair<String, List<WordTime>> {
        val tOut = lp.size / N_VOCAB
        val sb = StringBuilder()
        val words = ArrayList<WordTime>()
        // current word bookkeeping (in output frames)
        var curStart = -1
        var curLast = -1
        var curHasText = false
        // Flush the in-progress word using its LAST non-blank frame (curLast),
        // not the current frame — blank frames may separate words.
        fun flush() {
            if (curHasText && curStart >= 0 && curLast >= curStart) {
                val text = sb.toString().trim()
                val w = text.substringAfterLast(' ')
                if (w.isNotEmpty()) {
                    val s = (curStart * SUBSAMPLE * HOP).coerceIn(0, totalSamples)
                    val e = ((curLast + 1) * SUBSAMPLE * HOP).coerceIn(0, totalSamples)
                    words.add(WordTime(w, s, e))
                }
            }
            curStart = -1; curLast = -1; curHasText = false
        }
        var prev = BLANK_ID
        for (t in 0 until tOut) {
            var best = 0
            var bestV = lp[t * N_VOCAB]
            for (c in 1 until N_VOCAB) {
                val v = lp[t * N_VOCAB + c]
                if (v > bestV) { bestV = v; best = c }
            }
            if (best == prev || best == BLANK_ID) { prev = best; continue }
            val piece = pieces.getOrElse(best) { "" }
            if (piece.isEmpty() || piece == "<unk>") { prev = best; continue }
            if (piece.startsWith("▁")) {
                // word boundary: flush the previous word, start a new one
                flush()
                if (sb.isNotEmpty() && sb.last() != ' ') sb.append(' ')
                sb.append(piece.drop(1))
                curStart = t; curLast = t; curHasText = true
            } else {
                sb.append(piece)
                if (curStart < 0) curStart = t
                curLast = t; curHasText = true
            }
            prev = best
        }
        flush()
        return sb.toString().trim() to words
    }

    /** log-mel (N_MELS x nFrames) + per-band z-scoring, matches the exporter. */
    private fun logMel(samples: FloatArray, nFrames: Int): FloatArray {
        // STFT power: (N_BINS x nFrames), in-place radix-2 FFT (N_FFT = 512).
        val power = FloatArray(N_BINS * nFrames)
        val re = FloatArray(N_FFT)
        val im = FloatArray(N_FFT)
        for (f in 0 until nFrames) {
            val base = f * HOP
            val n = minOf(WIN_LEN, samples.size - base)
            var i = 0
            while (i < n) { re[i] = samples[base + i] * hanning[i]; im[i] = 0f; i++ }
            while (i < N_FFT) { re[i] = 0f; im[i] = 0f; i++ }
            fft(re, im)
            val po = f * N_BINS
            for (k in 0 until N_BINS) {
                power[po + k] = re[k] * re[k] + im[k] * im[k]
            }
        }
        // mel = fb @ power, log floor, then per-row z-score.
        val mel = FloatArray(N_MELS * nFrames)
        for (m in 0 until N_MELS) {
            val fbm = m * N_BINS
            val mm = m * nFrames
            for (f in 0 until nFrames) {
                var acc = 0.0
                val pf = f * N_BINS
                val fbmStart = fbm
                for (k in 0 until N_BINS) acc += melFilterbank[fbmStart + k] * power[pf + k]
                mel[mm + f] = (ln(acc + MEL_FLOOR) as Double).toFloat()
            }
        }
        for (m in 0 until N_MELS) {
            val mm = m * nFrames
            var mean = 0.0
            for (f in 0 until nFrames) mean += mel[mm + f]
            mean /= nFrames
            var varSum = 0.0
            for (f in 0 until nFrames) {
                val d = mel[mm + f] - mean
                varSum += d * d
            }
            val std = sqrt(varSum / nFrames) as Double + NORM_EPS
            for (f in 0 until nFrames) {
                mel[mm + f] = ((mel[mm + f] - mean) / std).toFloat()
            }
        }
        return mel
    }

    /** In-place radix-2 iterative FFT (size N_FFT). */
    private fun fft(re: FloatArray, im: FloatArray) {
        val n = re.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                var t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
            }
        }
        var len = 2
        while (len <= n) {
            val ang = -2.0 * PI / len
            val wRe = cos(ang).toFloat()
            val wIm = sin(ang).toFloat()
            var i = 0
            while (i < n) {
                var curRe = 1f
                var curIm = 0f
                val half = len / 2
                for (k in 0 until half) {
                    val a = i + k
                    val b = a + half
                    val yRe = re[b] * curRe - im[b] * curIm
                    val yIm = re[b] * curIm + im[b] * curRe
                    re[b] = re[a] - yRe
                    im[b] = im[a] - yIm
                    re[a] = re[a] + yRe
                    im[a] = im[a] + yIm
                    val ncRe = curRe * wRe - curIm * wIm
                    curIm = curRe * wIm + curIm * wRe
                    curRe = ncRe
                }
                i += len
            }
            len = len shl 1
        }
    }

    /** Greedy CTC decode: argmax per output frame, blank/repeat collapse. */
    private fun ctcDecode(lp: FloatArray): String {
        val tOut = lp.size / N_VOCAB
        val sb = StringBuilder()
        var prev = BLANK_ID
        for (t in 0 until tOut) {
            var best = 0
            var bestV = lp[t * N_VOCAB]
            for (c in 1 until N_VOCAB) {
                val v = lp[t * N_VOCAB + c]
                if (v > bestV) { bestV = v; best = c }
            }
            if (best == prev || best == BLANK_ID) { prev = best; continue }
            val piece = pieces.getOrElse(best) { "" }
            if (piece.isEmpty() || piece == "<unk>") { prev = best; continue }
            if (piece.startsWith("▁")) {
                if (sb.isNotEmpty() && sb.last() != ' ') sb.append(' ')
                sb.append(piece.drop(1))
            } else {
                sb.append(piece)
            }
            prev = best
        }
        return sb.toString().trim()
    }

    override fun close() {
        try { session.close() } catch (_: Throwable) {}
    }
}

private fun log10(x: Double): Double = kotlin.math.log(x, 10.0)

private fun sqrt(x: Double): Double = kotlin.math.sqrt(x)