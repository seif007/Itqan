package itqan.app.ml

import kotlin.math.*

/**
 * Port of the Python feature pipeline used to train TajweedNet
 * (src/train.py -> compute_logmel, clip_evidence_vector; src/rule_acoustics.py
 * -> clip_features). Must match byte-for-byte so Android inference equals training.
 *
 * Input : 16 kHz mono FloatArray (PCM, -1..1).
 * Output: Triple(logmel 40x200, durScalar, ev 5-dim) — exactly what the model sees.
 */
object TajweedFeatures {

    const val SR = 16000
    const val N_MELS = 40
    const val WIN_MS = 25
    const val HOP_MS = 10
    const val N_FFT = (SR * WIN_MS) / 1000          // 400
    const val HOP = (SR * HOP_MS) / 1000            // 160
    const val MAX_FRAMES = 200
    const val FMAX = 8000.0

    /** Rule order must match train.py RULES (also encoded in model_weights.json). */
    val RULES = listOf(
        "MADD", "GHUNNAH", "QALQALAH", "IDGHAM", "IKHFA",
        "ALLAM", "MEEM_SAKINAH", "IZHAR"
    )

    fun metaRules() = RULES

    // rule_acoustics constants
    private const val RA_N_FFT = 512
    private const val RA_HOP = 160
    private const val NASAL_LO = 200.0
    private const val NASAL_HI = 500.0
    private const val LOW_HI = 500.0
    private const val MID_HI = 2000.0
    private const val HIGH_HI = 8000.0

    // acoustic windows (Hann) precomputed
    private val LOGMEL_WIN = FloatArray(N_FFT) { i -> (0.5 - 0.5 * cos(2 * PI * i / (N_FFT - 1))).toFloat() }
    private val RA_WIN = FloatArray(RA_N_FFT) { i -> (0.5 - 0.5 * cos(2 * PI * i / (RA_N_FFT - 1))).toFloat() }

    // mel filterbank (n_mels x (n_fft/2+1)), built once
    private val MEL_FILTERS: Array<FloatArray> = buildMelFilters()

    // rule_acoustics STFT window depends on length; build per call lazily (cached by size)

    // -------------------- FFT-backed DFT (Bluestein chirp-z) --------------------
    // The previous implementation ran a naive O(N^2) DFT per frame: 201 bins x
    // 400 taps for the log-mel and 257 bins x 512 taps for the evidence STFT,
    // each tap a Double cos+sin. At ~200 frames per clip that alone cost
    // ~150-250 ms per analyze() call on the A05s — and the tajweed loop's
    // confirmation search runs dozens of analyze() calls per pass (measured
    // ~33 s per pass). Bluestein rewrites the DFT as a convolution computable
    // with power-of-two FFTs: O(N log N), numerically identical (Double
    // throughout, parity-checked against the naive DFT in unit tests).
    private data class BluesteinPlan(
        val n: Int,
        val m: Int,                       // convolution length (power of two)
        val chirpRe: DoubleArray,         // exp(-i*pi*k^2/n), k < n
        val chirpIm: DoubleArray,
        val bFftRe: DoubleArray,          // FFT of the conjugate-chirp kernel
        val bFftIm: DoubleArray
    )

    private val planCache = HashMap<Int, BluesteinPlan>()

    private fun planFor(n: Int): BluesteinPlan {
        synchronized(planCache) { planCache[n]?.let { return it } }
        var m = 1
        while (m < 2 * n - 1) m = m shl 1
        val chirpRe = DoubleArray(n)
        val chirpIm = DoubleArray(n)
        for (k in 0 until n) {
            // k^2 mod 2n keeps the angle in [0, 2pi) without precision loss
            val kk = ((k.toLong() * k) % (2L * n)).toDouble()
            val ang = -PI * kk / n
            chirpRe[k] = cos(ang)
            chirpIm[k] = sin(ang)
        }
        // b[k] = conj(chirp[k]) wrapped for circular convolution of length m
        val bRe = DoubleArray(m)
        val bIm = DoubleArray(m)
        bRe[0] = chirpRe[0]; bIm[0] = -chirpIm[0]
        for (k in 1 until n) {
            bRe[k] = chirpRe[k]; bIm[k] = -chirpIm[k]
            bRe[m - k] = chirpRe[k]; bIm[m - k] = -chirpIm[k]
        }
        fftInPlace(bRe, bIm, false)
        val plan = BluesteinPlan(n, m, chirpRe, chirpIm, bRe, bIm)
        synchronized(planCache) { planCache[n] = plan }
        return plan
    }

    /** In-place iterative radix-2 Cooley-Tukey FFT. Length must be a power of 2. */
    private fun fftInPlace(re: DoubleArray, im: DoubleArray, inverse: Boolean) {
        val n = re.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                val tr = re[i]; re[i] = re[j]; re[j] = tr
                val ti = im[i]; im[i] = im[j]; im[j] = ti
            }
        }
        var len = 2
        while (len <= n) {
            val ang = 2.0 * PI / len * (if (inverse) 1.0 else -1.0)
            val wr = cos(ang); val wi = sin(ang)
            val half = len shr 1
            var i = 0
            while (i < n) {
                var curR = 1.0; var curI = 0.0
                for (k in 0 until half) {
                    val ur = re[i + k]; val ui = im[i + k]
                    val vr = re[i + k + half] * curR - im[i + k + half] * curI
                    val vi = re[i + k + half] * curI + im[i + k + half] * curR
                    re[i + k] = ur + vr; im[i + k] = ui + vi
                    re[i + k + half] = ur - vr; im[i + k + half] = ui - vi
                    val nr = curR * wr - curI * wi
                    curI = curR * wi + curI * wr
                    curR = nr
                }
                i += len
            }
            len = len shl 1
        }
        if (inverse) {
            for (i in 0 until n) { re[i] /= n; im[i] /= n }
        }
    }

    /**
     * Reusable real-input DFT worker for a FIXED frame length [n] (one instance
     * per extract() call, shared across all frames). Writes the first [outSize]
     * complex DFT bins into [outRe]/[outIm].
     */
    private class RealDft(n: Int) {
        private val plan = planFor(n)
        private val aRe = DoubleArray(plan.m)
        private val aIm = DoubleArray(plan.m)

        fun transform(x: FloatArray, outRe: DoubleArray, outIm: DoubleArray, outSize: Int) {
            val n = plan.n
            val m = plan.m
            java.util.Arrays.fill(aRe, 0.0)
            java.util.Arrays.fill(aIm, 0.0)
            // a[k] = x[k] * chirp[k]
            val cRe = plan.chirpRe; val cIm = plan.chirpIm
            for (k in 0 until n) {
                val v = x[k].toDouble()
                aRe[k] = v * cRe[k]
                aIm[k] = v * cIm[k]
            }
            fftInPlace(aRe, aIm, false)
            val bRe = plan.bFftRe; val bIm = plan.bFftIm
            for (i in 0 until m) {
                val pr = aRe[i] * bRe[i] - aIm[i] * bIm[i]
                val pi = aRe[i] * bIm[i] + aIm[i] * bRe[i]
                aRe[i] = pr; aIm[i] = pi
            }
            fftInPlace(aRe, aIm, true)
            val lim = minOf(outSize, n)
            for (k in 0 until lim) {
                // X[k] = chirp[k] * conv[k]
                outRe[k] = cRe[k] * aRe[k] - cIm[k] * aIm[k]
                outIm[k] = cRe[k] * aIm[k] + cIm[k] * aRe[k]
            }
        }
    }

    data class Features(
        val logmel: Array<FloatArray>,   // [N_MELS][MAX_FRAMES]
        val dur: Float,                   // dur_sec/4 clipped to 4s
        val ev: FloatArray                // 5-dim [nasal, lam, closure, onset, tilt]
    )

    /** Full feature extraction for one clip [0..1] mono @16k. */
    fun extract(clip: FloatArray, durSec: Float): Features =
        extractImpl(clip, durSec, useFft = true)

    /**
     * Naive-DFT reference implementation of [extract], kept ONLY for the JVM
     * parity test (TajweedFeaturesFftParityTest) that proves the FFT-backed
     * pipeline is numerically identical to the original O(N^2) one. Never
     * called from the app.
     */
    internal fun extractNaive(clip: FloatArray, durSec: Float): Features =
        extractImpl(clip, durSec, useFft = false)

    private fun extractImpl(clip: FloatArray, durSec: Float, useFft: Boolean): Features {
        val wav = if (clip.isEmpty()) FloatArray(1) { 0f } else clip.copyOf()
        val lm = computeLogmel(wav, useFft)
        val dur = (min(durSec, 4.0f) / 4.0f)
        val ev = clipEvidenceVector(wav, useFft)
        return Features(lm, dur, ev)
    }

    // -------------------- log-mel (matches train.py compute_logmel) --------------------
    private fun computeLogmel(wav: FloatArray, useFft: Boolean): Array<FloatArray> {
        var x = wav
        if (x.size > 1) {
            // ensure mono (already mono) — nothing to do
        }
        var n = x.size
        var nFrames = 1 + max(0, (n - N_FFT) / HOP)
        if (nFrames <= 0) {
            x = x.copyOf(N_FFT)
            n = x.size
            nFrames = 1
        }
        // frame matrix via stride tricks (nFrames x N_FFT)
        val frames = Array(nFrames) { FloatArray(N_FFT) }
        for (i in 0 until nFrames) {
            val start = i * HOP
            for (j in 0 until N_FFT) {
                val idx = start + j
                frames[i][j] = if (idx < n) x[idx] * LOGMEL_WIN[j] else 0f
            }
        }
        // rfft magnitude^2 per frame
        val spec = Array(nFrames) { FloatArray(N_FFT / 2 + 1) }
        if (useFft) {
            val dft = RealDft(N_FFT)
            val re = DoubleArray(N_FFT / 2 + 1)
            val im = DoubleArray(N_FFT / 2 + 1)
            for (i in 0 until nFrames) {
                dft.transform(frames[i], re, im, N_FFT / 2 + 1)
                val sp = spec[i]
                for (k in sp.indices) sp[k] = (re[k] * re[k] + im[k] * im[k]).toFloat()
            }
        } else {
            for (i in 0 until nFrames) {
                rfftMag2(frames[i], spec[i])
            }
        }
        // mel = filters @ spec.T  -> (n_mels x nFrames)
        val mel = Array(N_MELS) { FloatArray(nFrames) }
        for (m in 0 until N_MELS) {
            val f = MEL_FILTERS[m]
            for (t in 0 until nFrames) {
                var s = 0f
                val sp = spec[t]
                for (k in f.indices) s += f[k] * sp[k]
                // clip + log10(s + 1e-8)
                s = if (s < 1e-8f) 1e-8f else s
                mel[m][t] = log10(s + 1e-8f).toFloat()
            }
        }
        // pad / trim to MAX_FRAMES
        val out = Array(N_MELS) { FloatArray(MAX_FRAMES) }
        val T = nFrames
        if (T < MAX_FRAMES) {
            for (m in 0 until N_MELS) {
                for (t in 0 until T) out[m][t] = mel[m][t]
                for (t in T until MAX_FRAMES) out[m][t] = 0f
            }
        } else {
            for (m in 0 until N_MELS) {
                for (t in 0 until MAX_FRAMES) out[m][t] = mel[m][t]
            }
        }
        return out
    }

    /** Real FFT magnitude squared of a real windowed frame (N_FFT points). */
    private fun rfftMag2(frame: FloatArray, out: FloatArray) {
        // naive DFT for N_FFT/2+1 bins; N_FFT=400 -> 201 bins, fine on-device
        val N = frame.size
        for (k in out.indices) {
            var re = 0.0
            var im = 0.0
            for (n in 0 until N) {
                val ang = -2.0 * PI * k * n / N
                re += frame[n] * cos(ang)
                im += frame[n] * sin(ang)
            }
            out[k] = (re * re + im * im).toFloat()
        }
    }

    // -------------------- rule_acoustics.ev (matches train.py clip_evidence_vector) ----
    private fun clipEvidenceVector(wav: FloatArray, useFft: Boolean): FloatArray {
        val f = clipFeatures(wav, useFft)
        if (f == null) return FloatArray(5)
        val raw = floatArrayOf(f.nasalRatio, f.lam, f.closure, f.onset, f.tilt)
        val scale = floatArrayOf(10.0f, 2.0f, 1.0f, 1.0f, 5.0f)
        val out = FloatArray(5)
        for (i in 0..4) out[i] = tanh(raw[i] / scale[i])
        return out
    }

    private data class AcousticDesc(
        val nasalRatio: Float, val lam: Float, val closure: Float,
        val onset: Float, val tilt: Float
    )

    /** Matches rule_acoustics.clip_features. */
    private fun clipFeatures(y: FloatArray, useFft: Boolean): AcousticDesc? {
        var yy = y
        if (yy.isEmpty()) return null
        if (yy.size < 256) return null
        // peak normalize
        var peak = 0f
        for (s in yy) peak = max(peak, abs(s))
        if (peak > 1e-4f) {
            val inv = 1.0f / peak
            yy = FloatArray(yy.size) { yy[it] * inv }
        }
        val mag = stftMag(yy, useFft)            // (nBins x nFrames)
        val nBins = mag.size
        val nFrames = mag[0].size
        val freqs = FloatArray(nBins) { i -> (i.toFloat() * SR / RA_N_FFT) }
        // mean mag per bin
        val meanMag = FloatArray(nBins)
        for (b in 0 until nBins) {
            var s = 0.0
            for (t in 0 until nFrames) s += mag[b][t]
            meanMag[b] = (if (nFrames > 0) s / nFrames else 0.0).toFloat()
        }
        fun band(lo: Double, hi: Double): Float {
            var s = 0.0
            var cnt = 0
            for (b in freqs.indices) {
                if (freqs[b] >= lo && freqs[b] <= hi) { s += meanMag[b]; cnt++ }
            }
            return if (cnt > 0) (s / cnt).toFloat() else 0f
        }
        val total = (meanMag.average().toFloat()) + 1e-8f
        val nasal = band(NASAL_LO, NASAL_HI)
        val low = band(0.0, LOW_HI)
        val mid = band(LOW_HI, MID_HI)
        val high = band(MID_HI, HIGH_HI)
        // spectral tilt: polyfit log(freq) vs log(meanMag)
        var tilt = 0.0
        var cnt = 0
        var sx = 0.0; var sy = 0.0; var sxx = 0.0; var sxy = 0.0
        for (b in freqs.indices) {
            if (freqs[b] > 0 && meanMag[b] > 0) {
                val lx = ln(freqs[b].toDouble())
                val ly = ln(max(meanMag[b].toDouble(), 1e-8))
                sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly; cnt++
            }
        }
        if (cnt > 2) tilt = (cnt * sxy - sx * sy) / (cnt * sxx - sx * sx)
        // energy envelope across frames -> closure + onset
        val env = FloatArray(nFrames)
        for (t in 0 until nFrames) {
            var s = 0.0
            for (b in 0 until nBins) s += mag[b][t]
            env[t] = (if (nBins > 0) s / nBins else 0.0).toFloat()
        }
        var envMax = 0f
        for (e in env) envMax = max(envMax, e)
        if (envMax > 0f) for (t in env.indices) env[t] = env[t] / envMax
        val nF = env.size
        // QALQALAH hallmark: a complete consonant closure (a *sustained* silent gap)
        // sharply released by a burst. Matches rule_acoustics.clip_features:
        //   closure = depth of the deepest INTERIOR silence window (~30 ms),
        //             restricted to the middle 4/6 of the clip so clip-edge
        //             padding/silence is not mistaken for a qalqalah stop;
        //   onset   = the sharp energy rise in the 1-2 frames right after that
        //             silent window (the "bounce"), only counted if the window
        //             is a genuinely quiet region (win_min < 0.35).
        // (The previous Kotlin port used the global quietest frame + any rising
        // edge -- the "old crude cue" explicitly rejected by rule_acoustics.py --
        // which produced out-of-distribution ev[2]/ev[3] for the Qalqalah head.)
        val lo_i = max(1, nF / 6).coerceAtMost((nF - 1).coerceAtLeast(0))
        val hi_i = max(2, (5 * nF) / 6).coerceAtMost(nF)
        val W = max(2, (0.03f * SR / HOP_MS).toInt())   // ~30 ms of frames
        var closure = 0f
        var onset = 0f
        if (hi_i > lo_i) {
            val interiorLen = hi_i - lo_i
            if (interiorLen >= W) {
                // sliding-window minimum over the interior envelope
                var bestWinMin = Float.MAX_VALUE
                var bestK = 0
                for (k in 0..interiorLen - W) {
                    var wmin = Float.MAX_VALUE
                    for (j in 0 until W) {
                        val v = env[lo_i + k + j]
                        if (v < wmin) wmin = v
                    }
                    if (wmin < bestWinMin) { bestWinMin = wmin; bestK = k }
                }
                closure = 1f - bestWinMin
                val after = lo_i + bestK + W
                val burst = when {
                    after + 1 < nF -> env[after + 1] - env[after]
                    after < nF -> env[after] - env[after - 1]
                    else -> 0f
                }
                onset = if (bestWinMin < 0.35f) max(0f, burst) else 0f
            } else {
                // interior shorter than the window: fall back to its minimum
                var interiorMin = Float.MAX_VALUE
                for (t in lo_i until hi_i) if (env[t] < interiorMin) interiorMin = env[t]
                closure = if (interiorLen > 0) 1f - interiorMin else 0f
                onset = 0f
            }
        }
        return AcousticDesc(
            nasalRatio = nasal / total,
            lam = high / (mid + 1e-8f),
            closure = closure,
            onset = onset,
            tilt = tilt.toFloat()
        )
    }

    private fun stftMag(y: FloatArray, useFft: Boolean): Array<FloatArray> {
        val n = y.size
        var nn = n
        var yy = y
        if (nn < RA_N_FFT) { yy = y.copyOf(RA_N_FFT); nn = yy.size }
        var nFrames = 1 + (nn - RA_N_FFT) / RA_HOP
        if (nFrames < 1) { yy = y.copyOf(RA_N_FFT + RA_HOP); nn = yy.size; nFrames = 1 }
        val frames = Array(nFrames) { FloatArray(RA_N_FFT) }
        for (i in 0 until nFrames) {
            val start = i * RA_HOP
            for (j in 0 until RA_N_FFT) {
                val idx = start + j
                frames[i][j] = if (idx < nn) yy[idx] * RA_WIN[j] else 0f
            }
        }
        val nBins = RA_N_FFT / 2 + 1
        val out = Array(nBins) { FloatArray(nFrames) }
        if (useFft) {
            val dft = RealDft(RA_N_FFT)
            val re = DoubleArray(nBins)
            val im = DoubleArray(nBins)
            for (i in 0 until nFrames) {
                dft.transform(frames[i], re, im, nBins)
                for (k in 0 until nBins) {
                    out[k][i] = sqrt(re[k] * re[k] + im[k] * im[k]).toFloat() + 1e-8f
                }
            }
        } else {
            for (i in 0 until nFrames) {
                val frame = frames[i]
                for (k in 0 until nBins) {
                    var re = 0.0; var im = 0.0
                    for (m in 0 until RA_N_FFT) {
                        val ang = -2.0 * PI * k * m / RA_N_FFT
                        re += frame[m] * cos(ang)
                        im += frame[m] * sin(ang)
                    }
                    out[k][i] = sqrt(re * re + im * im).toFloat() + 1e-8f
                }
            }
        }
        return out
    }

    // -------------------- mel filterbank (matches train.py _build_mel_filters) ----------
    private fun buildMelFilters(): Array<FloatArray> {
        fun hzToMel(h: Double) = 2595.0 * log10(1.0 + h / 700.0)
        fun melToHz(m: Double) = 700.0 * (10.0.pow(m / 2595.0) - 1.0)
        val melPts = DoubleArray(N_MELS + 2) {
            val t = it.toDouble() / (N_MELS + 1)
            hzToMel(0.0) + t * (hzToMel(FMAX) - hzToMel(0.0))
        }
        val hzPts = DoubleArray(N_MELS + 2) { melToHz(melPts[it]) }
        val bins = IntArray(N_MELS + 2) {
            floor((N_FFT + 1) * hzPts[it] / SR).toInt()
        }
        val filters = Array(N_MELS) { FloatArray(N_FFT / 2 + 1) }
        for (m in 0 until N_MELS) {
            val f0 = bins[m]; val f1 = bins[m + 1]; val f2 = bins[m + 2]
            if (f1 <= f0 || f2 <= f1) continue
            for (k in f0 until f1) {
                if (k < filters[m].size) filters[m][k] = ((k - f0).toFloat() / (f1 - f0))
            }
            for (k in f1 until f2) {
                if (k < filters[m].size) filters[m][k] = ((f2 - k).toFloat() / (f2 - f1))
            }
        }
        return filters
    }
}
