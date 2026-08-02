package com.example.tajweed

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.sin
import kotlin.math.sqrt

class AcousticFeatures(val sampleRate: Int = 16000) {

    fun extractRMSContour(audio: FloatArray, frameSize: Int = 256, hopSize: Int = 128): FloatArray {
        if (audio.isEmpty()) return floatArrayOf(0f)
        val numFrames = ((audio.size - frameSize) / hopSize) + 1
        if (numFrames <= 0) return floatArrayOf(rootMeanSquare(audio, 0, audio.size))
        val rms = FloatArray(numFrames)
        for (i in 0 until numFrames) {
            val start = i * hopSize
            val end = minOf(start + frameSize, audio.size)
            rms[i] = rootMeanSquare(audio, start, end)
        }
        return rms
    }

    private fun rootMeanSquare(audio: FloatArray, start: Int, end: Int): Float {
        var sum = 0f
        val len = end - start
        if (len <= 0) return 0f
        for (i in start until end) {
            val sample = audio[i]
            sum += sample * sample
        }
        return sqrt((sum / len).toDouble()).toFloat()
    }

    private fun preEmphasis(audio: FloatArray, coeff: Float = 0.97f): FloatArray {
        val out = FloatArray(audio.size)
        if (audio.isNotEmpty()) {
            out[0] = audio[0]
            for (i in 1 until audio.size) out[i] = audio[i] - coeff * audio[i - 1]
        }
        return out
    }

    private fun applyHammingWindow(frame: FloatArray) {
        val n = frame.size
        for (i in 0 until n) {
            frame[i] *= (0.54f - 0.46f * cos(2.0 * PI * i / (n - 1))).toFloat()
        }
    }

    fun computeFFTMag(frame: FloatArray, nfft: Int = 256): FloatArray {
        val real = FloatArray(nfft)
        val imag = FloatArray(nfft)
        System.arraycopy(frame, 0, real, 0, minOf(frame.size, nfft))

        applyHammingWindow(real)

        var j = 0
        for (i in 0 until nfft - 1) {
            if (i < j) {
                val tr = real[i]
                real[i] = real[j]
                real[j] = tr
            }
            var k = nfft / 2
            while (k >= 1 && k <= j) {
                j -= k
                k /= 2
            }
            j += k
        }

        var i = 1
        while (i < nfft) {
            val step = i * 2
            for (m in 0 until i) {
                val theta = -PI * m / i
                val wr = cos(theta).toFloat()
                val wi = sin(theta).toFloat()
                for (k in m until nfft step step) {
                    val idx = k + i
                    val tr = wr * real[idx] - wi * imag[idx]
                    val ti = wr * imag[idx] + wi * real[idx]
                    real[idx] = real[k] - tr
                    imag[idx] = imag[k] - ti
                    real[k] += tr
                    imag[k] += ti
                }
            }
            i = step
        }
        val mag = FloatArray(nfft / 2 + 1)
        for (k in mag.indices) {
            mag[k] = sqrt(real[k] * real[k] + imag[k] * imag[k])
        }
        return mag
    }

    private fun hzToMel(hz: Float) = 2595.0f * log10(1 + hz / 700.0f)
    private fun melToHz(mel: Float) = 700.0f * ((Math.pow(10.0, (mel / 2595.0f).toDouble())).toFloat() - 1.0f)

    private fun getMelFilterbanks(nfft: Int, numFilters: Int = 26): Array<FloatArray> {
        val minMel = hzToMel(0f)
        val maxMel = hzToMel(sampleRate / 2f)

        val melPoints = FloatArray(numFilters + 2)
        val hzPoints = FloatArray(numFilters + 2)
        val bins = IntArray(numFilters + 2)

        for (i in melPoints.indices) {
            melPoints[i] = minMel + i * (maxMel - minMel) / (numFilters + 1)
            hzPoints[i] = melToHz(melPoints[i])
            bins[i] = kotlin.math.floor((nfft + 1) * hzPoints[i] / sampleRate).toInt()
        }

        val fbank = Array(numFilters) { FloatArray(nfft / 2 + 1) }
        for (m in 1..numFilters) {
            for (k in bins[m - 1] until bins[m]) {
                fbank[m - 1][k] = (k - bins[m - 1]).toFloat() / (bins[m] - bins[m - 1])
            }
            for (k in bins[m] until bins[m + 1]) {
                fbank[m - 1][k] = (bins[m + 1] - k).toFloat() / (bins[m + 1] - bins[m])
            }
        }
        return fbank
    }

    private fun applyDCT(logEnergies: FloatArray, numCoefficients: Int): FloatArray {
        val numFilters = logEnergies.size
        val dct = FloatArray(numCoefficients)
        val factor = PI / numFilters
        for (i in 0 until numCoefficients) {
            var sum = 0f
            for (j in 0 until numFilters) {
                sum += logEnergies[j] * cos(i * (j + 0.5f) * factor).toFloat()
            }
            dct[i] = sum
        }
        return dct
    }

    suspend fun extractMFCC(audio: FloatArray, numCoefficients: Int = 13): List<FloatArray> {
        if (audio.isEmpty()) return listOf(FloatArray(numCoefficients))

        val frameSize = 256
        val hopSize = 128
        val nfft = 256
        val numFilters = 26

        val preEmphasized = preEmphasis(audio)
        val numFrames = maxOf(1, (preEmphasized.size - frameSize) / hopSize + 1)
        val filterbanks = getMelFilterbanks(nfft, numFilters)

        val mfccs = mutableListOf<FloatArray>()
        for (i in 0 until numFrames) {
            if (i % 50 == 0) kotlinx.coroutines.yield()
            val start = i * hopSize
            val end = minOf(preEmphasized.size, start + frameSize)
            val frame = FloatArray(frameSize)
            System.arraycopy(preEmphasized, start, frame, 0, end - start)

            val mag = computeFFTMag(frame, nfft)
            val logEnergies = FloatArray(numFilters)
            for (m in 0 until numFilters) {
                var sum = 0f
                for (k in mag.indices) sum += mag[k] * filterbanks[m][k]
                logEnergies[m] = if (sum > 1e-10f) log10(sum) else -10f
            }

            val coeffs = applyDCT(logEnergies, numCoefficients)
            mfccs.add(coeffs)
        }

        // Append delta (dynamics) and delta-delta (acceleration) so the matcher sees *how*
        // the spectrum changes over time, not just its static timbre. This makes the DTW far
        // more robust to speaker/pitch differences — key for telling tajweed rules apart.
        return withDeltas(mfccs, numCoefficients)
    }

    private fun computeDeltas(frames: List<FloatArray>): List<FloatArray> {
        if (frames.isEmpty()) return frames
        val n = frames.size
        val dim = frames[0].size
        val nWin = 2
        val denom = (1..nWin).sumOf { it * it }.toFloat() * 2f // 2*(1+4)=10
        return List(n) { i ->
            val out = FloatArray(dim)
            for (m in -nWin..nWin) {
                val idx = (i + m).coerceIn(0, n - 1)
                val w = m.toFloat()
                for (j in 0 until dim) out[j] += w * frames[idx][j]
            }
            for (j in 0 until dim) out[j] /= denom
            out
        }
    }

    private fun withDeltas(mfccs: List<FloatArray>, numCoefficients: Int): List<FloatArray> {
        val n = mfccs.size
        if (n < 2) return mfccs
        val deltas = computeDeltas(mfccs)
        val ddelta = computeDeltas(deltas)
        return List(n) { i ->
            val combined = FloatArray(numCoefficients * 3)
            System.arraycopy(mfccs[i], 0, combined, 0, numCoefficients)
            System.arraycopy(deltas[i], 0, combined, numCoefficients, numCoefficients)
            System.arraycopy(ddelta[i], 0, combined, numCoefficients * 2, numCoefficients)
            combined
        }
    }

    // =========================================================================
    // Source / articulatory features for nasal-transition rules (IDGHAM / IKHFA)
    // =========================================================================

    /**
     * Nasality degree in [0,1]: how much the segment's long-term spectrum resembles a
     * nasalised articulation. Nasal sounds create an anti-resonance (spectral valley)
     * in the low band plus extra "nasal murmur" energy above ~1 kHz. MFCC alone cannot
     * separate merging (IDGHAM), concealed (IKHFA) and clear (IZHAR) noon/meem realisations
     * because they share the same phonemes — only the *nasalisation degree* differs. This
     * cue lets [NasalTransitionAnalyzer] tell them apart.
     */
    fun extractNasality(audio: FloatArray): Float {
        if (audio.size < 256) return 0f
        val nfft = 256
        val freqStep = sampleRate / nfft.toFloat()
        val mag = FloatArray(nfft / 2 + 1)
        val step = nfft / 2
        val last = (audio.size - nfft).coerceAtLeast(0)
        var frames = 0
        var i = 0
        while (i <= last) {
            val frame = audio.sliceArray(i until i + nfft)
            val fm = computeFFTMag(frame, nfft)
            for (j in mag.indices) mag[j] += fm[j]
            frames++
            i += step
        }
        if (frames == 0) return 0f
        for (j in mag.indices) mag[j] /= frames

        var total = 0f
        for (v in mag) total += v
        if (total <= 1e-6f) return 0f

        // Low nasal band (200-800 Hz) energy ratio.
        var low = 0f
        for (idx in mag.indices) {
            val f = idx * freqStep
            if (f in 200f..800f) low += mag[idx]
        }
        val lowRatio = (low / total).coerceIn(0f, 1f)

        // Anti-resonance depth inside the low band: a nasal anti-resonance produces a
        // spectral valley, i.e. a large dip relative to the band's own peak.
        var mn = Float.MAX_VALUE
        var mx = 0f
        for (idx in mag.indices) {
            val f = idx * freqStep
            if (f in 200f..800f) {
                if (mag[idx] < mn) mn = mag[idx]
                if (mag[idx] > mx) mx = mag[idx]
            }
        }
        val depth = if (mx > 1e-6f) (1f - mn / mx).coerceIn(0f, 1f) else 0f

        // Nasal murmur prominence: extra energy around 1-2.5 kHz (above the oral formants).
        var high = 0f
        for (idx in mag.indices) {
            val f = idx * freqStep
            if (f in 1000f..2500f) high += mag[idx]
        }
        val highRatio = (high / total).coerceIn(0f, 1f)

        // Blend into a 0..1 degree. Low/high energy ratios are small (~0.1-0.3) so scale
        // them up; the anti-resonance depth is already 0..1 and carries most of the signal.
        return (0.35f * (lowRatio * 2.5f) + 0.40f * depth + 0.25f * (highRatio * 2.5f))
            .coerceIn(0f, 1f)
    }

    /**
     * Fundamental frequency + voicing probability via autocorrelation over a mid-window.
     * Returns (f0Hz, voicing) where voicing is the peak normalised correlation in [0,1].
     */
    fun extractF0(audio: FloatArray): Pair<Float, Float> {
        if (audio.size < 256) return 0f to 0f
        val winSize = minOf(512, audio.size)
        val start = maxOf(0, (audio.size - winSize) / 2)
        val frame = audio.sliceArray(start until start + winSize)
        val fMin = 80f
        val fMax = 350f
        val minLag = (sampleRate / fMax).toInt().coerceAtLeast(2)
        val maxLag = (sampleRate / fMin).toInt().coerceAtMost(winSize - 1)

        var g0 = 0f
        for (v in frame) g0 += v * v

        var bestLag = 0
        var bestCorr = -Float.MAX_VALUE
        for (lag in minLag..maxLag) {
            var corr = 0f
            for (k in 0 until winSize - lag) corr += frame[k] * frame[k + lag]
            if (corr > bestCorr) {
                bestCorr = corr
                bestLag = lag
            }
        }
        val f0 = if (bestLag > 0) sampleRate.toFloat() / bestLag else 0f
        val voiced = if (g0 > 1e-6f) (bestCorr / g0).coerceIn(0f, 1f) else 0f
        return f0 to voiced
    }

    /**
     * Harmonics-to-Noise Ratio proxy in [0,1]: strength of the periodic (voiced) component
     * versus aperiodic noise, derived from the peak normalised autocorrelation in the vocal
     * pitch range. Used to gate the nasal cue, since nasalisation is a voiced phenomenon —
     * an unvoiced/whispered or silent segment must not score high on "nasal".
     */
    fun extractHNR(audio: FloatArray): Float {
        if (audio.size < 256) return 0f
        val winSize = minOf(512, audio.size)
        val start = maxOf(0, (audio.size - winSize) / 2)
        val frame = audio.sliceArray(start until start + winSize)
        var g0 = 0f
        for (v in frame) g0 += v * v
        if (g0 <= 1e-6f) return 0f
        val fMin = 80f
        val fMax = 350f
        val minLag = (sampleRate / fMax).toInt().coerceAtLeast(2)
        val maxLag = (sampleRate / fMin).toInt().coerceAtMost(winSize - 1)
        var maxCorr = -Float.MAX_VALUE
        for (lag in minLag..maxLag) {
            var c = 0f
            for (k in 0 until winSize - lag) c += frame[k] * frame[k + lag]
            if (c > maxCorr) maxCorr = c
        }
        val r = (maxCorr / g0).coerceIn(0f, 1f)
        // HNR(dB) = -10*log10(1-r); map ~0..30 dB onto 0..1.
        return if (r >= 1f) 1f else (-10f * log10(1f - r)).coerceIn(0f, 60f) / 60f
    }

    fun extractSpectralCentroid(audio: FloatArray, nfft: Int = 256): Float {
        if (audio.isEmpty()) return 0f
        val mag = computeFFTMag(audio, nfft)
        var num = 0f
        var den = 0f
        for (i in mag.indices) {
            val freq = (i * sampleRate.toFloat()) / nfft
            num += freq * mag[i]
            den += mag[i]
        }
        return if (den > 0f) num / den else 0f
    }

    // Kept for backward compatibility if needed, but not used as centroid anymore
    fun extractZeroCrossingRate(audio: FloatArray, frameSize: Int = 256, hopSize: Int = 128): FloatArray {
        if (audio.isEmpty()) return floatArrayOf(0f)
        val numFrames = ((audio.size - frameSize) / hopSize) + 1
        if (numFrames <= 0) return floatArrayOf(zcr(audio, 0, audio.size))
        val zcrs = FloatArray(numFrames)
        for (i in 0 until numFrames) {
            val start = i * hopSize
            val end = minOf(start + frameSize, audio.size)
            zcrs[i] = zcr(audio, start, end)
        }
        return zcrs
    }

    private fun zcr(audio: FloatArray, start: Int, end: Int): Float {
        var crossings = 0
        val len = end - start
        if (len <= 1) return 0f
        for (i in start + 1 until end) {
            if (audio[i] * audio[i - 1] < 0) crossings++
        }
        return crossings.toFloat() / len
    }
}
