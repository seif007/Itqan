package itqan.app.tajweed

import kotlin.math.abs

/**
 * Meem Saakinah (labial rules: idghaam shafawi / ikhfa shafawi / izhaar shafawi).
 * Secondary cue: a bilabial closure (low-energy gap) followed by a burst, plus
 * low-band nasal energy. The MFCC best-of-N template carries most of the signal;
 * this adds the labial-articulation cue.
 */
class MeemSakinahAnalyzer(private val features: AcousticFeatures) {

    fun analyze(audioSegment: FloatArray, durationMs: Long = 0L): Float {
        if (audioSegment.size < 256) return 0.5f

        val rms = features.extractRMSContour(audioSegment, frameSize = 128, hopSize = 64)
        if (rms.size < 3) return 0.5f

        // Nasal / labial low-band energy ratio (m-like formants ~200-1000 Hz).
        val nfft = 256
        val mag = features.computeFFTMag(audioSegment.sliceArray(0 until minOf(nfft, audioSegment.size)), nfft)
        val freqStep = features.sampleRate / nfft.toFloat()
        var nasal = 0f
        var total = 0f
        for (i in mag.indices) {
            val f = i * freqStep
            total += mag[i]
            if (f in 200f..1000f) nasal += mag[i]
        }
        val nasalRatio = if (total > 0f) (nasal / total).coerceIn(0f, 1f) else 0f

        // Closure then burst: find a local RMS minimum followed by a rise.
        var minIdx = 0
        var minVal = Float.MAX_VALUE
        for (i in rms.indices) {
            if (rms[i] < minVal) { minVal = rms[i]; minIdx = i }
        }
        val right = if (minIdx < rms.size - 1) rms[minIdx + 1] else rms[minIdx]
        val left = if (minIdx > 0) rms[minIdx - 1] else rms[minIdx]
        val closureScore = if (minVal < 1e-4f) {
            // strong silence gap present
            1f
        } else {
            val surround = (left + right) / 2f
            if (surround > 0f) (1f - (minVal / surround)).coerceIn(0f, 1f) else 0f
        }

        return (0.5f * nasalRatio + 0.5f * closureScore).coerceIn(0f, 1f)
    }
}
