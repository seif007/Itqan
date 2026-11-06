package itqan.app.tajweed

/**
 * Izhar (clarity): the noon/meem saakinah is pronounced clearly with a gap
 * (no assimilation, no nasal murmur). Secondary cue: LOW nasal-band energy plus a
 * clear oral burst/closure. The MFCC best-of-N template is primary.
 */
class IzharAnalyzer(private val features: AcousticFeatures) {

    fun analyze(audioSegment: FloatArray): Float {
        if (audioSegment.size < 256) return 0.5f

        // Nasal murmur should be LOW for Izhar.
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
        val clarityFromNasal = (1f - nasalRatio).coerceIn(0f, 1f)

        // Clear oral burst: a distinct RMS peak.
        val rms = features.extractRMSContour(audioSegment, frameSize = 128, hopSize = 64)
        val burstScore = if (rms.size >= 3) {
            val maxE = rms.maxOrNull() ?: 1e-10f
            val avgE = rms.average().toFloat()
            if (avgE > 0f) ((maxE / avgE) / 5f).coerceIn(0f, 1f) else 0.5f
        } else 0.5f

        return (0.6f * clarityFromNasal + 0.4f * burstScore).coerceIn(0f, 1f)
    }
}
