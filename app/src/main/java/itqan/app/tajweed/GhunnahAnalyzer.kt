package itqan.app.tajweed

import kotlin.math.abs
import kotlin.math.sqrt

class GhunnahAnalyzer(
    private val features: AcousticFeatures,
    private val config: CalibrationDatabase.GhunnahConfig = CalibrationDatabase.GhunnahConfig()
) {

    // Analyzes nasal resonance and smoothness (duration ~ 2 harakat).
    fun analyze(audioSegment: FloatArray, durationMs: Long): Float {
        if (audioSegment.isEmpty() || durationMs <= 0) return 0f

        // 1. Duration score (Ghunnah should be around expectedMinDuration).
        val durationScore = (durationMs.toFloat() / config.expectedMinDurationMs).coerceIn(0f, 1.0f)

        // 2. Nasal resonance profile: low-frequency energy (nasal murmur).
        val nfft = 256
        val sampleRate = features.sampleRate
        var centroid = 0f
        var nasalEnergyRatio = 0f

        if (audioSegment.size >= nfft) {
            val mag = FloatArray(nfft / 2 + 1)
            var framesProcessed = 0
            for (i in 0 until (audioSegment.size - nfft) step nfft / 2) {
                val frame = audioSegment.sliceArray(i until i + nfft)
                val frameMag = features.computeFFTMag(frame, nfft)
                for (j in mag.indices) {
                    if (j < frameMag.size) mag[j] += frameMag[j]
                }
                val c = features.extractSpectralCentroid(frame, nfft)
                centroid += c
                framesProcessed++
            }
            if (framesProcessed > 0) centroid /= framesProcessed
            for (j in mag.indices) mag[j] /= framesProcessed

            var nasalEnergy = 0f
            var totalEnergy = 0f
            val freqStep = sampleRate / nfft.toFloat()
            for (i in mag.indices) {
                val freq = i * freqStep
                totalEnergy += mag[i]
                if (freq in config.nasalBandLowHz..config.nasalBandHighHz) {
                    nasalEnergy += mag[i]
                }
            }
            if (totalEnergy > 0) {
                nasalEnergyRatio = (nasalEnergy / totalEnergy).coerceIn(0f, 1f)
            }
        }

        // 3. RMS contour smoothness.
        val rmsContour = features.extractRMSContour(audioSegment)
        val smoothnessScore = if (rmsContour.size > 2) {
            var diffSum = 0f
            for (i in 1 until rmsContour.size) {
                diffSum += abs(rmsContour[i] - rmsContour[i - 1])
            }
            1f - (diffSum / rmsContour.size).coerceIn(0f, 1f)
        } else 0.5f

        // Blend nasal resonance, smoothness and duration using calibrated weights.
        val combined = (config.wNasal * nasalEnergyRatio + config.wSmoothness * smoothnessScore + config.wDuration * durationScore)
            .coerceIn(0f, 1f)

        return combined
    }
}
