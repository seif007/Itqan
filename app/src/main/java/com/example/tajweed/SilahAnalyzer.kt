package com.example.tajweed

import kotlin.math.abs

/**
 * Silah (Qalun only): continuation / light prolongation of the haa when followed by
 * a letter that permits it. Secondary cue: vowel continuity (low, stable RMS variance
 * across the segment). Duration is handled by the main analyzer via [targetRatio].
 */
class SilahAnalyzer(private val features: AcousticFeatures) {

    fun analyze(audioSegment: FloatArray): Float {
        if (audioSegment.size < 256) return 0.5f
        val rms = features.extractRMSContour(audioSegment)
        if (rms.size < 3) return 0.5f
        var diffSum = 0f
        for (i in 1 until rms.size) diffSum += abs(rms[i] - rms[i - 1])
        // Lower variance => more continuous (steady vowel). Map to 0..1.
        val variance = diffSum / rms.size
        return (1f - variance.coerceIn(0f, 1f)).coerceIn(0f, 1f)
    }
}
