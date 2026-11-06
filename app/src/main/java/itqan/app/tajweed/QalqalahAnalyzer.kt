package itqan.app.tajweed

import kotlin.math.abs
import kotlin.math.sqrt

class QalqalahAnalyzer(
    private val features: AcousticFeatures,
    private val config: CalibrationDatabase.QalqalahConfig = CalibrationDatabase.QalqalahConfig()
) {

    // Detect transient burst energy patterns (attack slope, decay slope, energy burst analysis).
    fun analyze(audioSegment: FloatArray): Float {
        if (audioSegment.isEmpty()) return 0f

        val rmsContour = features.extractRMSContour(audioSegment, hopSize = 64)
        if (rmsContour.size < 3) return 0f

        // A Qalqalah is a short burst of energy followed by quick decay.
        val maxEnergy = rmsContour.maxOrNull() ?: 1e-10f
        var maxIndex = 0
        var totalEnergy = 0f
        for (i in rmsContour.indices) {
            val v = rmsContour[i]
            if (v == maxEnergy) maxIndex = i
            totalEnergy += v
        }
        val avgEnergy = totalEnergy / rmsContour.size
        val burstScore = if (avgEnergy > 0) ((maxEnergy / avgEnergy) / config.burstDivisor).coerceIn(0f, 1f) else 0f

        // Attack slope
        var attackSlope = 0f
        if (maxIndex > 0) {
            val startIdx = maxOf(0, maxIndex - 3)
            val rise = maxEnergy - rmsContour[startIdx]
            val run = maxIndex - startIdx
            attackSlope = (rise / run) / maxEnergy
        }

        // Decay slope
        var decaySlope = 0f
        if (maxIndex < rmsContour.size - 1) {
            val endIdx = minOf(rmsContour.size - 1, maxIndex + 3)
            val drop = maxEnergy - rmsContour[endIdx]
            val run = endIdx - maxIndex
            decaySlope = (drop / run) / maxEnergy
        }

        // Transient peak analysis (Zero Crossing Rate spike at the burst)
        val zcrContour = features.extractZeroCrossingRate(audioSegment, hopSize = 64)
        var transientScore = 0f
        if (zcrContour.isNotEmpty() && maxIndex < zcrContour.size) {
            val peakZcr = zcrContour[maxIndex]
            val avgZcr = zcrContour.average().toFloat()
            transientScore = if (avgZcr > 0f) ((peakZcr / avgZcr) / config.transientDivisor).coerceIn(0f, 1f) else 0.5f
        }

        val attackScore = attackSlope.coerceIn(0f, 1f)
        val decayScore = decaySlope.coerceIn(0f, 1f)

        return (config.wBurst * burstScore + config.wAttack * attackScore + config.wDecay * decayScore + config.wTransient * transientScore)
            .coerceIn(0f, 1f)
    }
}
