package itqan.app.tajweed

class MaddAnalyzer(private val features: AcousticFeatures) {

    // Analyzes vowel elongation resonance and stability.
    fun analyzeEnergy(audioSegment: FloatArray): Float {
        if (audioSegment.isEmpty()) return 0f
        
        // Continuous energy profile (should be sustained, not bursty)
        val rmsContour = features.extractRMSContour(audioSegment)
        val energyScore = if (rmsContour.isNotEmpty()) {
            val maxRms = rmsContour.maxOrNull() ?: 1f
            val avgRms = rmsContour.average().toFloat()
            // High average compared to max means sustained energy
            (avgRms / (maxRms + 1e-6f)).coerceIn(0f, 1f)
        } else {
            0.5f
        }
        
        return energyScore
    }
}
