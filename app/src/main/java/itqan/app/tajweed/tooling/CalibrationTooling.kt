package itqan.app.tajweed.tooling

import itqan.app.tajweed.TajweedAnalyzer

class CalibrationTooling {
    
    data class CalibrationResult(
        val recommendedConfidenceThreshold: Float,
        val meanCorrectScore: Float,
        val meanIncorrectScore: Float,
        val suggestedMaddRatio: Float
    )

    fun calibrateThresholds(
        correctRecordings: List<Float>, 
        incorrectRecordings: List<Float>
    ): CalibrationResult {
        val meanCorrect = if (correctRecordings.isNotEmpty()) correctRecordings.average().toFloat() else 0.8f
        val meanIncorrect = if (incorrectRecordings.isNotEmpty()) incorrectRecordings.average().toFloat() else 0.3f
        
        // Midpoint between average correct and average incorrect
        val rawThreshold = (meanCorrect + meanIncorrect) / 2f
        // Add a small safety margin to avoid false positives (e.g. +0.05)
        val recommendedThreshold = (rawThreshold + 0.05f).coerceIn(0.1f, 0.9f)
        
        return CalibrationResult(
            recommendedConfidenceThreshold = recommendedThreshold,
            meanCorrectScore = meanCorrect,
            meanIncorrectScore = meanIncorrect,
            suggestedMaddRatio = 0.5f // Placeholder, can be calibrated based on duration tracking
        )
    }
}
