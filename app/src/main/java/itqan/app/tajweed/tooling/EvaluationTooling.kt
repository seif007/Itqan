package itqan.app.tajweed.tooling

class EvaluationTooling {
    
    data class EvaluationMetrics(
        val precision: Float,
        val recall: Float,
        val f1Score: Float,
        val falsePositiveRate: Float,
        val falseNegativeRate: Float
    )

    fun evaluate(
        predictions: List<Boolean>, // true if engine said it's correct
        groundTruth: List<Boolean>  // true if recitation was actually correct
    ): EvaluationMetrics {
        var truePositives = 0
        var falsePositives = 0
        var trueNegatives = 0
        var falseNegatives = 0
        
        for (i in predictions.indices) {
            val pred = predictions[i]
            val truth = groundTruth[i]
            
            if (pred && truth) truePositives++
            if (pred && !truth) falsePositives++
            if (!pred && !truth) trueNegatives++
            if (!pred && truth) falseNegatives++
        }
        
        val precision = if (truePositives + falsePositives > 0) {
            truePositives.toFloat() / (truePositives + falsePositives)
        } else 0f
        
        val recall = if (truePositives + falseNegatives > 0) {
            truePositives.toFloat() / (truePositives + falseNegatives)
        } else 0f
        
        val f1Score = if (precision + recall > 0) {
            2 * (precision * recall) / (precision + recall)
        } else 0f
        
        val fpr = if (falsePositives + trueNegatives > 0) {
            falsePositives.toFloat() / (falsePositives + trueNegatives)
        } else 0f
        
        val fnr = if (falseNegatives + truePositives > 0) {
            falseNegatives.toFloat() / (falseNegatives + truePositives)
        } else 0f
        
        return EvaluationMetrics(precision, recall, f1Score, fpr, fnr)
    }
}
