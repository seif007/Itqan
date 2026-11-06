package itqan.app.tajweed

import kotlin.math.sqrt
import kotlin.math.pow

class SimilarityEngine {

    suspend fun applyCMVN(mfccs: List<FloatArray>): List<FloatArray> {
        if (mfccs.isEmpty()) return mfccs
        val numCoefficients = mfccs[0].size
        val means = FloatArray(numCoefficients)
        val variances = FloatArray(numCoefficients)

        for (frame in mfccs) {
            for (j in 0 until numCoefficients) {
                means[j] += frame[j]
            }
        }
        for (j in 0 until numCoefficients) {
            means[j] /= mfccs.size
        }

        for (frame in mfccs) {
            for (j in 0 until numCoefficients) {
                val diff = frame[j] - means[j]
                variances[j] += diff * diff
            }
        }

        val stdDevs = FloatArray(numCoefficients)
        for (j in 0 until numCoefficients) {
            stdDevs[j] = kotlin.math.sqrt(variances[j] / mfccs.size).coerceAtLeast(1e-5f)
        }

        return mfccs.map { frame ->
            val newFrame = FloatArray(numCoefficients)
            for (j in 0 until numCoefficients) {
                newFrame[j] = (frame[j] - means[j]) / stdDevs[j]
            }
            newFrame
        }
    }

    /** DTW distance on already-CMVN-normalised MFCCs (caller normalises once). */
    fun rawDTWDistance(mfcc1: List<FloatArray>, mfcc2: List<FloatArray>): Float {
        val a = mfcc1.take(500)
        val b = mfcc2.take(500)
        if (a.isEmpty() || b.isEmpty()) return Float.MAX_VALUE

        val n = a.size
        val m = b.size
        val dtw = Array(n + 1) { FloatArray(m + 1) { Float.MAX_VALUE } }

        dtw[0][0] = 0f

        for (i in 1..n) {
            for (j in 1..m) {
                val cost = euclideanDistance(a[i - 1], b[j - 1])
                dtw[i][j] = cost + minOf(
                    dtw[i - 1][j],    // insertion
                    dtw[i][j - 1],    // deletion
                    dtw[i - 1][j - 1] // match
                )
            }
        }
        return dtw[n][m] / maxOf(n, m).toFloat()
    }

    suspend fun calculateDTWDistance(rawMfcc1: List<FloatArray>, rawMfcc2: List<FloatArray>): Float {
        // Prevent OOM and excessive processing time by capping frames to max ~4 seconds (500 frames)
        val mfcc1 = applyCMVN(rawMfcc1).take(500)
        val mfcc2 = applyCMVN(rawMfcc2).take(500)
        return rawDTWDistance(mfcc1, mfcc2)
    }

    private fun euclideanDistance(v1: FloatArray, v2: FloatArray): Float {
        var sum = 0f
        val len = minOf(v1.size, v2.size)
        for (i in 0 until len) {
            sum += (v1[i] - v2[i]).pow(2)
        }
        return sqrt(sum.toDouble()).toFloat()
    }
    
    fun scoreSimilarity(distance: Float, threshold: Float = 0.5f): Float {
        if (distance.isNaN() || threshold <= 0f) return 0.5f
        // Convert distance to a 0.0 - 1.0 confidence score
        val score = 1.0f - (distance / threshold)
        return score.coerceIn(0.0f, 1.0f)
    }
}
