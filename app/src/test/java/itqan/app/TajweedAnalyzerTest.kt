package itqan.app

import itqan.app.tajweed.TajweedAnalyzer
import org.junit.Assert.*
import org.junit.Test

class TajweedAnalyzerTest {

    @Test
    fun analyze_emptyAudio_returnsFailure() = kotlinx.coroutines.runBlocking {
        val analyzer = TajweedAnalyzer()
        val result = analyzer.analyze(
            word = "بِسْمِ",
            rule = "None",
            fullWordDurationMs = 500,
            fullAudioSlice = FloatArray(0),
            riwayah = "Hafs",
            startMs = 0,
            endMs = 500
        )
        assertFalse("Should fail for empty audio", result.isCorrect)
        assertEquals("Missing audio segment", result.debugLog)
    }

    @Test
    fun analyze_extremelyShortSegment_returnsFailure() = kotlinx.coroutines.runBlocking {
        val analyzer = TajweedAnalyzer()
        // Audio segment < 256 samples
        val result = analyzer.analyze(
            word = "بِسْمِ",
            rule = "None",
            fullWordDurationMs = 10,
            fullAudioSlice = FloatArray(100) { 0.1f },
            riwayah = "Hafs",
            startMs = 0,
            endMs = 10
        )
        assertFalse("Should fail for extremely short segment", result.isCorrect)
    }

    @Test
    fun analyze_invalidTimestamps_handlesGracefully() = kotlinx.coroutines.runBlocking {
        val analyzer = TajweedAnalyzer()
        // Negative duration, startMs > endMs
        val mockValidAudio = FloatArray(16000) { kotlin.math.sin(it.toDouble() * 0.1).toFloat() }
        var result: TajweedAnalyzer.AnalysisResult? = null
        try {
            result = analyzer.analyze(
                word = "الرَّحِيْمِ",
                rule = "Madd (Vowel Extension)",
                fullWordDurationMs = 500,
                fullAudioSlice = mockValidAudio,
                riwayah = "Hafs",
                startMs = 500,
                endMs = 100 // Invalid End
            )
        } catch (e: Exception) {
            e.printStackTrace()
            throw e
        }
        assertNotNull(result)
        // Should adjust to at least 1ms and run
    }

    @Test
    fun analyze_silenceOnlyAudio_handlesGracefully() = kotlinx.coroutines.runBlocking {
        val analyzer = TajweedAnalyzer()
        // Silence only
        var result: TajweedAnalyzer.AnalysisResult? = null
        try {
            result = analyzer.analyze(
                word = "الرَّحِيْمِ",
                rule = "Madd (Vowel Extension)",
                fullWordDurationMs = 500,
                fullAudioSlice = FloatArray(16000) { 0f },
                riwayah = "Hafs",
                startMs = 0,
                endMs = 500
            )
        } catch (e: Exception) {
            e.printStackTrace()
            throw e
        }
        // Because of trim silence, it might become empty, which returns AnalysisResult failure
        assertNotNull(result)
        // Actually, normalization on silence returns silence. Trim silence returns silence.
        // Similarity with target will be computed.
    }

    @Test
    fun analyze_nanFeatureVectors_handlesGracefully() = kotlinx.coroutines.runBlocking {
        val analyzer = TajweedAnalyzer()
        // NaN only
        var result: TajweedAnalyzer.AnalysisResult? = null
        try {
            result = analyzer.analyze(
                word = "الرَّحِيْمِ",
                rule = "Madd (Vowel Extension)",
                fullWordDurationMs = 500,
                fullAudioSlice = FloatArray(16000) { Float.NaN },
                riwayah = "Hafs",
                startMs = 0,
                endMs = 500
            )
        } catch (e: Exception) {
            e.printStackTrace()
            throw e
        }
        assertNotNull(result)
        // Similarity score may become NaN which is coerced to 0f
        assertEquals(0f, result.confidenceScore)
    }
}
