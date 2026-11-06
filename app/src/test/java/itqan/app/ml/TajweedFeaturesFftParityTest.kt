package itqan.app.ml

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

/**
 * Proves the FFT-backed (Bluestein) feature pipeline is numerically identical
 * to the original naive O(N^2) DFT pipeline. The app switched to the FFT path
 * to cut the tajweed confirmation-search cost from ~33 s to well under the
 * 2-3 s budget on the A05s; this test guards that the switch changed NOTHING
 * the model sees (log-mel, dur, evidence vector).
 */
class TajweedFeaturesFftParityTest {

    /** Deterministic pseudo-speech: a few sinusoids + noise, like a word clip. */
    private fun makeClip(samples: Int, seed: Long): FloatArray {
        var s = seed
        fun rnd(): Float {
            s = s * 6364136223846793005L + 1442695040888963407L
            return ((s ushr 33).toFloat() / Int.MAX_VALUE.toFloat()) - 1f
        }
        return FloatArray(samples) { i ->
            val t = i / 16000.0
            (0.4 * sin(2 * PI * 220 * t) +
                0.25 * sin(2 * PI * 880 * t) +
                0.15 * sin(2 * PI * 2400 * t) +
                0.2 * rnd()).toFloat()
        }
    }

    @Test
    fun fftFeatures_matchNaiveFeatures() {
        // Cover the clip lengths the app actually feeds: short word spans
        // (~0.6 s), the 2 s cross-word search window, and the 200-frame cap.
        for ((samples, seed) in listOf(9600 to 1L, 16000 to 2L, 32000 to 3L, 48000 to 4L)) {
            val clip = makeClip(samples, seed)
            val durSec = samples / 16000f
            val fft = TajweedFeatures.extract(clip, durSec)
            val naive = TajweedFeatures.extractNaive(clip, durSec)

            assertEquals("dur mismatch @$samples", naive.dur, fft.dur, 1e-6f)

            assertEquals("ev size @$samples", naive.ev.size, fft.ev.size)
            for (i in naive.ev.indices) {
                assertEquals("ev[$i] @$samples", naive.ev[i], fft.ev[i], 1e-4f)
            }

            assertEquals("logmel rows @$samples", naive.logmel.size, fft.logmel.size)
            var worst = 0f
            for (m in naive.logmel.indices) {
                assertEquals("logmel cols @$samples", naive.logmel[m].size, fft.logmel[m].size)
                for (t in naive.logmel[m].indices) {
                    val d = abs(naive.logmel[m][t] - fft.logmel[m][t])
                    if (d > worst) worst = d
                }
            }
            // log10 values are ~[-8, 2]; the FFT path computes the same Double
            // DFT (Bluestein) and only the final toFloat() rounding differs.
            assertTrue("logmel worst |d| too large @$samples: $worst", worst < 1e-3f)
        }
    }

    @Test
    fun fftFeatures_silenceAndTinyClips() {
        // Edge cases the app can hit: near-silence and sub-frame clips.
        for (samples in listOf(256, 400, 1000)) {
            val clip = FloatArray(samples)
            val fft = TajweedFeatures.extract(clip, samples / 16000f)
            val naive = TajweedFeatures.extractNaive(clip, samples / 16000f)
            for (m in naive.logmel.indices) {
                assertArrayEquals("silent logmel row $m @$samples",
                    naive.logmel[m], fft.logmel[m], 1e-4f)
            }
            assertArrayEquals("silent ev @$samples", naive.ev, fft.ev, 1e-4f)
        }
    }
}
