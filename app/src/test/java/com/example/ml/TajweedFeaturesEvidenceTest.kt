package com.example.ml

import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import kotlin.math.abs
import kotlin.math.sin

/**
 * Validates the acoustic-evidence vector (the 5-dim `ev` fed to every TajweedNet
 * head) computed from RAW audio by [TajweedFeatures.extract].
 *
 * The logits-only [TajweedNetParityTest] cannot catch feature-extraction bugs
 * because it consumes the fixture's PRE-computed logmel/dur/ev. This test feeds a
 * synthetic clip and asserts the Qalqalah hallmark cue (the interior-silence
 * closure depth, ev[2]) behaves correctly in a signal-relative way:
 *   - a clip with a sustained silent gap (a qalqalah "bounce") must yield a
 *     clearly positive closure that exceeds the closure of a flat steady vowel;
 *   - a flat steady vowel (no closure gap) must yield a near-zero closure.
 * This would have caught the previous bug where the Kotlin port used the global
 * quietest frame + any rising edge instead of the sliding-window interior
 * closure, which returned ~0 closure for every clip.
 */
@RunWith(RobolectricTestRunner::class)
class TajweedFeaturesEvidenceTest {

    private fun sinusoid(samples: Int, freq: Float, amp: Float): FloatArray =
        FloatArray(samples) { i -> (amp * sin(2 * Math.PI * freq * i / TajweedFeatures.SR)).toFloat() }

    /** A qalqalah-like clip: a voiced segment, a SUSTAINED silent closure, then a release. */
    private fun qalqalahLikeClip(): FloatArray {
        val sr = TajweedFeatures.SR
        val preMs = 220           // voiced lead-in
        val closureMs = 50        // sustained silent gap (interior) — ~50 ms
        val releaseMs = 40        // release
        val postMs = 140          // voiced tail
        val pre = sinusoid((sr * preMs / 1000), 220f, 0.6f)
        val silence = FloatArray(sr * closureMs / 1000)   // exact zeros -> a real stop
        val relN = sr * releaseMs / 1000
        val release = FloatArray(relN) { i ->
            val env = if (i < relN / 3) 1f else (1f - (i - relN / 3).toFloat() / (relN - relN / 3)).coerceIn(0f, 1f)
            (env * 0.9 * sin(2 * Math.PI * 2500.0 * i / sr)).toFloat()
        }
        val post = sinusoid((sr * postMs / 1000), 180f, 0.5f)
        return pre + silence + release + post
    }

    /** A flat steady vowel: no closure gap, no rise/fall. */
    private fun flatVowelClip(): FloatArray = sinusoid(TajweedFeatures.SR * 4 / 10, 220f, 0.5f)

    private fun closureOf(clip: FloatArray): Float {
        val feats = TajweedFeatures.extract(clip, clip.size / TajweedFeatures.SR.toFloat())
        return feats.ev[2]
    }

    @Test
    fun qalqalahClipProducesStrongerInteriorClosureThanFlatVowel() {
        val qalqalahClosure = closureOf(qalqalahLikeClip())
        val flatClosure = closureOf(flatVowelClip())
        // The OLD crude cue (global min envelope) returns ~0 for BOTH clips.
        // The corrected sliding-window interior-closure cue must return a clearly
        // positive value for the qalqalah clip AND strictly exceed the flat clip.
        assertTrue("Qalqalah closure should be clearly positive (got $qalqalahClosure)", qalqalahClosure > 0.3f)
        assertTrue("Qalqalah closure ($qalqalahClosure) should exceed flat vowel closure ($flatClosure)",
            qalqalahClosure > flatClosure + 0.15f)
    }

    @Test
    fun flatVowelClipProducesNearZeroClosure() {
        val flatClosure = closureOf(flatVowelClip())
        assertTrue("Flat vowel closure should be near zero (got $flatClosure)", abs(flatClosure) < 0.3f)
    }
}
