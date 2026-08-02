package com.example.tajweed

/**
 * Al-Laam (Solar / Lunar). The MFCC best-of-N template already encodes whether the
 * lam is assimilated (solar) or clearly pronounced (lunar). This analyzer adds a
 * secondary cue: lateral ("l") resonance in ~1.5-3.5 kHz, which is present for
 * a clearly articulated lunar lam and absent for an assimilated solar one.
 */
class AllamAnalyzer(private val features: AcousticFeatures) {

    fun analyze(audioSegment: FloatArray): Float {
        if (audioSegment.size < 256) return 0.5f
        val nfft = 256
        val frame = audioSegment.sliceArray(0 until minOf(nfft, audioSegment.size))
        val mag = features.computeFFTMag(frame, nfft)
        val freqStep = features.sampleRate / nfft.toFloat()
        var lateral = 0f
        var total = 0f
        for (i in mag.indices) {
            val f = i * freqStep
            total += mag[i]
            if (f in 1500f..3500f) lateral += mag[i]
        }
        return if (total > 0f) (lateral / total).coerceIn(0f, 1f) else 0.5f
    }
}
