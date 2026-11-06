package itqan.app.tajweed

/**
 * Nasal-transition analyzer for the noon/meem saakinah rules that MFCC alone confuses:
 *   - IDGHAM   : the consonant is *merged* into the following letter (strong nasality /
 *               ghunnah for the noon+meem/ya/waw cases).
 *   - IKHFA    : the consonant is *concealed* — a partial nasalisation between clear and merged.
 *   - IZHAR    : the consonant is pronounced clearly with (effectively) no nasality.
 *
 * Because the phonemes are identical, only the *degree of nasalisation* separates them.
 * This analyzer measures the segment's nasality degree ([AcousticFeatures.extractNasality])
 * and scores how close it is to the rule's expected target, gated by voicing/HNR so that
 * silence or noise cannot fake a high nasal score.
 */
class NasalTransitionAnalyzer(private val features: AcousticFeatures) {

    fun analyze(audioSegment: FloatArray, config: CalibrationDatabase.NasalConfig): Float {
        if (audioSegment.size < 256) return 0.5f

        val measured = features.extractNasality(audioSegment)
        val (_, voiced) = features.extractF0(audioSegment)
        val hnr = features.extractHNR(audioSegment)

        // Gaussian closeness of the measured nasality to the rule's target.
        val nasalityScore = closeness(measured, config.targetNasality, config.nasalTolerance)

        // Nasalisation is a voiced phenomenon. Down-weight if the segment is unvoiced /
        // breathy (a silent or noisy clip must not score high on "nasal").
        val voiceGate = (0.5f * voiced + 0.5f * hnr).coerceIn(0f, 1f)
        val voiceFactor = 0.4f + 0.6f * voiceGate

        return (nasalityScore * voiceFactor).coerceIn(0f, 1f)
    }

    private fun closeness(a: Float, b: Float, tol: Float): Float {
        val t = if (tol <= 0f) 0.35f else tol
        val d = (a - b) / t
        return kotlin.math.exp(-0.5f * d * d).coerceIn(0f, 1f)
    }
}
