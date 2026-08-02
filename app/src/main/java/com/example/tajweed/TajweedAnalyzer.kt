package com.example.tajweed

import android.content.Context
import android.util.Log

class TajweedAnalyzer(private val context: Context? = null) {

    private val features = AcousticFeatures()
    private val segmentExtractor = SegmentExtractor()
    private val similarityEngine = SimilarityEngine()

    // Sub-analyzers that need no per-call config.
    private val maddAnalyzer = MaddAnalyzer(features)
    private val allamAnalyzer = AllamAnalyzer(features)
    private val meemAnalyzer = MeemSakinahAnalyzer(features)
    private val silahAnalyzer = SilahAnalyzer(features)
    private val izharAnalyzer = IzharAnalyzer(features)

    data class AnalysisResult(
        val isCorrect: Boolean,
        val confidenceScore: Float,
        val feedback: String,
        val debugLog: String,
        val confidenceCutoff: Float = 0.5f,
        // Introspection fields used by the offline calibration runner.
        val mfccDistance: Float = 0f,
        val mfccSimilarity: Float = 0f,
        val durScore: Float = 0f,
        val extraScore: Float = 0f,
        val actualRatio: Float = 0f
    )

    private fun sigmoid(x: Float): Float {
        return if (x >= 0f) {
            val e = kotlin.math.exp(-x.toDouble()).toFloat()
            1f / (1f + e)
        } else {
            val e = kotlin.math.exp(x.toDouble()).toFloat()
            e / (1f + e)
        }
    }

    /**
     * [referenceMfccsOverride] / [referenceCentroidsOverride] let the calibration runner
     * inject real multi-reciter templates directly (so no Android assets are needed in a
     * JVM test). When null, templates are loaded from [ReferenceFeatureDatabase] (best-of-N).
     */
    suspend fun analyze(
        word: String,
        rule: String,
        fullWordDurationMs: Long,
        fullAudioSlice: FloatArray? = null,
        riwayah: String = "Hafs",
        startMs: Long = 0L,
        endMs: Long = fullWordDurationMs,
        referenceMfccsOverride: List<List<FloatArray>>? = null,
        referenceCentroidsOverride: List<Float>? = null
    ): AnalysisResult {
        val tajweedRule = TajweedRule.fromString(rule)
        if (word.isEmpty()) {
            return AnalysisResult(false, 0f, "Missing word text", "Invalid input word")
        }

        val segmentDurationMs = (endMs - startMs).coerceAtLeast(1L)

        var safeAudio = fullAudioSlice ?: FloatArray(0)

        if (safeAudio.size < 256) {
            return AnalysisResult(false, 0f, "Audio unavailable", "Missing audio segment")
        }

        val audioLength = safeAudio.size
        val expectedWordSamples = ((endMs - startMs) * features.sampleRate / 1000).toInt()

        val isWordBuffer = audioLength < expectedWordSamples + features.sampleRate || (startMs * features.sampleRate / 1000) >= audioLength

        var startSample: Int
        var endSample: Int

        if (isWordBuffer) {
            startSample = 0
            endSample = audioLength
        } else {
            startSample = (startMs * features.sampleRate / 1000).toInt()
            endSample = (endMs * features.sampleRate / 1000).toInt()
        }

        var audioSegment: FloatArray
        try {
            audioSegment = segmentExtractor.isolateTajweedSegment(word, rule, safeAudio, startSample, endSample)
        } catch (e: IllegalArgumentException) {
            return AnalysisResult(false, 0f, "Audio alignment error", e.message ?: "Invalid audio indices")
        }

        if (audioSegment.size < 256) {
            return AnalysisResult(false, 0f, "Audio too short", "Extracted segment shorter than one analysis window: ${audioSegment.size}")
        }

        audioSegment = segmentExtractor.trimSilence(audioSegment)
        audioSegment = segmentExtractor.normalizeAmplitude(audioSegment)

        // True length of the *isolated* tajweed segment (e.g. the sustained madd vowel) in
        // milliseconds. This — not the input word window — is what the duration score and
        // gate must measure; otherwise a rushed/abbreviated recitation is never caught.
        val isolatedDurationMs = (audioSegment.size.toLong() * 1000L / features.sampleRate).coerceAtLeast(1L)

        // --- Reference templates (best-of-N across reciters) ---
        // The calibration runner injects raw reference MFCCs (referenceMfccsOverride);
        // the on-device path uses CMVN-normalised references precomputed at load, and
        // normalises the user's voice only ONCE before scoring against every template
        // (this is the main latency win — previously CMVN re-ran per template).
        val centroidSet = referenceCentroidsOverride
            ?: ReferenceFeatureDatabase.getReferenceCentroidSet(word, tajweedRule.name, riwayah, context)

        val config = CalibrationDatabase.getConfig(tajweedRule, riwayah, context)

        val userMfcc = features.extractMFCC(audioSegment)

        val (mfccDistance, targetCentroid) = if (referenceMfccsOverride != null) {
            // Runner / test path: raw references, normalise per call.
            var minD = Float.MAX_VALUE
            var bestC = 0f
            for (i in referenceMfccsOverride.indices) {
                val d = similarityEngine.calculateDTWDistance(referenceMfccsOverride[i], userMfcc)
                if (d < minD) { minD = d; bestC = centroidSet.getOrElse(i) { 0f } }
            }
            minD to bestC
        } else {
            val normSet = ReferenceFeatureDatabase.getNormalizedReferenceMFCCSet(
                word, tajweedRule.name, riwayah, context)
            if (normSet.isNotEmpty()) {
                val userNorm = similarityEngine.applyCMVN(userMfcc)
                var minD = Float.MAX_VALUE
                var bestC = 0f
                for (i in normSet.indices) {
                    val d = similarityEngine.rawDTWDistance(normSet[i], userNorm)
                    if (d < minD) { minD = d; bestC = centroidSet.getOrElse(i) { 0f } }
                }
                minD to bestC
            } else {
                val single = ReferenceFeatureDatabase.getReferenceMFCC(word, tajweedRule.name, riwayah, context)
                similarityEngine.calculateDTWDistance(single, userMfcc) to 0f
            }
        }

        val mfccSimilarity = similarityEngine.scoreSimilarity(mfccDistance, config.dtwThreshold)

        val centroid = features.extractSpectralCentroid(audioSegment)
        val userRmsList = features.extractRMSContour(audioSegment).toList()
        val rmsAverage = if (userRmsList.isNotEmpty()) userRmsList.average().toFloat() else 0f
        val peakAmplitude = audioSegment.maxOfOrNull { Math.abs(it).toDouble() } ?: 0f

        // Duration ratio: length of the *isolated* segment relative to the full word.
        val actualRatio = if (fullWordDurationMs > 0) isolatedDurationMs.toFloat() / fullWordDurationMs else 0f

        // Time-focused duration score. For time-critical rules (e.g. MADD must be
        // prolonged 4-6 counts) being *too short* is the key failure mode, so we
        // penalise shortness much more steeply than a linear curve would: below the
        // calibrated target length the score falls off quadratically, while meeting or
        // exceeding the target still earns full credit.
        val durRatio = if (config.targetRatio > 0f) actualRatio / config.targetRatio else 0f
        val durScore = if (durRatio >= 1f) 1f else durRatio * durRatio

        // Centroid similarity vs best-matching reciter.
        val centroidDiff = if (targetCentroid > 0) Math.abs(centroid - targetCentroid) / targetCentroid else 0f
        val centroidSimilarity = (1f - centroidDiff).coerceIn(0f, 1f)

        // Per-rule extra acoustic cue.
        val extraScore: Float
        val extraName: String

        when (tajweedRule) {
            TajweedRule.GHUNNAH -> {
                val a = GhunnahAnalyzer(features, config.ghunnah).analyze(audioSegment, segmentDurationMs)
                extraScore = a; extraName = "Nasal"
            }
            TajweedRule.QALQALAH -> {
                val a = QalqalahAnalyzer(features, config.qalqalah).analyze(audioSegment)
                extraScore = a; extraName = "Burst"
            }
            TajweedRule.MADD -> {
                extraScore = maddAnalyzer.analyzeEnergy(audioSegment); extraName = "Energy"
            }
            TajweedRule.ALLAM -> {
                extraScore = allamAnalyzer.analyze(audioSegment); extraName = "Lateral"
            }
            TajweedRule.MEEM_SAKINAH -> {
                extraScore = meemAnalyzer.analyze(audioSegment, segmentDurationMs); extraName = "Labial"
            }
            TajweedRule.SILAH -> {
                extraScore = silahAnalyzer.analyze(audioSegment); extraName = "Continuity"
            }
            TajweedRule.IZHAR -> {
                extraScore = izharAnalyzer.analyze(audioSegment); extraName = "Clarity"
            }
            TajweedRule.IDGHAM -> {
                extraScore = NasalTransitionAnalyzer(features).analyze(audioSegment, config.nasal); extraName = "Nasal"
            }
            TajweedRule.IKHFA -> {
                extraScore = NasalTransitionAnalyzer(features).analyze(audioSegment, config.nasal); extraName = "Nasal"
            }
            else -> {
                extraScore = 0f; extraName = "None"
            }
        }

        // Scoring: when the data-driven calibration supplied a `bias`, the three
        // cues feed a logistic model p = sigmoid(bias + wMfcc*mfccSim + wDuration*
        // durScore + wExtra*extraScore), giving a calibrated probability. The legacy
        // additive weighted-sum (clamped) is kept when bias == 0 for backward
        // compatibility with hand-tuned calibration.json assets.
        val confidence = if (config.bias != 0f) {
            val z = config.bias +
                config.wMfcc * mfccSimilarity +
                config.wDuration * durScore +
                config.wExtra * extraScore
            sigmoid(z).coerceIn(0f, 1f)
        } else {
            (config.wMfcc * mfccSimilarity +
                config.wDuration * durScore +
                config.wExtra * extraScore).coerceIn(0f, 1f)
        }

        // Hard duration gate for time-critical rules. A recitation is forced incorrect if
        // EITHER:
        //  - the segment is proportionally shorter than the minimum required fraction of the
        //    calibrated target length (catches a madd that was not elongated enough), OR
        //  - its absolute duration is below the rule's minimum (catches a fast overall tempo
        //    that still elongates proportionally but for too few counts).
        // Timbre similarity can never rescue a too-short recitation.
        val tooShortRatio = config.durationGateRatio > 0f && durRatio < config.durationGateRatio
        val tooShortAbs = config.minDurationMs > 0f && isolatedDurationMs < config.minDurationMs
        val tooShort = tooShortRatio || tooShortAbs
        val gatedConfidence = if (tooShort) {
            val capped = config.confidenceCutoff * 0.85f
            if (confidence < capped) confidence else capped
        } else confidence

        val feedback = when {
            tooShort -> "Too short — prolong the $rule (held too briefly)"
            gatedConfidence >= config.confidenceCutoff -> "Good $rule application"
            else -> "Review your $rule"
        }
        val tooShortReason = when {
            tooShortRatio && tooShortAbs -> "TOO_SHORT(ratio+abs)"
            tooShortRatio -> "TOO_SHORT(ratio)"
            tooShortAbs -> "TOO_SHORT(abs)"
            else -> ""
        }
        val debugLog = "Rule: $rule ($riwayah) | MFCCSim: %.2f | DTW: %.2f | Dur: %.2f (ratio %.2f, iso %.0fms) | $extraName: %.2f | Conf: %.2f | Cut: %.2f%s".format(
            mfccSimilarity, mfccDistance, durScore, durRatio, isolatedDurationMs.toFloat(), extraScore, gatedConfidence, config.confidenceCutoff,
            if (tooShortReason.isNotEmpty()) " | $tooShortReason" else ""
        )

        println("Tajweed DEBUG: final confidence=$gatedConfidence (cutoff=${config.confidenceCutoff})${if (tooShortReason.isNotEmpty()) " $tooShortReason" else ""}")
        println("TajweedAcoustic: $debugLog")

        return AnalysisResult(
            isCorrect = gatedConfidence >= config.confidenceCutoff,
            confidenceScore = if (gatedConfidence.isNaN()) 0f else gatedConfidence,
            feedback = feedback,
            debugLog = debugLog,
            confidenceCutoff = config.confidenceCutoff,
            mfccDistance = mfccDistance,
            mfccSimilarity = mfccSimilarity,
            durScore = durScore,
            extraScore = extraScore,
            actualRatio = actualRatio
        )
    }
}
