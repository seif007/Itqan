package com.example.ml

import android.content.Context
import org.json.JSONObject

/**
 * Orchestrates the trained TajweedNet for on-device mistake detection.
 *
 * Pipeline per word:
 *   1. Derive the expected tajweed rules from the Arabic text (TajweedRules).
 *   2. Extract acoustic features from the recoded audio clip (TajweedFeatures):
 *      log-mel (40x200) + dur + 5-dim evidence vector.
 *   3. Run TajweedNet.forward -> per-rule logits -> sigmoid probabilities.
 *   4. Report, for each expected rule, P(correct) and a mistake flag.
 *
 * The model weights are loaded ONCE per app process from
 * assets/model_weights.json and cached, so repeated per-word analysis no longer
 * re-parses the ~1 MB JSON tensor blob each call (the main source of latency
 * in the previous per-call construction). Use [get] — guaranteed cheap after
 * the first hit — rather than constructing [RecitationAnalyzer] directly.
 */
class RecitationAnalyzer private constructor(context: Context) {

    private val net: TajweedNet

    init {
        val json = context.assets.open("model_weights.json").bufferedReader().use { it.readText() }
        net = TajweedNet(JSONObject(json))
    }

    data class RuleResult(
        val rule: String,
        val prob: Float,        // P(correct) from the model head
        val isMistake: Boolean  // prob below threshold
    )

    data class WordResult(
        val word: String,
        val expectedRules: List<String>,
        val ruleResults: List<RuleResult>
    )

    companion object {
        @Volatile private var instance: RecitationAnalyzer? = null

        /**
         * Returns the process-wide analyzer, loading + parsing the weights lazily
         * on the first call (thread-safe). Subsequent calls are nearly free.
         * Pass the application context to avoid leaking an Activity.
         */
        fun get(context: Context): RecitationAnalyzer {
            instance?.let { return it }
            synchronized(this) {
                instance?.let { return it }
                val appCtx = context.applicationContext
                return RecitationAnalyzer(appCtx).also { instance = it }
            }
        }
    }

    /**
     * Analyze a single recorded word.
     * @param audio   PCM float samples at TajweedFeatures.SR (16 kHz), mono.
     * @param word    The Arabic word text.
     * @param nextWord The following word (for cross-word rules), or null.
     * @param threshold Probability below which a rule is flagged as a mistake.
     */
    fun analyzeWord(
        audio: FloatArray,
        word: String,
        nextWord: String? = null,
        threshold: Float = 0.5f
    ): WordResult {
        val expected = TajweedRules.detectRulesForWord(word, nextWord)
        val results = analyze(audio, expected, threshold)
        return WordResult(word, expected, results)
    }

    /**
     * Analyze a phrase word-by-word using neighbouring context.
     */
    fun analyzePhrase(audioWords: List<FloatArray>, phrase: String, threshold: Float = 0.5f): List<WordResult> {
        val words = TajweedRules.splitWords(phrase)
        require(audioWords.size == words.size) {
            "audioWords size (${audioWords.size}) must match words size (${words.size})"
        }
        return words.mapIndexed { i, w ->
            val next = if (i + 1 < words.size) words[i + 1] else null
            analyzeWord(audioWords[i], w, next, threshold)
        }
    }

    /**
     * Run the model over [audio] for each rule in [expectedRules].
     * If expectedRules == ["NONE"] (no rule demonstrated) we still report the
     * raw 8-rule probabilities so the UI can show the model's full read-out.
     */
    fun analyze(audio: FloatArray, expectedRules: List<String>, threshold: Float = 0.5f): List<RuleResult> {
        val durSec = audio.size / TajweedFeatures.SR.toFloat()
        val feats = TajweedFeatures.extract(audio, durSec)
        val logits = net.forward(feats.logmel, feats.dur, feats.ev)
        val probs = net.sigmoidLogits(logits)

        val targets = if (expectedRules == listOf("NONE")) TajweedFeatures.RULES else expectedRules
        return targets.map { rule ->
            val p = probs[rule] ?: 0f
            RuleResult(rule, p, p < threshold)
        }
    }

    /**
     * Force any LATER lazy initialization inside [net] to finish NOW, on whatever thread
     * calls this. The app shows a "Loading Tajweed Model" popup on the first Tajweed-tab
     * entry or on flipping Memorize→Tajweed; if we only handed back the singleton after it
     * was constructed, that popup would disappear as soon as the weights JSON was parsed —
     * but the BiGRU/conv scratch tensors and any native-side buffers only materialise on the
     * first forward() call. Which means the first real user recording then stalls for a
     * second hidden init, right when the user just pressed the mic.
     *
     * We therefore run ONE short silent clip through the full pipeline right after
     * construction. This forces every lazy allocation to flush ahead of mic-time, making
     * the popup honest about "tajweed model being different from ready-to-use". Cost is
     * a few hundred ms once per process lifetime; the result is discarded.
     */
    fun warmUp() {
        try {
            // 0.125 s of silence at 16 kHz. Small enough to be fast, plenty to exercise
            // conv stack, BiGRU, per-rule heads, sigmoid activation.
            val silent = FloatArray(TajweedFeatures.SR / 8)
            analyze(silent, TajweedFeatures.RULES.toList(), 0.5f)
        } catch (e: Throwable) {
            android.util.Log.w("RecitationAnalyzer", "warmUp failed; will defer to first real call", e)
        }
    }
}
