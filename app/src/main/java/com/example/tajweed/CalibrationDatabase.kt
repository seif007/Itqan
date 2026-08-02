package com.example.tajweed

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.InputStreamReader
import java.util.concurrent.ConcurrentHashMap
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.Dispatchers

/**
 * Holds per-rule, per-riwayah calibration constants that were previously hard-coded
 * inside [TajweedAnalyzer] and the sub-analyzers.
 *
 * Values are loaded from `assets/calibration.json` (produced by the calibration runner from a
 * multi-reciter corpus). If a rule/riwayah is missing, sensible seeded defaults are used that
 * reproduce the engine's original hand-tuned behaviour, so the app keeps working before real
 * calibration data is available.
 */
object CalibrationDatabase {

    data class GhunnahConfig(
        val expectedMinDurationMs: Float = 350f,
        val targetCentroid: Float = 800f,
        val nasalBandLowHz: Float = 200f,
        val nasalBandHighHz: Float = 800f,
        val wDuration: Float = 0.35f,
        val wNasal: Float = 0.40f,
        val wSmoothness: Float = 0.25f
    )

    data class QalqalahConfig(
        val burstDivisor: Float = 5f,
        val transientDivisor: Float = 3f,
        val wBurst: Float = 0.30f,
        val wAttack: Float = 0.30f,
        val wDecay: Float = 0.20f,
        val wTransient: Float = 0.20f
    )

    data class NasalConfig(
        // Target nasality degree (see [AcousticFeatures.extractNasality]) the rule should
        // exhibit. Calibration may override this per rule from the corpus.
        val targetNasality: Float = 0.5f,
        // How close (in nasality units) the measured degree must be to the target to score 1.
        val nasalTolerance: Float = 0.35f,
        val wNasality: Float = 0.7f,
        val wVoicing: Float = 0.3f
    )

    data class RuleCalibrationConfig(
        val dtwThreshold: Float = 0.5f,
        val confidenceCutoff: Float = 0.5f,
        val targetRatio: Float = 0.4f,
        // For time-critical rules (e.g. MADD) this is the minimum fraction of the
        // calibrated target length the segment must reach. Below it, the recitation is
        // forced incorrect (a rushed/abbreviated application is wrong). 0f disables the gate.
        val durationGateRatio: Float = 0f,
        // Absolute minimum duration (ms) of the isolated segment. Catches a fast overall
        // tempo that still elongates proportionally but for too few counts. 0f disables it.
        val minDurationMs: Float = 0f,
        val wMfcc: Float = 0.4f,
        val wDuration: Float = 0.3f,
        val wExtra: Float = 0.3f,
        // Logistic intercept learned by the calibration runner. When non-zero the
        // analyzer applies p = sigmoid(bias + wMfcc*mfccSim + wDuration*durScore +
        // wExtra*extraScore) instead of a plain clamped weighted sum. 0f preserves
        // the legacy additive behaviour for hand-tuned calibration.json assets.
        val bias: Float = 0f,
        val nasal: NasalConfig = NasalConfig(),
        val ghunnah: GhunnahConfig = GhunnahConfig(),
        val qalqalah: QalqalahConfig = QalqalahConfig()
    )

    private val memoryCache = ConcurrentHashMap<String, RuleCalibrationConfig>()
    @Volatile private var isPreloaded = false
    private val mutex = Mutex()

    suspend fun ensurePreloaded(context: Context?) {
        if (isPreloaded || context == null) return
        mutex.withLock {
            if (isPreloaded) return
            withContext(Dispatchers.IO) {
                try {
                    // Primary: data-driven thresholds produced by the calibration
                    // runner. Fall back to the legacy hand-tuned calibration.json.
                    val assetName = if (context.assets.list("")?.contains("tajweed_thresholds.json") == true)
                        "tajweed_thresholds.json" else "calibration.json"
                    context.assets.open(assetName).use { inputStream ->
                        val root = JSONObject(InputStreamReader(inputStream).readText())
                        val keys = root.keys()
                        while (keys.hasNext()) {
                            val key = keys.next()
                            val obj = root.optJSONObject(key) ?: continue
                            memoryCache[key] = parseConfig(obj)
                        }
                    }
                    isPreloaded = true
                } catch (e: Exception) {
                    Log.e("CalibDB", "Failed to preload calibration.json: ${e.message}", e)
                }
            }
        }
    }

    private fun parseConfig(obj: JSONObject): RuleCalibrationConfig {
        val gh = obj.optJSONObject("ghunnah")
        val qa = obj.optJSONObject("qalqalah")
        val n = obj.optJSONObject("nasal")
        return RuleCalibrationConfig(
            dtwThreshold = obj.optDouble("dtwThreshold", 0.5).toFloat(),
            confidenceCutoff = obj.optDouble("confidenceCutoff", 0.5).toFloat(),
            targetRatio = obj.optDouble("targetRatio", 0.4).toFloat(),
            durationGateRatio = obj.optDouble("durationGateRatio", 0.0).toFloat(),
            minDurationMs = obj.optDouble("minDurationMs", 0.0).toFloat(),
            wMfcc = obj.optDouble("wMfcc", 0.4).toFloat(),
            wDuration = obj.optDouble("wDuration", 0.3).toFloat(),
            wExtra = obj.optDouble("wExtra", 0.3).toFloat(),
            bias = obj.optDouble("bias", 0.0).toFloat(),
            ghunnah = if (gh != null) GhunnahConfig(
                expectedMinDurationMs = gh.optDouble("expectedMinDurationMs", 350.0).toFloat(),
                targetCentroid = gh.optDouble("targetCentroid", 800.0).toFloat(),
                nasalBandLowHz = gh.optDouble("nasalBandLowHz", 200.0).toFloat(),
                nasalBandHighHz = gh.optDouble("nasalBandHighHz", 800.0).toFloat(),
                wDuration = gh.optDouble("wDuration", 0.35).toFloat(),
                wNasal = gh.optDouble("wNasal", 0.40).toFloat(),
                wSmoothness = gh.optDouble("wSmoothness", 0.25).toFloat()
            ) else GhunnahConfig(),
            qalqalah = if (qa != null) QalqalahConfig(
                burstDivisor = qa.optDouble("burstDivisor", 5.0).toFloat(),
                transientDivisor = qa.optDouble("transientDivisor", 3.0).toFloat(),
                wBurst = qa.optDouble("wBurst", 0.30).toFloat(),
                wAttack = qa.optDouble("wAttack", 0.30).toFloat(),
                wDecay = qa.optDouble("wDecay", 0.20).toFloat(),
                wTransient = qa.optDouble("wTransient", 0.20).toFloat()
            ) else QalqalahConfig(),
            nasal = if (n != null) NasalConfig(
                targetNasality = n.optDouble("targetNasality", 0.5).toFloat(),
                nasalTolerance = n.optDouble("nasalTolerance", 0.35).toFloat(),
                wNasality = n.optDouble("wNasality", 0.7).toFloat(),
                wVoicing = n.optDouble("wVoicing", 0.3).toFloat()
            ) else NasalConfig()
        )
    }

    suspend fun getConfig(rule: TajweedRule, riwayah: String, context: Context?): RuleCalibrationConfig {
        ensurePreloaded(context)
        val key = "${riwayah}_${rule.name}"
        memoryCache[key]?.let { return it }
        return defaultConfig(rule, riwayah)
    }

    /** Seeded defaults that mirror the original hand-tuned literals. */
    fun defaultConfig(rule: TajweedRule, riwayah: String): RuleCalibrationConfig {
        return when (rule) {
            TajweedRule.MADD -> RuleCalibrationConfig(
                dtwThreshold = 0.5f,
                confidenceCutoff = 0.5f,
                targetRatio = if (riwayah == "Qalun") 0.35f else 0.50f,
                durationGateRatio = 0.55f,
                minDurationMs = 250f,
                wMfcc = 0.3f, wDuration = 0.5f, wExtra = 0.2f
            )
            TajweedRule.GHUNNAH -> RuleCalibrationConfig(
                dtwThreshold = 0.5f, confidenceCutoff = 0.5f, targetRatio = 0.40f,
                durationGateRatio = 0.5f,
                wMfcc = 0.35f, wDuration = 0.25f, wExtra = 0.40f
            )
            TajweedRule.QALQALAH -> RuleCalibrationConfig(
                dtwThreshold = 0.5f, confidenceCutoff = 0.5f, targetRatio = 1.0f,
                wMfcc = 0.4f, wDuration = 0.0f, wExtra = 0.6f
            )
            TajweedRule.IDGHAM -> RuleCalibrationConfig(
                dtwThreshold = 0.5f, confidenceCutoff = 0.5f, targetRatio = 0.30f,
                wMfcc = 0.5f, wDuration = 0.2f, wExtra = 0.3f,
                nasal = NasalConfig(targetNasality = 0.75f, nasalTolerance = 0.35f, wNasality = 0.7f, wVoicing = 0.3f)
            )
            TajweedRule.IKHFA -> RuleCalibrationConfig(
                dtwThreshold = 0.5f, confidenceCutoff = 0.5f, targetRatio = 0.30f,
                wMfcc = 0.4f, wDuration = 0.3f, wExtra = 0.3f,
                nasal = NasalConfig(targetNasality = 0.55f, nasalTolerance = 0.35f, wNasality = 0.7f, wVoicing = 0.3f)
            )
            TajweedRule.ALLAM -> RuleCalibrationConfig(
                dtwThreshold = 0.5f, confidenceCutoff = 0.5f, targetRatio = 0.40f,
                wMfcc = 0.7f, wDuration = 0.3f, wExtra = 0.0f
            )
            TajweedRule.MEEM_SAKINAH -> RuleCalibrationConfig(
                dtwThreshold = 0.5f, confidenceCutoff = 0.5f, targetRatio = 0.40f,
                durationGateRatio = 0.5f,
                wMfcc = 0.5f, wDuration = 0.2f, wExtra = 0.3f
            )
            TajweedRule.SILAH -> RuleCalibrationConfig(
                dtwThreshold = 0.5f, confidenceCutoff = 0.5f, targetRatio = 0.40f,
                durationGateRatio = 0.5f,
                wMfcc = 0.4f, wDuration = 0.6f, wExtra = 0.0f
            )
            TajweedRule.IZHAR -> RuleCalibrationConfig(
                dtwThreshold = 0.5f, confidenceCutoff = 0.5f, targetRatio = 0.40f,
                wMfcc = 0.6f, wDuration = 0.0f, wExtra = 0.4f
            )
            else -> RuleCalibrationConfig(
                dtwThreshold = 0.5f, confidenceCutoff = 0.5f, targetRatio = 1.0f,
                wMfcc = 1.0f, wDuration = 0.0f, wExtra = 0.0f
            )
        }
    }
}
