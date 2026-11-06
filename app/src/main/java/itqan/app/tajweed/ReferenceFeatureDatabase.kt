package itqan.app.tajweed

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.InputStreamReader
import java.util.concurrent.ConcurrentHashMap
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.Dispatchers

/**
 * In-memory store of reference acoustic templates, keyed by `${riwayah}_${word}_${rule}`.
 *
 * Two JSON shapes are supported (auto-detected) so the existing single-reciter asset keeps
 * working while the calibration pipeline can emit **multi-reciter** data:
 *   - Legacy single template:  "Hafs_xxx_GHUNNAH": { "mfcc":[...], "rms":[...], "centroid": 800 }
 *   - Multi-reciter:         "Hafs_xxx_GHUNNAH": { "reciter1": {...}, "reciter2": {...} }
 *
 * [getFeatureSet] returns every reciter's template for a key; the analyzer scores the user's
 * voice against the **best matching reciter** (min DTW distance), which removes single-voice bias.
 */
object ReferenceFeatureDatabase {

    data class ReferenceFeature(
        val mfcc: List<FloatArray> = emptyList(),
        val rms: List<Float> = emptyList(),
        val centroid: Float = 0f
    )

    private val singleCache = ConcurrentHashMap<String, ReferenceFeature>()
    private val setCache = ConcurrentHashMap<String, List<ReferenceFeature>>()
    // CMVN-normalised reference MFCCs, precomputed once at load so the per-rule
    // DTW loop only normalises the user's voice a single time.
    private val normalizedSetCache = ConcurrentHashMap<String, List<List<FloatArray>>>()
    @Volatile
    private var isPreloaded = false
    private val mutex = Mutex()
    private val sim = SimilarityEngine()

    suspend fun ensurePreloaded(context: Context?) {
        if (isPreloaded || context == null) return
        mutex.withLock {
            if (isPreloaded) return
            withContext(Dispatchers.IO) {
                try {
                    context.assets.open("reference_features.json").use { inputStream ->
                        val root = JSONObject(InputStreamReader(inputStream).readText())
                        val riwayahs = root.keys()
                        while (riwayahs.hasNext()) {
                            val riwayah = riwayahs.next()
                            val riwayahObj = root.optJSONObject(riwayah) ?: continue
                            val keys = riwayahObj.keys()
                            while (keys.hasNext()) {
                                val keyName = keys.next()
                                val value = riwayahObj.opt(keyName)
                                val cacheKey = "${riwayah}_${keyName}"
                                when {
                                    value is JSONObject && value.has("mfcc") -> {
                                        val f = parseFeature(value)
                                        singleCache[cacheKey] = f
                                        setCache[cacheKey] = listOf(f)
                                        normalizedSetCache[cacheKey] = listOf(sim.applyCMVN(f.mfcc))
                                    }
                                    value is JSONObject -> {
                                        val list = mutableListOf<ReferenceFeature>()
                                        val rk = value.keys()
                                        while (rk.hasNext()) {
                                            val rid = rk.next()
                                            val rf = value.optJSONObject(rid)
                                            if (rf != null) list.add(parseFeature(rf))
                                        }
                                        if (list.isNotEmpty()) {
                                            singleCache[cacheKey] = list.first()
                                            setCache[cacheKey] = list
                                            normalizedSetCache[cacheKey] = list.map { sim.applyCMVN(it.mfcc) }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    isPreloaded = true
                } catch (e: Exception) {
                    Log.e("RefFeatureDB", "Failed to preload reference features: ${e.message}", e)
                }
            }
        }
    }

    private fun parseFeature(obj: JSONObject): ReferenceFeature {
        val mfccArr = obj.optJSONArray("mfcc") ?: JSONArray()
        val mfcc = mutableListOf<FloatArray>()
        for (i in 0 until mfccArr.length()) {
            val frameArr = mfccArr.optJSONArray(i) ?: continue
            val fa = FloatArray(frameArr.length())
            for (j in 0 until frameArr.length()) fa[j] = frameArr.optDouble(j, 0.0).toFloat()
            mfcc.add(fa)
        }
        val rmsArr = obj.optJSONArray("rms")
        val rms = mutableListOf<Float>()
        if (rmsArr != null) {
            for (i in 0 until rmsArr.length()) rms.add(rmsArr.optDouble(i, 0.0).toFloat())
        }
        val centroid = obj.optDouble("centroid", 0.0).toFloat()
        return ReferenceFeature(mfcc, rms, centroid)
    }

    /** All reciter templates for a (word, rule, riwayah); empty if none. */
    suspend fun getFeatureSet(word: String, rule: String, riwayah: String, context: Context?): List<ReferenceFeature> {
        ensurePreloaded(context)
        val cleanRule = TajweedRule.fromString(rule).name
        val candidates = listOf(
            "${riwayah}_${word}_${cleanRule}",
            "${riwayah}__${cleanRule}",
            "${riwayah}_${cleanRule}_${cleanRule}",
            "Hafs_${word}_${cleanRule}",
            "Hafs__${cleanRule}",
            "Hafs_${cleanRule}_${cleanRule}"
        )
        for (key in candidates) {
            setCache[key]?.let { if (it.isNotEmpty()) return it }
        }
        return emptyList()
    }

    suspend fun getReferenceMFCCSet(word: String, rule: String, riwayah: String, context: Context?): List<List<FloatArray>> {
        return getFeatureSet(word, rule, riwayah, context).map { it.mfcc }
    }

    /** CMVN-normalised reference templates (best-of-N) for fast DTW. */
    suspend fun getNormalizedReferenceMFCCSet(word: String, rule: String, riwayah: String, context: Context?): List<List<FloatArray>> {
        ensurePreloaded(context)
        val cleanRule = TajweedRule.fromString(rule).name
        val candidates = listOf(
            "${riwayah}_${word}_${cleanRule}",
            "${riwayah}__${cleanRule}",
            "${riwayah}_${cleanRule}_${cleanRule}",
            "Hafs_${word}_${cleanRule}",
            "Hafs__${cleanRule}",
            "Hafs_${cleanRule}_${cleanRule}"
        )
        for (key in candidates) {
            normalizedSetCache[key]?.let { if (it.isNotEmpty()) return it }
        }
        return emptyList()
    }

    suspend fun getReferenceCentroidSet(word: String, rule: String, riwayah: String, context: Context?): List<Float> {
        return getFeatureSet(word, rule, riwayah, context).map { it.centroid }
    }

    // Backward-compatible single-template accessors (first reciter / fallback).
    suspend fun getReferenceMFCC(word: String, rule: String, riwayah: String, context: Context?): List<FloatArray> {
        val set = getReferenceMFCCSet(word, rule, riwayah, context)
        if (set.isNotEmpty()) return set.first()
        return generateEmptyFallbackMFCC()
    }

    suspend fun getReferenceRMS(word: String, rule: String, riwayah: String, context: Context?): List<Float> {
        return getFeatureSet(word, rule, riwayah, context).firstOrNull()?.rms ?: emptyList()
    }

    suspend fun getReferenceCentroid(word: String, rule: String, riwayah: String, context: Context?): Float {
        return getReferenceCentroidSet(word, rule, riwayah, context).firstOrNull() ?: 0f
    }

    private fun generateEmptyFallbackMFCC(): List<FloatArray> {
        val numFrames = 20
        val coeffs = 39
        val result = mutableListOf<FloatArray>()
        for (i in 0 until numFrames) {
            val frame = FloatArray(coeffs)
            frame[0] = 1f
            result.add(frame)
        }
        return result
    }
}
