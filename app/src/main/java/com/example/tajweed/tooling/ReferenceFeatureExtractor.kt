package com.example.tajweed.tooling

import com.example.tajweed.AcousticFeatures
import com.example.tajweed.SegmentExtractor
import org.json.JSONArray
import org.json.JSONObject

/**
 * Tooling to generate the offline reference feature database from authentic audio.
 * This script extracts features from real Hafs and Qalun recitations.
 */
object ReferenceFeatureExtractor {

    suspend fun generateFeaturesForDatabase(audioSegments: Map<String, FloatArray>, outputPath: String) {
        val rootJson = JSONObject()
        val features = AcousticFeatures()

        val hafsJson = JSONObject()
        val qalunJson = JSONObject()

        for ((key, audio) in audioSegments) {
            // key format: "Hafs_بِسْمِ_GHUNNAH"
            val parts = key.split("_")
            val riwayah = parts[0]
            val word = parts[1]
            val rule = parts[2]

            // 1. Extract MFCC
            val mfcc = features.extractMFCC(audio)
            
            // 2. Extract RMS
            val rms = features.extractRMSContour(audio)
            
            // 3. Extract Centroids
            val centroid = features.extractSpectralCentroid(audio)

            // Convert MFCC to JSON Array
            val mfccArray = JSONArray()
            for (frame in mfcc) {
                val frameArray = JSONArray()
                for (value in frame) {
                    frameArray.put(value)
                }
                mfccArray.put(frameArray)
            }
            
            // Convert RMS
            val rmsArray = JSONArray()
            for (value in rms) {
                rmsArray.put(value.toDouble())
            }
            
            // We can store actual features as JSON for the model
            val ruleObj = JSONObject()
            ruleObj.put("mfcc", mfccArray)
            ruleObj.put("rms", rmsArray)
            ruleObj.put("centroid", centroid.toDouble())
            
            val wordRuleKey = "${word}_${rule}"
            
            if (riwayah == "Hafs") {
                hafsJson.put(wordRuleKey, ruleObj)
            } else {
                qalunJson.put(wordRuleKey, ruleObj)
            }
        }

        rootJson.put("Hafs", hafsJson)
        rootJson.put("Qalun", qalunJson)

        val jsonString = rootJson.toString(4)
        println("Generated JSON Asset")
        
        try {
            java.io.File(outputPath).parentFile?.mkdirs()
            java.io.File(outputPath).writeText(jsonString)
            println("Wrote JSON to $outputPath")
        } catch (e: Exception) {
            println("Failed to write to $outputPath: ${e.message}")
        }
    }
}
