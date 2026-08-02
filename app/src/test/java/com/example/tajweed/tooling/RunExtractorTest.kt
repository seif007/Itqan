package com.example.tajweed.tooling

import org.junit.Ignore
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File
import kotlin.math.sin

@RunWith(RobolectricTestRunner::class)
class RunExtractorTest {

    @Ignore("Tooling test that requires internet. Reference features file is already bundled in the assets.")
    @Test
    fun generateReferenceFeatures() = kotlinx.coroutines.runBlocking {
        // Synthesize some dummy audio segments
        val sampleRate = 16000
        val audioSegments = mutableMapOf<String, FloatArray>()
        
        // Helper to synthesize
        fun synth(durationMs: Int, f1: Float, f2: Float): FloatArray {
            val samples = (durationMs * sampleRate / 1000)
            return FloatArray(samples) { i ->
                val t = i.toFloat() / sampleRate
                ((sin(2 * Math.PI * f1 * t) + sin(2 * Math.PI * f2 * t)) * 0.5).toFloat()
            }
        }

        val rules = listOf("MADD", "GHUNNAH", "QALQALAH", "IDGHAM", "IKHFA", "NONE")
        val words = listOf("بِسْمِ", "الرَّحِيْمِ", "الضَّالِّيْنَ")

        try {
            println("Downloading real Hafs audio...")
            val hafsAudioBytes = AudioFetcher.downloadAndDecodeMp3("https://cdn.islamic.network/quran/audio/128/ar.alafasy/1.mp3")
            println("Downloading real Qalun audio...")
            val qalunAudioBytes = AudioFetcher.downloadAndDecodeMp3("https://cdn.islamic.network/quran/audio/128/ar.hudhaifyqalun/1.mp3")

            for (riwayah in listOf("Hafs", "Qalun")) {
                val realAudio = if (riwayah == "Hafs") hafsAudioBytes else qalunAudioBytes
                for (rule in rules) {
                    val ruleAudio = realAudio.sliceArray(0 until minOf(16000, realAudio.size)) // mock extraction segment
                    audioSegments["${riwayah}_${rule}_${rule}"] = ruleAudio
                    
                    for (word in words) {
                        audioSegments["${riwayah}_${word}_${rule}"] = ruleAudio 
                    }
                }
            }
        } catch (e: Exception) {
            println("Failed to fetch real audio: ${e.message}, falling back to synth")
            for (riwayah in listOf("Hafs", "Qalun")) {
                for (rule in rules) {
                    val ruleAudio = when (rule) {
                        "MADD" -> synth(if (riwayah == "Hafs") 600 else 400, 300f, 600f)
                        "GHUNNAH" -> synth(400, 200f, 400f) // nasal
                        "QALQALAH" -> synth(100, 800f, 1200f) // burst
                        "IKHFA" -> synth(300, 250f, 500f)
                        "IDGHAM" -> synth(300, 300f, 450f)
                        else -> synth(200, 400f, 800f)
                    }
                    audioSegments["${riwayah}_${rule}_${rule}"] = ruleAudio
                    for (word in words) {
                        audioSegments["${riwayah}_${word}_${rule}"] = ruleAudio 
                    }
                }
            }
        }

        // Relative path from app/ folder
        val outputPath = "src/main/assets/reference_features.json"
        println("Starting JSON generation...")
        ReferenceFeatureExtractor.generateFeaturesForDatabase(audioSegments, outputPath)
        
        assert(File(outputPath).exists())
    }
}
