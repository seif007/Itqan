package com.example.tajweed

import kotlin.math.abs

class SegmentExtractor {

    // Removed length ratio logic, purely rely on given correct sample indices derived from ms timings.
    fun isolateTajweedSegment(word: String, rule: String, fullWordAudio: FloatArray, startSample: Int, endSample: Int): FloatArray {
        if (fullWordAudio.size <= 1 || word.isEmpty()) return fullWordAudio
        
        if (startSample >= fullWordAudio.size) {
            throw IllegalArgumentException("Audio indexing error: startSample ($startSample) >= audioLength (${fullWordAudio.size})")
        }
        if (endSample > fullWordAudio.size) {
            throw IllegalArgumentException("Audio indexing error: endSample ($endSample) > audioLength (${fullWordAudio.size})")
        }
        
        var safeStart = startSample.coerceAtLeast(0)
        var safeEnd = endSample.coerceAtMost(fullWordAudio.size)
        
        if (safeStart >= safeEnd) {
            throw IllegalArgumentException("Audio indexing error: startSample ($safeStart) >= endSample ($safeEnd)")
        }
        
        val totalLen = safeEnd - safeStart
        val tajweedRule = TajweedRule.fromString(rule)
        
        // Compute energy envelope for sub-word estimation
        val frameSize = 128
        val fullSegment = fullWordAudio.sliceArray(safeStart until safeEnd)
        val numFrames = fullSegment.size / frameSize
        if (numFrames < 4) return fullSegment
        
        val energies = FloatArray(numFrames)
        var maxEnergy = 0f
        for (i in 0 until numFrames) {
            var sum = 0f
            for (j in 0 until frameSize) {
                val s = fullSegment[i * frameSize + j]
                sum += s * s
            }
            energies[i] = sum / frameSize
            if (energies[i] > maxEnergy) maxEnergy = energies[i]
        }
        
        when (tajweedRule) {
            TajweedRule.MADD -> {
                // Madd is an elongated vowel (sustained high energy).
                // Find the longest continuous region where energy > 30% of maxEnergy.
                val threshold = maxEnergy * 0.3f
                var bestStart = 0
                var bestEnd = numFrames - 1
                var maxLen = 0
                
                var currentStart = -1
                for (i in 0 until numFrames) {
                    if (energies[i] > threshold) {
                        if (currentStart == -1) currentStart = i
                    } else {
                        if (currentStart != -1) {
                            val len = i - currentStart
                            if (len > maxLen) {
                                maxLen = len
                                bestStart = currentStart
                                bestEnd = i - 1
                            }
                            currentStart = -1
                        }
                    }
                }
                if (currentStart != -1) {
                    val len = numFrames - currentStart
                    if (len > maxLen) {
                        bestStart = currentStart
                        bestEnd = numFrames - 1
                    }
                }
                
                if (maxLen > 0) {
                    safeStart += bestStart * frameSize
                    safeEnd = safeStart + (bestEnd - bestStart + 1) * frameSize
                }
            }
            TajweedRule.QALQALAH -> {
                // Qalqalah is a burst at the end of the word or syllable.
                // Look for an energy peak in the last 40% of the audio.
                val searchStartIdx = (numFrames * 0.6f).toInt()
                var peakIdx = searchStartIdx
                var localMax = 0f
                for (i in searchStartIdx until numFrames) {
                    if (energies[i] > localMax) {
                        localMax = energies[i]
                        peakIdx = i
                    }
                }
                val startIdx = maxOf(0, peakIdx - 5) // roughly 5 frames before peak
                val endIdx = minOf(numFrames - 1, peakIdx + 5)
                safeStart += startIdx * frameSize
                safeEnd = safeStart + (endIdx - startIdx + 1) * frameSize
            }
            TajweedRule.GHUNNAH -> {
                // Ghunnah is a sustained nasal resonance. Usually at the start/middle, moderate energy.
                // We'll focus on the first 60% of the word if it's a prefix, but Ghunnah can be anywhere.
                // For a more general approach, we can trim the extreme low-energy ends.
                val threshold = maxEnergy * 0.15f
                var startIdx = 0
                while (startIdx < numFrames && energies[startIdx] < threshold) startIdx++
                var endIdx = numFrames - 1
                while (endIdx > startIdx && energies[endIdx] < threshold) endIdx--
                
                if (startIdx < endIdx) {
                    safeStart += startIdx * frameSize
                    safeEnd = safeStart + (endIdx - startIdx + 1) * frameSize
                }
            }
            TajweedRule.IKHFA, TajweedRule.IDGHAM -> {
                // Usually occurs between letters/words, so we keep the middle transition part.
                val midStart = (numFrames * 0.2f).toInt()
                val midEnd = (numFrames * 0.8f).toInt()
                safeStart += midStart * frameSize
                safeEnd = safeStart + (midEnd - midStart + 1) * frameSize
            }
            else -> {}
        }
        
        safeStart = safeStart.coerceIn(0, fullWordAudio.size - 1)
        safeEnd = safeEnd.coerceIn(safeStart, fullWordAudio.size)
        
        return if (safeStart < safeEnd) {
            fullWordAudio.sliceArray(safeStart until safeEnd)
        } else {
            fullWordAudio
        }
    }

    // Advanced VAD-based silence trimming instead of naive threshold
    fun trimSilence(audio: FloatArray, threshold: Float = 0.01f): FloatArray {
        if (audio.size < 256) return audio
        val frameSize = 128
        val energies = FloatArray(audio.size / frameSize)
        
        for (i in energies.indices) {
            var sum = 0f
            for (j in 0 until frameSize) {
                val idx = i * frameSize + j
                if (idx < audio.size) {
                    sum += audio[idx] * audio[idx]
                }
            }
            energies[i] = sum / frameSize
        }
        
        val maxEnergy = energies.maxOrNull() ?: 1e-10f
        val vadThreshold = maxEnergy * 0.05f // adaptive threshold relative to max energy
        
        var startFrame = 0
        while (startFrame < energies.size && energies[startFrame] < vadThreshold) {
            startFrame++
        }
        
        var endFrame = energies.size - 1
        while (endFrame > startFrame && energies[endFrame] < vadThreshold) {
            endFrame--
        }
        
        var startSample = startFrame * frameSize
        var endSample = (endFrame + 1) * frameSize
        
        // Add a small buffer on both sides
        startSample = maxOf(0, startSample - frameSize * 2)
        endSample = minOf(audio.size, endSample + frameSize * 2)
        
        return if (startSample < endSample) {
            audio.sliceArray(startSample until endSample)
        } else {
            audio
        }
    }

    fun normalizeAmplitude(audio: FloatArray): FloatArray {
        if (audio.isEmpty()) return audio
        var sumSquares = 0f
        var maxAmp = 0f
        for (sample in audio) {
            val a = abs(sample)
            if (a > maxAmp) maxAmp = a
            sumSquares += a * a
        }
        val rms = kotlin.math.sqrt(sumSquares / audio.size)
        
        if (maxAmp <= 0f || rms <= 0f) return audio
        // normalize to a target RMS of 0.1 to avoid clipping and allow comparison
        val targetRms = 0.1f
        val gain = targetRms / rms
        
        val normalized = FloatArray(audio.size)
        for (i in audio.indices) {
            normalized[i] = (audio[i] * gain).coerceIn(-1.0f, 1.0f)
        }
        return normalized
    }
}
