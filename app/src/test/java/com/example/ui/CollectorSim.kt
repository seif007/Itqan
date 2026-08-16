package com.example.ui

/**
 * Faithful JVM mirror of the Memorize collector block in RecitationApp.kt
 * (the `pipe.partials.collect` section): filterMemorizeWords with the
 * unified committed-count suppression (a) + target-count cap (b) +
 * non-endpoint trailing fragment drop (c), the freshSession wipe +
 * resume-from-mistake re-seat, and the cumulative endpoint commit.
 *
 * Shared by:
 *  - MemorizeTranscriptTrackerTest (deterministic emission-stream scenarios)
 *  - PcQuranSttEndToEndTest (REAL recitation audio replayed through the real
 *    QuranSttEngine + matcher, then through this same collector)
 *
 * If you change the collector, update this fixture and re-run
 * `gradlew testDebugUnitTest`.
 */
class CollectorSim(
    private val riwayah: String,
    private val targetNorms: List<String>
) {
    var finalSpokenText = ""
    var partialSpokenText = ""
    var committedWordCount = 0
    var lastCommittedWords: List<String> = emptyList()
    var sessionFinalized = false
    var mistakeResumeIndex = -1
    var mistakeResumeLiveText = ""

    /**
     * The filterMemorizeWords output of the LAST emission — the exact word
     * list the device's collector builds its spokenTimings from (the
     * `pipe.partials.collect` block keys timings by the cumulative filtered
     * count). Exposed for PcTajweedFalsePositiveTest; unused elsewhere.
     */
    var lastFilteredWords: List<String> = emptyList()
        private set

    val spokenText: List<String>
        get() = (finalSpokenText + " " + partialSpokenText)
            .trim()
            .split(Regex("\\s+"))
            .filter { it.isNotEmpty() }

    private fun normalize(w: String) = normalizeForComparison(w, riwayah)

    /**
     * Mirrors filterMemorizeWords: ONE unified count rule for LIVE and
     * ENDPOINT emissions alike — (a) drop the first committedCount(n)
     * occurrences of each norm (the replay copies of already-committed
     * words), (b) cap kept occurrences at targetCount(n) (anything beyond is
     * a merge/decode artifact with no target slot to align to), (c) LIVE-only:
     * drop a trailing token that isn't a complete target word (window-edge
     * cut the next decode re-renders whole).
     */
    private fun filterMemorizeWords(raw: List<String>, endpoint: Boolean): List<String> {
        if (raw.isEmpty()) return raw
        val targetCounts = HashMap<String, Int>()
        for (tw in targetNorms) {
            val tn = normalize(tw)
            if (tn.isNotEmpty()) targetCounts[tn] = (targetCounts[tn] ?: 0) + 1
        }
        val committedCounts = HashMap<String, Int>()
        for (cw in lastCommittedWords) {
            val cn = normalize(cw)
            if (cn.isNotEmpty()) committedCounts[cn] = (committedCounts[cn] ?: 0) + 1
        }
        val committedOrig = HashMap(committedCounts)
        val keptCounts = HashMap<String, Int>()
        val out = ArrayList<String>(raw.size)
        for ((i, w) in raw.withIndex()) {
            val n = normalize(w)
            if (n.isEmpty()) continue
            if (!endpoint && i == raw.lastIndex && n !in targetCounts) continue
            val committedLeft = committedCounts[n] ?: 0
            if (committedLeft > 0) {
                committedCounts[n] = committedLeft - 1
                continue
            }
            // TOTAL cap: committed + kept may never exceed the target count
            // (a cumulative emission re-walks the committed log before the
            // new tail; the tail's re-render must not re-spend the cap).
            val cap = targetCounts[n] ?: 0
            val kept = keptCounts[n] ?: 0
            if ((committedOrig[n] ?: 0) + kept >= cap) continue
            keptCounts[n] = kept + 1
            out.add(w)
        }
        return out
    }

    fun onEmission(partial: List<String>, endpoint: Boolean, freshSession: Boolean, repaired: Boolean = false) {
        if (freshSession) {
            if (lastCommittedWords.size >= targetNorms.size) {
                finalSpokenText = ""
                lastCommittedWords = emptyList()
            }
            partialSpokenText = ""
            committedWordCount = 0
            sessionFinalized = false
            if (mistakeResumeIndex >= 0 && mistakeResumeLiveText.isNotEmpty()) {
                val resumeIdx = mistakeResumeIndex
                mistakeResumeIndex = -1
                val liveWords = mistakeResumeLiveText.split(Regex("\\s+")).filter { it.isNotEmpty() }
                mistakeResumeLiveText = ""
                if (liveWords.isNotEmpty()) {
                    val prefixWords = liveWords.take(resumeIdx.coerceIn(0, liveWords.size))
                    finalSpokenText = prefixWords.joinToString(" ")
                    lastCommittedWords = prefixWords
                }
            }
        }
        val words = filterMemorizeWords(partial, endpoint)
        lastFilteredWords = words
        val wordsLength = words.size
        val partialStr = words.joinToString(" ").trim()
        if (endpoint) {
            // FINAL-DECODE REPAIR SWAP (mirrors RecitationApp): the pipeline
            // flagged this endpoint as a merge REPAIR — the final tail decode
            // re-heard the whole session and REPLACED the scrambled live log,
            // so [partial] is the authoritative transcript in reading order.
            // Overwrite the committed transcript with it.
            if (repaired && partial.isNotEmpty()) {
                finalSpokenText = partial.joinToString(" ").trim()
                partialSpokenText = ""
                committedWordCount = partial.size
                lastCommittedWords = partial
                sessionFinalized = true
                return
            }
            val commitText = partialStr.ifEmpty { partialSpokenText.trim() }
            val commitWords = commitText.split(Regex("\\s+")).filter { it.isNotEmpty() }
            if (commitWords.isNotEmpty()) {
                finalSpokenText = "$finalSpokenText ${commitWords.joinToString(" ")}".trim()
                committedWordCount = commitWords.size
                lastCommittedWords = lastCommittedWords + commitWords
            }
            if (words.isNotEmpty()) sessionFinalized = true
            partialSpokenText = ""
        } else {
            if (partialStr.isNotEmpty()) partialSpokenText = partialStr
        }
    }
}
