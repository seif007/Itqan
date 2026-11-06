package itqan.app.ui

/**
 * Faithful JVM mirror of the Memorize collector block in RecitationApp.kt
 * (the `pipe.partials.collect` section): filterMemorizeWords with the
 * unified committed-count suppression (a) + target-count cap (b) +
 * non-endpoint trailing fragment drop (c), the freshSession wipe +
 * resume-from-mistake re-seat, the CONFIRMED-MISTAKE FREEZE (non-fresh
 * emissions dropped while the resume latch is armed), the final-decode
 * repair-swap acceptance by POSITIONAL PREFIX QUALITY vs the live
 * transcript (never raw size), and the endpoint commit's overlap-strip of
 * duplicated committed replays.
 *
 * Shared by:
 *  - MemorizeTranscriptTrackerTest (deterministic emission-stream scenarios)
 *  - PcSttEndToEndTest / PcTajweedFalsePositiveTest (REAL recitation
 *    audio replayed through the real SttEngine + matcher, then through
 *    this same collector)
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

    /** Pre-normalized target sequence for positional prefix scoring (repair acceptance). */
    private val targetNormSequence: List<String> = targetNorms.map { normalize(it) }

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
        // CONFIRMED-MISTAKE FREEZE (mirrors RecitationApp): while a confirmed
        // WRONG/SKIPPED stop has armed the resume latch, every non-fresh
        // emission is dead pre-stop audio (in-flight window decodes +
        // pause()'s own final-tail decode) and MUST be dropped — committing
        // it would resurrect the flagged word as CORRECT. The fresh-session
        // event IS the resume and was consumed just above.
        if (!freshSession && mistakeResumeIndex >= 0) {
            lastFilteredWords = emptyList()
            return
        }
        val words = filterMemorizeWords(partial, endpoint)
        lastFilteredWords = words
        val wordsLength = words.size
        val partialStr = words.joinToString(" ").trim()
        if (endpoint) {
            // FINAL-DECODE REPAIR SWAP (mirrors RecitationApp): the pipeline
            // flagged this endpoint as a merge REPAIR — the final tail decode
            // re-heard the whole session and REPLACED the scrambled live log,
            // so [partial] claims to be the authoritative transcript. Accept
            // it only when it is at least as GOOD as the live transcript,
            // scored as the LONGEST LEADING RUN of words matching the target
            // sequence (positional quality), never by raw word count: a
            // monotonic-merge echo can leave the live log LONGER but
            // MISORDERED, which used to make the correct deduplicated repair
            // get rejected. On reject, commit the uncommitted live tail and
            // finalize — never shrink the transcript.
            if (repaired && partial.isNotEmpty()) {
                // LONGEST COMMON SUBSEQUENCE with the target — how many target
                // words the candidate holds IN READING ORDER. Mirrors
                // RecitationApp: the old longest-leading-run metric punished a
                // transcript missing one early word while holding the whole rest
                // (fatihah_c34: live prefix 3 beat the repair's 0 even though the
                // repair held 27 of the 29 target words in order against the
                // log's 20).
                fun targetCoverageLen(words: List<String>): Int {
                    val a = words.map { normalize(it) }
                    val b = targetNormSequence
                    val dp = IntArray(b.size + 1)
                    for (i in a.indices) {
                        var prevDiag = 0
                        for (j in 1..b.size) {
                            val tmp = dp[j]
                            dp[j] = if (a[i].isNotEmpty() && b[j - 1].isNotEmpty() && a[i] == b[j - 1]) {
                                prevDiag + 1
                            } else {
                                maxOf(dp[j], dp[j - 1])
                            }
                            prevDiag = tmp
                        }
                    }
                    return dp[b.size]
                }
                // Longest leading run: the ORDER tie-break when both candidates
                // hold the same NUMBER of target words (see RecitationApp).
                fun targetRunLen(words: List<String>): Int {
                    var i = 0
                    while (i < words.size && i < targetNormSequence.size &&
                        targetNormSequence[i].isNotEmpty() &&
                        normalize(words[i]) == targetNormSequence[i]
                    ) i++
                    return i
                }
                val liveWords = spokenText
                val livePrefix = targetCoverageLen(liveWords)
                val repairPrefix = targetCoverageLen(partial)
                val liveRun = targetRunLen(liveWords)
                val repairRun = targetRunLen(partial)
                val takeRepair = when {
                    repairPrefix != livePrefix -> repairPrefix > livePrefix
                    repairRun != liveRun -> repairRun > liveRun
                    else -> partial.size >= liveWords.size
                }
                if (takeRepair) {
                    finalSpokenText = partial.joinToString(" ").trim()
                    partialSpokenText = ""
                    committedWordCount = partial.size
                    lastCommittedWords = partial
                    sessionFinalized = true
                    return
                } else {
                    val tailWords = partialSpokenText.trim()
                        .split(Regex("\\s+")).filter { it.isNotEmpty() }
                    if (tailWords.isNotEmpty()) {
                        finalSpokenText = "$finalSpokenText ${tailWords.joinToString(" ")}".trim()
                        partialSpokenText = ""
                        committedWordCount = tailWords.size
                        lastCommittedWords = lastCommittedWords + tailWords
                    }
                    // A REJECTED REPAIR MAY STILL HOLD THE ONLY COPY OF A WORD
                    // THE LIVE LOG NEVER HEARD (mirrors RecitationApp,
                    // fatihah_c33). Rejecting the repair means "do not REPLACE
                    // the live transcript with this worse ordering" — it does
                    // NOT mean "discard everything this decode heard".
                    // Recompute the delta against the now-committed log and
                    // append whatever survives: those are words no sliding
                    // window ever rendered, and dropping them lost real speech
                    // silently (the walk ended 28/29 with no row for the last
                    // word, no popup, and a "clean" summary while the word sat
                    // greyed out on screen).
                    val recovered = filterMemorizeWords(partial, endpoint = true)
                    if (recovered.isNotEmpty()) {
                        finalSpokenText = "$finalSpokenText ${recovered.joinToString(" ")}".trim()
                        committedWordCount = recovered.size
                        lastCommittedWords = lastCommittedWords + recovered
                    }
                    sessionFinalized = true
                    return
                }
            }
            val commitText = partialStr.ifEmpty { partialSpokenText.trim() }
            var commitWords = commitText.split(Regex("\\s+")).filter { it.isNotEmpty() }
            if (commitWords.isNotEmpty()) {
                // DOUBLE-COUNT GUARD (mirrors RecitationApp): the final tail
                // decode re-hears the whole session, so its merged log can
                // RE-EMIT words a previous endpoint already committed — the
                // count-filter strips one copy per committed word, but when
                // the merge duplicated them a replay survives in the delta.
                // Strip any leading run of commitWords that duplicates the
                // tail of the already-committed transcript.
                val existing = finalSpokenText.split(Regex("\\s+")).filter { it.isNotEmpty() }
                if (existing.isNotEmpty()) {
                    val exNorm = existing.map { normalize(it) }
                    val cmNorm = commitWords.map { normalize(it) }
                    var overlap = 0
                    val maxOv = minOf(exNorm.size, cmNorm.size)
                    for (k in maxOv downTo 1) {
                        var match = true
                        for (i in 0 until k) {
                            if (exNorm[exNorm.size - k + i] != cmNorm[i]) { match = false; break }
                        }
                        if (match) { overlap = k; break }
                    }
                    if (overlap > 0) commitWords = commitWords.drop(overlap)
                }
                if (commitWords.isNotEmpty()) {
                    finalSpokenText = "$finalSpokenText ${commitWords.joinToString(" ")}".trim()
                    committedWordCount = commitWords.size
                    lastCommittedWords = lastCommittedWords + commitWords
                }
            }
            if (wordsLength > 0) sessionFinalized = true
            partialSpokenText = ""
        } else {
            if (partialStr.isNotEmpty()) partialSpokenText = partialStr
        }
    }
}
